from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from urbanflow.ingest import REQUIRED, validate_months
from urbanflow.io import sha256, write_json


def normalize_pickups(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    """TLC timestamps are NYC wall time. Refuse ambiguous fall-back observations."""
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    zone = pd.to_numeric(frame["PULocationID"], errors="coerce")
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    valid = pickup.ge(start) & pickup.lt(end) & zone.between(1, 263) & zone.eq(zone.round())
    local = pickup.loc[valid].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    if local.isna().any():
        disambiguated = pickup.loc[valid].dt.tz_localize(
            "America/New_York", ambiguous=True, nonexistent="NaT"
        )
        if (local.isna() & disambiguated.notna()).any():
            raise ValueError(
                "Ambiguous NYC fall-back timestamps cannot be assigned to UTC reliably; "
                "use months without the autumn DST transition"
            )
    result = pd.DataFrame(
        {
            "hour": local.dt.tz_convert("UTC").dt.floor("h").dt.tz_localize(None),
            "zone_id": zone.loc[valid].astype("int32"),
        }
    ).dropna()
    return result


def arrow_counts(path: Path, month: str) -> tuple[pd.DataFrame, dict]:
    chunks = []
    total = valid = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250_000, columns=REQUIRED):
        frame = batch.to_pandas()
        clean = normalize_pickups(frame, month)
        total += len(frame)
        valid += len(clean)
        chunks.append(clean.groupby(["hour", "zone_id"]).size().rename("demand"))
    if not chunks:
        raise ValueError(f"No rows in {path}")
    counts = pd.concat(chunks).groupby(level=[0, 1]).sum().reset_index()
    return counts, {
        "month": month,
        "raw_rows": total,
        "valid_rows": valid,
        "rejected_rows": total - valid,
    }


def spark_counts(paths: list[Path], months: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Native Spark SQL implements the Arrow eligibility/timezone contract."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.appName("UrbanFlow-Bronze-Silver")
        .master("local[2]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    frames, quality = [], []
    try:
        for path, month in zip(paths, months, strict=True):
            source = spark.read.parquet(str(path.resolve())).select(*REQUIRED)

            # Source wall-time values are initially parsed in a UTC session. Conversion
            # below explicitly interprets those lexical values as NYC local time.
            pickup = F.col("tpep_pickup_datetime").cast("timestamp")
            zone = F.col("PULocationID").cast("double")
            start = pd.Timestamp(f"{month}-01")
            end = start + pd.offsets.MonthBegin(1)
            valid = source.filter(
                (pickup >= F.lit(str(start)).cast("timestamp"))
                & (pickup < F.lit(str(end)).cast("timestamp"))
                & zone.between(1, 263)
                & (zone == F.floor(zone))
            )
            utc = F.to_utc_timestamp(pickup, "America/New_York")
            ambiguous = (
                F.from_utc_timestamp(utc + F.expr("INTERVAL 1 HOUR"), "America/New_York") == pickup
            ) | (
                F.from_utc_timestamp(utc - F.expr("INTERVAL 1 HOUR"), "America/New_York") == pickup
            )
            if valid.filter(ambiguous).limit(1).count():
                raise ValueError(
                    "Ambiguous NYC fall-back timestamps cannot be assigned to UTC "
                    "reliably; use months without the autumn DST transition"
                )
            # A round trip rejects nonexistent spring times that Java otherwise shifts.
            clean = (
                valid.filter(F.from_utc_timestamp(utc, "America/New_York") == pickup)
                .select(F.date_trunc("hour", utc).alias("hour"), zone.cast("int").alias("zone_id"))
                .cache()
            )
            raw_rows, valid_rows = source.count(), clean.count()
            frames.append(
                clean.groupBy("hour", "zone_id").agg(F.count("*").alias("demand")).toPandas()
            )
            quality.append(
                {
                    "month": month,
                    "raw_rows": raw_rows,
                    "valid_rows": valid_rows,
                    "rejected_rows": raw_rows - valid_rows,
                }
            )
            clean.unpersist()
    finally:
        spark.stop()
    return pd.concat(frames, ignore_index=True), quality


def build_panel(counts: pd.DataFrame, months: list[str], zones: list[int]) -> pd.DataFrame:
    months = validate_months(months)
    if not zones or len(zones) != len(set(zones)) or any(z < 1 or z > 263 for z in zones):
        raise ValueError("Zones must be unique integers from 1 to 263")
    start = pd.Timestamp(f"{months[0]}-01", tz="America/New_York").tz_convert("UTC")
    end = (
        (pd.Timestamp(f"{months[-1]}-01") + pd.offsets.MonthBegin(1))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    hours = pd.date_range(start, end, freq="h", inclusive="left").tz_localize(None)
    grid = pd.MultiIndex.from_product([sorted(zones), hours], names=["zone_id", "hour"])
    panel = counts.set_index(["zone_id", "hour"])["demand"].reindex(grid, fill_value=0)
    if panel.isna().any() or (panel < 0).any():
        raise ValueError("Invalid demand counts")
    return panel.astype(np.int32).reset_index().sort_values(["zone_id", "hour"])


def process(
    months: list[str],
    raw: Path,
    output: Path,
    engine: str = "arrow",
    zones: list[int] | None = None,
) -> dict:
    months = validate_months(months)
    output.mkdir(parents=True, exist_ok=True)
    paths = [raw / f"yellow_tripdata_{m}.parquet" for m in months]
    if engine == "spark":
        counts, quality = spark_counts(paths, months)
    elif engine == "arrow":
        pairs = [arrow_counts(path, month) for path, month in zip(paths, months, strict=True)]
        counts = pd.concat([pair[0] for pair in pairs], ignore_index=True)
        quality = [pair[1] for pair in pairs]
    else:
        raise ValueError("engine must be arrow or spark")
    panel = build_panel(counts, months, zones or list(range(1, 264)))
    destination = output / "hourly.parquet"
    panel.to_parquet(destination, index=False)
    report = {
        "engine": engine,
        "months": months,
        "quality": quality,
        "panel_rows": len(panel),
        "zones": panel.zone_id.nunique(),
        "timezone": "UTC; NYC wall-time source localized before aggregation",
        "sha256": sha256(destination),
        "source_sha256": {p.name: sha256(p) for p in paths},
        "zero_fraction": float(panel.demand.eq(0).mean()),
    }
    write_json(output / "quality.json", report)
    return report
