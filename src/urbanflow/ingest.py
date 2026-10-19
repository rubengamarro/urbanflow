import re
import ssl
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq

from urbanflow.io import sha256, write_json

BASE = "https://d37ci6vzurychx.cloudfront.net"
SOURCE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
REQUIRED = ["tpep_pickup_datetime", "PULocationID"]


def validate_months(months: list[str]) -> list[str]:
    if not months or len(months) != len(set(months)):
        raise ValueError("Provide unique consecutive months in YYYY-MM format")
    for month in months:
        if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
            raise ValueError(f"Invalid month: {month}")
    ordered = sorted(months)
    periods = pd.PeriodIndex(ordered, freq="M")
    if len(periods) > 1 and (periods.asi8[1:] - periods.asi8[:-1] != 1).any():
        raise ValueError("Months must be consecutive; missing months are not zero demand")
    return ordered


def download(url: str, destination: Path) -> dict:
    """Stream to a temporary file, validate, then publish. Reuse verified cached files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() and sidecar.exists():
        import json

        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record["url"] == url and record["sha256"] == sha256(destination):
            return record
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with httpx.stream(
            "GET", url, timeout=120, follow_redirects=True, verify=ssl.create_default_context()
        ) as response:
            response.raise_for_status()
            with tmp.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk)
            headers = dict(response.headers)
        if destination.suffix == ".parquet":
            # Own the file handle even if the Parquet constructor itself fails (Windows).
            with tmp.open("rb") as source, pq.ParquetFile(source) as parquet:
                missing = set(REQUIRED) - set(parquet.schema.names)
                if missing:
                    raise ValueError(f"Source schema is missing {missing}")
        else:
            lookup = pd.read_csv(tmp)
            if not {"LocationID", "Zone", "Borough"}.issubset(lookup):
                raise ValueError("Invalid zone lookup schema")
        tmp.replace(destination)
        record = {
            "url": url,
            "source_page": SOURCE,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "etag": headers.get("etag"),
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(sidecar, record)
        return record
    finally:
        tmp.unlink(missing_ok=True)


def ingest(months: list[str], raw: Path) -> dict:
    months = validate_months(months)
    records = [download(f"{BASE}/misc/taxi_zone_lookup.csv", raw / "taxi_zone_lookup.csv")]
    for month in months:
        name = f"yellow_tripdata_{month}.parquet"
        records.append(download(f"{BASE}/trip-data/{name}", raw / name))
    manifest = {"months": months, "files": records}
    write_json(raw / "manifest.json", manifest)
    return manifest
