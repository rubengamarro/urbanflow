import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from urbanflow.cli import app
from urbanflow.eda import analyze
from urbanflow.features import make_features
from urbanflow.modeling import temporal_split
from urbanflow.monitoring import monitor
from urbanflow.processing import process
from urbanflow.warehouse import load_warehouse


def test_processing_eda_and_monitoring_outputs(tmp_path, panel):
    raw = tmp_path / "raw"
    raw.mkdir()
    months = ["2024-01", "2024-02"]
    # Convert the UTC fixture into TLC's documented naive NYC pickup schema.
    local = panel.hour.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    trips = pd.DataFrame({"tpep_pickup_datetime": local, "PULocationID": panel.zone_id})
    for month in months:
        trips[trips.tpep_pickup_datetime.dt.strftime("%Y-%m").eq(month)].to_parquet(
            raw / f"yellow_tripdata_{month}.parquet", coerce_timestamps="us"
        )
    lookup = raw / "taxi_zone_lookup.csv"
    pd.DataFrame(
        {"LocationID": [4, 161, 263], "Zone": ["A", "B", "C"], "Borough": ["Manhattan"] * 3}
    ).to_csv(lookup, index=False)
    processed = tmp_path / "processed"
    report = process(months, raw, processed, zones=[4, 161, 263])
    assert report["panel_rows"] == len(panel)
    assert sum(r["rejected_rows"] for r in report["quality"]) == 0
    with pytest.raises(ValueError, match="engine"):
        process(months, raw, processed, engine="invalid")
    reports = tmp_path / "reports"
    summary = analyze(processed / "hourly.parquet", lookup, reports)
    assert summary["total_valid_pickups_in_scope"] == len(panel)
    assert (reports / "eda.png").stat().st_size > 1000
    split = temporal_split(make_features(panel))
    evaluation = {
        "partitions": {
            "train": {"end": str(split.train.hour.max())},
            "test": {"start": str(split.test.hour.min())},
        },
        "test_coverage_90": 0.8,
    }
    diagnostic = monitor(processed / "hourly.parquet", evaluation, reports)
    assert diagnostic["coverage_below_85"]
    assert json.loads((reports / "monitoring.json").read_text())["coverage_90"] == 0.8


def test_warehouse_requires_explicit_postgres_uri(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        load_warehouse(tmp_path / "missing", tmp_path / "missing")
    with pytest.raises(ValueError, match="PostgreSQL"):
        load_warehouse(tmp_path / "missing", tmp_path / "missing", "sqlite://")


def test_cli_exposes_the_reproducible_workflows():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "pipeline",
        "ingest",
        "process",
        "train",
        "warehouse",
        "serve",
        "verify",
        "promote",
    ):
        assert name in result.stdout
