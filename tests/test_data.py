import json

import httpx
import pandas as pd
import pyarrow as pa
import pytest

from urbanflow.ingest import download, validate_months
from urbanflow.processing import arrow_counts, build_panel, normalize_pickups


@pytest.mark.parametrize(
    "months", [[], ["2024-00"], ["../bad"], ["2024-01", "2024-03"], ["2024-01", "2024-01"]]
)
def test_reject_invalid_months(months):
    with pytest.raises(ValueError):
        validate_months(months)


def test_clean_utc_zero_grid_and_preserve_identical_pickups(tmp_path):
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01 00:20"] * 2 + ["2024-01-02", "2023-12-31", None],
                format="mixed",
            ),
            "PULocationID": [161, 161, 265, 161, 161],
        }
    )
    path = tmp_path / "trips.parquet"
    raw.to_parquet(path)
    counts, quality = arrow_counts(path, "2024-01")
    assert quality == {"month": "2024-01", "raw_rows": 5, "valid_rows": 2, "rejected_rows": 3}
    assert counts.hour.iloc[0] == pd.Timestamp("2024-01-01 05:00")
    grid = build_panel(counts, ["2024-01"], [4, 161])
    assert len(grid) == 31 * 24 * 2
    assert grid.demand.sum() == 2
    assert grid[grid.zone_id.eq(4)].demand.sum() == 0
    assert grid.demand.dtype == "int32"


def test_spring_dst_real_elapsed_grid():
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-03-10 01:30", "2024-03-10 02:30", "2024-03-10 03:30"]
            ),
            "PULocationID": [4, 4, 4],
        }
    )
    clean = normalize_pickups(raw, "2024-03")
    assert clean.hour.tolist() == [
        pd.Timestamp("2024-03-10 06:00"),
        pd.Timestamp("2024-03-10 07:00"),
    ]
    counts = clean.groupby(["hour", "zone_id"]).size().rename("demand").reset_index()
    assert len(build_panel(counts, ["2024-03"], [4])) == 31 * 24 - 1


def test_ambiguous_autumn_fails_instead_of_silent_zero():
    raw = pd.DataFrame(
        {"tpep_pickup_datetime": [pd.Timestamp("2024-11-03 01:30")], "PULocationID": [4]}
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        normalize_pickups(raw, "2024-11")


def test_download_stream_validates_and_reuses_checksum(tmp_path, monkeypatch):
    calls = []
    content = b"LocationID,Borough,Zone,service_zone\n4,Manhattan,Alphabet City,Yellow\n"

    def stream(method, url, **kwargs):
        calls.append(url)
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=content, headers={"etag": "test"})
            )
        ).stream(method, url)

    monkeypatch.setattr(httpx, "stream", stream)
    path = tmp_path / "zones.csv"
    first = download("https://example.test/zones.csv", path)
    assert download("https://example.test/zones.csv", path) == first
    assert len(calls) == 1
    path.write_text("corrupted")
    download("https://example.test/zones.csv", path)
    assert len(calls) == 2
    assert json.loads(path.with_suffix(".csv.json").read_text())["sha256"] == first["sha256"]


def test_failed_download_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"not a parquet"))
        ).stream("GET", "https://example.test/trips"),
    )
    path = tmp_path / "trips.parquet"
    with pytest.raises(pa.ArrowInvalid):
        download("https://example.test/trips", path)
    assert not path.exists()
    assert not path.with_suffix(".parquet.part").exists()
