import os

import pandas as pd
import pytest

from urbanflow.processing import arrow_counts, spark_counts


@pytest.mark.spark
@pytest.mark.parametrize("month", ["2024-01", "2024-03", "2024-11"])
def test_spark_arrow_parity(tmp_path, month):
    if os.getenv("RUN_SPARK_TESTS") != "1":
        pytest.skip("Set RUN_SPARK_TESTS=1 with the spark extra and Java 17")
    pytest.importorskip("pyspark")
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2023-12-31"]
            ),
            "PULocationID": [4, 4, 161, 4],
        }
    )
    if month == "2024-03":
        raw = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.to_datetime(
                    ["2024-03-10 01:30", "2024-03-10 02:30", "2024-03-10 03:30"]
                ),
                "PULocationID": [4, 4, 4],
            }
        )
    elif month == "2024-11":
        raw = pd.DataFrame(
            {"tpep_pickup_datetime": [pd.Timestamp("2024-11-03 01:30")], "PULocationID": [4]}
        )
    path = tmp_path / "input.parquet"
    raw.to_parquet(path, coerce_timestamps="us")
    if month == "2024-11":
        with pytest.raises(ValueError, match="Ambiguous"):
            arrow_counts(path, month)
        with pytest.raises(ValueError, match="Ambiguous"):
            spark_counts([path], [month])
        return
    arrow, quality = arrow_counts(path, month)
    spark, reports = spark_counts([path], [month])
    pd.testing.assert_frame_equal(
        arrow.sort_values(["hour", "zone_id"]).reset_index(drop=True),
        spark.sort_values(["hour", "zone_id"]).reset_index(drop=True),
        check_dtype=False,
    )
    assert reports == [quality]
