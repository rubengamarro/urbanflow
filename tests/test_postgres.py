import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from urbanflow.warehouse import load_warehouse


@pytest.mark.postgres
def test_transactional_idempotent_warehouse(tmp_path, panel):
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("PostgreSQL service and DATABASE_URL required")
    path = tmp_path / "hourly.parquet"
    panel.to_parquet(path, index=False)
    lookup = tmp_path / "zones.csv"
    pd.DataFrame(
        {
            "LocationID": [4, 161, 263],
            "Borough": ["Manhattan"] * 3,
            "Zone": ["A", "B", "C"],
            "service_zone": ["Yellow"] * 3,
        }
    ).to_csv(lookup, index=False)
    for _ in range(2):
        load_warehouse(path, lookup, url)
    engine = create_engine(url)
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT COUNT(*) FROM hourly_demand")) == len(panel)
        assert conn.scalar(text("SELECT SUM(demand) FROM hourly_demand")) == int(panel.demand.sum())
    # FK failure halfway through a load must roll back the earlier delete.
    invalid = panel.copy()
    invalid.loc[invalid.index[-1], "zone_id"] = 1
    invalid.to_parquet(path, index=False)
    with pytest.raises(IntegrityError):
        load_warehouse(path, lookup, url)
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT COUNT(*) FROM hourly_demand")) == len(panel)
    engine.dispose()
