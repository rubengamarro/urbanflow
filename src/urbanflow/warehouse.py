import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


def load_warehouse(panel_path: Path, lookup_path: Path, database_url: str | None = None) -> dict:
    """Transactional, repeatable upserts, with explicit UTC conversion at the boundary."""
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("Set DATABASE_URL to a PostgreSQL SQLAlchemy URI")
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        raise ValueError("The warehouse contract requires PostgreSQL")
    panel = pd.read_parquet(panel_path)
    zones = pd.read_csv(lookup_path).fillna("")
    schema = (Path(__file__).parent / "sql/schema.sql").read_text(encoding="utf-8")
    try:
        with engine.begin() as connection:
            # Concurrent writers serialize rather than interleave snapshots.
            connection.execute(text("SELECT pg_advisory_xact_lock(872410)"))
            for statement in schema.split(";"):
                if statement.strip():
                    connection.execute(text(statement))
            connection.execute(
                text("""
                INSERT INTO taxi_zones (zone_id, borough, zone_name, service_zone)
                VALUES (:zone_id, :borough, :zone_name, :service_zone)
                ON CONFLICT (zone_id) DO UPDATE SET borough=EXCLUDED.borough,
                    zone_name=EXCLUDED.zone_name, service_zone=EXCLUDED.service_zone
            """),
                [
                    {
                        "zone_id": int(r.LocationID),
                        "borough": r.Borough,
                        "zone_name": r.Zone,
                        "service_zone": r.service_zone,
                    }
                    for r in zones.itertuples()
                ],
            )
            # Replace only the explicit scope; rows outside it remain untouched.
            connection.execute(
                text("""
                DELETE FROM hourly_demand WHERE hour >= :start AND hour <= :end
                    AND zone_id = ANY(:zones)
            """),
                {
                    "start": panel.hour.min().tz_localize("UTC").to_pydatetime(),
                    "end": panel.hour.max().tz_localize("UTC").to_pydatetime(),
                    "zones": [int(z) for z in panel.zone_id.unique()],
                },
            )
            insert = text("""
                INSERT INTO hourly_demand (hour, zone_id, demand) VALUES (:hour, :zone_id, :demand)
                ON CONFLICT (hour, zone_id) DO UPDATE SET demand=EXCLUDED.demand
            """)
            for start in range(0, len(panel), 5000):
                chunk = panel.iloc[start : start + 5000]
                connection.execute(
                    insert,
                    [
                        {
                            "hour": r.hour.tz_localize("UTC").to_pydatetime(),
                            "zone_id": int(r.zone_id),
                            "demand": int(r.demand),
                        }
                        for r in chunk.itertuples()
                    ],
                )
    finally:
        engine.dispose()
    return {"loaded_rows": len(panel), "zones": len(zones)}
