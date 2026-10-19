from prefect import flow, task

from urbanflow.config import Settings
from urbanflow.eda import analyze
from urbanflow.ingest import ingest
from urbanflow.modeling import train
from urbanflow.monitoring import monitor
from urbanflow.processing import process
from urbanflow.warehouse import load_warehouse

ingest_task = task(ingest, name="download-verified-tlc", retries=3, retry_delay_seconds=10)
process_task = task(process, name="bronze-to-hourly-silver")
eda_task = task(analyze, name="exploratory-analysis")
train_task = task(train, name="temporal-benchmark-and-promote")
warehouse_task = task(
    load_warehouse, name="transactional-postgres-load", retries=2, retry_delay_seconds=5
)
monitor_task = task(monitor, name="offline-drift-and-coverage")


@flow(name="urbanflow-end-to-end", log_prints=True)
def run_pipeline(
    months: list[str] | None = None,
    engine: str = "arrow",
    root: str = ".",
    zones: list[int] | None = None,
    warehouse: bool = False,
    holdout_days: int = 7,
) -> dict:
    settings = Settings(root=root)
    settings.prepare()
    months = months or ["2024-01", "2024-02"]
    ingest_task(months, settings.raw)
    quality = process_task(months, settings.raw, settings.processed, engine, zones)
    panel = settings.processed / "hourly.parquet"
    if warehouse:
        warehouse_task(panel, settings.raw / "taxi_zone_lookup.csv")
    eda_task(panel, settings.raw / "taxi_zone_lookup.csv", settings.reports)
    evaluation = train_task(panel, settings.artifacts, settings.reports, holdout_days)
    monitor_task(panel, evaluation, settings.reports)
    print(
        {
            "selected_model": evaluation["selected_model"],
            "test": evaluation["test"],
            "valid_rows": sum(x["valid_rows"] for x in quality["quality"]),
        }
    )
    return evaluation
