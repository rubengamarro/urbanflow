import json
from typing import Annotated

import typer

from urbanflow.config import Settings

app = typer.Typer(no_args_is_help=True, help="UrbanFlow AI reproducible data and ML workflows")


@app.command()
def pipeline(
    months: Annotated[
        str, typer.Option(help="Comma-separated consecutive YYYY-MM values")
    ] = "2024-01,2024-02",
    engine: str = "arrow",
    root: str = ".",
    zones: str = "",
    warehouse: bool = False,
    holdout_days: int = 7,
):
    """Run real ingestion, processing, EDA, evaluation and model publication."""
    from urbanflow.pipeline import run_pipeline

    selected = [int(z) for z in zones.split(",")] if zones else None
    run_pipeline(months.split(","), engine, root, selected, warehouse, holdout_days)


@app.command()
def ingest(months: str = "2024-01,2024-02", root: str = "."):
    """Download and validate real TLC Parquet and zone lookup files."""
    from urbanflow.ingest import ingest as download

    settings = Settings(root=root)
    settings.prepare()
    typer.echo(json.dumps(download(months.split(","), settings.raw), indent=2))


@app.command()
def process(
    months: str = "2024-01,2024-02", engine: str = "arrow", root: str = ".", zones: str = ""
):
    """Build a complete zone-hour grid with Arrow or Spark."""
    from urbanflow.processing import process as build

    settings = Settings(root=root)
    settings.prepare()
    selected = [int(z) for z in zones.split(",")] if zones else None
    typer.echo(
        json.dumps(
            build(months.split(","), settings.raw, settings.processed, engine, selected), indent=2
        )
    )


@app.command()
def train(root: str = ".", holdout_days: int = 7):
    """Compare models and publish a verified immutable serving bundle."""
    from urbanflow.modeling import train as fit

    settings = Settings(root=root)
    typer.echo(
        json.dumps(
            fit(
                settings.processed / "hourly.parquet",
                settings.artifacts,
                settings.reports,
                holdout_days,
            ),
            indent=2,
        )
    )


@app.command()
def warehouse(root: str = "."):
    """Load data into PostgreSQL using DATABASE_URL."""
    from urbanflow.warehouse import load_warehouse

    settings = Settings(root=root)
    typer.echo(
        load_warehouse(settings.processed / "hourly.parquet", settings.raw / "taxi_zone_lookup.csv")
    )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the REST service."""
    import uvicorn

    uvicorn.run("urbanflow.api:app", host=host, port=port)


@app.command()
def schedule(
    months: str = "2024-01,2024-02", engine: str = "arrow", root: str = ".", cron: str = "0 6 5 * *"
):
    """Serve a Prefect deployment, rerunning the explicitly configured monthly snapshot."""
    from urbanflow.pipeline import run_pipeline

    run_pipeline.serve(
        name="monthly-snapshot",
        cron=cron,
        parameters={"months": months.split(","), "engine": engine, "root": root},
    )


@app.command()
def promote(version: str, root: str = "."):
    """Verify and activate an existing trusted bundle (also supports rollback)."""
    from urbanflow.io import write_json
    from urbanflow.serving import Predictor

    settings = Settings(root=root)
    model = Predictor(settings.artifacts, model_version=version)
    model.forecast(model.zones[:1])
    write_json(settings.artifacts / "champion.json", {"version": version})
    typer.echo(f"Activated {version}. Restart the API to load this immutable snapshot.")


@app.command()
def verify(root: str = "."):
    """Validate bundle integrity and exercise next-hour inference."""
    from urbanflow.serving import Predictor

    model = Predictor(Settings(root=root).artifacts)
    typer.echo(json.dumps(model.forecast(model.zones[:3]), indent=2))


if __name__ == "__main__":
    app()
