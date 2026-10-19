# UrbanFlow AI 🚕

**One-hour-ahead NYC yellow taxi pickup forecasting, from public trip records to a versioned REST API.**

UrbanFlow processes real NYC TLC Parquet files, builds an hourly feature panel, compares count-regression models against a weekly baseline, tracks experiments in MLflow, and serves predictions through FastAPI and Streamlit.

![Exploratory analysis of the real reference dataset](reports/reference/eda.png)

## Verified reference benchmark

The reference run uses **January–February 2024**, all **263 geographic taxi zones**, and **5,949,006 eligible pickups** from **5,972,150 source rows**. The panel contains 378,720 zone-hour observations, including zero-pickup hours. No sampled or synthetic training data are used for these results.

| Final test metric | Selected XGBoost | Weekly seasonal baseline |
|---|---:|---:|
| MAE, pickups / zone-hour | 3.096 | 3.963 |
| RMSE | 10.704 | 13.959 |
| WAPE | 17.98% | 23.02% |
| Poisson deviance | 1.495 | 6.335 |

Test MAE decreases by **21.88%**. Selection uses validation MAE; the final test never chooses the model. Nominal 90% residual intervals achieve **88.28%** test coverage. Sparse zones make global MAE favorable, so reports also contain per-zone errors, WAPE, and a top-20 cohort selected using training demand alone.

Inspect the [evaluation](reports/reference/evaluation.json), [source manifest](reports/reference/manifest.json), [quality report](reports/reference/quality.json), [held-out predictions](reports/reference/test_predictions.parquet), and [checksums](reports/reference/evidence.json). These describe a historical benchmark, not live operations or annual performance. Read the [methodology](docs/evaluation.md) and [model card](docs/model-card.md).

## Run locally

Requirements: **Python 3.11/3.12**, [uv](https://docs.astral.sh/uv/getting-started/installation/), internet access for the initial ~100 MB TLC download, and approximately 2 GB free disk for dependencies, data and artifacts. Run from the cloned repository root.

```bash
uv sync --frozen --extra dev
uv run urbanflow pipeline
uv run urbanflow verify
```

The default Arrow engine reads full monthly files in batches without Java. Spark implements the same contract. Seeds are fixed and `uv.lock` pins dependencies; hardware/library differences can introduce small numeric differences, so compare metric tolerances rather than binary model hashes across machines.

Start services in separate terminals:

```bash
uv run urbanflow serve
uv run streamlit run apps/dashboard.py
```

Open the [dashboard](http://localhost:8501), [OpenAPI documentation](http://localhost:8000/docs), and [readiness](http://localhost:8000/health/ready). The dashboard shows forecasts, intervals, errors, splits, demand patterns and drift diagnostics.

Inspect experiments:

```bash
uv run mlflow server --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000
```

### Spark and PostgreSQL

Use **Java 17**, set `JAVA_HOME`, and install the spark extra. Clear inherited `SPARK_HOME`/`PYTHONPATH` if an existing global Spark installation overrides the project's pinned distribution.

```bash
uv sync --frozen --extra dev --extra spark
uv run urbanflow pipeline --engine spark
```

PostgreSQL is an optional analytical warehouse; the local quickstart is fully functional without it. Start PostgreSQL 16, set `DATABASE_URL`, and load the hourly panel:

```bash
# bash
export DATABASE_URL=postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow
uv run urbanflow warehouse
```

```powershell
# PowerShell
$env:DATABASE_URL = 'postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow'
uv run urbanflow warehouse
```

The loader uses one transaction, an advisory lock, scoped replacement and upserts under a `(hour, zone_id)` primary key. Repeat loading and rollback after an FK failure are integration-tested. SQL definitions and analytics are in [src/urbanflow/sql](src/urbanflow/sql).

## Docker Compose

Requires a running Docker Engine/Desktop, Compose v2 and x86-64 Linux containers. Ports bind to localhost. The development database password is configurable with `POSTGRES_PASSWORD`. If Windows reserves port 8000, set `URBANFLOW_API_PORT=18000` in an ignored `.env` file; the API documentation then opens at `http://localhost:18000/docs`.

```bash
docker compose build
docker compose up -d postgres mlflow
docker compose run --rm trainer
docker compose up -d api dashboard
```

Compose builds one shared application image through the MLflow service. The trainer executes the real Spark pipeline and SQL load, logs to MLflow and publishes its bundle into shared storage. It must finish before API readiness. After retraining, run `docker compose restart api dashboard` to adopt the new snapshot. Health checks and named volumes are included. `docker compose down` stops services; adding `-v` also deletes training data, models and database volumes.

To keep the actual training run visible in the persistent [Prefect UI](http://localhost:4200), start its server before running the trainer:

```bash
docker compose --profile orchestration up -d --wait prefect
docker compose run --rm -e PREFECT_API_URL=http://prefect:4200/api trainer
```

See [operations](docs/operations.md) for container verification and the optional resident scheduler.

## Forecast contract

Predict eligible pickups for the next **single hour** using the previous 168 complete observations. Predictions are continuous expected counts, not rounded allocations. Input timestamps need a timezone offset and whole UTC hour alignment.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"zone_ids": [161, 132, 138]}'
```

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType 'application/json' -Body '{"zone_ids":[161,132,138]}'
```

Omitting `target_hour` predicts immediately after stored history. For the reference snapshot this means **2024-03-01T05:00:00Z**, irrespective of today's date. Historical replay accepts targets after calibration and within available history:

```json
{"zone_ids": [161], "target_hour": "2024-02-25T15:00:00Z"}
```

Responses include target time, observation cutoff, version, replay/next-hour mode, expected counts and nominal intervals. See the [actual inference example](reports/reference/inference_example.json). Targets beyond the next hour yield 422; missing/corrupt models produce readiness 503 while liveness remains 200.

## Architecture

```mermaid
flowchart LR
    TLC[NYC TLC monthly Parquet + zone CSV] --> Ingest[Verified downloads + SHA-256 manifest]
    Ingest --> Bronze[Raw source cache]
    Bronze --> Process[Arrow batches / PySpark]
    Process --> Silver[Complete UTC zone-hour panel]
    Silver --> SQL[PostgreSQL analytical warehouse]
    Silver --> Features[Past-only features + NYC calendar]
    Features --> Eval[Expanding CV → validation → calibration → test]
    Eval --> MLflow[MLflow runs + registered model]
    Eval --> Bundle[Versioned model + history + checksums]
    Bundle --> API[FastAPI snapshot inference]
    API --> UI[Streamlit dashboard]
    Eval --> Reports[EDA + predictions + uncertainty + drift]
    Prefect[Prefect flow and tasks] -. orchestrates .-> Ingest
    Prefect -. orchestrates .-> Eval
```

Read [architecture](docs/architecture.md), [data contract](docs/data-card.md), [operations](docs/operations.md), and the [interview guide](docs/interview-guide.md).

## Reproduction and automation

```bash
# Alternate consecutive months, excluding ambiguous autumn DST
uv run urbanflow pipeline --months 2024-04,2024-05

# Explicit smaller development scope
uv run urbanflow pipeline --zones 4,48,79,132,138,161,162,170,186,230

# Resident Prefect deployment for an explicitly configured snapshot
uv run urbanflow schedule --months 2024-01,2024-02 --cron "0 6 5 * *"

# Activate a prior verified version, then restart API
uv run urbanflow promote --version VERSION_FROM_ARTIFACTS
```

Scheduling uses explicit months, not guesses about TLC release availability. Update the configured months when moving to a new snapshot. A monthly rerun is not a live observation feed. `docker compose --profile orchestration up -d prefect scheduler` runs the optional Prefect server and deployment process; the cron uses Prefect's default UTC timezone.

## Tests and CI/CD

```bash
uv run ruff check src tests apps scripts
uv run ruff format --check src tests apps scripts
uv run pytest -m "not spark and not postgres" --cov=urbanflow
# With Java 17, the spark extra and a disposable PostgreSQL DATABASE_URL:
RUN_SPARK_TESTS=1 uv run pytest -m "spark or postgres"
```

Tests cover future-label leakage, timezone/DST behavior, missing grids, verified downloads, grouped splits, calibration, train/serve parity, corrupt artifacts, failed training preserving a champion, API requests and SQL transactions. Synthetic fixtures exist only in tests and never provide benchmark results.

GitHub Actions includes Python 3.11/3.12 on Linux/Windows, Spark/PostgreSQL integration on Linux, and a Docker build and readiness smoke test. The manual **Real TLC benchmark** workflow regenerates full-data evidence. Explicit `v*` tags rerun unit checks and publish containers to GHCR. No cloud deployment is automatic.

The [validation record](docs/validation.md) distinguishes executed checks from configured CI checks. GitHub Actions results only exist after pushing the repository.

## Repository layout

```text
src/urbanflow/       data pipeline, features, models, flows, API, SQL
apps/dashboard.py   Streamlit UI
tests/              unit, component and service integration tests
notebooks/          executed analysis on real data
scripts/            notebook execution and evidence export
docs/               architecture, cards, methodology and operations
reports/reference/  committed small reference outputs and checksums
.github/workflows/  quality, integration, benchmark and container release
```

Raw data, models, local databases, caches and secrets are ignored. Export a new reviewed benchmark with `uv run python scripts/export_evidence.py`. Execute the notebook with `uv run python scripts/execute_notebook.py` after training. Training never reads committed reference metrics.

## Scope and limitations

- The target measures **observed yellow taxi pickups**, not unmet passenger demand, all taxi services or causal effects.
- The final test is seven winter days. Seasonal, holiday and regime backtests are required for operational decisions.
- TLC timestamps lack offsets. Spring nonexistent times are rejected; autumn ambiguous times fail processing. November snapshots containing the fall-back hour are unsupported.
- Zeros assume monthly input completeness. Vendor outages are not identifiable. Identical timestamp/zone pairs remain because they are not unique trip IDs.
- Numeric zone IDs are tree features, not distances. Categorical representations merit a controlled comparison.
- Live observations, authentication, distributed feature storage, multi-instance metrics and a production promotion gate remain future work.

## Attribution and license

Data come from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), its linked zone lookup and the [yellow taxi dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf). TLC cautions that vendor-supplied data may contain inaccuracies. Raw data are not redistributed; this code license does not cover third-party data.

Code is [MIT licensed](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md).
