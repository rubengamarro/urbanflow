# Executed validation record

Validated on **8 October 2026**, Windows, Python 3.12.14, uv 0.9.8, Java 17.0.20.1, PySpark 3.5.5, Prefect 3.8.8, MLflow 3.17.0 and PostgreSQL 16.15. The exact source hashes, test case names, package versions, flow name and model version are in [validation.json](../reports/reference/validation.json).

| Check actually executed | Outcome |
|---|---|
| Ruff lint and formatting | Passed |
| Complete pytest suite with Spark and PostgreSQL | **29 passed, 0 failures/errors/skips** |
| Statement coverage | **86.73%**, including service integrations; 75% CI gate |
| Full real-data Prefect flow with Spark, SQL load, EDA, MLflow and monitoring | Completed |
| Spark vs Arrow on every real zone-hour observation | Exact equality on **378,720 rows** |
| Spark spring/autumn DST parity | Passed |
| PostgreSQL repeat loading and failure rollback | Passed against a disposable database |
| Full PostgreSQL reference panel | **378,720 rows**, **5,949,006 pickups**, **263 zones** |
| MLflow registered-model loading | Predictions match the serving estimator |
| Live HTTP API | Readiness, next-hour forecast, historical replay and OpenAPI passed |
| Streamlit AppTest against live HTTP API | All tabs rendered and forecast interaction passed |
| Notebook | **13 code cells** executed on the real processed data |
| Distribution packaging | Wheel/sdist built; both SQL files included in wheel |
| Docker Compose configuration | Passed `docker compose config --quiet` |
| Docker image build and complete Spark/SQL/MLflow/Prefect pipeline | Passed on Docker Engine **28.4.0**, Linux x86-64 |
| Complete pytest suite inside Docker with isolated PostgreSQL database | **29 passed**, 86.73% coverage |
| Docker API, Streamlit AppTest and MLflow model registry reload | Passed; all five services healthy |

Detailed evidence: [Spark](../reports/reference/spark_verification.json), [SQL](../reports/reference/warehouse_verification.json), [HTTP/UI](../reports/reference/service_verification.json), and [prediction rows](../reports/reference/test_predictions.parquet). The reference benchmark's final test MAE was 3.0956 versus 3.9627 for the weekly baseline; nominal 90% intervals achieved 88.2763% coverage. Synthetic unit fixtures do not generate these results.

## Docker execution

The complete Docker deployment was subsequently executed on **8 October 2026**, using Linux Python **3.12.15** and the same locked dependencies. The actual Prefect run `prophetic-limpet` completed ingestion, Spark processing, PostgreSQL loading, EDA, model comparison, registration and monitoring. It published model **717262bd933c**. Its test MAE was **3.0956188292505136** versus **3.962656165127648** for the weekly baseline, with **88.2763%** interval coverage. These closely match the independent Windows reference run; model UUIDs are expected to differ.

[Docker verification](../reports/reference/docker_verification.json) records the image ID, source checksums, real source hashes, quality counts, completed flow ID, measured metrics, tests, non-root UID, SQL counts, health statuses, host HTTP checks and model reload. Streamlit AppTest rendered all four tabs and exercised forecasting against the Docker API. All five running services were healthy. Windows reserved port 8000, so this workstation's ignored `.env` sets `URBANFLOW_API_PORT=18000`; the container API still uses port 8000 internally.

The executed [test log](../reports/reference/docker_tests.txt) and [training log](../reports/reference/docker_training.txt) retain assertions, coverage, completed tasks and upstream warnings. The checksum manifest includes these records.

## Checks not executed on GitHub

The Linux/Windows Python 3.11/3.12 CI matrix and GHCR tag publication are configured but have not yet run on GitHub. This repository was prepared locally; no remote repository, public deployment or successful hosted CI run is implied.

Upstream dependency deprecation warnings appear under the locked environment. The assertions, model reload and request behavior pass; warnings are preserved during testing rather than globally hidden. Broader seasonal robustness and live dispatch suitability remain outside this reference validation.
