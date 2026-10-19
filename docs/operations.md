# Operations

## Running and recovery

Use `uv sync --frozen --extra dev` and `uv run urbanflow pipeline`. Prefect starts a temporary local server unless `PREFECT_API_URL` points to an existing server. Set `PREFECT_HOME` to a writable directory on restricted machines. Behind a corporate TLS proxy, `uv --native-tls sync` uses system certificates; do not disable TLS verification.

CLI stages `ingest`, `process`, `train`, `warehouse`, `verify` isolate failures. No stage or dashboard substitutes synthetic data. Ingestion is retryable and verified cached files are reused. Failed-run report files are not a successful promotion: `champion.json` and the API loaded version are authoritative.

Spark requires Java 17 and the spark extra, which includes setuptools for PySpark's Python 3.12 distutils compatibility. Clear global `SPARK_HOME`/`PYTHONPATH` overrides. Use microsecond Parquet timestamps. Native Windows environments may require Hadoop native binaries; Linux Docker/CI avoids that dependency.

## Publication and rollback

Training registers in MLflow and completes a versioned bundle before atomically changing the local pointer. A failed run preserves the old pointer. This is local snapshot publication; a production policy comparing the current service model on a common future window is not implemented.

1. Review evaluation, zone errors, bootstrap and coverage.
2. Run `uv run urbanflow verify`.
3. Restart API/dashboard to adopt the snapshot.
4. Check readiness/model metadata for version and history cutoff.
5. Roll back with `uv run urbanflow promote --version <prior-version>` and restart.

Joblib/cloudpickle artifacts must be trusted. Checksums detect corruption, not an unknown provider's authenticity. Restore with the exact environment. Do not run overlapping training writers on the same local directories; the SQL loader serializes writers, but local report/model publication is single-writer.

## Warehouse

Set a PostgreSQL `DATABASE_URL`; use `pipeline --warehouse` or `warehouse`. TIMESTAMPTZ inputs are explicitly UTC regardless of session timezone. Zones, scoped deletion and 5,000-row insertion batches share one transaction and an advisory lock; failures roll back. Rows outside the requested zone/time scope remain. A COPY staging approach would improve throughput for larger histories.

## Scheduling

`urbanflow schedule` runs a resident deployment for explicit months; default cron is 06:00 UTC on the fifth day of each month. For persistent UI, start `prefect server start`, set `PREFECT_API_URL=http://localhost:4200/api`, and keep the deployment process running. Compose's orchestration profile supplies both processes.

Update months when TLC releases another snapshot. The schedule does not guess release availability and cannot provide live hourly observations from delayed monthly data.

## Verify the Docker deployment

After the real trainer finishes and API/dashboard are healthy, execute the integration probe inside the MLflow container. It checks SQL totals, API forecasts against the immutable bundle, historical replay, invalid future requests, model registry reload, and actual Streamlit tab rendering and forecast interaction. The probe targets the documented January–February 2024 reference snapshot.

```bash
docker compose cp scripts/verify_docker.py mlflow:/tmp/verify_docker.py
docker compose exec -T \
  -e URBANFLOW_API_URL=http://api:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e DATABASE_URL=postgresql+psycopg://urbanflow:urbanflow@postgres:5432/urbanflow \
  mlflow python /tmp/verify_docker.py
docker compose cp mlflow:/tmp/docker_verification.json reports/reference/docker_verification.json
```

Use your configured database password when it differs from the local development default. In PowerShell, place the `exec` arguments on one line or use PowerShell backticks for continuation. The resulting JSON belongs to that execution; regenerate evidence checksums when publishing changed reports.

## Diagnostics

| Symptom | Action |
|---|---|
| Live 200, ready 503 | Train/verify a valid bundle and restart |
| Prediction 422 | Check requested zone, target alignment/cutoff and available history |
| Checksum mismatch | Quarantine and regenerate trusted artifacts |
| PSI > .2 | Investigate distribution change; heuristic, not an automatic deployment decision |
| Interval coverage < 85% | Investigate/calibrate on another independent window |
| Old observation cutoff | Historical snapshot; run a newer-data pipeline |
| Docker API unhealthy before training | Finish trainer, restart API |
| Windows denies binding port 8000 | Check `netsh interface ipv4 show excludedportrange protocol=tcp`; set `URBANFLOW_API_PORT=18000` in `.env` and recreate API |

`/metrics` exports request/failure counters, total request seconds and readiness. There is no distributed aggregation, latency histogram or external alert manager. Live deployment needs authentication, delayed-label monitoring, infrastructure controls and an explicit promotion gate.

## GitHub release

No remote/cloud credentials are stored. Push to your own repository after reviewing tracked files. CI checks pushes/PRs; the manual benchmark workflow uploads regenerated evidence. Explicit `v*` tags publish containers to GHCR under GitHub package permissions. No cloud resources are created. Configured checks do not imply they already executed here; see the validation record and review CI after pushing.
