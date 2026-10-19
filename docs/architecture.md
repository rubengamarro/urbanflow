# Architecture and decisions

## Artifact boundaries

The src-layout Python package separates verified raw input, a complete hourly observation panel and immutable serving bundles. Prefect coordinates functions also available individually through the CLI. Ingestion retries; training is not blindly retried because tracking and model versions have side effects.

| Layer | Responsibility | Output |
|---|---|---|
| Bronze | Stream official sources, validate schema, verify cached hashes | Parquet/CSV and manifest |
| Silver | Clean pickup keys, NYC→UTC conversion, hourly grid | hourly.parquet, quality.json |
| Analytics | Optional transactional SQL projection | Zone dimension and hourly fact |
| Features | Shifted lags/windows and local calendar | Deterministic recomputation |
| Experiments | Expanding CV, selection, calibration, held-out test | MLflow, reports, predictions |
| Serving | Verified immutable estimator and history | Version directory and pointer |
| Product | HTTP validation, one-hour forecast, diagnostic UI | OpenAPI and dashboard |

## ADR 001: Arrow and Spark share a cleaning contract

At ~100 MB compressed, Arrow batches suit a laptop without Java and bound trip-level memory. PySpark implements the same eligibility and timezone contract with native Spark SQL expressions and distributed aggregation. The driver collects only hourly aggregates, not millions of trips. Spark is optional locally and default in the Docker trainer. Its default local master has two workers; a cluster/storage adapter is beyond this deployment's scope.

Native Spark expressions avoid Python trip-level worker serialization. A service test compares both engines on identical records, including spring and autumn DST behavior. Round-trip timezone conversion rejects nonexistent times and duplicate local representations detect ambiguous fall-back pickups. Reading each source month separately avoids unrelated fare-schema evolution.

## ADR 002: UTC storage, NYC calendar

TLC timestamps lack offsets. Localize before flooring to UTC hours. Internal Parquet timestamps are naive under an explicit UTC storage convention; SQL uses TIMESTAMPTZ; API requests require aware timestamps. Elapsed lags retain their meaning across spring DST. Ambiguous autumn timestamps cannot be recovered reliably, so processing fails instead of inventing UTC offsets or zeros.

## ADR 003: one shared information boundary

Target hour t uses observations through t−1. Features: zone ID, cyclic NYC hour/day, weekend, lags 1/2/24/168, shifted means 24/168 and shifted standard deviation 24. API and training share the implementation. Inference supplies only the 168 preceding hours plus an unlabeled target row. Tests perturb future labels and verify unchanged earlier and target features.

This assumes completed hourly observations. Actual monthly TLC publication is delayed, so the service explicitly demonstrates historical replay/next hour after a stored snapshot. A continuously updated live feed is not implemented.

## ADR 004: count objectives and an eligible baseline

The weekly baseline may win selection. HistGradientBoosting and two fixed XGBoost configurations use Poisson losses, seeds and nonnegative inference. Random early-stopping validation is disabled. A small declared configuration set is appropriate for two winter months. Expanding CV measures stability; the independent validation block selects the winner. Calibration and final test stay separate.

## ADR 005: SQL analytics, file-based serving

PostgreSQL enforces constraints and enables SQL queries. One transaction, advisory lock, scoped replacement and upserts make reruns repeatable and preserve outside data. Serving uses a checksum-verified observation snapshot, avoiding database availability in the forecast path. This is consistent offline/online computation, not a distributed continuously refreshed feature store.

## ADR 006: immutable versions and atomic publication

A bundle contains estimator, history, features, radii, cutoffs, hashes, package versions and evaluation. Successful MLflow registration precedes atomic `champion.json` replacement. Failed training preserves the old pointer. The API loads one snapshot at startup and requires restart to adopt changes. `promote` validates an existing bundle before activation or rollback.

Checksums detect corruption, not the trustworthiness of an unknown author. Deserialize only this trusted pipeline's joblib/cloudpickle artifacts and restore with pinned dependencies. Local training/report directories use a single-writer design.

## Deployment and observability

Docker runs non-root with Java 17, locked dependencies and persistent volumes. FastAPI has separate live/ready routes and Prometheus-format per-process counters. Offline PSI compares training/test features and separately flags low interval coverage. These mechanics do not constitute distributed monitoring or a production promotion decision.

CI runs lint, tests, service integration and image smoke checks. GHCR publication is tag-triggered. Full real-data benchmarking is manual to keep PR checks independent of a large external download. No external infrastructure is provisioned.
