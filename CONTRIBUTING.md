# Contributing

Use Python 3.11/3.12 and `uv sync --frozen --extra dev`. Run Ruff and unit/component tests. Spark service tests use Java 17 and the spark extra; SQL tests need a disposable PostgreSQL DATABASE_URL.

Eligibility, timezone, feature cutoff and split changes need behavioral tests. Future-label perturbation and train/serve parity must remain valid. Synthetic fixtures belong only in tests; benchmark reports require actual source files and lineage hashes.

Do not commit raw monthly files, model pickles, databases, caches or credentials. To replace reference evidence, run the full pipeline, execute the notebook and export reports; explain scope/cutoff differences. Never hand-edit generated metrics. Dependency changes update both pyproject.toml and uv.lock.
