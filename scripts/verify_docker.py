"""Run inside the MLflow container to validate the complete Compose deployment.

Copy this script to /tmp/verify_docker.py and execute with URBANFLOW_API_URL,
MLFLOW_TRACKING_URI and DATABASE_URL configured. Writes JSON evidence to /tmp.
"""

import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx
import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from streamlit.testing.v1 import AppTest

from urbanflow.config import Settings
from urbanflow.features import FEATURES, make_features
from urbanflow.serving import Predictor


def verify() -> dict:
    settings = Settings()
    model = Predictor(settings.artifacts)
    with httpx.Client(base_url=settings.api_url, timeout=30) as client:
        for path in ("/health/live", "/health/ready", "/model", "/evaluation", "/openapi.json"):
            response = client.get(path)
            response.raise_for_status()
        forecast = client.post("/predict", json={"zone_ids": [161, 132, 138]})
        forecast.raise_for_status()
        assert forecast.json() == model.forecast([161, 132, 138])
        replay = client.post(
            "/predict",
            json={"zone_ids": [161], "target_hour": "2024-02-29T12:00:00Z"},
        )
        replay.raise_for_status()
        assert replay.json()["mode"] == "historical_replay"
        invalid = client.post(
            "/predict", json={"zone_ids": [161], "target_hour": "2026-10-08T12:00:00Z"}
        )
        assert invalid.status_code == 422
        assert "urbanflow_model_ready 1" in client.get("/metrics").text
    for url in ("http://dashboard:8501/_stcore/health", "http://mlflow:5000/health"):
        httpx.get(url, timeout=30).raise_for_status()

    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        counts = dict(
            conn.execute(
                text(
                    "SELECT COUNT(*) AS rows, SUM(demand) AS pickups, "
                    "COUNT(DISTINCT zone_id) AS zones FROM hourly_demand"
                )
            )
            .mappings()
            .one()
        )
    engine.dispose()
    assert counts == {"rows": 378720, "pickups": 5949006, "zones": 263}

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    versions = mlflow.MlflowClient().search_model_versions("name='UrbanFlowDemand'")
    registered = next(v for v in versions if v.run_id == model.metadata["run_id"])
    loaded = mlflow.pyfunc.load_model(f"models:/UrbanFlowDemand/{registered.version}")
    features = make_features(pd.read_parquet(settings.processed / "hourly.parquet"))
    example = features.dropna(subset=FEATURES).tail(20)[FEATURES]
    np.testing.assert_allclose(loaded.predict(example), model.model.predict(example), rtol=1e-6)

    dashboard = AppTest.from_file("/app/apps/dashboard.py").run(timeout=60)
    assert not dashboard.exception, dashboard.exception
    assert not dashboard.error, dashboard.error
    assert len(dashboard.tabs) == 4
    dashboard.button[0].click().run(timeout=60)
    assert not dashboard.exception, dashboard.exception
    assert not dashboard.error, dashboard.error
    assert len(dashboard.dataframe) >= 1
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uid": os.getuid(),
        "packages": {p: version(p) for p in ("pyspark", "prefect", "mlflow", "xgboost")},
        "api": "readiness, OpenAPI, forecast equality, historical replay, rejected future, metrics",
        "dashboard": "HTTP health, four AppTest tabs and forecast interaction passed",
        "warehouse": counts,
        "model_version": model.metadata["version"],
        "mlflow_run_id": model.metadata["run_id"],
        "mlflow_registered_version": registered.version,
        "mlflow_reload": "20 predictions match the serving estimator",
        "test": model.evaluation["test"],
        "test_baseline": model.evaluation["test_baseline"],
        "coverage_90": model.evaluation["test_coverage_90"],
    }


if __name__ == "__main__":
    report = verify()
    Path("/tmp/docker_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
