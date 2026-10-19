import json

import mlflow.pyfunc
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from urbanflow.api import create_app
from urbanflow.config import Settings
from urbanflow.features import FEATURES, make_features
from urbanflow.modeling import SeasonalNaive, predict, train
from urbanflow.serving import Predictor


@pytest.fixture
def trained(tmp_path, panel, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        "urbanflow.modeling.candidates",
        lambda: {
            "seasonal_naive": SeasonalNaive(),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                loss="poisson", max_iter=8, early_stopping=False, random_state=42
            ),
            "xgboost_small": XGBRegressor(
                objective="count:poisson", n_estimators=8, max_depth=3, n_jobs=2, random_state=42
            ),
        },
    )
    settings = Settings(root=tmp_path)
    settings.prepare()
    path = settings.processed / "hourly.parquet"
    panel.to_parquet(path, index=False)
    report = train(path, settings.artifacts, settings.reports)
    return settings, report


def test_real_components_train_register_and_serve(trained, panel):
    settings, report = trained
    predictor = Predictor(settings.artifacts)
    target = pd.Timestamp(report["partitions"]["test"]["start"]).tz_localize("UTC")
    response = predictor.forecast([161], target)
    offline = make_features(panel)
    row = offline[(offline.zone_id == 161) & (offline.hour == target.tz_localize(None))]
    registered = mlflow.pyfunc.load_model("models:/UrbanFlowDemand/1")
    assert float(registered.predict(row[FEATURES])[0]) == pytest.approx(
        predict(predictor.model, row[FEATURES])[0]
    )
    assert response["predictions"][0]["predicted_pickups"] == pytest.approx(
        predict(predictor.model, row[FEATURES])[0]
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        result = client.post("/predict", json={"zone_ids": [161]})
        assert result.status_code == 200
        assert result.json()["mode"] == "next_hour"
        assert result.json()["predictions"][0]["lower_90"] >= 0
        for body in (
            {"zone_ids": [265]},
            {"zone_ids": [161, 161]},
            {"zone_ids": [161], "target_hour": "2030-01-01T00:00:00Z"},
            {"zone_ids": [161], "target_hour": "2024-02-25T00:00:00"},
            {"zone_ids": [161], "target_hour": "2024-02-25T00:30:00Z"},
            {"zone_ids": [161], "target_hour": "2024-01-05T00:00:00Z"},
        ):
            assert client.post("/predict", json=body).status_code == 422
        assert "urbanflow_model_ready 1" in client.get("/metrics").text
        assert client.get("/evaluation").json()["selected_model"] == report["selected_model"]


def test_corrupt_bundle_rejected(trained):
    settings, _ = trained
    pointer = json.loads((settings.artifacts / "champion.json").read_text())
    path = settings.artifacts / "versions" / pointer["version"] / "model.joblib"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(settings.artifacts)


def test_missing_model_has_liveness_but_not_readiness(tmp_path):
    with TestClient(create_app(Settings(root=tmp_path))) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"zone_ids": [161]}).status_code == 503


def test_failed_training_preserves_champion(trained, monkeypatch):
    settings, _ = trained
    before = (settings.artifacts / "champion.json").read_bytes()
    monkeypatch.setattr(
        "urbanflow.modeling.mlflow.sklearn.log_model",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tracking unavailable")),
    )
    with pytest.raises(RuntimeError, match="tracking unavailable"):
        train(settings.processed / "hourly.parquet", settings.artifacts, settings.reports)
    assert (settings.artifacts / "champion.json").read_bytes() == before
