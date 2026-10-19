import os
import platform
import time
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from urbanflow.features import FEATURES, make_features
from urbanflow.io import sha256, write_json


class SeasonalNaive(RegressorMixin, BaseEstimator):
    """Strong weekly baseline, using the identical serving feature contract."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return X["lag_168"].to_numpy(dtype=float)


def candidates() -> dict:
    return {
        "seasonal_naive": SeasonalNaive(),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=120,
            max_leaf_nodes=31,
            learning_rate=0.08,
            early_stopping=False,
            random_state=42,
        ),
        "xgboost_small": XGBRegressor(
            objective="count:poisson",
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        ),
        "xgboost_deep": XGBRegressor(
            objective="count:poisson",
            n_estimators=240,
            max_depth=7,
            learning_rate=0.05,
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        ),
    }


def predict(model, X: pd.DataFrame) -> np.ndarray:
    result = np.maximum(np.asarray(model.predict(X), dtype=float), 0)
    if not np.isfinite(result).all():
        raise ValueError("Model returned non-finite predictions")
    return result


def metrics(y, prediction) -> dict[str, float]:
    y, prediction = np.asarray(y), np.asarray(prediction)
    denominator = float(np.abs(y).sum())
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "wape": float(np.abs(y - prediction).sum() / max(denominator, 1)),
        "poisson_deviance": float(mean_poisson_deviance(y, np.maximum(prediction, 1e-6))),
    }


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame, holdout_days: int = 7) -> Split:
    """All zones for an hour share a partition. Never random-split panel rows."""
    if holdout_days < 2:
        raise ValueError("At least two holdout days are required")
    hours = pd.DatetimeIndex(sorted(frame.hour.unique()))
    holdout = holdout_days * 24
    if len(hours) < 3 * holdout + 14 * 24:
        raise ValueError("Need 14 training days after lag burn-in plus three holdout periods")
    val_start, cal_start, test_start = hours[-3 * holdout], hours[-2 * holdout], hours[-holdout]
    return Split(
        frame[frame.hour < val_start],
        frame[(frame.hour >= val_start) & (frame.hour < cal_start)],
        frame[(frame.hour >= cal_start) & (frame.hour < test_start)],
        frame[frame.hour >= test_start],
    )


def time_folds(frame: pd.DataFrame, n_splits: int = 3):
    hours = np.array(sorted(frame.hour.unique()))
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=n_splits).split(hours):
        yield frame[frame.hour.isin(hours[train_idx])], frame[frame.hour.isin(hours[valid_idx])]


def conformal_radius(residuals, coverage: float = 0.9) -> float:
    residuals = np.asarray(residuals)
    if not 0 < coverage < 1 or not len(residuals):
        raise ValueError("Invalid coverage or empty calibration set")
    rank = min(int(np.ceil((len(residuals) + 1) * coverage)), len(residuals))
    return float(np.sort(residuals)[rank - 1])


def daily_bootstrap(test: pd.DataFrame, baseline, champion, iterations: int = 2000) -> dict:
    blocks = ((test.hour - test.hour.min()).dt.total_seconds() // 86400).astype(int)
    errors = pd.DataFrame(
        {
            "day": blocks,
            "mae": np.abs(test.demand.to_numpy() - champion),
            "gain": np.abs(test.demand.to_numpy() - baseline)
            - np.abs(test.demand.to_numpy() - champion),
        }
    )
    daily = errors.groupby("day")[["mae", "gain"]].mean().to_numpy()
    rng = np.random.default_rng(42)
    boot = daily[rng.integers(0, len(daily), size=(iterations, len(daily)))].mean(axis=1)
    return {
        "unit": "24-hour block anchored at test start, all zones kept together",
        "days": len(daily),
        "iterations": iterations,
        "mae_ci95": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
        "paired_mae_gain_ci95": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
    }


def train(panel_path: Path, artifacts: Path, reports: Path, holdout_days: int = 7) -> dict:
    started = time.perf_counter()
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    frame = make_features(panel)
    split = temporal_split(frame, holdout_days)
    tracking = os.getenv("MLFLOW_TRACKING_URI") or (
        "sqlite:///" + (artifacts / "mlflow.db").resolve().as_posix()
    )
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment("UrbanFlow-hourly-demand")
    scores, cv = {}, []
    models = candidates()
    with mlflow.start_run(run_name="temporal-benchmark") as run:
        mlflow.log_params(
            {
                "seed": 42,
                "holdout_days": holdout_days,
                "features": ",".join(FEATURES),
                "data_sha256": sha256(panel_path),
            }
        )
        for name, estimator in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                for fold, (past, future) in enumerate(time_folds(split.train)):
                    fitted = clone(estimator).fit(past[FEATURES], past.demand)
                    score = metrics(future.demand, predict(fitted, future[FEATURES]))
                    cv.append(
                        {
                            "model": name,
                            "fold": fold,
                            "train_end": str(past.hour.max()),
                            "validation_start": str(future.hour.min()),
                            **score,
                        }
                    )
                    mlflow.log_metrics({f"cv_{key}": val for key, val in score.items()}, step=fold)
                fitted = clone(estimator).fit(split.train[FEATURES], split.train.demand)
                scores[name] = metrics(
                    split.validation.demand, predict(fitted, split.validation[FEATURES])
                )
                mlflow.log_metrics({f"validation_{key}": val for key, val in scores[name].items()})
                mlflow.log_params(estimator.get_params())
        # Selection happens before calibration and test. Baseline can win.
        winner = min(scores, key=lambda name: scores[name]["mae"])
        interpretation_model = clone(models[winner]).fit(split.train[FEATURES], split.train.demand)
        importance = permutation_importance(
            interpretation_model,
            split.validation[FEATURES],
            split.validation.demand,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=42,
            n_jobs=1,
        )
        pd.DataFrame(
            {
                "feature": FEATURES,
                "validation_mae_increase": importance.importances_mean,
                "std": importance.importances_std,
            }
        ).sort_values("validation_mae_increase", ascending=False).to_csv(
            reports / "validation_feature_importance.csv", index=False
        )
        fit_data = pd.concat([split.train, split.validation])
        model = clone(models[winner]).fit(fit_data[FEATURES], fit_data.demand)
        cal = split.calibration.copy()
        cal["residual"] = np.abs(cal.demand - predict(model, cal[FEATURES]))
        radii = {
            str(zone): conformal_radius(group.residual) for zone, group in cal.groupby("zone_id")
        }
        test = split.test.copy()
        test["prediction"] = predict(model, test[FEATURES])
        test["baseline"] = test.lag_168
        test["radius"] = test.zone_id.astype(str).map(radii)
        test["lower"] = np.maximum(0, test.prediction - test.radius)
        test["upper"] = test.prediction + test.radius
        test["covered"] = test.demand.between(test.lower, test.upper)
        test_scores = metrics(test.demand, test.prediction)
        baseline_scores = metrics(test.demand, test.baseline)
        per_zone = []
        for zone, group in test.groupby("zone_id"):
            per_zone.append(
                {
                    "zone_id": int(zone),
                    "actual_total": int(group.demand.sum()),
                    "coverage": float(group.covered.mean()),
                    **metrics(group.demand, group.prediction),
                    "baseline_mae": float(mean_absolute_error(group.demand, group.baseline)),
                }
            )
        busy_zones = split.train.groupby("zone_id").demand.mean().nlargest(20).index
        busy = test[test.zone_id.isin(busy_zones)]
        version_id = uuid.uuid4().hex[:12]
        bundle = artifacts / "versions" / version_id
        bundle.mkdir(parents=True)
        joblib.dump(model, bundle / "model.joblib")
        panel.to_parquet(bundle / "history.parquet", index=False)
        report = {
            "model_version": version_id,
            "selected_model": winner,
            "run_id": run.info.run_id,
            "selection_metric": "validation MAE (test never used for selection)",
            "validation": scores,
            "cross_validation": cv,
            "test": test_scores,
            "test_baseline": baseline_scores,
            "test_top20_training_demand_zones": {
                "zones": [int(z) for z in busy_zones],
                "model": metrics(busy.demand, busy.prediction),
                "baseline": metrics(busy.demand, busy.baseline),
            },
            "test_coverage_90": float(test.covered.mean()),
            "test_mean_interval_width": float((test.upper - test.lower).mean()),
            "bootstrap": daily_bootstrap(
                test, test.baseline.to_numpy(), test.prediction.to_numpy()
            ),
            "partitions": {
                name: {
                    "start": str(getattr(split, name).hour.min()),
                    "end": str(getattr(split, name).hour.max()),
                    "rows": len(getattr(split, name)),
                }
                for name in ("train", "validation", "calibration", "test")
            },
            "data_sha256": sha256(panel_path),
            "features": FEATURES,
            "versions": {p: version(p) for p in ("pandas", "scikit-learn", "xgboost", "mlflow")},
            "python": platform.python_version(),
            "duration_seconds": time.perf_counter() - started,
        }
        metadata = {
            "version": version_id,
            "model": winner,
            "features": FEATURES,
            "interval_radii": radii,
            "trained_through_utc": str(fit_data.hour.max()),
            "calibrated_through_utc": str(cal.hour.max()),
            "history_start_utc": str(panel.hour.min()),
            "history_end_utc": str(panel.hour.max()),
            "zones": sorted(int(x) for x in panel.zone_id.unique()),
            "model_sha256": sha256(bundle / "model.joblib"),
            "history_sha256": sha256(bundle / "history.parquet"),
            "data_sha256": report["data_sha256"],
            "run_id": run.info.run_id,
            "versions": report["versions"],
        }
        write_json(bundle / "metadata.json", metadata)
        write_json(bundle / "evaluation.json", report)
        write_json(reports / "evaluation.json", report)
        pd.DataFrame(per_zone).to_csv(reports / "per_zone.csv", index=False)
        test[["hour", "zone_id", "demand", "prediction", "baseline", "lower", "upper"]].to_parquet(
            reports / "test_predictions.parquet", index=False
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items()})
        mlflow.log_metric("test_coverage_90", report["test_coverage_90"])
        mlflow.log_param("selected_model", winner)
        mlflow.log_artifacts(str(reports), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=fit_data[FEATURES].head(3),
            signature=infer_signature(
                fit_data[FEATURES].head(3), predict(model, fit_data[FEATURES].head(3))
            ),
            registered_model_name="UrbanFlowDemand",
            pip_requirements=[
                f"{p}=={version(p)}"
                for p in ("scikit-learn", "xgboost", "pandas", "numpy", "joblib")
            ],
            code_paths=[str(Path(__file__).parent)],
        )
        # Publish only a complete, tracked bundle. A failed training run leaves champion intact.
        write_json(artifacts / "champion.json", {"version": version_id})
    return report
