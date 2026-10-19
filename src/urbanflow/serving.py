import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from urbanflow.features import FEATURES, make_features
from urbanflow.io import sha256
from urbanflow.modeling import predict


class Predictor:
    """Immutable, checksum-verified snapshot loaded once at process startup."""

    def __init__(self, artifacts: Path, model_version: str | None = None):
        pointer = json.loads((artifacts / "champion.json").read_text(encoding="utf-8"))
        version = model_version or pointer["version"]
        if not isinstance(version, str) or not version.isalnum():
            raise ValueError("Invalid bundle version")
        bundle = artifacts / "versions" / version
        self.metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        for name, key in (("model.joblib", "model_sha256"), ("history.parquet", "history_sha256")):
            if sha256(bundle / name) != self.metadata[key]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        if self.metadata["features"] != FEATURES:
            raise ValueError("Feature contract mismatch; retrain with the current code")
        # Only locally generated trusted bundles may be deserialized.
        self.model = joblib.load(bundle / "model.joblib")
        self.history = pd.read_parquet(bundle / "history.parquet")
        self.evaluation = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
        self.latest = self.history.hour.max()
        self.zones = self.metadata["zones"]

    def forecast(self, zone_ids: list[int], target: pd.Timestamp | None = None) -> dict:
        if not zone_ids or len(set(zone_ids)) != len(zone_ids) or len(zone_ids) > 263:
            raise ValueError("Provide 1..263 unique zones")
        if not set(zone_ids).issubset(self.zones):
            raise ValueError("Requested zone is outside the trained scope")
        target = (
            target if target is not None else self.latest.tz_localize("UTC") + pd.Timedelta("1h")
        )
        if target.tzinfo is None:
            raise ValueError("target_hour must include a timezone offset")
        target = target.tz_convert("UTC").tz_localize(None)
        if target != target.floor("h"):
            raise ValueError("target_hour must be aligned to a whole UTC hour")
        if target <= pd.Timestamp(self.metadata["calibrated_through_utc"]):
            raise ValueError("Target must be after the model's calibration period")
        if target > self.latest + pd.Timedelta("1h"):
            raise ValueError(
                "Only one-hour-ahead forecasts are supported; newer observations needed"
            )
        start = target - pd.Timedelta("168h")
        history = self.history[
            self.history.zone_id.isin(zone_ids)
            & self.history.hour.between(start, target, inclusive="left")
        ]
        if len(history) != 168 * len(zone_ids):
            raise ValueError("A complete 168-hour observation window is required")
        # No target labels enter the shared feature function.
        target_rows = pd.DataFrame({"zone_id": zone_ids, "hour": target, "demand": np.nan})
        features = make_features(pd.concat([history, target_rows], ignore_index=True))
        features = features[features.hour.eq(target)].sort_values("zone_id")
        predictions = predict(self.model, features[FEATURES])
        results = []
        for zone, value in zip(features.zone_id, predictions, strict=True):
            radius = self.metadata["interval_radii"][str(zone)]
            results.append(
                {
                    "zone_id": int(zone),
                    "predicted_pickups": float(value),
                    "lower_90": float(max(0, value - radius)),
                    "upper_90": float(value + radius),
                }
            )
        return {
            "target_hour": target.tz_localize("UTC").isoformat(),
            "observed_through": (target - pd.Timedelta("1h")).tz_localize("UTC").isoformat(),
            "model_version": self.metadata["version"],
            "mode": "next_hour" if target > self.latest else "historical_replay",
            "predictions": results,
        }
