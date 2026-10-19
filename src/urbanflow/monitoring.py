from pathlib import Path

import numpy as np
import pandas as pd

from urbanflow.features import FEATURES, make_features
from urbanflow.io import write_json


def population_stability(reference, current) -> float:
    """PSI using reference quantiles; infinities capture values outside training range."""
    internal = np.unique(np.quantile(reference, np.linspace(0.1, 0.9, 9)))
    bins = np.concatenate(([-np.inf], internal, [np.inf]))
    old = np.maximum(np.histogram(reference, bins)[0] / len(reference), 1e-6)
    new = np.maximum(np.histogram(current, bins)[0] / len(current), 1e-6)
    return float(np.sum((new - old) * np.log(new / old)))


def monitor(panel_path: Path, evaluation: dict, reports: Path) -> dict:
    features = make_features(pd.read_parquet(panel_path))
    train_end = pd.Timestamp(evaluation["partitions"]["train"]["end"])
    test_start = pd.Timestamp(evaluation["partitions"]["test"]["start"])
    reference = features[features.hour <= train_end]
    recent = features[features.hour >= test_start]
    psi = {f: population_stability(reference[f], recent[f]) for f in FEATURES if f != "zone_id"}
    result = {
        "reference": "training period",
        "current": "test period (offline diagnostic)",
        "psi": psi,
        "psi_alert_threshold": 0.2,
        "alert_features": [f for f, score in psi.items() if score > 0.2],
        "coverage_90": evaluation["test_coverage_90"],
        "coverage_below_85": evaluation["test_coverage_90"] < 0.85,
        "note": "Heuristic alerts; drift alone is not a reason to deploy a new model.",
    }
    write_json(reports / "monitoring.json", result)
    return result
