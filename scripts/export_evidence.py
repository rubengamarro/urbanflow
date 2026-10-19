"""Publish a small, auditable report without bundling raw data or pickled models."""

import argparse
import json
import shutil
from pathlib import Path

from urbanflow.config import Settings
from urbanflow.io import sha256, write_json
from urbanflow.serving import Predictor


def export(root: Path, destination: Path) -> None:
    settings = Settings(root=root)
    model = Predictor(settings.artifacts)
    destination.mkdir(parents=True, exist_ok=True)
    files = [settings.raw / "manifest.json", settings.processed / "quality.json"]
    files.extend(
        settings.reports / name
        for name in (
            "evaluation.json",
            "eda.json",
            "monitoring.json",
            "per_zone.csv",
            "top_zones.csv",
            "validation_feature_importance.csv",
            "eda.png",
            "test_predictions.parquet",
        )
    )
    for path in files:
        shutil.copy2(path, destination / path.name)
    write_json(destination / "inference_example.json", model.forecast([161, 132, 138]))
    report = model.evaluation
    manifest = {
        "model_version": model.metadata["version"],
        "pipeline_status": "Completed",
        "files": {
            p.name: sha256(p)
            for p in destination.iterdir()
            if p.is_file() and p.name != "evidence.json"
        },
        "note": "Outputs from a real TLC benchmark; test fixtures are never exported.",
    }
    write_json(destination / "evidence.json", manifest)
    gain = 1 - report["test"]["mae"] / report["test_baseline"]["mae"]
    print(
        json.dumps(
            {
                "selected_model": report["selected_model"],
                "test": report["test"],
                "mae_improvement": gain,
                "coverage_90": report["test_coverage_90"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, default=Path("reports/reference"))
    options = parser.parse_args()
    export(options.root, options.destination)
