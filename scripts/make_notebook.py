"""Build an executable notebook whose tables always come from the real processed data."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbf.v4.new_markdown_cell("""# UrbanFlow AI — auditable data and forecasting analysis

Run `urbanflow pipeline` first. Every table below reads the real TLC outputs; no synthetic fallback.
The target is observed eligible yellow-taxi pickups, not unserved passenger demand.
All stored timestamps are UTC. NYC calendar features are converted explicitly.
"""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from urbanflow.features import FEATURES, make_features
from urbanflow.modeling import temporal_split
root = Path.cwd()
if not (root / 'data/processed/hourly.parquet').exists():
    root = root.parent
panel = pd.read_parquet(root / 'data/processed/hourly.parquet')
quality = json.loads((root / 'data/processed/quality.json').read_text())
quality"""),
    nbf.v4.new_code_cell("""panel.describe(include='all')"""),
    nbf.v4.new_code_cell("""utc = panel.hour.dt.tz_localize('UTC')
local = utc.dt.tz_convert('America/New_York')
daily = panel.groupby(local.dt.tz_localize(None).dt.floor('D')).demand.sum()
daily.plot(figsize=(12, 3), title='Observed pickups by NYC day', ylabel='Pickups');"""),
    nbf.v4.new_code_cell("""zones = pd.read_csv(root / 'data/raw/taxi_zone_lookup.csv')
top = panel.groupby('zone_id').demand.sum().nlargest(20).reset_index()
joined = top.merge(zones, left_on='zone_id', right_on='LocationID')
joined[['zone_id','Zone','Borough','demand']]"""),
    nbf.v4.new_markdown_cell("""## The information boundary

Every lag and rolling window ends before the target hour. The seven-day burn-in is dropped.
Chronological folds contain all zones for each hour; random row splits would leak time.
Test evaluation is rolling one-step: the previous hour's actual count is assumed to have arrived.
"""),
    nbf.v4.new_code_cell("""features = make_features(panel)
split = temporal_split(features)
pd.DataFrame({name: {'start': getattr(split, name).hour.min(),
                     'end': getattr(split, name).hour.max(),
                     'rows': len(getattr(split, name))}
              for name in ('train', 'validation', 'calibration', 'test')}).T"""),
    nbf.v4.new_code_cell("""cutoff = features.hour.max() - pd.Timedelta(hours=24)
changed = panel.copy()
changed.loc[changed.hour >= cutoff, 'demand'] += 10000
changed_features = make_features(changed)
pd.testing.assert_frame_equal(features.loc[features.hour <= cutoff, FEATURES],
                              changed_features.loc[changed_features.hour <= cutoff, FEATURES])
print('Future-label perturbation leaves earlier and target-hour features unchanged.')"""),
    nbf.v4.new_code_cell("""evaluation_path = root / 'reports/latest/evaluation.json'
evaluation = json.loads(evaluation_path.read_text())
pd.DataFrame(evaluation['validation']).T.sort_values('mae')"""),
    nbf.v4.new_code_cell(
        """pd.DataFrame(evaluation['cross_validation']).groupby('model').mae.agg(['mean','std'])"""
    ),
    nbf.v4.new_code_cell(
        """results = {'champion': evaluation['test'],
           'weekly_baseline': evaluation['test_baseline']}
pd.DataFrame(results).T"""
    ),
    nbf.v4.new_code_cell(
        """pd.read_csv(root / 'reports/latest/validation_feature_importance.csv')"""
    ),
    nbf.v4.new_code_cell("""prediction_path = root / 'reports/latest/test_predictions.parquet'
predictions = pd.read_parquet(prediction_path)
zone = predictions[predictions.zone_id.eq(161)].set_index('hour')
ax = zone[['demand', 'prediction', 'baseline']].plot(
    figsize=(14, 4), title='Midtown Center — rolling one-hour test forecasts')
ax.fill_between(zone.index, zone.lower, zone.upper, alpha=0.15, label='Nominal 90% interval')
ax.legend();"""),
    nbf.v4.new_code_cell("""per_zone = pd.read_csv(root / 'reports/latest/per_zone.csv')
per_zone.sort_values('actual_total', ascending=False).head(20)"""),
    nbf.v4.new_code_cell("""{'coverage_90': evaluation['test_coverage_90'],
 'bootstrap': evaluation['bootstrap'],
 'busy_zones': evaluation['test_top20_training_demand_zones']}"""),
    nbf.v4.new_markdown_cell("""## Interpretation and limits

Global MAE includes sparse zones; inspect WAPE and the top-20 cohort chosen using training counts.
Permutation importance is measured on validation with a model fitted only on training.
Correlated lag features dilute individual importances; shuffled rows are an explanatory diagnostic.
Conformal-style residual intervals do not have an exchangeability guarantee for a time series.
Seven test days are a demonstration, not evidence of full-year robustness. More seasons, holiday
backtests, weather forecasts available at issue time, categorical zone encoding, and hourly online
observation ingestion are explicit next steps. TLC monthly release data cannot power live dispatch.
"""),
]
destination = Path("notebooks/01_urbanflow_analysis.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, destination)
