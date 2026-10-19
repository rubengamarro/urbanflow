# Model card

**Use:** historical NYC yellow taxi pickup forecasting, engineering evaluation and interviews. Output: expected eligible pickups in one zone for the next hour.

**Reference estimator:** XGBoost Poisson regression, 240 histogram trees, depth 7, learning rate .05, seed 42, two CPU workers. Numeric zone ID, NYC calendar cycles, weekend, past counts and shifted summaries. Parameters are tracked in MLflow.

**Scope:** all 263 geographic zones, January–February 2024. Fitted through 2024-02-16 04:00 UTC, calibrated through 2024-02-23 04:00 UTC. Stored observations end at 2024-03-01 04:00 UTC.

**Measured test:** rolling one-hour final week. MAE 3.0956, RMSE 10.7044, WAPE 17.9843%; baseline MAE 3.9627. Nominal 90% intervals cover 88.2763%. See reference prediction rows, zone metrics and bootstrap estimates.

**Out of scope:** live fleet dispatch, unmet-demand inference, causal mobility policies, arbitrary multi-hour forecasts, FHV/green taxis, unseen zones, autumn ambiguous timestamps, annual robustness claims or guaranteed fleet allocations.

**Failure modes:** source omissions, events, weather/holidays, policy/regime changes, zone mix drift and conditions outside winter 2024. Sparse rows can mask airport/busy-zone errors. Numeric tree splits on zone IDs do not represent geography. Residual intervals may under-cover after a shift.

**Serving controls:** reject unsupported/duplicate zones, naive or fractional timestamps, targets before calibration, incomplete history or unavailable future observations. Verified trusted bundles only; readiness 503 without a model. Restart after pointer changes, retain old bundles for rollback.

**Monitoring:** per-process service counters plus offline PSI/coverage. Stable distributions do not prove accuracy; live error monitoring needs delayed labels and genuine hourly observations.

**Fairness:** historical pickups reflect access and supply. Low recorded activity is not evidence of low mobility need; the model should not automatically deprioritize communities or stand in for causal fairness analysis.
