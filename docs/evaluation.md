# Evaluation methodology

## Information available at issuance

Each row is a zone/target hour. All history ends at the preceding hour. Rolling features shift before aggregation; calendar features are known in advance. The weekly lag causes 168 hours of burn-in. No target-hour trips, observed fare totals or realized future weather are features.

The final evaluation is **rolling one-step-ahead**, assuming the preceding hour's actual pickups have arrived. It is not one seven-day forecast issued at the test start. Earlier test labels entering later lags are expected information under this contract. Reporting delays or a multi-hour horizon require a new feature/evaluation design.

## Reference split

| Partition | UTC start | UTC end | Rows |
|---|---|---|---:|
| Train | 2024-01-08 05:00 | 2024-02-09 04:00 | 201,984 |
| Validation | 2024-02-09 05:00 | 2024-02-16 04:00 | 44,184 |
| Calibration | 2024-02-16 05:00 | 2024-02-23 04:00 | 44,184 |
| Test | 2024-02-23 05:00 | 2024-03-01 04:00 | 44,184 |

Three expanding TimeSeriesSplit folds run inside training. Splits operate on unique hours, then map back to all zones; a timestamp cannot appear in both fitting and fold validation. There is no gap because the one-hour label is assumed available at the next origin. A reporting-delay contract would require corresponding delayed lags.

## Declared candidates

| Candidate | Main configuration | Validation MAE |
|---|---|---:|
| Weekly naive | t−168 count | 3.7757 |
| HistGradientBoosting | Poisson, 120 iterations, 31 leaves, lr .08 | 3.2365 |
| XGBoost small | Poisson, 120 trees, depth 5, lr .08 | 3.3056 |
| XGBoost deep | Poisson, 240 trees, depth 7, lr .05 | 3.1933 |

Validation MAE selects the model, including the baseline if appropriate. CV describes stability. Refit the winner on training+validation, calibrate on a distinct held-out block and evaluate once on test. The exported weights are not refitted on test labels, preserving the evaluated model. Training never reads the committed reference report.

## Metrics

MAE measures absolute pickups per zone-hour. RMSE emphasizes large misses. WAPE divides total absolute error by total actual pickups; it avoids individual percentage errors on zero counts. The denominator is floored at one for a zero-total cohort; that degenerate result represents absolute error, not a meaningful percentage. Poisson deviance clips predicted rates at 1e−6 only for metric computation. Per-zone reports and a top-20 cohort chosen by training demand reveal masking by sparse zones.

## Calibration and uncertainty

Each zone's absolute calibration residuals define its interval radius. For n residuals and nominal coverage c, use sorted rank `ceil((n+1)c)`, capped at n. Lower endpoints are clipped to zero. Exchangeability is not guaranteed for dependent time series; actual test coverage is **88.28%**, below the nominal 90%. Per-zone coverage is exported.

A seeded 2,000-resample bootstrap uses complete 24-hour blocks anchored at the test start, keeping all zones together. It estimates champion MAE and paired MAE improvement. Only seven blocks are available; this is descriptive uncertainty and cannot establish seasonal robustness or eliminate interday dependence.

Permutation importance increases MAE on validation for a model fitted only on training. It is explanatory, not causal. Correlated lags can dilute individual scores. Importance does not select models on the test period.

## Evidence and further validation

Every held-out prediction, source/data hash, cutoff, package version and run ID is exported. UUIDs and durations change across runs; compare numerical tolerances rather than identical model binaries. Expand disjoint seasonal/regime/holiday/airport backtests, compare categorical zone encodings, model reporting latency and use only external forecasts available at the original issuance time. Repeatedly tuning against this same held-out week would compromise its independence; create new forward holdouts when iterating.
