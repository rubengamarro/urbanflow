# Technical interview guide

Start with the prediction unit and observation assumption. Show official manifests and the full hourly grid; distinguish zero demand from missing months. Use the executed notebook to show that future-label perturbation leaves available features unchanged. Explain the baseline, grouped temporal folds, validation selection and separate calibration/test.

Show actual held-out rows and MAE/WAPE. Explain that test evaluation is rolling one-step, not a seven-day open-loop forecast, and nominal 90% interval coverage actually reached 88.3%. Call the API, inspect its version/cutoff, and explain atomic publication, restart adoption and preserving an old model on failure. Demonstrate the SQL transaction contract.

| Question | Defensible answer |
|---|---|
| Why observed pickups? | Directly measured; unmet demand requires request/supply information not present here. |
| Why no random split? | It leaks future regimes and nearby temporal patterns into fitting. |
| Why previous test labels in lags? | At each new origin the preceding hour is assumed observed; reporting delay changes the contract. |
| Why baseline eligible to win? | Complexity alone should not justify serving a weaker model. |
| Why Arrow and Spark? | Arrow suits this volume; Spark demonstrates equivalent distributed aggregation. |
| Why no fare/distance filter? | It would redefine pickup eligibility without a target requirement. |
| Why preserve repeated keys? | Zone/timestamp pairs are not unique trips. |
| Why not assume UTC source times? | TLC wall time is NYC; wrong localization shifts all demand/calendar features. |
| What does 21.9% improvement prove? | A measured reduction on this held-out winter week, not annual or live robustness. |
| Are intervals guaranteed? | No exchangeability guarantee for dependent time series; report actual coverage. |
| Is importance independent? | It uses validation with a training-only fitted model, not test. Correlation still limits interpretation. |
| What is missing for production? | Live observations, reporting-latency contract, auth, distributed storage/metrics, delayed labels and a promotion gate. |

Next experiments: new seasonal/holiday holdouts, categorical zone encoding, airport-specific errors, and weather forecasts available at issue time. Multi-horizon models require a matching product/evaluation contract. Reuse new forward holdouts when iterating rather than repeatedly tuning on the same test week. Separate measured findings from assumptions during the interview.
