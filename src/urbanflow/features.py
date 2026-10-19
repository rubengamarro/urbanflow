import numpy as np
import pandas as pd

FEATURES = [
    "zone_id",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "weekend",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_168",
    "mean_24",
    "mean_168",
    "std_24",
]


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Target hour t uses pickups <= t-1. UTC grid preserves real elapsed lags."""
    frame = panel.sort_values(["zone_id", "hour"]).copy()
    if frame.duplicated(["zone_id", "hour"]).any():
        raise ValueError("Duplicate zone-hour observations")
    deltas = frame.groupby("zone_id").hour.diff().dropna()
    if not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Panel must have a complete hourly grid")
    local = frame.hour.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    frame["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7)
    frame["weekend"] = (local.dt.dayofweek >= 5).astype(int)
    grouped = frame.groupby("zone_id", sort=False).demand
    for lag in (1, 2, 24, 168):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in (24, 168):
        frame[f"mean_{window}"] = grouped.transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=w).mean()
        )
    frame["std_24"] = grouped.transform(lambda x: x.shift(1).rolling(24).std())
    return frame.dropna(subset=FEATURES).reset_index(drop=True)
