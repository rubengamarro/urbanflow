import numpy as np
import pandas as pd
import pytest

from urbanflow.features import FEATURES, make_features
from urbanflow.modeling import conformal_radius, temporal_split, time_folds
from urbanflow.monitoring import population_stability


def test_future_labels_cannot_change_past_or_target_features(panel):
    target = panel.hour.max() - pd.Timedelta("24h")
    original = make_features(panel)
    mutated = panel.copy()
    mutated.loc[mutated.hour >= target, "demand"] += 10000
    updated = make_features(mutated)
    pd.testing.assert_frame_equal(
        original.loc[original.hour <= target, FEATURES],
        updated.loc[updated.hour <= target, FEATURES],
    )


def test_lags_and_windows_are_zone_local(panel):
    result = make_features(panel)
    row = result.iloc[0]
    history = panel[(panel.zone_id == row.zone_id) & (panel.hour < row.hour)]
    assert row.lag_1 == history.demand.iloc[-1]
    assert row.lag_168 == history.demand.iloc[-168]
    assert row.mean_24 == history.demand.tail(24).mean()
    assert len(result) == len(panel) - 168 * 3


def test_missing_and_duplicate_hours_fail(panel):
    with pytest.raises(ValueError, match="Duplicate"):
        make_features(pd.concat([panel, panel.iloc[:1]]))
    with pytest.raises(ValueError, match="complete"):
        make_features(panel.drop(index=10))


def test_temporal_split_keeps_hours_and_zones_together(panel):
    split = temporal_split(make_features(panel))
    partitions = [split.train, split.validation, split.calibration, split.test]
    for past, future in zip(partitions[:-1], partitions[1:], strict=True):
        assert past.hour.max() < future.hour.min()
    for frame in partitions:
        assert (frame.groupby("hour").zone_id.nunique() == 3).all()
    for past, future in time_folds(split.train):
        assert past.hour.max() < future.hour.min()


def test_insufficient_history_fails(panel):
    with pytest.raises(ValueError, match="Need"):
        temporal_split(make_features(panel[panel.hour < pd.Timestamp("2024-01-25")]))


def test_conformal_finite_sample_rank():
    assert conformal_radius(np.arange(10), 0.9) == 9
    with pytest.raises(ValueError):
        conformal_radius([], 0.9)


def test_psi_detects_distribution_change():
    values = np.arange(1000)
    assert population_stability(values, values) == pytest.approx(0)
    assert population_stability(values, values + 10000) > 0.2
