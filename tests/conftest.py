import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel():
    """Explicitly synthetic test fixture; never used for published benchmarks."""
    rng = np.random.default_rng(42)
    hours = pd.date_range("2024-01-01 05:00", periods=60 * 24, freq="h")
    frames = []
    for zone in (4, 161, 263):
        intensity = 10 + 8 * np.sin(2 * np.pi * (hours.hour - 7) / 24)
        frames.append(
            pd.DataFrame({"hour": hours, "zone_id": zone, "demand": rng.poisson(intensity)})
        )
    return pd.concat(frames, ignore_index=True)
