from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def market_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=300)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)
