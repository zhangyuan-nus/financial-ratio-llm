from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def chart_payload() -> dict:
    """A real get-chart response for AAPL, trimmed to five trading days."""
    return json.loads((FIXTURES / "aapl_chart.json").read_text())


@pytest.fixture
def market_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=300)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)
