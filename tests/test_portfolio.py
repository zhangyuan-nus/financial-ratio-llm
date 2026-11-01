import pandas as pd
import pytest

from finratio.portfolio import normalize_weights, portfolio_returns


def test_percentages_and_fractions_describe_the_same_portfolio():
    expected = {"AAPL": 0.6, "MSFT": 0.4}
    assert normalize_weights({"AAPL": 60, "MSFT": 40}) == pytest.approx(expected)
    assert normalize_weights({"AAPL": 0.6, "MSFT": 0.4}) == pytest.approx(expected)


@pytest.mark.parametrize("weights", [{}, {"AAPL": -10, "MSFT": 110}, {"AAPL": 0, "MSFT": 0}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        normalize_weights(weights)


def test_portfolio_returns_use_dates_shared_by_all_holdings():
    dates = pd.bdate_range("2026-01-05", periods=4)
    returns = {
        "AAPL": pd.Series([0.01, 0.02, -0.01, 0.03], index=dates),
        "MSFT": pd.Series([0.03, -0.01, 0.02], index=dates[1:]),
    }
    result = portfolio_returns(returns, {"AAPL": 0.5, "MSFT": 0.5})
    assert list(result.index) == list(dates[1:])
    assert result.tolist() == pytest.approx([0.025, -0.01, 0.025])
