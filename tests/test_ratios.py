import math

import pandas as pd
import pytest

from finratio.ratios import (
    MIN_OBSERVATIONS,
    TRADING_DAYS,
    InsufficientDataError,
    annualized_return,
    beta,
    compute_ratios,
    downside_deviation,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    treynor_ratio,
)


def daily(values, start="2026-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


RETURNS = daily([0.010, -0.020, 0.015, -0.005, 0.030, -0.010, 0.000, 0.020, -0.015, 0.005] * 3)


def test_simple_returns_from_prices():
    assert simple_returns(daily([100.0, 110.0, 99.0])).tolist() == pytest.approx([0.10, -0.10])


def test_sharpe_matches_the_textbook_definition():
    rf = 0.05
    excess = RETURNS - rf / TRADING_DAYS
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(TRADING_DAYS)
    assert sharpe_ratio(RETURNS, rf) == pytest.approx(expected)


def test_sharpe_is_undefined_without_volatility():
    assert sharpe_ratio(daily([0.001] * 30), 0.0) is None


def test_downside_deviation_averages_over_all_observations():
    # Two of the four days fall short of a 0% target, by 2% and 4%.
    returns = daily([0.01, -0.02, 0.03, -0.04])
    expected = math.sqrt((0.02**2 + 0.04**2) / 4) * math.sqrt(TRADING_DAYS)
    assert downside_deviation(returns, target=0.0) == pytest.approx(expected)


def test_sortino_divides_excess_return_by_downside_deviation():
    rf = 0.03
    expected = (RETURNS.mean() * TRADING_DAYS - rf) / downside_deviation(RETURNS, rf)
    assert sortino_ratio(RETURNS, rf) == pytest.approx(expected)


def test_sortino_is_undefined_without_downside():
    assert sortino_ratio(daily([0.01, 0.02] * 10), 0.0) is None


def test_beta_recovers_a_known_sensitivity(market_returns):
    assert beta(2 * market_returns, market_returns) == pytest.approx(2.0)
    assert beta(market_returns, market_returns) == pytest.approx(1.0)


def test_beta_uses_overlapping_dates_when_histories_differ(market_returns):
    asset = 1.5 * market_returns.iloc[20:]
    assert beta(asset, market_returns) == pytest.approx(1.5)


def test_beta_needs_enough_overlap(market_returns):
    assert beta(market_returns.iloc[: MIN_OBSERVATIONS - 1], market_returns) is None


def test_treynor_is_excess_return_per_unit_of_beta(market_returns):
    asset, rf = 2 * market_returns, 0.04
    expected = (asset.mean() * TRADING_DAYS - rf) / 2.0
    assert treynor_ratio(asset, market_returns, rf) == pytest.approx(expected)


def test_annualized_return_compounds_daily_returns():
    assert annualized_return(daily([0.001] * TRADING_DAYS)) == pytest.approx(
        1.001**TRADING_DAYS - 1
    )


def test_max_drawdown_measures_peak_to_trough():
    # Wealth goes 1.0 -> 1.2 -> 0.9 -> 1.1, so the worst decline is 0.9 / 1.2 - 1.
    returns = simple_returns(daily([1.0, 1.2, 0.9, 1.1]))
    assert max_drawdown(returns) == pytest.approx(-0.25)


def test_max_drawdown_counts_a_loss_on_the_first_day():
    assert max_drawdown(daily([-0.10, 0.05])) == pytest.approx(-0.10)


def test_compute_ratios_rejects_short_histories():
    with pytest.raises(InsufficientDataError):
        compute_ratios(daily([0.01] * (MIN_OBSERVATIONS - 1)), risk_free_rate=0.0)


def test_compute_ratios_without_benchmark_skips_market_metrics():
    results = compute_ratios(RETURNS, risk_free_rate=0.02)
    assert results.beta is None
    assert results.treynor is None
    assert results.observations == len(RETURNS)
    assert results.start == RETURNS.index[0].date()
    assert results.end == RETURNS.index[-1].date()
