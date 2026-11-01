"""Risk-adjusted performance metrics computed from daily simple returns.

Every function here is pure (pandas in, floats out), so the maths can be unit-tested
without network access or a language model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_OBSERVATIONS = 15
_EPS = 1e-12


class InsufficientDataError(ValueError):
    """Raised when a return series is too short for meaningful statistics."""


@dataclass(frozen=True)
class RatioResults:
    """Annualized metrics for one return series. Ratios are None when undefined."""

    sharpe: float | None
    sortino: float | None
    treynor: float | None
    beta: float | None
    annualized_return: float
    annualized_volatility: float
    downside_deviation: float
    max_drawdown: float
    observations: int
    start: date
    end: date


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns from a price series."""
    prices = prices.dropna().sort_index()
    return (prices / prices.shift(1) - 1).dropna()


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR) implied by the return series."""
    growth = float((1 + returns).prod())
    return growth ** (TRADING_DAYS / len(returns)) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Sample standard deviation of daily returns, annualized."""
    return float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS)


def annualized_excess_return(returns: pd.Series, risk_free_rate: float) -> float:
    """Arithmetic mean daily return, annualized, minus the annual risk-free rate."""
    return float(returns.mean()) * TRADING_DAYS - risk_free_rate


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float | None:
    """Annualized excess return per unit of total volatility."""
    volatility = annualized_volatility(returns)
    if volatility < _EPS:
        return None
    return annualized_excess_return(returns, risk_free_rate) / volatility


def downside_deviation(returns: pd.Series, target: float) -> float:
    """Annualized downside deviation below an annual target return.

    Shortfalls are squared and averaged over *all* observations, not only the losing
    days. Averaging over the losing days alone overstates downside risk.
    """
    shortfall = np.minimum(returns - target / TRADING_DAYS, 0.0)
    return math.sqrt(float((shortfall**2).mean()) * TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float | None:
    """Annualized excess return per unit of downside deviation below the risk-free rate."""
    downside = downside_deviation(returns, risk_free_rate)
    if downside < _EPS:
        return None
    return annualized_excess_return(returns, risk_free_rate) / downside


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Sensitivity to the benchmark, estimated on the dates both series have returns."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        return None
    asset, market = aligned.iloc[:, 0], aligned.iloc[:, 1]
    variance = float(market.var(ddof=1))
    if variance < _EPS:
        return None
    return float(asset.cov(market, ddof=1)) / variance


def treynor_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float
) -> float | None:
    """Annualized excess return per unit of beta (market risk)."""
    asset_beta = beta(returns, benchmark_returns)
    if asset_beta is None or abs(asset_beta) < _EPS:
        return None
    return annualized_excess_return(returns, risk_free_rate) / asset_beta


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of cumulative wealth, as a negative fraction."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return float((wealth / peak - 1).min())


def compute_ratios(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    risk_free_rate: float,
) -> RatioResults:
    """Compute every metric for one daily return series indexed by date."""
    returns = returns.dropna()
    if len(returns) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_OBSERVATIONS} daily returns, got {len(returns)}; "
            "try a longer period"
        )
    has_benchmark = benchmark_returns is not None
    return RatioResults(
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        treynor=treynor_ratio(returns, benchmark_returns, risk_free_rate)
        if has_benchmark
        else None,
        beta=beta(returns, benchmark_returns) if has_benchmark else None,
        annualized_return=annualized_return(returns),
        annualized_volatility=annualized_volatility(returns),
        downside_deviation=downside_deviation(returns, risk_free_rate),
        max_drawdown=max_drawdown(returns),
        observations=len(returns),
        start=returns.index[0].date(),
        end=returns.index[-1].date(),
    )
