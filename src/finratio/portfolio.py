"""Portfolio construction from individual holdings."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Scale weights to sum to 1, so 60/40 and 0.6/0.4 describe the same portfolio."""
    if not weights:
        raise ValueError("portfolio weights are empty")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("negative weights (short positions) are not supported")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("portfolio weights must add up to more than zero")
    return {symbol: weight / total for symbol, weight in weights.items()}


def portfolio_returns(returns: Mapping[str, pd.Series], weights: Mapping[str, float]) -> pd.Series:
    """Daily returns of a portfolio rebalanced back to `weights` at every close.

    Only dates on which every holding has a return are used, so a missing price
    never silently shifts the allocation.
    """
    frame = pd.concat({symbol: returns[symbol] for symbol in weights}, axis=1, join="inner")
    frame = frame.dropna()
    return (frame * pd.Series(weights)).sum(axis=1).rename("portfolio")
