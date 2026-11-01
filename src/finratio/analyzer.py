"""Core pipeline: fetch prices for a request and compute ratios for every holding."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .market_data import PERIODS, PriceSource
from .portfolio import normalize_weights, portfolio_returns
from .ratios import InsufficientDataError, RatioResults, compute_ratios, simple_returns

MAX_SYMBOLS = 10
_TICKER = re.compile(r"\^?[A-Z0-9][A-Z0-9.=\-]{0,19}")


def normalize_symbol(symbol: str) -> str:
    """Upper-case a ticker and check that it looks like a Yahoo Finance symbol."""
    cleaned = symbol.strip().upper()
    if not _TICKER.fullmatch(cleaned):
        raise ValueError(f"{symbol!r} does not look like a ticker symbol")
    return cleaned


@dataclass(frozen=True)
class AnalysisRequest:
    """What to analyze. Values are normalized and validated on construction."""

    symbols: Sequence[str]
    period: str = "1y"
    benchmark: str = "^GSPC"
    risk_free_rate: float = 0.05
    weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        raw = (self.symbols,) if isinstance(self.symbols, str) else self.symbols
        symbols = tuple(dict.fromkeys(normalize_symbol(s) for s in raw))
        if not symbols:
            raise ValueError("at least one ticker is required")
        if len(symbols) > MAX_SYMBOLS:
            raise ValueError(f"at most {MAX_SYMBOLS} tickers can be analyzed at once")
        if self.period not in PERIODS:
            raise ValueError(
                f"unsupported period {self.period!r}; choose from {', '.join(PERIODS)}"
            )
        if not -0.05 <= self.risk_free_rate <= 0.25:
            raise ValueError("the risk-free rate must be an annual decimal, e.g. 0.045 for 4.5%")

        weights = None
        if self.weights is not None and len(symbols) > 1:
            weights = normalize_weights({normalize_symbol(s): w for s, w in self.weights.items()})
            if set(weights) != set(symbols):
                raise ValueError("portfolio weights must cover exactly the requested tickers")

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "benchmark", normalize_symbol(self.benchmark))
        object.__setattr__(self, "weights", weights)

    @property
    def kind(self) -> Literal["single", "comparison", "portfolio"]:
        if self.weights is not None:
            return "portfolio"
        return "single" if len(self.symbols) == 1 else "comparison"


@dataclass(frozen=True)
class Analysis:
    """Ratios for each requested symbol, the portfolio (if any), and the benchmark."""

    request: AnalysisRequest
    assets: dict[str, RatioResults]
    benchmark: RatioResults
    portfolio: RatioResults | None = None


async def analyze(request: AnalysisRequest, source: PriceSource) -> Analysis:
    """Fetch prices for every ticker plus the benchmark, then compute all ratios."""
    tickers = list(dict.fromkeys([*request.symbols, request.benchmark]))
    prices = await source.get_prices(tickers, request.period)
    returns = {symbol: simple_returns(series) for symbol, series in prices.items()}
    market = returns[request.benchmark]

    def ratios(name: str, series: pd.Series) -> RatioResults:
        try:
            return compute_ratios(series, market, risk_free_rate=request.risk_free_rate)
        except InsufficientDataError as exc:
            raise InsufficientDataError(f"{name}: {exc}") from None

    portfolio = None
    if request.weights is not None:
        portfolio = ratios("portfolio", portfolio_returns(returns, request.weights))
    return Analysis(
        request=request,
        assets={symbol: ratios(symbol, returns[symbol]) for symbol in request.symbols},
        benchmark=ratios(request.benchmark, market),
        portfolio=portfolio,
    )
