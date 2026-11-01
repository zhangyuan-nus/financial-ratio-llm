from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
import pytest

from finratio.config import Settings
from finratio.llm import OllamaAssistant

FIXTURES = Path(__file__).parent / "fixtures"


class FakePriceSource:
    """In-memory stand-in for RapidAPIClient that records each request."""

    def __init__(self, prices: dict[str, pd.Series]) -> None:
        self.prices = prices
        self.requests: list[tuple[tuple[str, ...], str]] = []

    async def get_prices(self, symbols, period):
        self.requests.append((tuple(symbols), period))
        return {symbol: self.prices[symbol] for symbol in symbols}


class FakeOllama:
    """Replays canned replies, or raises canned errors, in order."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            reply = json.dumps(reply)
        return ollama.ChatResponse(message=ollama.Message(role="assistant", content=reply))


@pytest.fixture
def chart_payload() -> dict:
    """A real get-chart response for AAPL, trimmed to five trading days."""
    return json.loads((FIXTURES / "aapl_chart.json").read_text())


@pytest.fixture
def market_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=300)
    return pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)


@pytest.fixture
def price_source(market_returns) -> FakePriceSource:
    """A synthetic market plus a high-beta (1.5) and a low-beta (0.5) stock."""
    rng = np.random.default_rng(11)

    def stock(true_beta: float) -> pd.Series:
        return true_beta * market_returns + rng.normal(0, 0.002, len(market_returns))

    returns = {"MKT": market_returns, "HIGH": stock(1.5), "LOW": stock(0.5)}
    return FakePriceSource({symbol: 100 * (1 + r).cumprod() for symbol, r in returns.items()})


@pytest.fixture
def settings() -> Settings:
    return Settings(rapidapi_key="test-key", risk_free_rate=0.05, benchmark="MKT", period="1y")


@pytest.fixture
def make_assistant(settings):
    def factory(*replies):
        client = FakeOllama(replies)
        return OllamaAssistant(settings, client), client

    return factory
