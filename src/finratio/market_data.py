"""Daily price history from the Yahoo Finance API on RapidAPI (apidojo)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, get_args

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

Period = Literal["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y"]
PERIODS: tuple[str, ...] = get_args(Period)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class MarketDataError(RuntimeError):
    """Raised when price data cannot be fetched or parsed."""


class PriceSource(Protocol):
    async def get_prices(self, symbols: Sequence[str], period: str) -> dict[str, pd.Series]:
        """Return daily adjusted closing prices for each symbol, indexed by date."""
        ...


def parse_chart(payload: Mapping[str, Any]) -> pd.Series:
    """Extract adjusted closing prices from a chart response, indexed by trading date.

    Yahoo timestamps are UTC instants. Converting them to the exchange's own time zone
    before dropping the time keeps every bar on its local trading date, so series from
    different exchanges line up correctly.
    """
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise MarketDataError(chart["error"].get("description") or "request rejected")
    results = chart.get("result") or []
    if not results:
        raise MarketDataError("no data returned")

    result = results[0]
    meta = result.get("meta") or {}
    granularity = meta.get("dataGranularity", "1d")
    if granularity != "1d":
        raise MarketDataError(f"expected daily bars, got {granularity!r}")

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    closes = adjusted or (indicators.get("quote") or [{}])[0].get("close") or []
    if not timestamps or len(closes) != len(timestamps):
        raise MarketDataError("response has no usable price history")

    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(meta.get("exchangeTimezoneName") or "UTC")
        .normalize()
        .tz_localize(None)
    )
    prices = pd.Series(closes, index=dates, dtype="float64", name=meta.get("symbol")).dropna()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices.index.name = "date"
    return prices


class RapidAPIClient:
    """Async client for daily prices, with bounded concurrency and retries."""

    host = "apidojo-yahoo-finance-v1.p.rapidapi.com"

    def __init__(
        self,
        api_key: str,
        *,
        max_concurrency: int = 4,
        retries: int = 3,
        backoff: float = 1.0,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise MarketDataError("RAPIDAPI_KEY is not set; copy .env.example to .env and add it")
        self._http = httpx.AsyncClient(
            base_url=f"https://{self.host}",
            headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": self.host},
            timeout=timeout,
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._retries = retries
        self._backoff = backoff

    async def __aenter__(self) -> RapidAPIClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_prices(self, symbols: Sequence[str], period: str) -> dict[str, pd.Series]:
        """Fetch every symbol concurrently; report all failures together."""
        if period not in PERIODS:
            raise ValueError(f"unsupported period {period!r}; choose from {', '.join(PERIODS)}")
        results = await asyncio.gather(
            *(self._get_symbol(symbol, period) for symbol in symbols), return_exceptions=True
        )
        failures = [
            f"{symbol}: {result}"
            for symbol, result in zip(symbols, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise MarketDataError("; ".join(failures))
        return dict(zip(symbols, results, strict=True))

    async def _get_symbol(self, symbol: str, period: str) -> pd.Series:
        params = {"symbol": symbol, "range": period, "interval": "1d", "region": "US"}
        async with self._semaphore:
            payload = await self._get_json("/stock/v2/get-chart", params)
        prices = parse_chart(payload)
        logger.debug("fetched %d prices for %s", len(prices), symbol)
        return prices.rename(symbol)

    async def _get_json(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        reason = ""
        for attempt in range(1, self._retries + 1):
            try:
                response = await self._http.get(path, params=params)
            except httpx.TransportError as exc:
                reason = f"network error ({type(exc).__name__})"
            else:
                if response.status_code == httpx.codes.OK:
                    return response.json()
                reason = f"HTTP {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise MarketDataError(f"{reason}: {response.text[:200]}")
            if attempt < self._retries:
                delay = self._backoff * 2 ** (attempt - 1)
                logger.warning("%s: %s, retrying in %.1fs", params["symbol"], reason, delay)
                await asyncio.sleep(delay)
        raise MarketDataError(f"{reason} after {self._retries} attempts")
