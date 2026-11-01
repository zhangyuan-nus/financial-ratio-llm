import asyncio
import copy

import httpx
import pandas as pd
import pytest

from finratio.market_data import MarketDataError, RapidAPIClient, parse_chart

NOT_FOUND = {
    "chart": {
        "result": None,
        "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"},
    }
}


def fetch(handler, symbols=("AAPL",), period="1mo"):
    async def run():
        client = RapidAPIClient("test-key", transport=httpx.MockTransport(handler), backoff=0)
        async with client:
            return await client.get_prices(list(symbols), period)

    return asyncio.run(run())


def test_parse_chart_returns_adjusted_closes_by_trading_date(chart_payload):
    prices = parse_chart(chart_payload)
    adjclose = chart_payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
    assert prices.tolist() == pytest.approx(adjclose)
    assert prices.index[0] == pd.Timestamp("2026-09-30")
    assert prices.index.is_monotonic_increasing


def test_parse_chart_dates_bars_in_the_exchange_time_zone(chart_payload):
    # 03:00 UTC on 1 October is still 30 September in New York, but 1 October in Singapore.
    payload = copy.deepcopy(chart_payload)
    result = payload["chart"]["result"][0]
    result["timestamp"] = [int(pd.Timestamp("2026-10-01 03:00", tz="UTC").timestamp())]
    result["indicators"]["adjclose"][0]["adjclose"] = [100.0]
    assert parse_chart(payload).index[0] == pd.Timestamp("2026-09-30")

    result["meta"]["exchangeTimezoneName"] = "Asia/Singapore"
    assert parse_chart(payload).index[0] == pd.Timestamp("2026-10-01")


def test_parse_chart_drops_missing_prices(chart_payload):
    payload = copy.deepcopy(chart_payload)
    payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][2] = None
    assert len(parse_chart(payload)) == 4


def test_parse_chart_reports_api_errors():
    with pytest.raises(MarketDataError, match="symbol may be delisted"):
        parse_chart(NOT_FOUND)


def test_client_sends_the_key_and_query(chart_payload):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=chart_payload)

    prices = fetch(handler)
    assert seen[0].headers["X-RapidAPI-Key"] == "test-key"
    assert seen[0].url.params["symbol"] == "AAPL"
    assert seen[0].url.params["range"] == "1mo"
    assert prices["AAPL"].name == "AAPL"


def test_client_retries_rate_limited_requests(chart_payload):
    responses = iter([httpx.Response(429), httpx.Response(200, json=chart_payload)])
    prices = fetch(lambda request: next(responses))
    assert len(prices["AAPL"]) == 5


def test_client_gives_up_after_repeated_server_errors():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    with pytest.raises(MarketDataError, match="HTTP 503 after 3 attempts"):
        fetch(handler)
    assert len(calls) == 3


def test_client_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, text="Forbidden")

    with pytest.raises(MarketDataError, match="HTTP 403"):
        fetch(handler)
    assert len(calls) == 1


def test_client_reports_every_failed_symbol(chart_payload):
    def handler(request):
        found = request.url.params["symbol"] == "AAPL"
        return httpx.Response(200, json=chart_payload if found else NOT_FOUND)

    with pytest.raises(MarketDataError) as excinfo:
        fetch(handler, symbols=("AAPL", "XXXX", "YYYY"))
    assert "XXXX: No data found" in str(excinfo.value)
    assert "YYYY: No data found" in str(excinfo.value)


def test_client_requires_an_api_key():
    with pytest.raises(MarketDataError, match="RAPIDAPI_KEY"):
        RapidAPIClient("")
