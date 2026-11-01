import asyncio

import ollama
import pytest

from finratio.analyzer import AnalysisRequest, analyze
from finratio.llm import LLMError


def parse(make_assistant, reply):
    assistant, client = make_assistant(reply)
    return asyncio.run(assistant.parse("question")), client


def test_parse_builds_a_portfolio_request(make_assistant):
    reply = {
        "holdings": [
            {"symbol": "aapl", "weight_percent": 60},
            {"symbol": "MSFT", "weight_percent": 40},
        ],
        "is_portfolio": True,
        "period": "2y",
        "benchmark": None,
        "risk_free_rate_percent": 4.5,
    }
    request, client = parse(make_assistant, reply)

    assert request.symbols == ("AAPL", "MSFT")
    assert request.weights == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})
    assert request.period == "2y"
    assert request.risk_free_rate == pytest.approx(0.045)
    assert request.benchmark == "MKT"
    assert client.calls[0]["format"]["title"] == "ParsedQuery"
    assert client.calls[0]["options"]["temperature"] == 0


def test_parse_fills_in_defaults_for_a_comparison(make_assistant):
    reply = {"holdings": [{"symbol": "SPY"}, {"symbol": "QQQ"}], "is_portfolio": False}
    request, _ = parse(make_assistant, reply)

    assert request.kind == "comparison"
    assert (request.period, request.risk_free_rate, request.weights) == ("1y", 0.05, None)


def test_unweighted_holdings_share_the_remaining_allocation(make_assistant):
    reply = {
        "holdings": [
            {"symbol": "AAPL", "weight_percent": 50},
            {"symbol": "MSFT"},
            {"symbol": "NVDA"},
        ],
        "is_portfolio": True,
    }
    request, _ = parse(make_assistant, reply)
    assert request.weights == pytest.approx({"AAPL": 0.5, "MSFT": 0.25, "NVDA": 0.25})


def test_weights_given_as_fractions_are_understood(make_assistant):
    reply = {
        "holdings": [
            {"symbol": "AAPL", "weight_percent": 0.7},
            {"symbol": "MSFT", "weight_percent": 0.3},
        ],
        "is_portfolio": True,
    }
    request, _ = parse(make_assistant, reply)
    assert request.weights == pytest.approx({"AAPL": 0.7, "MSFT": 0.3})


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        {"holdings": [], "is_portfolio": False},
        {"holdings": [{"symbol": "AAPL"}], "is_portfolio": False, "period": "3y"},
        {"holdings": [{"symbol": "not a ticker"}], "is_portfolio": False},
    ],
)
def test_unusable_answers_raise_llm_error(make_assistant, reply):
    with pytest.raises(LLMError, match="could not turn that question"):
        parse(make_assistant, reply)


def test_connection_failures_are_reported(make_assistant):
    with pytest.raises(LLMError, match="could not reach Ollama"):
        parse(make_assistant, ConnectionError("connection refused"))


def test_missing_model_suggests_pulling_it(make_assistant):
    error = ollama.ResponseError("model 'llama3.1:8b' not found", 404)
    with pytest.raises(LLMError, match=r"ollama pull llama3\.1:8b"):
        parse(make_assistant, error)


def test_explain_grounds_the_model_in_computed_results(make_assistant, price_source):
    analysis = asyncio.run(
        analyze(AnalysisRequest(symbols=["HIGH"], benchmark="MKT"), price_source)
    )
    assistant, client = make_assistant("HIGH beat the market.")

    text = asyncio.run(assistant.explain("How did HIGH do?", analysis))

    assert text == "HIGH beat the market."
    prompt = client.calls[0]["messages"][1]["content"]
    assert "How did HIGH do?" in prompt
    assert f"{analysis.assets['HIGH'].sharpe:.2f}" in prompt
