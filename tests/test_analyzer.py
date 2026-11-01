import asyncio

import pytest

from finratio.analyzer import AnalysisRequest, analyze
from finratio.report import format_table, summarize


def run(request, source):
    return asyncio.run(analyze(request, source))


def test_request_normalizes_and_deduplicates_symbols():
    request = AnalysisRequest(symbols=[" aapl", "MSFT", "AAPL"])
    assert request.symbols == ("AAPL", "MSFT")
    assert request.kind == "comparison"


def test_request_with_weights_is_a_portfolio():
    request = AnalysisRequest(symbols=["aapl", "msft"], weights={"aapl": 60, "msft": 40})
    assert request.kind == "portfolio"
    assert request.weights == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symbols": []}, "at least one ticker"),
        ({"symbols": ["AAPL; DROP TABLE"]}, "does not look like a ticker"),
        ({"symbols": ["AAPL"], "period": "3y"}, "unsupported period"),
        ({"symbols": ["AAPL"], "risk_free_rate": 4.5}, "annual decimal"),
        ({"symbols": ["AAPL", "MSFT"], "weights": {"AAPL": 100}}, "cover exactly"),
    ],
)
def test_invalid_requests_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AnalysisRequest(**kwargs)


def test_portfolio_analysis_covers_holdings_and_benchmark(price_source):
    request = AnalysisRequest(
        symbols=["HIGH", "LOW"], benchmark="MKT", weights={"HIGH": 50, "LOW": 50}
    )
    analysis = run(request, price_source)

    assert price_source.requests == [(("HIGH", "LOW", "MKT"), "1y")]
    assert analysis.assets["HIGH"].beta == pytest.approx(1.5, abs=0.1)
    assert analysis.assets["LOW"].beta == pytest.approx(0.5, abs=0.1)
    assert analysis.portfolio.beta == pytest.approx(1.0, abs=0.1)
    assert analysis.benchmark.beta == pytest.approx(1.0)


def test_benchmark_is_fetched_once_when_also_requested(price_source):
    run(AnalysisRequest(symbols=["MKT", "HIGH"], benchmark="MKT"), price_source)
    assert price_source.requests == [(("MKT", "HIGH"), "1y")]


def test_table_lists_portfolio_holdings_then_benchmark(price_source):
    request = AnalysisRequest(
        symbols=["HIGH", "LOW"], benchmark="MKT", weights={"HIGH": 60, "LOW": 40}
    )
    lines = format_table(run(request, price_source)).splitlines()

    assert lines[0] == "Portfolio: 60% HIGH, 40% LOW"
    header = next(i for i, line in enumerate(lines) if "Sharpe" in line)
    rows = lines[header + 1 : header + 5]
    expected = ["Portfolio ", "  HIGH ", "  LOW ", "MKT (benchmark) "]
    assert all(row.startswith(label) for label, row in zip(expected, rows, strict=True))


def test_summary_uses_the_same_rounding_as_the_table(price_source):
    analysis = run(AnalysisRequest(symbols=["HIGH"], benchmark="MKT"), price_source)
    summary = summarize(analysis)

    sharpe = summary["results"]["HIGH"]["sharpe"]
    assert sharpe == f"{analysis.assets['HIGH'].sharpe:.2f}"
    assert sharpe in format_table(analysis)
    assert set(summary["results"]) == {"HIGH", "MKT (benchmark)"}
