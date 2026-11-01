import asyncio

import pytest

from finratio import cli


@pytest.fixture
def run_cli(monkeypatch, settings, price_source):
    class FakeClient:
        def __init__(self, api_key):
            pass

        async def __aenter__(self):
            return price_source

        async def __aexit__(self, *exc_info):
            pass

    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(cli, "RapidAPIClient", FakeClient)
    return cli.main


def test_analyze_prints_a_portfolio_table(run_cli, capsys):
    assert run_cli(["analyze", "HIGH", "LOW", "--weights", "60", "40"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Portfolio: 60% HIGH, 40% LOW")
    assert "MKT (benchmark)" in out


def test_analyze_rejects_mismatched_weights(run_cli):
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["analyze", "HIGH", "LOW", "--weights", "60"])
    assert excinfo.value.code == 2


def test_invalid_input_is_reported_without_a_traceback(run_cli, capsys):
    assert run_cli(["analyze", "HIGH", "--risk-free-rate", "5"]) == 1
    assert "annual decimal" in capsys.readouterr().err


def test_answer_shows_the_table_then_the_explanation(make_assistant, price_source):
    assistant, _ = make_assistant(
        {"holdings": [{"symbol": "HIGH"}], "is_portfolio": False}, "HIGH beat the market."
    )
    output = asyncio.run(cli._answer("How did HIGH do?", assistant, price_source))
    assert output.startswith("HIGH\n")
    assert output.endswith("HIGH beat the market.")


def test_answer_keeps_the_table_when_the_explanation_fails(make_assistant, price_source):
    assistant, _ = make_assistant(
        {"holdings": [{"symbol": "HIGH"}], "is_portfolio": False}, ConnectionError("down")
    )
    output = asyncio.run(cli._answer("How did HIGH do?", assistant, price_source))
    assert "MKT (benchmark)" in output
    assert "Explanation unavailable" in output
