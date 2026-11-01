"""Ollama integration: questions become validated requests, results become prose.

The model never computes a metric. It fills in a JSON schema (enforced by Ollama's
structured outputs) and later explains numbers that the ratio engine produced.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import ollama
from pydantic import BaseModel, Field

from .analyzer import Analysis, AnalysisRequest
from .config import Settings
from .market_data import Period
from .report import summarize

PARSE_PROMPT = """\
You translate investment questions into a request for a risk-metrics engine.
Fill in the JSON schema:

- holdings: every security the user wants analyzed, as Yahoo Finance tickers
  (Apple -> AAPL, Berkshire Hathaway B -> BRK-B, DBS -> D05.SI, S&P 500 ETF -> SPY).
  Set weight_percent only when the user states an allocation ("60% AAPL" -> 60).
- is_portfolio: true when the holdings are combined into one portfolio ("my portfolio",
  "60/40", "an equal-weight basket"); false for one security or a side-by-side comparison.
- period: one of 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y, 10y ("two years" -> 2y,
  "this year" -> ytd), or null if not stated.
- benchmark: an index ticker only if the user names one (S&P 500 -> ^GSPC,
  Nasdaq Composite -> ^IXIC, Dow Jones -> ^DJI, Straits Times Index -> ^STI), else null.
  When the user compares a security with an index, the index is the benchmark, not a holding.
- risk_free_rate_percent: only if the user states a risk-free rate ("assume 4%" -> 4),
  else null.
"""

EXPLAIN_PROMPT = """\
You are an investment analyst explaining risk-adjusted performance to a non-specialist.
You receive the user's question and metrics computed by a deterministic engine.

- Use only the numbers provided. Never invent, recompute or extrapolate figures.
- Answer the question directly in the first two or three sentences.
- Then explain the result: Sharpe is return per unit of total volatility, Sortino per unit
  of downside volatility, Treynor per unit of market risk (beta). Relate each holding to
  the benchmark row.
- Rules of thumb: a Sharpe or Sortino below 0 means the asset trailed the risk-free rate;
  0 to 1 is modest, 1 to 2 good, above 2 excellent. Beta above 1 amplifies market moves.
  A Treynor ratio above the benchmark's means more excess return per unit of market risk.
- For a portfolio, comment on diversification by comparing its volatility and drawdown
  with those of its holdings.
- Mention the period, benchmark and risk-free rate. Keep it under 250 words, with no
  tables and no investment advice.
"""


class LLMError(RuntimeError):
    """Raised when the model is unreachable or its answer cannot be used."""


class Holding(BaseModel):
    symbol: str = Field(description="Yahoo Finance ticker, e.g. AAPL, BRK-B, D05.SI")
    weight_percent: float | None = Field(
        default=None, description="Allocation in percent, only if the user stated one"
    )


class ParsedQuery(BaseModel):
    """The structure the model must return for a question."""

    holdings: list[Holding] = Field(min_length=1)
    is_portfolio: bool
    period: Period | None = None
    benchmark: str | None = None
    risk_free_rate_percent: float | None = None

    def to_request(self, settings: Settings) -> AnalysisRequest:
        weights = _portfolio_weights(self.holdings) if self.is_portfolio else None
        rate = self.risk_free_rate_percent
        return AnalysisRequest(
            symbols=[holding.symbol for holding in self.holdings],
            period=self.period or settings.period,
            benchmark=self.benchmark or settings.benchmark,
            risk_free_rate=settings.risk_free_rate if rate is None else rate / 100,
            weights=weights,
        )


def _portfolio_weights(holdings: list[Holding]) -> dict[str, float]:
    """Stated allocations, with any remainder split equally across unweighted holdings."""
    stated = {h.symbol: h.weight_percent for h in holdings if h.weight_percent is not None}
    if stated and sum(stated.values()) <= 1:  # the model answered in fractions, not percent
        stated = {symbol: weight * 100 for symbol, weight in stated.items()}
    unweighted = [h.symbol for h in holdings if h.weight_percent is None]
    if unweighted:
        remainder = 100 - sum(stated.values())
        if remainder <= 0:
            raise ValueError("the stated weights leave nothing for the other holdings")
        stated.update({symbol: remainder / len(unweighted) for symbol in unweighted})
    return stated


class OllamaAssistant:
    def __init__(self, settings: Settings, client: ollama.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or ollama.AsyncClient(host=settings.ollama_host)

    async def parse(self, question: str) -> AnalysisRequest:
        """Turn a natural-language question into a validated analysis request."""
        content = await self._chat(
            PARSE_PROMPT, question, format=ParsedQuery.model_json_schema(), temperature=0
        )
        try:
            return ParsedQuery.model_validate_json(content).to_request(self._settings)
        except ValueError as exc:  # includes pydantic's ValidationError
            raise LLMError(
                "could not turn that question into an analysis. Name the tickers, "
                f"e.g. 'Compare AAPL and MSFT over 2 years'. Details: {exc}"
            ) from exc

    async def explain(self, question: str, analysis: Analysis) -> str:
        """Explain the computed results in plain English."""
        results = json.dumps(summarize(analysis), indent=2)
        prompt = f"Question: {question}\n\nComputed results:\n{results}"
        return await self._chat(EXPLAIN_PROMPT, prompt, temperature=0.2)

    async def _chat(
        self, system: str, user: str, *, temperature: float, format: dict[str, Any] | None = None
    ) -> str:
        model = self._settings.ollama_model
        try:
            response = await self._client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=format,
                options={"temperature": temperature},
            )
        except ollama.ResponseError as exc:
            hint = f" Run `ollama pull {model}` first." if exc.status_code == 404 else ""
            raise LLMError(f"Ollama error: {exc.error}.{hint}") from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMError(
                f"could not reach Ollama at {self._settings.ollama_host}: {exc}"
            ) from exc
        content = (response.message.content or "").strip()
        if not content:
            raise LLMError("Ollama returned an empty response")
        return content
