"""Rendering of an analysis as a text table, and as JSON-friendly data for the LLM.

Both views use the same rounding, so the numbers in the LLM's explanation match the table.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .analyzer import Analysis
from .ratios import RatioResults


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _weight(value: float) -> str:
    return f"{value * 100:.1f}".rstrip("0").rstrip(".") + "%"


# (table header, data key, formatter)
COLUMNS: tuple[tuple[str, str, Callable[[RatioResults], str]], ...] = (
    ("Sharpe", "sharpe", lambda r: _ratio(r.sharpe)),
    ("Sortino", "sortino", lambda r: _ratio(r.sortino)),
    ("Treynor", "treynor", lambda r: _percent(r.treynor)),
    ("Beta", "beta", lambda r: _ratio(r.beta)),
    ("CAGR", "cagr", lambda r: _percent(r.annualized_return)),
    ("Volatility", "volatility", lambda r: _percent(r.annualized_volatility)),
    ("Max DD", "max_drawdown", lambda r: _percent(r.max_drawdown)),
)


def _title(analysis: Analysis) -> str:
    request = analysis.request
    if request.weights is not None:
        holdings = ", ".join(f"{_weight(w)} {s}" for s, w in request.weights.items())
        return f"Portfolio: {holdings}"
    if request.kind == "comparison":
        return "Comparison: " + ", ".join(request.symbols)
    return request.symbols[0]


def _rows(analysis: Analysis) -> list[tuple[str, str, RatioResults]]:
    """(table label, data key, results) for each row, benchmark last."""
    request = analysis.request
    rows = []
    indent = ""
    if analysis.portfolio is not None:
        rows.append(("Portfolio", "portfolio", analysis.portfolio))
        indent = "  "
    rows += [(indent + s, s, analysis.assets[s]) for s in request.symbols]
    benchmark = f"{request.benchmark} (benchmark)"
    rows.append((benchmark, benchmark, analysis.benchmark))
    return rows


def format_table(analysis: Analysis) -> str:
    request = analysis.request
    window = analysis.benchmark
    table = [("", *(header for header, _, _ in COLUMNS))]
    table += [(label, *(fmt(r) for _, _, fmt in COLUMNS)) for label, _, r in _rows(analysis)]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]

    lines = [
        _title(analysis),
        f"Period: {request.period} ({window.start} to {window.end}, "
        f"{window.observations} trading days)",
        f"Benchmark: {request.benchmark} | Risk-free rate: {request.risk_free_rate:.2%}",
        "",
    ]
    for row in table:
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        lines.append("  ".join(cells).rstrip())
    lines += ["", "Ratios and returns are annualized. Treynor = excess return per unit of beta."]
    return "\n".join(lines)


def summarize(analysis: Analysis) -> dict[str, Any]:
    """The analysis as plain data, for the LLM to explain."""
    request = analysis.request
    window = analysis.benchmark
    summary: dict[str, Any] = {
        "analysis_type": request.kind,
        "period": request.period,
        "window": f"{window.start} to {window.end} ({window.observations} trading days)",
        "benchmark": request.benchmark,
        "risk_free_rate": f"{request.risk_free_rate:.2%}",
    }
    if request.weights is not None:
        summary["weights"] = {s: _weight(w) for s, w in request.weights.items()}
    summary["results"] = {
        key: {data_key: fmt(results) for _, data_key, fmt in COLUMNS}
        for _, key, results in _rows(analysis)
    }
    return summary
