"""Command-line interface: `finratio analyze`, `finratio ask` and `finratio chat`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import __version__
from .analyzer import AnalysisRequest, analyze
from .config import Settings
from .llm import LLMError, OllamaAssistant
from .market_data import PERIODS, MarketDataError, PriceSource, RapidAPIClient
from .report import format_table

logger = logging.getLogger(__name__)

EXAMPLES = """\
examples:
  finratio analyze AAPL
  finratio analyze AAPL MSFT NVDA --period 2y
  finratio analyze AAPL MSFT --weights 60 40
  finratio ask "Compare the Sharpe ratios of SPY and QQQ over 5 years"
  finratio chat
"""


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finratio",
        description="Risk-adjusted performance analysis for stocks and portfolios.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    direct = commands.add_parser("analyze", help="compute ratios for tickers directly (no LLM)")
    direct.add_argument("symbols", nargs="+", metavar="TICKER")
    direct.add_argument(
        "--weights",
        nargs="+",
        type=float,
        metavar="W",
        help="portfolio weights in ticker order, e.g. 60 40 (omit for a comparison)",
    )
    direct.add_argument("--period", choices=PERIODS, default=settings.period)
    direct.add_argument("--benchmark", default=settings.benchmark, help="default: %(default)s")
    direct.add_argument(
        "--risk-free-rate",
        type=float,
        default=settings.risk_free_rate,
        metavar="RATE",
        help="annual rate as a decimal (default: %(default)s)",
    )

    ask = commands.add_parser("ask", help="ask a question in plain English (requires Ollama)")
    ask.add_argument("question", nargs="+")
    commands.add_parser("chat", help="interactive session (requires Ollama)")
    return parser


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def _run_analyze(request: AnalysisRequest, settings: Settings) -> None:
    async with RapidAPIClient(settings.rapidapi_key) as source:
        analysis = await analyze(request, source)
    print(format_table(analysis))


async def _answer(question: str, assistant: OllamaAssistant, source: PriceSource) -> str:
    _status("Interpreting the question...")
    request = await assistant.parse(question)
    _status(f"Fetching {request.period} of prices for {', '.join(request.symbols)}...")
    analysis = await analyze(request, source)
    _status("Writing the explanation...")
    try:
        explanation = await assistant.explain(question, analysis)
    except LLMError as exc:
        explanation = f"(Explanation unavailable: {exc})"
    return f"{format_table(analysis)}\n\n{explanation}"


async def _run_ask(question: str, settings: Settings) -> None:
    async with RapidAPIClient(settings.rapidapi_key) as source:
        print(await _answer(question, OllamaAssistant(settings), source))


async def _run_chat(settings: Settings) -> None:
    assistant = OllamaAssistant(settings)
    print("Ask about any ticker or portfolio, e.g. 'Compare AAPL and MSFT over 2 years'.")
    print("Type 'exit' to quit.")
    async with RapidAPIClient(settings.rapidapi_key) as source:
        while True:
            try:
                question = input("\n> ").strip()
            except EOFError:
                return
            if question.lower() in {"exit", "quit", "q"}:
                return
            if not question:
                continue
            try:
                print(await _answer(question, assistant, source))
            except (MarketDataError, LLMError, ValueError) as exc:
                print(f"error: {exc}")


def main(argv: list[str] | None = None) -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser = build_parser(settings)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "analyze":
            if args.weights is not None and len(args.weights) != len(args.symbols):
                parser.error("--weights needs exactly one value per ticker")
            weights = dict(zip(args.symbols, args.weights, strict=True)) if args.weights else None
            request = AnalysisRequest(
                symbols=args.symbols,
                period=args.period,
                benchmark=args.benchmark,
                risk_free_rate=args.risk_free_rate,
                weights=weights,
            )
            asyncio.run(_run_analyze(request, settings))
        elif args.command == "ask":
            asyncio.run(_run_ask(" ".join(args.question), settings))
        else:
            asyncio.run(_run_chat(settings))
    except (MarketDataError, LLMError, ValueError) as exc:
        logger.debug("command failed", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0
