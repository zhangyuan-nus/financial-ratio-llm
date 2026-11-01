"""Settings read from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Settings:
    rapidapi_key: str = ""
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    risk_free_rate: float = 0.05
    benchmark: str = "SPY"
    period: str = "1y"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment, loading .env from the working directory."""
        load_dotenv(find_dotenv(usecwd=True))
        defaults = cls()
        raw_rate = os.getenv("RISK_FREE_RATE", str(defaults.risk_free_rate))
        try:
            risk_free_rate = float(raw_rate)
        except ValueError:
            raise ValueError(
                f"RISK_FREE_RATE must be a decimal such as 0.05, got {raw_rate!r}"
            ) from None
        return cls(
            rapidapi_key=os.getenv("RAPIDAPI_KEY", "").strip(),
            ollama_host=os.getenv("OLLAMA_BASE_URL", defaults.ollama_host),
            ollama_model=os.getenv("OLLAMA_MODEL", defaults.ollama_model),
            risk_free_rate=risk_free_rate,
            benchmark=os.getenv("DEFAULT_BENCHMARK", defaults.benchmark),
            period=os.getenv("DEFAULT_PERIOD", defaults.period),
        )
