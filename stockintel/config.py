"""Runtime settings, read from environment variables (and a .env file if python-dotenv is installed)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    model: str = field(default_factory=lambda: os.getenv("STOCKINTEL_MODEL", "claude-opus-5-5"))
    web_search: bool = field(default_factory=lambda: _flag("STOCKINTEL_WEB_SEARCH", "1"))
    finnhub_key: str | None = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY") or None)
    newsapi_key: str | None = field(default_factory=lambda: os.getenv("NEWSAPI_KEY") or None)
    fred_key: str | None = field(default_factory=lambda: os.getenv("FRED_API_KEY") or None)
    db_path: str = field(default_factory=lambda: os.getenv("STOCKINTEL_DB", "stockintel.db"))
    backtest_folds: int = field(default_factory=lambda: int(os.getenv("STOCKINTEL_BACKTEST_FOLDS", "6")))
    enable_lstm: bool = field(default_factory=lambda: _flag("STOCKINTEL_ENABLE_LSTM", "1"))
    enable_prophet: bool = field(default_factory=lambda: _flag("STOCKINTEL_ENABLE_PROPHET", "1"))
    history_years: int = 5


def get_settings() -> Settings:
    return Settings()
