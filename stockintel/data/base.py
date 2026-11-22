"""Market-data provider interface.

Agents depend only on this interface. A method that cannot obtain data raises ``DataUnavailable``.
It never returns made-up values.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd


class DataUnavailable(Exception):
    """Raised when a data item cannot be retrieved. The message says what and why."""


@dataclass
class NewsItem:
    id: str
    title: str
    publisher: str | None
    url: str | None
    published_at: datetime | None
    summary: str | None = None
    origin: str = "yfinance"  # yfinance | finnhub | newsapi | web_search | synthetic


@dataclass
class TickerMatch:
    symbol: str
    name: str | None
    exchange: str | None
    quote_type: str | None


@dataclass
class OptionChain:
    expiration: str
    calls: pd.DataFrame
    puts: pd.DataFrame


@dataclass
class Quote:
    price: float
    previous_close: float | None
    change_pct: float | None
    volume: float | None
    avg_volume: float | None
    market_cap: float | None
    currency: str | None
    as_of: datetime
    delayed: bool = True
    exchange: str | None = None


class _TTLCache:
    def __init__(self, ttl_s: float = 600):
        self.ttl = ttl_s
        self._d: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[Any, threading.Lock] = {}

    def get_or_set(self, key: Any, fn: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._d.get(key)
            if hit and time.time() - hit[0] < self.ttl:
                return hit[1]
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:  # one fetch per key even when agents run in parallel
            with self._lock:
                hit = self._d.get(key)
                if hit and time.time() - hit[0] < self.ttl:
                    return hit[1]
            val = fn()
            with self._lock:
                self._d[key] = (time.time(), val)
            return val


@dataclass
class ProviderInfo:
    name: str
    is_synthetic: bool = False
    notes: list[str] = field(default_factory=list)


class MarketDataProvider(ABC):
    info: ProviderInfo

    @abstractmethod
    def search(self, query: str) -> list[TickerMatch]: ...

    @abstractmethod
    def history(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV with a tz-naive DatetimeIndex, columns Open/High/Low/Close/Volume (split/dividend adjusted)."""

    @abstractmethod
    def quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def profile(self, symbol: str) -> dict: ...

    @abstractmethod
    def financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Keys: income_annual, income_quarterly, balance_annual, balance_quarterly, cashflow_annual, cashflow_quarterly.
        Frames are line-items x period-end columns (newest first), like yfinance."""

    @abstractmethod
    def analyst(self, symbol: str) -> dict:
        """Keys: price_targets (dict), recommendations (DataFrame), upgrades_downgrades (DataFrame)."""

    @abstractmethod
    def news(self, symbol: str, limit: int = 30) -> list[NewsItem]: ...

    @abstractmethod
    def holders(self, symbol: str) -> dict:
        """Keys: major_holders (dict), institutional_holders (DataFrame), mutualfund_holders (DataFrame),
        insider_transactions (DataFrame)."""

    @abstractmethod
    def options(self, symbol: str, max_expirations: int = 6) -> list[OptionChain]: ...

    @abstractmethod
    def calendar(self, symbol: str) -> dict:
        """Keys (when known): earnings_date (datetime), ex_dividend_date (datetime)."""


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataUnavailable("empty price history")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
    cols = {c: c.title() for c in df.columns}
    df = df.rename(columns=cols)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataUnavailable(f"price history missing columns {missing}")
    out = df[needed].astype(float).copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    if out.empty:
        raise DataUnavailable("price history has no valid closes")
    return out
