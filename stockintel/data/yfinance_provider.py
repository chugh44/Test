"""Real market data via yfinance (Yahoo Finance). Quotes are delayed (~15 min) and labeled that way."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .base import (
    DataUnavailable, MarketDataProvider, NewsItem, OptionChain, ProviderInfo, Quote, TickerMatch,
    _TTLCache, normalize_ohlcv,
)

log = logging.getLogger(__name__)


def _ts(v: Any) -> datetime | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        t = pd.Timestamp(v)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.to_pydatetime()
    except Exception:
        return None


class YFinanceProvider(MarketDataProvider):
    def __init__(self, ttl_s: float = 600):
        import yfinance as yf  # imported lazily so tests don't need network

        self._yf = yf
        self._cache = _TTLCache(ttl_s)
        self.info = ProviderInfo(
            name="Yahoo Finance (yfinance)",
            notes=["Quotes and option chains are delayed (~15 min).",
                   "Holder data derives from quarterly 13F filings, which lag by up to 45 days."],
        )

    def _t(self, symbol: str):
        return self._cache.get_or_set(("ticker", symbol), lambda: self._yf.Ticker(symbol))

    # ------------------------------------------------------------------ search / prices
    def search(self, query: str) -> list[TickerMatch]:
        def run():
            try:
                res = self._yf.Search(query, max_results=8, news_count=0)
                quotes = res.quotes or []
            except Exception as e:  # pragma: no cover - network
                log.warning("yfinance search failed: %s", e)
                quotes = []
            return [TickerMatch(q.get("symbol"), q.get("longname") or q.get("shortname"), q.get("exchange"),
                                q.get("quoteType")) for q in quotes if q.get("symbol")]
        return self._cache.get_or_set(("search", query.lower()), run)

    def history(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        def run():
            df = self._t(symbol).history(period=period, interval=interval, auto_adjust=True)
            return normalize_ohlcv(df)
        return self._cache.get_or_set(("hist", symbol, period, interval), run)

    def quote(self, symbol: str) -> Quote:
        t = self._t(symbol)
        fi: dict = {}
        try:
            fast = t.fast_info
            fi = {k: fast.get(k) for k in ("lastPrice", "previousClose", "lastVolume", "marketCap", "currency",
                                           "threeMonthAverageVolume", "exchange")}
        except Exception:
            pass
        price = fi.get("lastPrice")
        if not price:
            h = self.history(symbol, "5d")
            price = float(h["Close"].iloc[-1])
            fi["previousClose"] = float(h["Close"].iloc[-2]) if len(h) > 1 else None
            fi["lastVolume"] = float(h["Volume"].iloc[-1])
        prev = fi.get("previousClose")
        return Quote(
            price=float(price), previous_close=prev,
            change_pct=(price / prev - 1) * 100 if prev else None,
            volume=fi.get("lastVolume"), avg_volume=fi.get("threeMonthAverageVolume"),
            market_cap=fi.get("marketCap"), currency=fi.get("currency"),
            as_of=datetime.now(timezone.utc), delayed=True, exchange=fi.get("exchange"),
        )

    def profile(self, symbol: str) -> dict:
        def run():
            try:
                info = self._t(symbol).get_info() or {}
            except Exception as e:
                raise DataUnavailable(f"profile: {e}") from e
            if not info or len(info) < 3:
                raise DataUnavailable("profile: Yahoo returned no company info")
            return info
        return self._cache.get_or_set(("profile", symbol), run)

    # ------------------------------------------------------------------ fundamentals
    def financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        def run():
            t = self._t(symbol)
            out: dict[str, pd.DataFrame] = {}
            for key, fn in {
                "income_annual": lambda: t.income_stmt, "income_quarterly": lambda: t.quarterly_income_stmt,
                "balance_annual": lambda: t.balance_sheet, "balance_quarterly": lambda: t.quarterly_balance_sheet,
                "cashflow_annual": lambda: t.cashflow, "cashflow_quarterly": lambda: t.quarterly_cashflow,
            }.items():
                try:
                    df = fn()
                    if df is not None and not df.empty:
                        out[key] = df
                except Exception as e:  # pragma: no cover
                    log.info("financials %s %s: %s", symbol, key, e)
            if not out:
                raise DataUnavailable("financial statements unavailable")
            return out
        return self._cache.get_or_set(("fin", symbol), run)

    def analyst(self, symbol: str) -> dict:
        def run():
            t = self._t(symbol)
            out: dict[str, Any] = {}
            for key, fn in {
                "price_targets": lambda: t.analyst_price_targets,
                "recommendations": lambda: t.recommendations,
                "upgrades_downgrades": lambda: t.upgrades_downgrades,
            }.items():
                try:
                    v = fn()
                    if v is not None and (not hasattr(v, "empty") or not v.empty) and (not isinstance(v, dict) or v):
                        out[key] = v
                except Exception as e:  # pragma: no cover
                    log.info("analyst %s %s: %s", symbol, key, e)
            return out
        return self._cache.get_or_set(("analyst", symbol), run)

    def news(self, symbol: str, limit: int = 30) -> list[NewsItem]:
        def run():
            try:
                raw = self._t(symbol).get_news(count=limit) or []
            except Exception:
                raw = getattr(self._t(symbol), "news", None) or []
            items: list[NewsItem] = []
            for r in raw:
                c = r.get("content") if isinstance(r.get("content"), dict) else r
                url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
                       or c.get("link"))
                provider = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) \
                    else c.get("publisher")
                items.append(NewsItem(
                    id=str(r.get("id") or c.get("id") or c.get("uuid") or url),
                    title=(c.get("title") or "").strip(), publisher=provider, url=url,
                    published_at=_ts(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")),
                    summary=c.get("summary") or c.get("description"), origin="yfinance",
                ))
            return [i for i in items if i.title]
        return self._cache.get_or_set(("news", symbol), run)

    def holders(self, symbol: str) -> dict:
        def run():
            t = self._t(symbol)
            out: dict[str, Any] = {}
            try:
                mh = t.major_holders
                if mh is not None and not mh.empty:
                    col = "Value" if "Value" in mh.columns else mh.columns[0]
                    out["major_holders"] = {str(k): float(v) for k, v in mh[col].items() if pd.notna(v)}
            except Exception:
                pass
            for key, fn in {
                "institutional_holders": lambda: t.institutional_holders,
                "mutualfund_holders": lambda: t.mutualfund_holders,
                "insider_transactions": lambda: t.insider_transactions,
            }.items():
                try:
                    v = fn()
                    if v is not None and not v.empty:
                        out[key] = v
                except Exception:
                    pass
            return out
        return self._cache.get_or_set(("holders", symbol), run)

    def options(self, symbol: str, max_expirations: int = 6) -> list[OptionChain]:
        def run():
            t = self._t(symbol)
            try:
                exps = list(t.options or [])
            except Exception as e:
                raise DataUnavailable(f"options: {e}") from e
            chains = []
            for exp in exps[:max_expirations]:
                try:
                    ch = t.option_chain(exp)
                    chains.append(OptionChain(exp, ch.calls, ch.puts))
                except Exception:
                    continue
            if not chains:
                raise DataUnavailable("no listed options")
            return chains
        return self._cache.get_or_set(("options", symbol, max_expirations), run)

    def calendar(self, symbol: str) -> dict:
        def run():
            out: dict[str, Any] = {}
            try:
                cal = self._t(symbol).calendar or {}
            except Exception:
                cal = {}
            ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if isinstance(ed, (list, tuple)) and ed:
                ed = ed[0]
            if ed is not None:
                out["earnings_date"] = _ts(ed)
            xd = cal.get("Ex-Dividend Date") if isinstance(cal, dict) else None
            if xd is not None:
                out["ex_dividend_date"] = _ts(xd)
            return out
        return self._cache.get_or_set(("cal", symbol), run)
