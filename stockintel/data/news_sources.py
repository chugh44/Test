"""Optional news and macro adapters (Finnhub, NewsAPI, FRED). Each returns [] / None when unconfigured."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from .base import NewsItem

log = logging.getLogger(__name__)


def finnhub_news(symbol: str, key: str | None, days: int = 14) -> list[NewsItem]:
    if not key:
        return []
    to = datetime.now(timezone.utc).date()
    frm = to - timedelta(days=days)
    try:
        r = requests.get("https://finnhub.io/api/v1/company-news",
                         params={"symbol": symbol, "from": frm.isoformat(), "to": to.isoformat(), "token": key},
                         timeout=15)
        r.raise_for_status()
        return [NewsItem(id=f"fh-{x.get('id')}", title=x.get("headline", ""), publisher=x.get("source"),
                         url=x.get("url"), published_at=datetime.fromtimestamp(x.get("datetime", 0), tz=timezone.utc),
                         summary=x.get("summary"), origin="finnhub") for x in r.json()[:40] if x.get("headline")]
    except Exception as e:  # pragma: no cover - network
        log.warning("finnhub news failed: %s", e)
        return []


def newsapi_news(query: str, key: str | None, days: int = 14) -> list[NewsItem]:
    if not key:
        return []
    try:
        r = requests.get("https://newsapi.org/v2/everything", timeout=15, params={
            "q": query, "from": (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat(),
            "language": "en", "sortBy": "publishedAt", "pageSize": 40, "apiKey": key})
        r.raise_for_status()
        out = []
        for i, a in enumerate(r.json().get("articles", [])):
            ts = a.get("publishedAt")
            out.append(NewsItem(id=f"na-{i}-{hash(a.get('url'))}", title=a.get("title") or "",
                                publisher=(a.get("source") or {}).get("name"), url=a.get("url"),
                                published_at=datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None,
                                summary=a.get("description"), origin="newsapi"))
        return [x for x in out if x.title]
    except Exception as e:  # pragma: no cover - network
        log.warning("newsapi failed: %s", e)
        return []


def fred_series(series_id: str, key: str | None, limit: int = 24) -> list[tuple[str, float]] | None:
    """Return [(date, value)] newest-last, or None when FRED is not configured / fails."""
    if not key:
        return None
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=15, params={
            "series_id": series_id, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": limit})
        r.raise_for_status()
        obs = [(o["date"], float(o["value"])) for o in r.json().get("observations", []) if o.get("value") not in (".", None)]
        return list(reversed(obs))
    except Exception as e:  # pragma: no cover - network
        log.warning("FRED %s failed: %s", series_id, e)
        return None
