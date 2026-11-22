"""yfinance provider parsing, tested against mocked yfinance objects (no network)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from stockintel.data.base import DataUnavailable, normalize_ohlcv
from stockintel.data.yfinance_provider import YFinanceProvider


def test_normalize_ohlcv_handles_tz_and_duplicates():
    idx = pd.DatetimeIndex(["2025-01-02 00:00-05:00", "2025-01-02 00:00-05:00", "2025-01-03 00:00-05:00"])
    df = pd.DataFrame({"open": [1, 1, 2], "high": [2, 2, 3], "low": [0.5, 0.5, 1], "close": [1.5, 1.6, 2.5],
                       "volume": [10, 10, 20], "Dividends": 0}, index=idx)
    out = normalize_ohlcv(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.tz is None and len(out) == 2 and out["Close"].iloc[0] == 1.6
    with pytest.raises(DataUnavailable):
        normalize_ohlcv(pd.DataFrame())


def _provider(fake_ticker):
    p = YFinanceProvider()
    p._t = lambda sym: fake_ticker
    return p


def test_news_parses_new_and_legacy_formats():
    new = {"id": "n1", "content": {"title": "Acme beats", "pubDate": "2026-11-20T13:00:00Z",
                                   "provider": {"displayName": "Reuters"},
                                   "canonicalUrl": {"url": "https://reuters.com/x"}, "summary": "s"}}
    legacy = {"uuid": "n2", "title": "Acme launches", "publisher": "CNBC", "link": "https://cnbc.com/y",
              "providerPublishTime": 1763643600}
    p = _provider(SimpleNamespace(get_news=lambda count: [new, legacy, {"id": "n3", "content": {"title": ""}}]))
    items = p.news("ACME")
    assert [i.title for i in items] == ["Acme beats", "Acme launches"]
    assert items[0].publisher == "Reuters" and items[0].url == "https://reuters.com/x" and items[0].published_at.year == 2026
    assert items[1].publisher == "CNBC" and items[1].published_at is not None


def test_calendar_and_holders():
    mh = pd.DataFrame({"Value": [0.01, 0.6]}, index=["insidersPercentHeld", "institutionsPercentHeld"])
    fake = SimpleNamespace(calendar={"Earnings Date": [pd.Timestamp("2026-12-01").date()]}, major_holders=mh,
                           institutional_holders=pd.DataFrame({"Holder": ["V"]}), mutualfund_holders=None,
                           insider_transactions=pd.DataFrame())
    p = _provider(fake)
    assert p.calendar("ACME")["earnings_date"].date().isoformat() == "2026-12-01"
    h = p.holders("ACME")
    assert h["major_holders"]["institutionsPercentHeld"] == 0.6 and "insider_transactions" not in h
