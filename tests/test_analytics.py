import numpy as np
import pandas as pd

from stockintel.analytics.indicators import add_all, macd, rsi
from stockintel.analytics.levels import find_levels
from stockintel.analytics.risk_metrics import max_drawdown, var_cvar


def test_rsi_bounds(provider):
    r = rsi(provider.history("AAPL")["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_indicators_have_no_lookahead(provider):
    df = provider.history("AAPL")
    full = add_all(df)
    cut = add_all(df.iloc[:800])
    cols = ["sma50", "rsi14", "macd", "bb_upper", "atr14", "vwap20"]
    # The value at day 799 must be identical whether or not later data exists.
    pd.testing.assert_series_equal(full[cols].iloc[799], cut[cols].iloc[799], check_names=False)


def test_macd_columns(provider):
    m = macd(provider.history("AAPL")["Close"])
    assert list(m.columns) == ["macd", "signal", "hist"]
    assert np.allclose((m["macd"] - m["signal"]).dropna(), m["hist"].dropna())


def test_levels_split_by_price(provider):
    df = provider.history("AAPL")
    price = df["Close"].iloc[-1]
    for lv in find_levels(df):
        assert (lv.price < price) == (lv.kind == "support")


def test_drawdown_and_var():
    s = pd.Series([100, 120, 90, 110, 60, 80], index=pd.bdate_range("2024-01-01", periods=6))
    dd, peak, trough = max_drawdown(s)
    assert dd == 60 / 120 - 1
    v, cv = var_cvar(pd.Series(np.linspace(100, 200, 400), index=pd.bdate_range("2023-01-01", periods=400)), 21)
    assert cv <= v
