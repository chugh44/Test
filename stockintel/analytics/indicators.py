"""Technical indicators computed with pandas. Every value at time t uses only data up to t, so nothing looks ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower,
                         "pct_b": (close - lower) / (upper - lower), "width": (upper - lower) / mid})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Rolling n-day VWAP using the typical price. (Daily bars only allow an approximation of intraday VWAP.)"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).rolling(n, min_periods=n).sum() / df["Volume"].rolling(n, min_periods=n).sum()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    for n in (20, 50, 200):
        out[f"sma{n}"] = sma(c, n)
    out["ema12"], out["ema26"] = ema(c, 12), ema(c, 26)
    out["rsi14"] = rsi(c)
    out = out.join(macd(c))
    bb = bollinger(c)
    out["bb_mid"], out["bb_upper"], out["bb_lower"], out["bb_pct_b"], out["bb_width"] = (
        bb["mid"], bb["upper"], bb["lower"], bb["pct_b"], bb["width"])
    out["atr14"] = atr(out)
    out["vwap20"] = rolling_vwap(out)
    out["vol_sma20"] = sma(out["Volume"], 20)
    out["obv"] = obv(out)
    return out
