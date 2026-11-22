"""Rule-based chart pattern and momentum-event detection on an indicator-enriched frame (see indicators.add_all)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .levels import swing_points


@dataclass
class Pattern:
    name: str
    bias: str  # bullish | bearish | neutral
    date: str
    detail: str


def _last_cross(a: pd.Series, b: pd.Series, lookback: int) -> tuple[str, pd.Timestamp] | None:
    diff = (a - b).dropna().tail(lookback + 1)
    s = np.sign(diff)
    flips = s[s != s.shift()].iloc[1:]
    if flips.empty:
        return None
    ts = flips.index[-1]
    return ("up" if flips.iloc[-1] > 0 else "down"), ts


def detect(df: pd.DataFrame) -> list[Pattern]:
    out: list[Pattern] = []
    c = df["Close"]
    last = df.index[-1]

    x = _last_cross(df["sma50"], df["sma200"], 30)
    if x:
        out.append(Pattern("Golden cross" if x[0] == "up" else "Death cross",
                           "bullish" if x[0] == "up" else "bearish", str(x[1].date()),
                           "50-day SMA crossed the 200-day SMA within the last 30 sessions"))
    x = _last_cross(df["macd"], df["signal"], 5)
    if x:
        out.append(Pattern(f"MACD {'bullish' if x[0] == 'up' else 'bearish'} crossover",
                           "bullish" if x[0] == "up" else "bearish", str(x[1].date()), "MACD crossed its signal line"))

    hi55, lo55 = df["High"].rolling(55).max().shift(), df["Low"].rolling(55).min().shift()
    vol_ok = df["Volume"].iloc[-1] > 1.3 * df["vol_sma20"].iloc[-1]
    if c.iloc[-1] > hi55.iloc[-1]:
        out.append(Pattern("55-day breakout" + (" on volume" if vol_ok else ""), "bullish", str(last.date()),
                           f"Close above prior 55-day high {hi55.iloc[-1]:.2f}"))
    elif c.iloc[-1] < lo55.iloc[-1]:
        out.append(Pattern("55-day breakdown" + (" on volume" if vol_ok else ""), "bearish", str(last.date()),
                           f"Close below prior 55-day low {lo55.iloc[-1]:.2f}"))

    if df["bb_width"].iloc[-1] <= df["bb_width"].tail(126).quantile(0.1):
        out.append(Pattern("Bollinger squeeze", "neutral", str(last.date()),
                           "Band width is in the lowest decile of the last 6 months; volatility expansion likely"))

    highs, lows = swing_points(df.tail(126), 5)
    if len(highs) >= 2:
        h1, h2 = highs.iloc[-2], highs.iloc[-1]
        if abs(h2 / h1 - 1) < 0.02 and c.iloc[-1] < min(h1, h2) * 0.97:
            out.append(Pattern("Possible double top", "bearish", str(highs.index[-1].date()),
                               f"Twin highs near {h2:.2f} with price now below them"))
    if len(lows) >= 2:
        l1, l2 = lows.iloc[-2], lows.iloc[-1]
        if abs(l2 / l1 - 1) < 0.02 and c.iloc[-1] > max(l1, l2) * 1.03:
            out.append(Pattern("Possible double bottom", "bullish", str(lows.index[-1].date()),
                               f"Twin lows near {l2:.2f} with price now above them"))

    # RSI divergence across the last two swing points
    if len(highs) >= 2 and df.loc[highs.index[-1], "rsi14"] < df.loc[highs.index[-2], "rsi14"] - 3 \
            and highs.iloc[-1] > highs.iloc[-2]:
        out.append(Pattern("Bearish RSI divergence", "bearish", str(highs.index[-1].date()),
                           "Higher price high with a lower RSI high"))
    if len(lows) >= 2 and df.loc[lows.index[-1], "rsi14"] > df.loc[lows.index[-2], "rsi14"] + 3 \
            and lows.iloc[-1] < lows.iloc[-2]:
        out.append(Pattern("Bullish RSI divergence", "bullish", str(lows.index[-1].date()),
                           "Lower price low with a higher RSI low"))
    return out
