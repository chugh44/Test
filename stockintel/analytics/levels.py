"""Support and resistance levels: swing points clustered by price, ranked by touches and recency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Level:
    price: float
    kind: str  # "support" | "resistance"
    touches: int
    strength: float  # 0-1
    last_touch: str


def swing_points(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    hi, lo = df["High"], df["Low"]
    is_hi = hi == hi.rolling(2 * window + 1, center=True).max()
    is_lo = lo == lo.rolling(2 * window + 1, center=True).min()
    return hi[is_hi].dropna(), lo[is_lo].dropna()


def find_levels(df: pd.DataFrame, lookback: int = 252, window: int = 5, tol: float = 0.015,
                max_levels: int = 4) -> list[Level]:
    d = df.tail(lookback)
    if len(d) < 3 * window:
        return []
    price = float(d["Close"].iloc[-1])
    highs, lows = swing_points(d, window)
    pts = pd.concat([highs, lows]).sort_index()
    if pts.empty:
        return []
    vals = pts.to_numpy()
    order = np.argsort(vals)
    clusters: list[list[int]] = []
    for i in order:
        if clusters and abs(vals[i] / np.mean(vals[clusters[-1]]) - 1) <= tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    n = len(d)
    pos = {ts: k for k, ts in enumerate(d.index)}
    levels: list[Level] = []
    for c in clusters:
        lvl = float(np.mean(vals[c]))
        last_ts = max(pts.index[i] for i in c)
        recency = pos.get(last_ts, 0) / n
        strength = min(1.0, 0.25 * len(c) + 0.5 * recency)
        kind = "support" if lvl < price else "resistance"
        levels.append(Level(round(lvl, 2), kind, len(c), round(strength, 2), str(pd.Timestamp(last_ts).date())))
    sup = sorted([l for l in levels if l.kind == "support"], key=lambda l: (-l.strength, price - l.price))[:max_levels]
    res = sorted([l for l in levels if l.kind == "resistance"], key=lambda l: (-l.strength, l.price - price))[:max_levels]
    return sorted(sup, key=lambda l: -l.price) + sorted(res, key=lambda l: l.price)


def nearest(levels: list[Level], kind: str, price: float) -> Level | None:
    c = [l for l in levels if l.kind == kind]
    if not c:
        return None
    return min(c, key=lambda l: abs(l.price - price))
