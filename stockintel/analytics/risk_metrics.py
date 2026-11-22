"""Return-distribution risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def annualized_vol(close: pd.Series, window: int | None = None) -> float:
    r = log_returns(close)
    if window:
        r = r.tail(window)
    return float(r.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> tuple[float, str | None, str | None]:
    """Return (max drawdown as a negative fraction, peak date, trough date)."""
    peak = close.cummax()
    dd = close / peak - 1
    trough = dd.idxmin()
    pk = close.loc[:trough].idxmax()
    return float(dd.min()), str(pd.Timestamp(pk).date()), str(pd.Timestamp(trough).date())


def var_cvar(close: pd.Series, horizon: int = 21, level: float = 0.95) -> tuple[float, float]:
    """Historical VaR/CVaR over overlapping horizon-day log returns, as negative simple-return fractions."""
    lr = np.log(close)
    h = (lr.shift(-horizon) - lr).dropna()
    if len(h) < 50:
        r = log_returns(close)
        h = pd.Series(np.random.default_rng(0).choice(r.values, (2000, horizon)).sum(axis=1))
    q = np.quantile(h, 1 - level)
    tail = h[h <= q]
    return float(np.expm1(q)), float(np.expm1(tail.mean() if len(tail) else q))


def beta(close: pd.Series, bench: pd.Series, window: int = 252) -> float | None:
    a = log_returns(close).tail(window)
    b = log_returns(bench).reindex(a.index).dropna()
    a = a.reindex(b.index)
    if len(a) < 60 or b.var() == 0:
        return None
    return float(np.cov(a, b)[0, 1] / b.var())


def period_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-days - 1] - 1)
