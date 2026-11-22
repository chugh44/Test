"""Forecaster interface.

Every model forecasts the distribution of the *h-day forward log return*: a normal with mean ``mu`` and
standard deviation ``sigma``. Price forecasts follow as ``P_t * exp(mu ± z*sigma)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS: dict[str, int] = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


@dataclass
class HorizonForecast:
    mu: float
    sigma: float


class Forecaster(ABC):
    name: str = "base"
    family: str = "statistical"

    @abstractmethod
    def fit(self, close: pd.Series) -> "Forecaster": ...

    @abstractmethod
    def predict(self, horizons: list[int]) -> dict[int, HorizonForecast]: ...


def daily_vol(close: pd.Series, lam: float = 0.94, long_window: int = 504) -> float:
    """Blend of EWMA (recent) and long-run volatility of daily log returns."""
    r = np.log(close).diff().dropna()
    if len(r) < 20:
        return float(r.std() or 0.02)
    ew = float(np.sqrt((r ** 2).ewm(alpha=1 - lam).mean().iloc[-1]))
    lr = float(r.tail(long_window).std())
    return 0.5 * ew + 0.5 * lr


def sigma_for(close: pd.Series, h: int) -> float:
    # Short horizons lean on current vol; long horizons revert toward long-run vol.
    r = np.log(close).diff().dropna()
    long_run = float(r.tail(756).std())
    cur = daily_vol(close)
    w = np.exp(-h / 63)
    return float((w * cur + (1 - w) * long_run) * np.sqrt(h))
