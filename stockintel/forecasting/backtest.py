"""Walk-forward (rolling-origin) backtest.

At each cutoff, the model is fit on ``close[:cutoff]`` only, predicts every horizon, and is scored against
the realized prices at ``cutoff + h``. A horizon is scored only where that realized price exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Forecaster
from .models import fit_model

log = logging.getLogger(__name__)
Z80 = 1.2816


@dataclass
class HorizonScore:
    n: int = 0
    rmse: float | None = None        # of log return
    mape: float | None = None        # of price, %
    hit_rate: float | None = None    # directional accuracy vs realized sign
    coverage80: float | None = None  # share of realized prices inside the 80% interval
    z2: float | None = None          # mean squared standardized error, used for calibration
    skill: float | None = None       # 1 - rmse / rmse_naive


@dataclass
class ModelBacktest:
    model: str
    family: str
    by_horizon: dict[int, HorizonScore] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def cutoffs(n: int, folds: int, min_train: int = 504, min_h: int = 5) -> list[int]:
    last = n - 1 - min_h
    if last <= min_train:
        return []
    step = max(1, (last - min_train) // folds)
    return sorted({last - k * step for k in range(folds) if last - k * step >= min_train})


def walk_forward(close: pd.Series, volume: pd.Series | None, model_cls: type[Forecaster], horizons: list[int],
                 folds: int = 6) -> tuple[ModelBacktest, dict[int, list[tuple[float, float, float]]]]:
    """Return a scored backtest plus the raw (mu, sigma, realized) triples for each horizon."""
    n = len(close)
    lc = np.log(close.to_numpy())
    raw: dict[int, list[tuple[float, float, float]]] = {h: [] for h in horizons}
    bt = ModelBacktest(model_cls.name, model_cls.family)
    # Long horizons need older cutoffs, so spread them across the whole usable range.
    cuts = sorted(set(cutoffs(n, folds, min_h=5)) | set(cutoffs(n, max(2, folds // 2), min_h=max(horizons))))
    for c in cuts:
        hs = [h for h in horizons if c + h < n]
        if not hs:
            continue
        try:
            m = fit_model(model_cls, close.iloc[: c + 1], None if volume is None else volume.iloc[: c + 1], hs)
            preds = m.predict(hs)
        except Exception as e:
            # Some models can't run at early cutoffs (e.g. too little data for long horizons); retry per horizon.
            preds = {}
            for h in hs:
                try:
                    m = fit_model(model_cls, close.iloc[: c + 1], None if volume is None else volume.iloc[: c + 1], [h])
                    preds.update(m.predict([h]))
                except Exception as e2:
                    bt.errors.append(f"cutoff {close.index[c].date()} h={h}: {str(e2)[:80]}")
            if not preds:
                log.debug("%s failed at cutoff %s: %s", model_cls.name, c, e)
        for h, p in preds.items():
            realized = lc[c + h] - lc[c]
            raw[h].append((p.mu, p.sigma, realized))
    for h, rows in raw.items():
        if not rows:
            bt.by_horizon[h] = HorizonScore()
            continue
        a = np.array(rows)
        mu, sd, y = a[:, 0], np.maximum(a[:, 1], 1e-6), a[:, 2]
        err = y - mu
        bt.by_horizon[h] = HorizonScore(
            n=len(a), rmse=float(np.sqrt(np.mean(err ** 2))),
            mape=float(np.mean(np.abs(np.expm1(mu) - np.expm1(y)) / np.exp(y)) * 100),
            hit_rate=float(np.mean(np.sign(mu) == np.sign(y))) if np.any(mu != 0) else None,
            coverage80=float(np.mean(np.abs(err) <= Z80 * sd)), z2=float(np.mean((err / sd) ** 2)),
        )
    return bt, raw


def attach_skill(results: list[ModelBacktest], naive_name: str) -> None:
    naive = next((r for r in results if r.model == naive_name), None)
    if not naive:
        return
    for r in results:
        for h, s in r.by_horizon.items():
            base = naive.by_horizon.get(h)
            if s.rmse and base and base.rmse:
                s.skill = float(1 - s.rmse / base.rmse)
