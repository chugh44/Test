"""Skill-weighted ensemble, interval calibration and bull/base/bear scenarios.

Models are not averaged equally. For each horizon, a model's weight is proportional to 1/RMSE² from the
walk-forward backtest, and only models within 5% of the naive benchmark's error (skill ≥ -0.05) are
eligible. The naive random walk always remains eligible, so the ensemble falls back to "no edge" when no
model has demonstrated skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backtest import ModelBacktest
from .base import HorizonForecast

Z = {"80": 1.2816, "95": 1.96, "scenario": 1.0364}  # scenario z = 85th/15th percentile


@dataclass
class EnsembleHorizon:
    h: int
    mu: float
    sigma: float
    calibration: float
    weights: dict[str, float] = field(default_factory=dict)
    model_dispersion: float = 0.0  # stdev of model mus, a measure of model disagreement
    best_skill: float | None = None


def combine(h: int, preds: dict[str, HorizonForecast], backtests: dict[str, ModelBacktest],
            naive_name: str) -> EnsembleHorizon:
    raw_w: dict[str, float] = {}
    for name, p in preds.items():
        s = backtests.get(name).by_horizon.get(h) if backtests.get(name) else None
        if s is None or not s.rmse or s.n < 2:
            continue
        if name != naive_name and (s.skill is None or s.skill < -0.05):
            continue
        raw_w[name] = 1 / s.rmse ** 2
    if not raw_w:
        raw_w = {naive_name: 1.0} if naive_name in preds else {k: 1.0 for k in preds}
    tot = sum(raw_w.values())
    w = {k: v / tot for k, v in raw_w.items()}
    mu = sum(w[k] * preds[k].mu for k in w)
    var = sum(w[k] * (preds[k].sigma ** 2 + (preds[k].mu - mu) ** 2) for k in w)
    # Calibration: weighted mean of each model's backtest E[z²]. sqrt(E[z²]) > 1 means intervals were too narrow.
    z2 = [(w[k], backtests[k].by_horizon[h].z2) for k in w if backtests.get(k) and backtests[k].by_horizon[h].z2]
    cal = float(np.sqrt(sum(a * b for a, b in z2) / sum(a for a, _ in z2))) if z2 else 1.0
    cal = float(np.clip(cal, 0.8, 2.0))
    mus = np.array([p.mu for p in preds.values()])
    # Shrink each skill estimate toward 0 by sample size; a few lucky folds shouldn't look like an edge.
    skills = [backtests[k].by_horizon[h].skill * backtests[k].by_horizon[h].n / (backtests[k].by_horizon[h].n + 12)
              for k in preds if k != naive_name and backtests.get(k) and backtests[k].by_horizon[h].skill is not None]
    return EnsembleHorizon(h, float(mu), float(np.sqrt(var) * cal), cal, w,
                           float(mus.std()) if len(mus) > 1 else 0.0, max(skills) if skills else None)


def price_band(p0: float, mu: float, sigma: float) -> dict[str, float]:
    return {
        "base": p0 * np.exp(mu),
        "bull": p0 * np.exp(mu + Z["scenario"] * sigma),
        "bear": p0 * np.exp(mu - Z["scenario"] * sigma),
        "lo80": p0 * np.exp(mu - Z["80"] * sigma), "hi80": p0 * np.exp(mu + Z["80"] * sigma),
        "lo95": p0 * np.exp(mu - Z["95"] * sigma), "hi95": p0 * np.exp(mu + Z["95"] * sigma),
        "mean": p0 * np.exp(mu + sigma ** 2 / 2),
    }


def interpolate_path(p0: float, horizons: list[EnsembleHorizon], step: int = 1) -> list[dict]:
    """Daily fan-chart path. mu interpolates linearly in h and variance linearly in h between anchors."""
    anchors = [(0, 0.0, 0.0)] + [(e.h, e.mu, e.sigma ** 2) for e in sorted(horizons, key=lambda e: e.h)]
    out = []
    for d in range(0, anchors[-1][0] + 1, step):
        for (h0, m0, v0), (h1, m1, v1) in zip(anchors, anchors[1:]):
            if h0 <= d <= h1:
                t = (d - h0) / (h1 - h0)
                mu, var = m0 + t * (m1 - m0), v0 + t * (v1 - v0)
                break
        band = price_band(p0, mu, float(np.sqrt(max(var, 0))))
        out.append({"day": d, **band})
    return out
