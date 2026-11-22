"""Agent 1: Stock Price Forecasting."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ...analytics.levels import find_levels
from ...forecasting.backtest import attach_skill, walk_forward
from ...forecasting.base import HORIZONS
from ...forecasting.ensemble import combine, interpolate_path, price_band
from ...forecasting.models import RandomWalk, available_models, fit_model
from ..base import AgentContext, BaseAgent
from .prompts import METHOD
from .schema import ForecastData, ForecastFinding, HorizonOut, LevelOut, ModelMetrics


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class ForecastingAgent(BaseAgent):
    name, title, finding_cls = "forecasting", "Stock Price Forecasting Agent", ForecastFinding

    def analyze(self, ctx: AgentContext) -> ForecastFinding:
        df: pd.DataFrame = ctx.shared["prices"]
        close, volume = df["Close"], df["Volume"]
        hs = list(HORIZONS.values())
        models, skipped = available_models(ctx.settings.enable_prophet, ctx.settings.enable_lstm)
        warnings: list[str] = []
        if len(close) < 600:
            warnings.append(f"Only {len(close)} sessions of history; long-horizon backtests are thin.")

        def run_one(cls):
            bt, _ = walk_forward(close, volume, cls, hs, folds=ctx.settings.backtest_folds)
            try:
                live = fit_model(cls, close, volume, hs)
                preds = {}
                for h in hs:
                    try:
                        preds[h] = live.predict([h])[h]
                    except Exception as e:
                        bt.errors.append(f"live h={h}: {str(e)[:80]}")
                return cls, bt, preds, None
            except Exception as e:
                return cls, bt, {}, str(e)

        with ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(run_one, models))

        backtests = {cls.name: bt for cls, bt, _, _ in results}
        attach_skill(list(backtests.values()), RandomWalk.name)
        live_preds = {cls.name: preds for cls, _, preds, _ in results}
        metrics = []
        for cls, bt, preds, err in results:
            status = "ok" if preds and not err else ("failed: " + (err or "no predictions"))[:120]
            metrics.append(ModelMetrics(
                name=cls.name, family=cls.family, status=status, errors=bt.errors[:5],
                by_horizon={lbl: {k: (None if v is None else round(v, 4)) for k, v in vars(bt.by_horizon[h]).items()}
                            for lbl, h in HORIZONS.items()}))

        p0 = float(close.iloc[-1])
        last_date = close.index[-1]
        out_h: list[HorizonOut] = []
        ens = []
        for lbl, h in HORIZONS.items():
            preds_h = {m: p[h] for m, p in live_preds.items() if h in p}
            e = combine(h, preds_h, backtests, RandomWalk.name)
            ens.append(e)
            band = price_band(p0, e.mu, e.sigma)
            out_h.append(HorizonOut(
                label=lbl, days=h, target_date=str((last_date + pd.tseries.offsets.BDay(h)).date()),
                base=round(band["base"], 2), bull=round(band["bull"], 2), bear=round(band["bear"], 2),
                lo80=round(band["lo80"], 2), hi80=round(band["hi80"], 2),
                lo95=round(band["lo95"], 2), hi95=round(band["hi95"], 2),
                expected_return_pct=round((band["base"] / p0 - 1) * 100, 2),
                bull_upside_pct=round((band["bull"] / p0 - 1) * 100, 2),
                bear_downside_pct=round((band["bear"] / p0 - 1) * 100, 2),
                prob_up=round(_phi(e.mu / e.sigma) if e.sigma > 0 else 0.5, 3),
                calibration=round(e.calibration, 3), model_dispersion=round(e.model_dispersion, 4),
                best_skill=None if e.best_skill is None else round(e.best_skill, 4),
                weights={k: round(v, 3) for k, v in e.weights.items()},
                model_point={m: round(p0 * np.exp(p.mu), 2) for m, p in preds_h.items()},
            ))
        levels = [LevelOut(**vars(l)) for l in find_levels(df)]

        # Signal: risk-adjusted expected return over 1-6 months, scaled by demonstrated skill.
        mid = [e for e in ens if e.h in (21, 63, 126)]
        z = float(np.mean([e.mu / e.sigma for e in mid if e.sigma > 0])) if mid else 0.0
        skill = max([e.best_skill for e in mid if e.best_skill is not None] or [0.0])
        non_naive_w = float(np.mean([1 - e.weights.get(RandomWalk.name, 0) for e in mid])) if mid else 0
        signal = float(np.tanh(1.5 * z))
        confidence = float(np.clip(0.25 + 4 * max(skill, 0) + 0.2 * non_naive_w, 0.1, 0.85))
        if skill <= 0:
            warnings.append("No model beat the naive random walk out-of-sample; forecasts carry little directional edge.")

        h3 = next(h for h in out_h if h.label == "3M")
        h1y = next(h for h in out_h if h.label == "1Y")
        ok_models = [m.name for m in metrics if m.status == "ok"]
        key_points = [
            f"3-month base case {h3.base:.2f} ({h3.expected_return_pct:+.1f}%), 80% range {h3.lo80:.2f}–{h3.hi80:.2f}.",
            f"1-year base case {h1y.base:.2f} ({h1y.expected_return_pct:+.1f}%); bull {h1y.bull:.2f}, bear {h1y.bear:.2f}.",
            f"Probability of a higher price in 3 months: {h3.prob_up:.0%}.",
            f"Best out-of-sample skill vs naive (1–6M): {skill:+.1%}. Ensemble weights for 3M: "
            + ", ".join(f"{k} {v:.0%}" for k, v in sorted(h3.weights.items(), key=lambda x: -x[1])),
        ]
        sup = [l for l in levels if l.kind == "support"]
        res = [l for l in levels if l.kind == "resistance"]
        if sup or res:
            key_points.append("Key levels — support: " + (", ".join(f"{l.price:.2f}" for l in sup[:2]) or "n/a")
                              + "; resistance: " + (", ".join(f"{l.price:.2f}" for l in res[:2]) or "n/a"))
        return ForecastFinding(
            agent=self.name, ticker=ctx.ticker, status="ok" if len(ok_models) >= 3 else "partial",
            signal=signal, confidence=confidence,
            summary=(f"Ensemble of {len(ok_models)} models projects {h3.expected_return_pct:+.1f}% over 3 months and "
                     f"{h1y.expected_return_pct:+.1f}% over 1 year, with wide uncertainty."),
            key_points=key_points,
            risks=[f"95% 1-year range spans {h1y.lo95:.2f}–{h1y.hi95:.2f}.",
                   "Statistical models extrapolate past behaviour and cannot anticipate news shocks."],
            sources=[ctx.source("Daily OHLCV history", "market_data", published_at=last_date.to_pydatetime()),
                     ctx.source("Walk-forward backtest of forecasting models", "model", delayed=False)],
            data_as_of=last_date.to_pydatetime(), warnings=warnings,
            unavailable=[f"{k} ({v})" for k, v in skipped.items()],
            data=ForecastData(last_price=round(p0, 2), last_date=str(last_date.date()), horizons=out_h,
                              models=metrics, skipped_models=skipped, path=interpolate_path(p0, ens, step=1),
                              levels=levels, method=METHOD),
        )
