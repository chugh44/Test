"""Agent 8: Risk Management. Runs after the consensus draft and reviews the proposed thesis."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ...analytics.indicators import atr as atr_fn
from ...analytics.risk_metrics import annualized_vol, beta, log_returns, max_drawdown, var_cvar
from ..base import AgentContext, BaseAgent
from .prompts import (EARNINGS_CAP, EARNINGS_WAIT_DAYS, EXTENDED_ATR, HIGH_VOL, HIGH_VOL_CAP, MAX_STOP_PCT, MIN_RR,
                      PROFILES)
from .schema import PositionSize, RiskData, RiskFinding, RiskReview


class RiskAgent(BaseAgent):
    name, title, finding_cls = "risk", "Risk Management Agent", RiskFinding

    def analyze(self, ctx: AgentContext) -> RiskFinding:
        px = ctx.shared["prices"]
        close = px["Close"]
        price = float(ctx.shared["quote"].price)
        bench = ctx.shared.get("benchmark")
        unavailable = []

        vol1y, vol3m = annualized_vol(close, 252), annualized_vol(close, 63)
        mdd1, pk1, tr1 = max_drawdown(close.tail(252))
        mdd5, pk5, tr5 = max_drawdown(close)
        var21, cvar21 = var_cvar(close.tail(756), 21)
        var5, cvar5 = var_cvar(close.tail(756), 5)
        b = beta(close, bench) if bench is not None else None
        if bench is None:
            unavailable.append("benchmark for beta")
        r = log_returns(close).tail(252)
        downside_dev = float(r[r < 0].std() * np.sqrt(252))
        monthly = close.resample("ME").last().pct_change().dropna()
        atr = float(atr_fn(px).iloc[-1])
        metrics = {"annual_vol_1y": vol1y, "annual_vol_3m": vol3m, "max_drawdown_1y": mdd1, "max_drawdown_1y_peak": pk1,
                   "max_drawdown_1y_trough": tr1, "max_drawdown_5y": mdd5, "var95_1m": var21, "cvar95_1m": cvar21,
                   "var95_1w": var5, "cvar95_1w": cvar5, "beta": b, "downside_deviation": downside_dev,
                   "worst_month": float(monthly.min()) if len(monthly) else None, "atr14": atr,
                   "atr_pct": atr / price}

        # --- event risk
        events = []
        try:
            cal = ctx.provider.calendar(ctx.ticker)
        except Exception:
            cal = {}
            unavailable.append("earnings calendar")
        now = datetime.now(timezone.utc)
        ed = cal.get("earnings_date")
        days_to_earn = None
        if ed:
            days_to_earn = (ed - now).days
            if days_to_earn >= -1:
                events.append({"event": "Earnings report", "date": ed.date().isoformat(), "days_away": days_to_earn,
                               "typical_move_pct": None})
        flow = ctx.findings.get("flow")
        if flow and flow.data and flow.data.options_summary.get("atm_iv_near") and days_to_earn is not None and events:
            iv = flow.data.options_summary["atm_iv_near"]
            events[-1]["typical_move_pct"] = round(iv * np.sqrt(1 / 52) * 100, 1)  # ~1-week implied move
        news = ctx.findings.get("news")
        if news and news.data:
            for rk in news.data.risks[:3]:
                events.append({"event": rk.text, "date": rk.expected_date, "days_away": None, "typical_move_pct": None})

        # --- stop and risk/reward
        tech = ctx.findings.get("technical")
        plan = tech.data.plan if tech and tech.data else None
        stop = plan.stop_loss if plan else price - 2 * atr
        stop_dist = (price - stop) / price if stop else None
        fc = ctx.findings.get("forecasting")
        rr = {"technical_t1": plan.risk_reward_t1 if plan else None, "technical_t2": plan.risk_reward_t2 if plan else None}
        if fc and fc.data:
            h3 = next((h for h in fc.data.horizons if h.label == "3M"), None)
            if h3 and h3.bear < price:
                rr["forecast_3m_bull_vs_bear"] = round((h3.bull - price) / (price - h3.bear), 2)
        an = ctx.findings.get("analyst")
        if an and an.data and an.data.targets.get("median") and stop and price > stop:
            rr["analyst_median_vs_stop"] = round((an.data.targets["median"] - price) / (price - stop), 2)

        # --- suitability and sizing
        suit = ("conservative" if vol1y <= PROFILES["Conservative"][2] and (b or 1) <= 1.1 else
                "moderate" if vol1y <= PROFILES["Moderate"][2] else "aggressive")
        sizes = []
        for prof, (rpt, cap, maxvol) in PROFILES.items():
            alloc = min(cap, rpt / stop_dist) if stop_dist and stop_dist > 0 else 0.0
            ok = vol1y <= maxvol
            sizes.append(PositionSize(profile=prof, risk_per_trade_pct=rpt * 100, max_position_pct=cap * 100,
                                      position_pct_of_portfolio=round(alloc * 100 if ok else 0.0, 2), suitable=ok,
                                      note=(f"Risking {rpt:.1%} of the portfolio with a stop {stop_dist:.1%} below the price"
                                            if ok and stop_dist else f"Volatility {vol1y:.0%} exceeds this profile's limit")))

        scen = []
        for label, mv in (("1-week 95% VaR", var5), ("1-month 95% VaR", var21), ("1-month 95% CVaR (tail average)", cvar21),
                          ("Repeat of worst 1-year drawdown", mdd1), ("Repeat of worst 5-year drawdown", mdd5)):
            scen.append({"scenario": label, "move_pct": round(mv * 100, 1), "price": round(price * (1 + mv), 2)})
        if b:
            scen.append({"scenario": "Market falls 10% (beta-implied)", "move_pct": round(-10 * b, 1),
                         "price": round(price * (1 - 0.1 * b), 2)})

        inval = [f"Daily close below the stop at {stop:.2f}"]
        if tech and tech.data and tech.data.indicators.get("sma200"):
            inval.append(f"Sustained break below the 200-day SMA ({tech.data.indicators['sma200']:.2f})")
        inval += ["Guidance cut or an earnings miss at the next report",
                  "A cluster of analyst downgrades or target cuts",
                  "Market regime turning risk-off (S&P 500 below its 200-day average with VIX above 25)"]

        # --- review of the draft thesis
        review = RiskReview()
        draft = ctx.draft
        if draft is not None:
            if days_to_earn is not None and 0 <= days_to_earn <= EARNINGS_WAIT_DAYS:
                review.force_wait, review.conviction_cap = True, EARNINGS_CAP
                review.notes.append(f"Earnings in {days_to_earn} days: wait for the report or size down; conviction capped at {EARNINGS_CAP}.")
            if vol1y > HIGH_VOL:
                review.conviction_cap = min(review.conviction_cap or 100, HIGH_VOL_CAP)
                review.notes.append(f"Very high volatility ({vol1y:.0%}); conviction capped at {HIGH_VOL_CAP}.")
            if draft.decision == "BUY":
                if rr.get("technical_t1") is not None and rr["technical_t1"] < MIN_RR:
                    review.force_wait = True
                    review.notes.append(f"Risk/reward to the first target is {rr['technical_t1']:.2f} (< {MIN_RR}); "
                                        "wait for a better entry.")
                if plan and price > plan.entry_high + EXTENDED_ATR * atr:
                    review.force_wait = True
                    review.notes.append("Price is extended above the entry zone.")
                if stop_dist and stop_dist > MAX_STOP_PCT:
                    review.veto = True
                    review.notes.append(f"Nearest sensible stop is {stop_dist:.0%} away; the risk is too large to define.")
            if not review.notes:
                review.notes.append("No risk objections to the draft thesis.")

        rr_val = rr.get("technical_t1") or 1.0
        signal = float(np.clip(np.tanh((rr_val - MIN_RR) / 1.5) * 0.4 - max(0, vol1y - 0.35), -1, 1))
        return RiskFinding(
            agent=self.name, ticker=ctx.ticker, signal=signal, confidence=0.6,
            summary=(f"Annual volatility {vol1y:.0%}, beta {b:.2f}" if b else f"Annual volatility {vol1y:.0%}")
                    + f", 1-month 95% VaR {var21:.1%}. Suitable for {suit} investors and above. "
                    + " ".join(review.notes),
            key_points=[f"Max drawdown: {mdd1:.1%} (1y), {mdd5:.1%} (5y)",
                        f"Stop {stop:.2f} ({stop_dist:.1%} below the price)" if stop_dist else "No stop available",
                        "R/R: " + ", ".join(f"{k} {v}" for k, v in rr.items() if v is not None)]
                       + [f"{e['event']} on {e['date']}" for e in events if e.get("date")][:2],
            risks=[e["event"] for e in events][:5],
            sources=[ctx.source("Daily OHLCV + benchmark (risk metrics computed locally)", "derived")],
            data_as_of=close.index[-1].to_pydatetime(), unavailable=unavailable,
            data=RiskData(metrics=metrics, event_risks=events, stop_loss=round(stop, 2) if stop else None,
                          stop_distance_pct=round(stop_dist * 100, 2) if stop_dist else None, risk_reward=rr,
                          position_sizing=sizes, suitability=suit, downside_scenarios=scen, invalidation=inval,
                          review=review),
        )
