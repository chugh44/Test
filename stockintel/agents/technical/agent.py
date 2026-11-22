"""Agent 5: Technical Analysis & Trading Signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...analytics.indicators import add_all
from ...analytics.levels import find_levels
from ...analytics.patterns import detect
from ..base import AgentContext, BaseAgent
from .prompts import MAX_STOP_ATR, MIN_STOP_ATR, RSI_OVERBOUGHT, RSI_OVERSOLD, SIGNAL_THRESHOLD, STOP_ATR_BUFFER
from .schema import TechnicalData, TechnicalFinding, TradePlan


def _sig(x: float) -> str:
    return "BUY" if x > SIGNAL_THRESHOLD else "SELL" if x < -SIGNAL_THRESHOLD else "HOLD"


def build_plan(price: float, atr: float, levels: list, sma20: float | None) -> TradePlan:
    sup = sorted([l.price for l in levels if l.kind == "support" and l.price < price], reverse=True)
    res = sorted([l.price for l in levels if l.kind == "resistance" and l.price > price])
    s1 = sup[0] if sup else None
    # Entry zone: from the nearest support (or the 20-day SMA) up to just under the price.
    lo_anchor = max(x for x in [s1, sma20, price - 1.5 * atr] if x is not None and x < price) \
        if any(x is not None and x < price for x in [s1, sma20]) else price - 1.0 * atr
    entry_low, entry_high = round(min(lo_anchor, price - 0.25 * atr), 2), round(price, 2)
    stop = (s1 - STOP_ATR_BUFFER * atr) if s1 else price - 2 * atr
    stop = min(stop, price - MIN_STOP_ATR * atr)
    stop = max(stop, price - MAX_STOP_ATR * atr)
    risk = price - stop
    t1 = res[0] if res and res[0] - price > 0.5 * risk else price + 2 * risk
    t2 = next((r for r in res if r > t1 + 0.5 * risk), price + 3 * risk)
    return TradePlan(entry_low=entry_low, entry_high=entry_high, stop_loss=round(stop, 2), target_1=round(t1, 2),
                     target_2=round(t2, 2), risk_reward_t1=round((t1 - price) / risk, 2) if risk > 0 else None,
                     risk_reward_t2=round((t2 - price) / risk, 2) if risk > 0 else None,
                     rationale=("Stop placed below nearest support" if s1 else "Stop at 2×ATR (no support found)")
                     + ("; targets at overhead resistance" if res else "; targets at 2R/3R (no resistance overhead)"))


class TechnicalAgent(BaseAgent):
    name, title, finding_cls = "technical", "Technical Analysis & Trading Signals Agent", TechnicalFinding

    def analyze(self, ctx: AgentContext) -> TechnicalFinding:
        df = add_all(ctx.shared["prices"])
        last = df.iloc[-1]
        price = float(last["Close"])
        g = lambda k: None if pd.isna(last.get(k)) else float(last[k])  # noqa: E731
        sma20, sma50, sma200, rsi_v = g("sma20"), g("sma50"), g("sma200"), g("rsi14")
        atr = g("atr14") or price * 0.02

        comp: dict[str, float] = {}
        # Long-term trend
        lt = 0.0
        if sma200:
            lt += 0.4 if price > sma200 else -0.4
            slope = df["sma200"].iloc[-1] / df["sma200"].iloc[-21] - 1 if len(df) > 221 else 0
            lt += float(np.clip(slope * 20, -0.3, 0.3))
        if sma50 and sma200:
            lt += 0.3 if sma50 > sma200 else -0.3
        comp["trend"] = round(lt, 3)
        # Short-term momentum
        st = 0.0
        if rsi_v is not None:
            st += 0.25 if 50 < rsi_v < RSI_OVERBOUGHT else -0.25 if RSI_OVERSOLD < rsi_v < 50 else 0
            st += -0.2 if rsi_v >= RSI_OVERBOUGHT else 0.2 if rsi_v <= RSI_OVERSOLD else 0
        hist = df["hist"].dropna()
        if len(hist) > 3:
            st += 0.2 if hist.iloc[-1] > 0 else -0.2
            st += 0.1 if hist.iloc[-1] > hist.iloc[-3] else -0.1
        if sma20:
            st += 0.15 if price > sma20 else -0.15
        vwap = g("vwap20")
        if vwap:
            st += 0.1 if price > vwap else -0.1
        comp["momentum"] = round(st, 3)
        # Volume confirmation: OBV slope agreeing with price slope
        obv_s = df["obv"].diff(20).iloc[-1]
        px_s = df["Close"].diff(20).iloc[-1]
        vol_c = 0.15 if np.sign(obv_s) == np.sign(px_s) and px_s > 0 else -0.15 if np.sign(obv_s) == np.sign(px_s) else 0
        comp["volume"] = vol_c
        pats = detect(df)
        pat_score = float(np.clip(sum(0.15 if p.bias == "bullish" else -0.15 if p.bias == "bearish" else 0 for p in pats), -0.4, 0.4))
        comp["patterns"] = round(pat_score, 3)
        bb = g("bb_pct_b")
        comp["bollinger"] = round(-0.1 if bb and bb > 1 else 0.1 if bb is not None and bb < 0 else 0, 3)

        short_score = float(np.clip(st + vol_c + pat_score + comp["bollinger"], -1, 1))
        long_score = float(np.clip(lt + 0.5 * pat_score + 0.5 * vol_c, -1, 1))
        levels = find_levels(df)
        plan = build_plan(price, atr, levels, sma20)

        trend = ("uptrend" if lt > 0.3 else "downtrend" if lt < -0.3 else "sideways / mixed")
        momentum = ("overbought" if rsi_v and rsi_v >= RSI_OVERBOUGHT else "oversold" if rsi_v and rsi_v <= RSI_OVERSOLD
                    else "positive" if st > 0.2 else "negative" if st < -0.2 else "neutral")
        setups = []
        if lt > 0.3 and sma20 and abs(price / sma20 - 1) < 0.02:
            setups.append("Pullback to the 20-day SMA within an uptrend (swing long setup)")
        if rsi_v and rsi_v <= RSI_OVERSOLD and lt > 0:
            setups.append("Oversold within a long-term uptrend (mean-reversion bounce candidate)")
        if any("breakout" in p.name.lower() for p in pats):
            setups.append("Breakout in progress; wait for a retest of the breakout level before entering")
        if any("breakdown" in p.name.lower() for p in pats):
            setups.append("Breakdown in progress; avoid longs until price recovers the prior range")
        if rsi_v and rsi_v >= RSI_OVERBOUGHT:
            setups.append("Overbought: chasing here has poor risk/reward; prefer the lower part of the entry zone")
        ind = {k: g(k) for k in ("rsi14", "macd", "signal", "hist", "sma20", "sma50", "sma200", "ema12", "ema26",
                                 "bb_upper", "bb_lower", "bb_pct_b", "bb_width", "atr14", "vwap20")}
        ind["volume_ratio_20d"] = float(last["Volume"] / last["vol_sma20"]) if last["vol_sma20"] else None
        blend = 0.5 * short_score + 0.5 * long_score
        return TechnicalFinding(
            agent=self.name, ticker=ctx.ticker, signal=blend,
            confidence=float(np.clip(0.35 + 0.3 * abs(blend) + (0.1 if sma200 else -0.15), 0.1, 0.8)),
            summary=(f"{trend.capitalize()}, momentum {momentum}. Short-term signal {_sig(short_score)} "
                     f"({short_score:+.2f}); long-term signal {_sig(long_score)} ({long_score:+.2f})."),
            key_points=[
                f"RSI {rsi_v:.0f}" if rsi_v is not None else "RSI n/a",
                f"Price vs SMA50/200: {price:.2f} / {sma50 or float('nan'):.2f} / {sma200 or float('nan'):.2f}",
                f"MACD histogram {ind['hist']:+.3f}" if ind["hist"] is not None else "MACD n/a",
                f"Entry zone {plan.entry_low:.2f}–{plan.entry_high:.2f}, stop {plan.stop_loss:.2f}, "
                f"targets {plan.target_1:.2f} / {plan.target_2:.2f} (R/R {plan.risk_reward_t1}, {plan.risk_reward_t2})",
            ] + [f"Pattern: {p.name} ({p.bias}, {p.date})" for p in pats[:4]],
            risks=[s for s in setups if "avoid" in s.lower() or "overbought" in s.lower()],
            sources=[ctx.source("Daily OHLCV history (indicators computed locally)", "market_data",
                                published_at=df.index[-1].to_pydatetime())],
            data_as_of=df.index[-1].to_pydatetime(),
            data=TechnicalData(last_close=price, indicators=ind, components=comp, short_term_score=short_score,
                               long_term_score=long_score, short_term_signal=_sig(short_score),
                               long_term_signal=_sig(long_score), trend=trend, momentum=momentum,
                               patterns=[vars(p) for p in pats], levels=[vars(l) for l in levels], setups=setups,
                               plan=plan),
        )
