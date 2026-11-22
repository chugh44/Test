"""Agent 6: Market & Sector Intelligence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ...analytics.risk_metrics import period_return
from ...data.news_sources import fred_series
from ...data.peers import SECTOR_ETF, get_peers
from ..base import AgentContext, BaseAgent
from .prompts import SYSTEM_MACRO, USER_MACRO
from .schema import MacroData, MacroFinding

PERIODS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
ALL_SECTOR_ETFS = sorted(set(SECTOR_ETF.values()))


def _rets(close: pd.Series | None) -> dict[str, float | None]:
    if close is None:
        return {k: None for k in PERIODS}
    return {k: period_return(close, d) for k, d in PERIODS.items()}


class MacroAgent(BaseAgent):
    name, title, finding_cls = "macro", "Market & Sector Intelligence Agent", MacroFinding

    def _hist(self, ctx: AgentContext, sym: str) -> pd.Series | None:
        try:
            return ctx.provider.history(sym, "2y")["Close"]
        except Exception:
            return None

    def analyze(self, ctx: AgentContext) -> MacroFinding:
        prof = ctx.shared.get("profile") or {}
        sector = prof.get("sector")
        etf = SECTOR_ETF.get(sector or "")
        peers, _ = get_peers(ctx.ticker, ctx.settings.finnhub_key)
        syms = ["^GSPC", "^IXIC", "^VIX", "^TNX", "^IRX"] + ALL_SECTOR_ETFS + peers
        with ThreadPoolExecutor(max_workers=6) as ex:
            hist = dict(zip(syms, ex.map(lambda s: self._hist(ctx, s), syms)))
        unavailable = [s for s in ("^GSPC", "^VIX", "^TNX") if hist.get(s) is None]
        stock = ctx.shared["prices"]["Close"]
        spx = hist.get("^GSPC")

        mk: dict[str, float | None] = {}
        regime_score = 0.0
        if spx is not None:
            mk["sp500_1m"], mk["sp500_3m"] = period_return(spx, 21), period_return(spx, 63)
            sma200 = spx.rolling(200).mean().iloc[-1]
            mk["sp500_vs_200dma"] = float(spx.iloc[-1] / sma200 - 1) if not np.isnan(sma200) else None
            regime_score += 0.35 if (mk["sp500_vs_200dma"] or 0) > 0 else -0.35
            regime_score += float(np.clip((mk["sp500_3m"] or 0) * 3, -0.25, 0.25))
        if hist.get("^IXIC") is not None:
            mk["nasdaq_1m"], mk["nasdaq_3m"] = period_return(hist["^IXIC"], 21), period_return(hist["^IXIC"], 63)
        if hist.get("^VIX") is not None:
            v = hist["^VIX"]
            mk["vix"] = float(v.iloc[-1])
            mk["vix_percentile_1y"] = float((v.tail(252) < v.iloc[-1]).mean())
            regime_score += 0.2 if mk["vix"] < 18 else -0.25 if mk["vix"] > 25 else 0
        if hist.get("^TNX") is not None:
            t = hist["^TNX"]
            mk["us10y"] = float(t.iloc[-1])
            mk["us10y_change_3m_bp"] = float((t.iloc[-1] - t.iloc[-64]) * 100) if len(t) > 64 else None
            if mk["us10y_change_3m_bp"] is not None:
                regime_score -= float(np.clip(mk["us10y_change_3m_bp"] / 400, -0.15, 0.15))
        if hist.get("^IRX") is not None and mk.get("us10y") is not None:
            mk["us3m"] = float(hist["^IRX"].iloc[-1])
            mk["curve_10y_3m_bp"] = (mk["us10y"] - mk["us3m"]) * 100
        regime = ("risk-on" if regime_score > 0.3 else "risk-off" if regime_score < -0.3 else "mixed")

        returns = {"stock": _rets(stock), "market": _rets(spx), "sector": _rets(hist.get(etf) if etf else None)}
        rs = {}
        for k in PERIODS:
            s, m_, sec = returns["stock"][k], returns["market"][k], returns["sector"][k]
            rs[f"vs_market_{k}"] = (s - m_) if s is not None and m_ is not None else None
            rs[f"vs_sector_{k}"] = (s - sec) if s is not None and sec is not None else None
        ranking = sorted(
            [{"etf": e, "return_3m": period_return(hist[e], 63), "return_1m": period_return(hist[e], 21)}
             for e in ALL_SECTOR_ETFS if hist.get(e) is not None], key=lambda r: -(r["return_3m"] or -9))
        for i, r in enumerate(ranking):
            r["rank"] = i + 1
        sec_rank = next((r["rank"] for r in ranking if r["etf"] == etf), None)
        peer_rows = [{"symbol": p, **{k: v for k, v in _rets(hist.get(p)).items()}} for p in peers if hist.get(p) is not None]

        inflation: dict = {}
        cpi = fred_series("CPIAUCSL", ctx.settings.fred_key, 14)
        if cpi and len(cpi) >= 13:
            inflation["cpi_yoy"] = cpi[-1][1] / cpi[-13][1] - 1
            inflation["cpi_as_of"] = cpi[-1][0]
        ff = fred_series("FEDFUNDS", ctx.settings.fred_key, 3)
        if ff:
            inflation["fed_funds"], inflation["fed_funds_as_of"] = ff[-1][1], ff[-1][0]
        if not inflation:
            unavailable.append("CPI / Fed funds series (set FRED_API_KEY)")

        brief, brief_src = None, []
        if not ctx.provider.info.is_synthetic:
            ws = ctx.llm.web_search(purpose="macro_brief", system=SYSTEM_MACRO.format(sector=sector, ticker=ctx.ticker),
                                    prompt=USER_MACRO.format(today=datetime.now(timezone.utc).date(), ticker=ctx.ticker,
                                                             name=prof.get("longName", ctx.ticker), sector=sector),
                                    max_uses=4)
            if ws and ws.text:
                brief, brief_src = ws.text, list(dict.fromkeys(ws.cited_urls or [r.url for r in ws.results]))[:8]

        rs3m, rs6m = rs.get("vs_market_3M"), rs.get("vs_market_6M")
        rel = np.nanmean([x for x in [rs3m, rs6m, rs.get("vs_sector_3M")] if x is not None]) if any(
            x is not None for x in [rs3m, rs6m, rs.get("vs_sector_3M")]) else 0.0
        sector_score = (0.2 if sec_rank and sec_rank <= 3 else -0.2 if sec_rank and sec_rank >= len(ranking) - 2 else 0)
        signal = float(np.clip(0.45 * regime_score + 0.4 * np.tanh(rel * 5) + sector_score, -1, 1))
        verdict = ("outperforming" if (rs3m or 0) > 0.02 else "underperforming" if (rs3m or 0) < -0.02 else "in line with") \
            + " the market over 3 months" + (f", and {'ahead of' if (rs.get('vs_sector_3M') or 0) > 0 else 'behind'} "
                                             f"its sector ({etf})" if rs.get("vs_sector_3M") is not None else "")
        kp = [f"Market regime: {regime} (S&P 500 {'above' if (mk.get('sp500_vs_200dma') or 0) > 0 else 'below'} its "
              f"200-day average; VIX {mk.get('vix', float('nan')):.1f})",
              f"10-year yield {mk.get('us10y', float('nan')):.2f}% ({(mk.get('us10y_change_3m_bp') or 0):+.0f}bp over 3 months)",
              f"{ctx.ticker} is {verdict}."]
        if sec_rank:
            kp.append(f"Sector {etf} ranks {sec_rank}/{len(ranking)} by 3-month return (sector rotation).")
        if "cpi_yoy" in inflation:
            kp.append(f"CPI {inflation['cpi_yoy']:.1%} YoY (as of {inflation['cpi_as_of']}); "
                      f"Fed funds {inflation.get('fed_funds', float('nan')):.2f}%")
        last_dt = spx.index[-1].to_pydatetime() if spx is not None else None
        return MacroFinding(
            agent=self.name, ticker=ctx.ticker, status="ok" if spx is not None else "partial", signal=signal,
            confidence=float(np.clip(0.3 + (0.2 if spx is not None else 0) + (0.1 if etf else 0) + (0.1 if brief else 0), 0, 0.7)),
            summary=f"{regime.capitalize()} backdrop; {ctx.ticker} is {verdict}.", key_points=kp,
            risks=(["Rising long-term yields weigh on equity valuations."] if (mk.get("us10y_change_3m_bp") or 0) > 40 else [])
                  + (["Elevated volatility (VIX > 25)."] if (mk.get("vix") or 0) > 25 else [])
                  + (["Inverted yield curve (10y below 3m)."] if (mk.get("curve_10y_3m_bp") or 0) < 0 else []),
            sources=[ctx.source(f"{ctx.provider.info.name}: indices, yields, sector ETFs", "macro")]
                    + ([ctx.source("FRED (St. Louis Fed)", "macro", url="https://fred.stlouisfed.org", delayed=False)] if inflation else [])
                    + [ctx.source("Claude web search", "llm_web_search", url=u, delayed=False) for u in brief_src],
            data_as_of=last_dt, unavailable=unavailable,
            data=MacroData(market=mk, regime=regime, sector=sector, sector_etf=etf, returns=returns, relative_strength=rs,
                           sector_ranking=ranking, peers=peer_rows, inflation=inflation, macro_brief=brief,
                           macro_brief_sources=brief_src, performance_verdict=verdict),
        )
