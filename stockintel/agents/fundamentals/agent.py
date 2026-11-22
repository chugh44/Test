"""Agent 4: Financial & Fundamental Analysis."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ...data.peers import get_peers
from ..base import AgentContext, BaseAgent
from .prompts import THRESHOLDS as T
from .schema import FundamentalsData, FundamentalsFinding, PeerRow


def _num(v) -> float | None:
    try:
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index(ascending=False)
    return None


def _at(s: pd.Series | None, i: int) -> float | None:
    if s is None or len(s) <= i:
        return None
    return _num(s.iloc[i])


def _g(a, b) -> float | None:
    return (a / b - 1) if a is not None and b not in (None, 0) and b > 0 else None


def _ratio(a, b) -> float | None:
    return a / b if a is not None and b not in (None, 0) else None


def _median(vals) -> float | None:
    v = [x for x in vals if x is not None and x > 0]
    return float(np.median(v)) if v else None


VAL_KEYS = {"trailing_pe": "trailingPE", "forward_pe": "forwardPE", "peg": "pegRatio",
            "ev_ebitda": "enterpriseToEbitda", "price_to_sales": "priceToSalesTrailing12Months",
            "price_to_book": "priceToBook"}


class FundamentalsAgent(BaseAgent):
    name, title, finding_cls = "fundamentals", "Financial & Fundamental Analysis Agent", FundamentalsFinding

    def analyze(self, ctx: AgentContext) -> FundamentalsFinding:
        info = ctx.shared.get("profile") or {}
        unavailable: list[str] = []
        try:
            fin = ctx.provider.financials(ctx.ticker)
        except Exception as e:
            fin = {}
            unavailable.append(f"financial statements ({e})")
        ia, iq = fin.get("income_annual"), fin.get("income_quarterly")
        ca, cq = fin.get("cashflow_annual"), fin.get("cashflow_quarterly")
        ba, bq = fin.get("balance_annual"), fin.get("balance_quarterly")

        rev_a, rev_q = row(ia, "Total Revenue", "Operating Revenue"), row(iq, "Total Revenue", "Operating Revenue")
        ni_a = row(ia, "Net Income", "Net Income Common Stockholders")
        op_a = row(ia, "Operating Income", "EBIT")
        gp_a = row(ia, "Gross Profit")
        ebitda_a = row(ia, "EBITDA", "Normalized EBITDA")
        eps_a = row(ia, "Diluted EPS", "Basic EPS")
        ocf_a = row(ca, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        capex_a = row(ca, "Capital Expenditure")
        fcf_a = row(ca, "Free Cash Flow")
        debt = row(bq if bq is not None else ba, "Total Debt")
        cash = row(bq if bq is not None else ba, "Cash And Cash Equivalents",
                   "Cash Cash Equivalents And Short Term Investments")
        equity = row(bq if bq is not None else ba, "Stockholders Equity", "Common Stock Equity")
        ca_ = row(bq if bq is not None else ba, "Current Assets")
        cl_ = row(bq if bq is not None else ba, "Current Liabilities")
        shares = row(ba, "Ordinary Shares Number", "Share Issued")

        fcf0 = _at(fcf_a, 0)
        if fcf0 is None and _at(ocf_a, 0) is not None and _at(capex_a, 0) is not None:
            fcf0 = _at(ocf_a, 0) + _at(capex_a, 0)
        rev0, rev1 = _at(rev_a, 0), _at(rev_a, 1)
        m = {
            "revenue_ttm_or_fy": rev0,
            "revenue_growth_yoy": _g(rev0, rev1) if rev0 else _num(info.get("revenueGrowth")),
            "revenue_growth_q_yoy": _g(_at(rev_q, 0), _at(rev_q, 4)) if rev_q is not None and len(rev_q) >= 5
            else _num(info.get("revenueGrowth")),
            "revenue_cagr_3y": ((rev0 / _at(rev_a, 3)) ** (1 / 3) - 1) if rev0 and _at(rev_a, 3) else None,
            "eps_growth_yoy": _g(_at(eps_a, 0), _at(eps_a, 1)) or _num(info.get("earningsGrowth")),
            "net_income_growth_yoy": _g(_at(ni_a, 0), _at(ni_a, 1)),
            "gross_margin": _ratio(_at(gp_a, 0), rev0) or _num(info.get("grossMargins")),
            "operating_margin": _ratio(_at(op_a, 0), rev0) or _num(info.get("operatingMargins")),
            "operating_margin_prior": _ratio(_at(op_a, 1), rev1),
            "net_margin": _ratio(_at(ni_a, 0), rev0) or _num(info.get("profitMargins")),
            "roe": _num(info.get("returnOnEquity")),
            "free_cash_flow": fcf0 if fcf0 is not None else _num(info.get("freeCashflow")),
            "fcf_margin": _ratio(fcf0, rev0),
            "fcf_conversion": _ratio(fcf0, _at(ni_a, 0)) if (_at(ni_a, 0) or 0) > 0 else None,
            "total_debt": _at(debt, 0) or _num(info.get("totalDebt")),
            "cash": _at(cash, 0) or _num(info.get("totalCash")),
            "debt_to_equity": _ratio(_at(debt, 0), _at(equity, 0)) or (
                _num(info.get("debtToEquity")) / 100 if _num(info.get("debtToEquity")) else None),
            "current_ratio": _ratio(_at(ca_, 0), _at(cl_, 0)) or _num(info.get("currentRatio")),
            "share_count_change_yoy": _g(_at(shares, 0), _at(shares, 1)),
        }
        nd = (m["total_debt"] or 0) - (m["cash"] or 0) if m["total_debt"] is not None else None
        m["net_debt"] = nd
        m["net_debt_to_ebitda"] = _ratio(nd, _at(ebitda_a, 0)) if nd is not None and (_at(ebitda_a, 0) or 0) > 0 else None
        val = {k: _num(info.get(v)) for k, v in VAL_KEYS.items()}
        if val["peg"] is None:
            val["peg"] = _num(info.get("trailingPegRatio"))
        if not any(v is not None for v in val.values()):
            unavailable.append("valuation multiples")

        # --- peers
        peers_syms, peer_src = get_peers(ctx.ticker, ctx.settings.finnhub_key)
        peers: list[PeerRow] = []

        def peer(sym):
            try:
                p = ctx.provider.profile(sym)
                return PeerRow(symbol=sym, name=p.get("shortName") or p.get("longName"),
                               market_cap=_num(p.get("marketCap")), trailing_pe=_num(p.get("trailingPE")),
                               forward_pe=_num(p.get("forwardPE")), ev_ebitda=_num(p.get("enterpriseToEbitda")),
                               price_to_sales=_num(p.get("priceToSalesTrailing12Months")),
                               revenue_growth=_num(p.get("revenueGrowth")), operating_margin=_num(p.get("operatingMargins")))
            except Exception:
                return None
        if peers_syms:
            with ThreadPoolExecutor(max_workers=5) as ex:
                peers = [p for p in ex.map(peer, peers_syms) if p]
        else:
            unavailable.append("peer comparison (no peer list for this ticker; set FINNHUB_API_KEY)")
        pm = {k: _median([getattr(p, k) for p in peers]) for k in
              ("trailing_pe", "forward_pe", "ev_ebitda", "price_to_sales", "revenue_growth", "operating_margin")}
        prem = {k: (round((val[k] / pm[k] - 1) * 100, 1) if val.get(k) and pm.get(k) else None)
                for k in ("trailing_pe", "forward_pe", "ev_ebitda", "price_to_sales")}

        # --- assessment
        strengths, weak, flags = [], [], []
        g, om = m["revenue_growth_yoy"], m["operating_margin"]
        if g is not None:
            (strengths if g >= T["strong_growth"] else weak if g < T["weak_growth"] else []).append(
                f"Revenue growth {g:+.1%} YoY")
        if om is not None:
            if om >= T["high_op_margin"]:
                strengths.append(f"High operating margin {om:.1%}")
            elif om < T["low_op_margin"]:
                weak.append(f"Thin operating margin {om:.1%}")
        if om is not None and m["operating_margin_prior"] is not None:
            d = (om - m["operating_margin_prior"]) * 100
            if d <= -T["margin_compression_pp"]:
                flags.append(f"Operating margin compressed {d:.1f}pp YoY")
            elif d >= T["margin_compression_pp"]:
                strengths.append(f"Operating margin expanded {d:+.1f}pp YoY")
        if m["free_cash_flow"] is not None:
            if m["free_cash_flow"] < 0:
                flags.append("Negative free cash flow")
            elif m["fcf_margin"] and m["fcf_margin"] > 0.15:
                strengths.append(f"Strong FCF margin {m['fcf_margin']:.1%}")
        if m["fcf_conversion"] is not None and m["fcf_conversion"] < T["min_fcf_conversion"]:
            flags.append(f"Low cash conversion: FCF is {m['fcf_conversion']:.0%} of net income")
        if m["net_debt_to_ebitda"] is not None:
            if m["net_debt_to_ebitda"] > T["max_net_debt_ebitda"]:
                flags.append(f"High leverage: net debt/EBITDA {m['net_debt_to_ebitda']:.1f}x")
            elif m["net_debt_to_ebitda"] < 0:
                strengths.append("Net cash balance sheet")
        if m["current_ratio"] is not None and m["current_ratio"] < T["min_current_ratio"]:
            weak.append(f"Current ratio {m['current_ratio']:.2f} below 1")
        if m["share_count_change_yoy"] is not None:
            if m["share_count_change_yoy"] > T["share_dilution"]:
                flags.append(f"Share dilution {m['share_count_change_yoy']:+.1%} YoY")
            elif m["share_count_change_yoy"] < -0.01:
                strengths.append(f"Share count reduced {m['share_count_change_yoy']:.1%} YoY (buybacks)")

        q = 0.0
        q += np.tanh((g or 0) / 0.15) * 0.35
        q += np.tanh(((om or 0) - 0.1) / 0.15) * 0.25
        q += (0.2 if (m["fcf_margin"] or 0) > 0.1 else -0.2 if (m["free_cash_flow"] or 0) < 0 else 0)
        q -= 0.12 * len(flags)
        quality = float(np.clip(q, -1, 1))

        # Valuation: premium vs peers, adjusted for the growth gap. PEG is the reference point.
        vs = []
        if prem.get("forward_pe") is not None:
            growth_gap = ((g or 0) - (pm.get("revenue_growth") or 0)) * 100  # pp
            vs.append(-np.tanh((prem["forward_pe"] - 2.0 * growth_gap) / 50))
        if val["peg"]:
            vs.append(float(np.clip((1.5 - val["peg"]) / 1.5, -1, 1)))
        if prem.get("ev_ebitda") is not None:
            vs.append(-np.tanh(prem["ev_ebitda"] / 60))
        vscore = float(np.mean(vs)) if vs else 0.0
        verdict = ("Valuation looks justified or cheap relative to growth and peers" if vscore > 0.15 else
                   "Valuation looks stretched relative to growth and peers" if vscore < -0.15 else
                   "Valuation roughly in line with growth and peers") if vs else "Insufficient data to judge valuation"

        def series_table(inc, cf, n):
            rv, ni_, op_ = row(inc, "Total Revenue"), row(inc, "Net Income"), row(inc, "Operating Income")
            fc = row(cf, "Free Cash Flow")
            out = []
            if rv is None:
                return out
            for i in range(min(n, len(rv))):
                r = _at(rv, i)
                out.append({"period": str(rv.index[i].date()), "revenue": r, "net_income": _at(ni_, i),
                            "free_cash_flow": _at(fc, i) if fc is not None and len(fc) > i else None,
                            "operating_margin": _ratio(_at(op_, i), r)})
            return list(reversed(out))

        coverage = sum(v is not None for v in m.values()) / len(m)
        latest = rev_q.index[0] if rev_q is not None else rev_a.index[0] if rev_a is not None else None
        kp = []
        if g is not None:
            kp.append(f"Revenue {g:+.1%} YoY" + (f", 3-yr CAGR {m['revenue_cagr_3y']:.1%}" if m["revenue_cagr_3y"] else ""))
        if om is not None:
            kp.append(f"Margins: gross {m['gross_margin'] or 0:.1%}, operating {om:.1%}, net {m['net_margin'] or 0:.1%}")
        if val["forward_pe"]:
            kp.append(f"Forward P/E {val['forward_pe']:.1f}" + (f" vs peer median {pm['forward_pe']:.1f} "
                                                                 f"({prem['forward_pe']:+.0f}%)" if pm.get("forward_pe") else ""))
        kp.append(verdict + ".")
        return FundamentalsFinding(
            agent=self.name, ticker=ctx.ticker, status="ok" if coverage > 0.6 else "partial" if coverage > 0.2 else "unavailable",
            signal=float(np.clip(0.55 * quality + 0.45 * vscore, -1, 1)),
            confidence=float(np.clip(0.2 + 0.5 * coverage + (0.15 if peers else 0), 0, 0.85)),
            summary=f"Quality score {quality:+.2f}, valuation score {vscore:+.2f}. {verdict}.",
            key_points=kp, risks=flags + weak,
            sources=[ctx.source(f"{ctx.provider.info.name}: financial statements & key statistics", "fundamentals",
                                delayed=False, published_at=latest.to_pydatetime() if latest is not None else None),
                     ctx.source(f"Peer set ({peer_src})", "derived", delayed=False)],
            data_as_of=latest.to_pydatetime() if latest is not None else None, unavailable=unavailable,
            data=FundamentalsData(metrics=m, valuation=val, peer_median=pm, premium_vs_peers=prem, peers=peers,
                                  peer_source=peer_src, annual=series_table(ia, ca, 4), quarterly=series_table(iq, cq, 5),
                                  strengths=strengths, weaknesses=weak, red_flags=flags, quality_score=quality,
                                  valuation_score=vscore, valuation_verdict=verdict,
                                  latest_period=str(latest.date()) if latest is not None else None),
        )
