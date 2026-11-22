"""Agent 7: Institutional & Options Flow."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ...analytics.risk_metrics import annualized_vol
from ..base import AgentContext, BaseAgent
from .prompts import LABELS, UNUSUAL_MIN_VOLUME, UNUSUAL_VOL_OI
from .schema import FlowData, FlowFinding


def _col(df: pd.DataFrame, *names):
    return next((n for n in names if n in df.columns), None)


def max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    strikes = sorted(set(calls["strike"]).union(puts["strike"]))
    if not strikes:
        return None
    c = calls.set_index("strike")["openInterest"].fillna(0)
    p = puts.set_index("strike")["openInterest"].fillna(0)
    ck, cv, pk, pv = c.index.to_numpy(float), c.to_numpy(float), p.index.to_numpy(float), p.to_numpy(float)
    pain = {k: float((np.clip(k - ck, 0, None) * cv).sum() + (np.clip(pk - k, 0, None) * pv).sum()) for k in strikes}
    return min(pain, key=pain.get)


class FlowAgent(BaseAgent):
    name, title, finding_cls = "flow", "Institutional & Options Flow Agent", FlowFinding

    def analyze(self, ctx: AgentContext) -> FlowFinding:
        price = ctx.shared["quote"].price
        unavailable, kp, risks = [], [], []
        comps: list[tuple[float, float]] = []
        try:
            h = ctx.provider.holders(ctx.ticker)
        except Exception as e:
            h = {}
            unavailable.append(f"holder data ({e})")
        mh = h.get("major_holders") or {}
        own = {"insiders_pct": mh.get("insidersPercentHeld"), "institutions_pct": mh.get("institutionsPercentHeld"),
               "institutions_float_pct": mh.get("institutionsFloatPercentHeld"), "institutions_count": mh.get("institutionsCount")}
        prof = ctx.shared.get("profile") or {}
        own["short_pct_float"] = prof.get("shortPercentOfFloat")

        inst = h.get("institutional_holders")
        top, net_chg = [], None
        if isinstance(inst, pd.DataFrame) and not inst.empty:
            pc, ph = _col(inst, "pctChange", "% Change"), _col(inst, "pctHeld", "% Out")
            for _, r in inst.head(10).iterrows():
                top.append({"holder": r.get("Holder"), "pct_held": r.get(ph) if ph else None, "shares": r.get("Shares"),
                            "value": r.get("Value"), "pct_change": r.get(pc) if pc else None,
                            "date_reported": str(pd.Timestamp(r.get("Date Reported")).date()) if r.get("Date Reported") is not None else None})
            if pc and ph:
                w = inst[ph].astype(float).fillna(0)
                net_chg = float((inst[pc].astype(float).fillna(0) * w).sum() / w.sum()) if w.sum() else None
                if net_chg is not None:
                    comps.append((float(np.tanh(net_chg / 0.05)), 0.3))
                    kp.append(f"Top-10 institutions changed holdings by {net_chg:+.1%} (holding-weighted, latest 13F).")
        else:
            unavailable.append("institutional holders")

        ins = h.get("insider_transactions")
        isum, irows = {}, []
        if isinstance(ins, pd.DataFrame) and not ins.empty:
            dcol = _col(ins, "Start Date", "Date")
            df = ins.copy()
            df["_d"] = pd.to_datetime(df[dcol]) if dcol else pd.NaT
            df = df[df["_d"] >= pd.Timestamp.today() - pd.Timedelta(days=180)]
            text = (df.get("Text", pd.Series("", index=df.index)).fillna("") + " "
                    + df.get("Transaction", pd.Series("", index=df.index)).fillna("")).str.lower()
            buys, sells = df[text.str.contains("purchase|buy")], df[text.str.contains("sale|sell")]
            bv, sv = float(buys["Value"].fillna(0).sum()), float(sells["Value"].fillna(0).sum())
            isum = {"buys": len(buys), "sells": len(sells), "buy_value": bv, "sell_value": sv, "net_value": bv - sv,
                    "window_days": 180}
            for _, r in df.sort_values("_d", ascending=False).head(15).iterrows():
                irows.append({"date": str(r["_d"].date()) if pd.notna(r["_d"]) else None, "insider": r.get("Insider"),
                              "position": r.get("Position"), "text": r.get("Text"), "shares": r.get("Shares"),
                              "value": r.get("Value")})
            if len(buys) >= 2 and bv > 0.2 * sv:
                comps.append((0.6, 0.25))
                kp.append(f"Insider buying: {len(buys)} open-market purchases (${bv/1e6:.1f}M) in 180 days.")
            elif len(sells):
                comps.append((-0.15 if sv > 5 * max(bv, 1) else 0.0, 0.15))
                kp.append(f"Insiders: {len(sells)} sales (${sv/1e6:.1f}M) vs {len(buys)} purchases in 180 days "
                          "(sales are often pre-planned 10b5-1 trades).")
        else:
            unavailable.append("insider transactions")

        osum: dict = {}
        term, walls, unusual, oi_strike = [], {}, [], []
        try:
            chains = ctx.provider.options(ctx.ticker, 6)
            now = datetime.now(timezone.utc).date()
            cv = pv = coi = poi = 0.0
            for ch in chains:
                cv += ch.calls["volume"].fillna(0).sum()
                pv += ch.puts["volume"].fillna(0).sum()
                coi += ch.calls["openInterest"].fillna(0).sum()
                poi += ch.puts["openInterest"].fillna(0).sum()
                atm = ch.calls.iloc[(ch.calls["strike"] - price).abs().argsort()[:2]]
                atm_p = ch.puts.iloc[(ch.puts["strike"] - price).abs().argsort()[:2]]
                iv = float(pd.concat([atm["impliedVolatility"], atm_p["impliedVolatility"]]).mean())
                dte = (pd.Timestamp(ch.expiration).date() - now).days
                term.append({"expiration": ch.expiration, "dte": dte, "atm_iv": round(iv, 4)})
                for side, frame in (("call", ch.calls), ("put", ch.puts)):
                    f = frame.fillna({"volume": 0, "openInterest": 0})
                    u = f[(f["volume"] >= UNUSUAL_VOL_OI * f["openInterest"].clip(lower=1))
                          & (f["volume"] >= UNUSUAL_MIN_VOLUME)]
                    for _, r in u.iterrows():
                        unusual.append({"expiration": ch.expiration, "type": side, "strike": float(r["strike"]),
                                        "volume": int(r["volume"]), "open_interest": int(r["openInterest"]),
                                        "vol_oi": round(r["volume"] / max(r["openInterest"], 1), 1)})
            near = chains[0]
            for side, frame in (("calls", near.calls), ("puts", near.puts)):
                t = frame.nlargest(3, "openInterest")
                walls[side] = [{"strike": float(r["strike"]), "open_interest": int(r["openInterest"] or 0)} for _, r in t.iterrows()]
            m = near.calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"}).merge(
                near.puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"}), on="strike", how="outer").fillna(0)
            oi_strike = m[(m["strike"] > price * 0.75) & (m["strike"] < price * 1.25)].sort_values("strike").to_dict("records")
            rv30 = annualized_vol(ctx.shared["prices"]["Close"], 21)
            osum = {"put_call_volume": pv / cv if cv else None, "put_call_oi": poi / coi if coi else None,
                    "call_volume": cv, "put_volume": pv, "atm_iv_near": term[0]["atm_iv"] if term else None,
                    "realized_vol_30d": rv30, "max_pain_near": max_pain(near.calls, near.puts),
                    "iv_premium": (term[0]["atm_iv"] / rv30 - 1) if term and rv30 else None,
                    "nearest_expiration": near.expiration}
            unusual = sorted(unusual, key=lambda u: -u["volume"])[:8]
            pcv = osum["put_call_volume"]
            if pcv is not None:
                comps.append((float(np.clip((0.9 - pcv) / 0.6, -1, 1)) * 0.6, 0.25))
                kp.append(f"Put/call volume {pcv:.2f}, put/call OI {osum['put_call_oi']:.2f}; near-term ATM IV "
                          f"{osum['atm_iv_near']:.0%} vs 30-day realized {rv30:.0%}.")
            uc = sum(u["volume"] for u in unusual if u["type"] == "call")
            up = sum(u["volume"] for u in unusual if u["type"] == "put")
            if unusual:
                comps.append((float(np.tanh((uc - up) / max(uc + up, 1) * 2)) * 0.5, 0.2))
                kp.append(f"{len(unusual)} unusual contracts (volume ≥ {UNUSUAL_VOL_OI:.0f}× OI): "
                          f"call volume {uc:,} vs put volume {up:,}.")
            if osum.get("iv_premium") and osum["iv_premium"] > 0.5:
                risks.append("Options price a large move (implied vol well above realized); likely event risk ahead.")
            if len(term) >= 2 and term[0]["atm_iv"] > term[-1]["atm_iv"] * 1.15:
                risks.append("Inverted IV term structure: near-term event priced in.")
        except Exception as e:
            unavailable.append(f"options chains ({e})")

        sig = sum(v * w for v, w in comps) / sum(w for _, w in comps) if comps else 0.0
        verdict = ("Signs of accumulation" if sig > 0.15 else "Signs of distribution" if sig < -0.15
                   else "No clear accumulation or distribution")
        return FlowFinding(
            agent=self.name, ticker=ctx.ticker, status="ok" if len(comps) >= 3 else "partial" if comps else "unavailable",
            signal=float(sig), confidence=float(np.clip(0.15 + 0.12 * len(comps), 0, 0.6)),
            summary=f"{verdict} across ownership, insider and options data.", key_points=kp, risks=risks,
            sources=[ctx.source(f"{ctx.provider.info.name}: 13F-derived holders", "filing"),
                     ctx.source(f"{ctx.provider.info.name}: Form 4 insider transactions", "filing"),
                     ctx.source(f"{ctx.provider.info.name}: listed option chains (delayed)", "options")],
            data_as_of=datetime.now(timezone.utc), unavailable=unavailable + ["real-time options order flow (paid feed required)"],
            data=FlowData(ownership=own, top_institutions=top, institutional_net_change=net_chg, insider_summary=isum,
                          insider_transactions=irows, options_summary=osum, iv_term_structure=term, oi_walls=walls,
                          unusual_activity=unusual, oi_by_strike=oi_strike, accumulation_verdict=verdict,
                          data_labels=LABELS),
        )
