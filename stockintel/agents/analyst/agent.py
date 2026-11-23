"""Agent 3: Analyst Ratings & Price Targets."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..base import AgentContext, BaseAgent
from .prompts import LABEL
from .schema import AnalystData, AnalystFinding, RatingChange

ACTION_NAMES = {"up": "upgrade", "down": "downgrade", "init": "initiation", "main": "target change",
                "reit": "reiteration"}


def _f(v) -> float | None:
    try:
        v = float(v)
        return None if math.isnan(v) or v <= 0 else v
    except (TypeError, ValueError):
        return None


def rec_score(row: pd.Series) -> tuple[float | None, int]:
    sb, b, h, s, ss = (int(row.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
    n = sb + b + h + s + ss
    return ((2 * sb + b - s - 2 * ss) / (2 * n) if n else None), n


class AnalystAgent(BaseAgent):
    name, title, finding_cls = "analyst", "Analyst Ratings & Price Target Agent", AnalystFinding

    def analyze(self, ctx: AgentContext) -> AnalystFinding:
        price = ctx.shared["quote"].price
        try:
            raw = ctx.provider.analyst(ctx.ticker)
        except Exception as e:
            raw = {}
            err = str(e)
        else:
            err = None
        unavailable = []
        pt = raw.get("price_targets") or {}
        targets = {k: _f(pt.get(k)) for k in ("mean", "median", "high", "low")}
        if not any(targets.values()):
            unavailable.append("consensus price targets")
        upside = {k: (round((v / price - 1) * 100, 2) if v else None) for k, v in targets.items()}

        recs = raw.get("recommendations")
        counts, score, trend, n_an = {}, None, None, None
        if isinstance(recs, pd.DataFrame) and not recs.empty:
            r0 = recs.iloc[0]
            score, n_an = rec_score(r0)
            counts = {k: int(r0.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
            if len(recs) >= 4:
                s3, _ = rec_score(recs.iloc[3])
                trend = None if (score is None or s3 is None) else score - s3
        else:
            unavailable.append("recommendation counts")

        ud = raw.get("upgrades_downgrades")
        changes: list[RatingChange] = []
        counts90: dict[str, int] = {}
        rev90 = None
        history = []
        if isinstance(ud, pd.DataFrame) and not ud.empty:
            ud = ud.copy()
            ud.index = pd.to_datetime(ud.index)
            if getattr(ud.index, "tz", None) is not None:
                ud.index = ud.index.tz_localize(None)
            ud = ud.sort_index(ascending=False)
            now = pd.Timestamp.today().normalize()
            last90 = ud[ud.index >= now - pd.Timedelta(days=90)]
            counts90 = {ACTION_NAMES.get(a, a): int(n) for a, n in last90["Action"].value_counts().items()}
            if "currentPriceTarget" in ud.columns:
                cur = ud["currentPriceTarget"].where(ud["currentPriceTarget"] > 0)
                prior = (ud["priorPriceTarget"].where(ud["priorPriceTarget"] > 0) if "priorPriceTarget" in ud.columns
                         else pd.Series(np.nan, index=ud.index))
                chg = (cur / prior - 1).where(prior.notna())
                r90 = chg[ud.index >= now - pd.Timedelta(days=90)].dropna()
                rev90 = float(r90.median() * 100) if len(r90) else None
                m = cur.dropna().groupby(cur.dropna().index.to_period("M")).agg(["median", "count"]).tail(12)
                history = [{"month": str(p), "median_target": round(float(row["median"]), 2), "n": int(row["count"])}
                           for p, row in m.iterrows()]
            else:
                cur = prior = chg = pd.Series(np.nan, index=ud.index)
                unavailable.append("per-firm target revisions (provider does not report targets)")
            for i, (ts, row) in enumerate(ud.head(25).iterrows()):
                c, p = _f(cur.iloc[i]), _f(prior.iloc[i])
                pct = round((c / p - 1) * 100, 1) if c and p else None
                action = ACTION_NAMES.get(row.get("Action"), row.get("Action") or "")
                changes.append(RatingChange(
                    date=str(ts.date()), firm=str(row.get("Firm")), action=action,
                    from_grade=row.get("FromGrade") or None, to_grade=row.get("ToGrade") or None,
                    prior_target=p, current_target=c, target_change_pct=pct,
                    notable=action in ("upgrade", "downgrade", "initiation") or (pct is not None and abs(pct) >= 15)))
        else:
            unavailable.append("upgrade/downgrade history")

        # Consensus targets lag. If firms that revised in the last 30 days sit far below the consensus median,
        # use their median for the signal and flag the consensus as stale.
        stale_note = None
        recent = [c.current_target for c in changes if c.current_target
                  and pd.Timestamp(c.date) >= pd.Timestamp.today() - pd.Timedelta(days=30)]
        recent_median = float(np.median(recent)) if len(recent) >= 2 else None
        signal_upside = upside["median"]
        if recent_median and targets["median"] and recent_median < 0.7 * targets["median"]:
            signal_upside = (recent_median / price - 1) * 100
            stale_note = (f"Consensus looks stale: the {len(recent)} targets revised in the last 30 days have a median of "
                          f"{recent_median:.2f} ({signal_upside:+.1f}%), well below the consensus median {targets['median']:.2f}.")
        dispersion = ((targets["high"] - targets["low"]) / targets["median"]
                      if targets["high"] and targets["low"] and targets["median"] else None)
        comps = []
        if signal_upside is not None:
            comps.append((np.tanh(signal_upside / 20), 0.5))
        if score is not None:
            comps.append((score, 0.25))
        if rev90 is not None:
            comps.append((np.tanh(rev90 / 10), 0.25))
        signal = sum(v * w for v, w in comps) / sum(w for _, w in comps) if comps else 0.0
        conf = 0.15 + (0.35 if targets["median"] else 0) + (0.2 * min(1, (n_an or 0) / 20)) \
            - (0.15 if dispersion and dispersion > 0.8 else 0)
        ups = counts90.get("upgrade", 0)
        downs = counts90.get("downgrade", 0)
        kp = []
        if targets["median"]:
            kp.append(f"Median target {targets['median']:.2f} ({upside['median']:+.1f}% vs {price:.2f}); "
                      f"mean {targets['mean']:.2f}, range {targets['low']:.2f}–{targets['high']:.2f}.")
        if score is not None:
            kp.append(f"{n_an} analysts: {counts.get('strongBuy', 0) + counts.get('buy', 0)} buy, "
                      f"{counts.get('hold', 0)} hold, {counts.get('sell', 0) + counts.get('strongSell', 0)} sell"
                      + (f"; rating score {'up' if trend > 0 else 'down'} {abs(trend):.2f} vs 3 months ago" if trend else ""))
        if counts90:
            kp.append(f"Last 90 days: {ups} upgrades, {downs} downgrades"
                      + (f"; median target revision {rev90:+.1f}%" if rev90 is not None else ""))
        if stale_note:
            kp.insert(0, stale_note)
        for c in [c for c in changes if c.notable][:3]:
            kp.append(f"{c.date} {c.firm}: {c.action}" + (f" {c.from_grade}→{c.to_grade}" if c.from_grade else
                                                           f" to {c.to_grade}" if c.to_grade else "")
                      + (f", target {c.prior_target:.0f}→{c.current_target:.0f}" if c.prior_target and c.current_target else ""))
        latest = changes[0].date if changes else None
        status = "ok" if targets["median"] and changes else "partial" if (targets["median"] or changes) else "unavailable"
        return AnalystFinding(
            agent=self.name, ticker=ctx.ticker, status=status, signal=float(signal), confidence=float(max(0.05, conf)),
            summary=(f"Analyst consensus implies {upside['median']:+.1f}% to the median target." if upside["median"] is not None
                     else "No consensus price target is available.") + f" ({LABEL})",
            key_points=kp,
            risks=([stale_note] if stale_note else []) + (["Wide target dispersion; analysts disagree materially."] if dispersion and dispersion > 0.6 else [])
                  + (["Net downgrades over the last 90 days."] if downs > ups else []),
            sources=[ctx.source(f"{ctx.provider.info.name}: analyst price targets & rating changes", "analyst")],
            data_as_of=pd.Timestamp(latest).to_pydatetime() if latest else None,
            unavailable=unavailable + ([f"analyst data ({err})"] if err else []),
            data=AnalystData(current_price=price, targets=targets, implied_upside_pct=upside, n_analysts=n_an,
                             recommendation_counts=counts, recommendation_score=score, recommendation_trend=trend,
                             changes=changes, counts_90d=counts90, target_revision_90d_pct=rev90,
                             target_dispersion=dispersion, revision_history=history),
        )
