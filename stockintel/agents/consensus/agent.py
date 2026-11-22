"""Agent 9: Investment Decision & Consensus (master coordinator).

It runs twice. ``draft()`` comes after the research agents: it weighs the evidence, finds agreement and
conflicts, and proposes a decision. ``finalize()`` comes after the risk agent has reviewed that draft: it
applies the risk overlay, decides entry timing and writes the thesis.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from ...schemas.common import AgentFinding, stance_of
from ..base import AgentContext, BaseAgent
from .prompts import (MIN_DIRECTIONAL_CONVICTION, MIN_ENTER_NOW_CONVICTION, BASE_WEIGHTS, CONFLICT_MIN_SIGNAL, DECISION_THRESHOLD, HALF_LIFE_DAYS, METHOD, SYSTEM_THESIS,
                      USER_THESIS)
from .schema import AgentVote, ConsensusData, ConsensusFinding, Conflict, DraftDecision, LLMThesis

TITLES = {"forecasting": "Price Forecasting", "news": "News Intelligence", "analyst": "Analyst Consensus",
          "fundamentals": "Fundamentals", "technical": "Technical Signals", "macro": "Market & Sector",
          "flow": "Institutional & Options", "risk": "Risk Management"}
STATUS_MULT = {"ok": 1.0, "partial": 0.7, "unavailable": 0.0, "error": 0.0}


def _freshness(f: AgentFinding) -> float:
    if not f.data_as_of:
        return 0.6
    ts = f.data_as_of if f.data_as_of.tzinfo else f.data_as_of.replace(tzinfo=timezone.utc)
    age = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400)
    # Weekends and holidays make daily bars look older than they are, so allow 3 days of grace.
    return float(max(0.3, 0.5 ** (max(0, age - 3) / HALF_LIFE_DAYS.get(f.agent, 30))))


def build_draft(findings: dict[str, AgentFinding]) -> DraftDecision:
    votes: list[AgentVote] = []
    for name, bw in BASE_WEIGHTS.items():
        f = findings.get(name)
        if f is None:
            continue
        fr = _freshness(f)
        ew = bw * f.confidence * fr * STATUS_MULT.get(f.status, 0)
        votes.append(AgentVote(agent=name, title=TITLES[name], status=f.status, signal=round(f.signal, 3),
                               confidence=round(f.confidence, 3), freshness=round(fr, 3), base_weight=bw,
                               effective_weight=round(ew, 4), stance=stance_of(f.signal), summary=f.summary[:300]))
    tot = sum(v.effective_weight for v in votes)
    score = sum(v.signal * v.effective_weight for v in votes) / tot if tot else 0.0
    disp = math.sqrt(sum(v.effective_weight * (v.signal - score) ** 2 for v in votes) / tot) if tot else 1.0
    agreement = float(max(0.0, 1 - disp / 0.6))
    coverage = sum(1 for v in votes if v.effective_weight > 0) / len(BASE_WEIGHTS)

    conflicts: list[Conflict] = []
    for a, b in combinations([v for v in votes if v.effective_weight > 0], 2):
        if a.signal * b.signal < 0 and abs(a.signal) >= CONFLICT_MIN_SIGNAL and abs(b.signal) >= CONFLICT_MIN_SIGNAL:
            win = a if a.effective_weight >= b.effective_weight else b
            lose = b if win is a else a
            why = []
            if win.confidence > lose.confidence + 0.05:
                why.append("higher confidence / evidence quality")
            if win.freshness > lose.freshness + 0.05:
                why.append("fresher data")
            if win.base_weight > lose.base_weight:
                why.append("larger base weight")
            conflicts.append(Conflict(
                agents=[a.agent, b.agent],
                description=f"{a.title} is {a.stance} ({a.signal:+.2f}) while {b.title} is {b.stance} ({b.signal:+.2f}).",
                resolution=f"{win.title} carries more weight ({win.effective_weight:.3f} vs {lose.effective_weight:.3f}) "
                           f"due to {', '.join(why) or 'comparable evidence'}; the net effect is reflected in the score.",
                severity=round(min(1.0, abs(a.signal - b.signal) / 2 * (a.confidence + b.confidence)), 3)))
    agreements = []
    for stance in ("bullish", "bearish", "neutral"):
        names = [v.title for v in votes if v.stance == stance and v.effective_weight > 0]
        if len(names) >= 2:
            agreements.append(f"{len(names)} agents are {stance}: " + ", ".join(names))

    # Explicit check: StockIntel's AI forecast vs the Wall Street consensus, never blended
    fva: dict = {}
    fc, an = findings.get("forecasting"), findings.get("analyst")
    if fc and fc.data and an and an.data and an.data.implied_upside_pct.get("median") is not None:
        h1y = next(h for h in fc.data.horizons if h.label == "1Y")
        fva = {"ai_1y_base_return_pct": h1y.expected_return_pct,
               "analyst_median_implied_pct": an.data.implied_upside_pct["median"],
               "gap_pp": round(an.data.implied_upside_pct["median"] - h1y.expected_return_pct, 2)}
        fva["note"] = ("Analyst targets are well above the model's base case. Targets are typically 12-month views "
                       "and lean optimistic, while the model has limited out-of-sample skill; treat the gap as a range of "
                       "outcomes, not a number to average." if fva["gap_pp"] > 15 else
                       "Analyst targets sit below the model's base case." if fva["gap_pp"] < -15 else
                       "AI forecast and analyst consensus are broadly consistent.")

    decision = "BUY" if score > DECISION_THRESHOLD else "SELL" if score < -DECISION_THRESHOLD else "HOLD"
    strength = min(1.0, abs(score) / 0.45) if decision != "HOLD" else max(0.0, 1 - abs(score) / DECISION_THRESHOLD) * 0.6
    # Conflicts lower conviction, capped so that contested evidence reads as "low conviction" and not as zero.
    penalty = min(30.0, sum(4 + 8 * c.severity for c in conflicts))
    conviction = float(np.clip(100 * (0.45 * strength + 0.35 * agreement + 0.20 * coverage) - penalty, 0, 100))
    return DraftDecision(decision=decision, score=round(score, 4), conviction=round(conviction, 1),
                         agreement=round(agreement, 3), coverage=round(coverage, 3), votes=votes, conflicts=conflicts,
                         agreements=agreements, forecast_vs_analyst=fva)


def _template_thesis(ticker, draft: DraftDecision, findings, timing, decision: str) -> LLMThesis:
    bulls = [v for v in draft.votes if v.stance == "bullish" and v.effective_weight > 0]
    bears = [v for v in draft.votes if v.stance == "bearish" and v.effective_weight > 0]
    maps = (f"which maps to {decision}" if decision == draft.decision else
            f"which maps to {draft.decision}, softened to {decision} because conviction is low")
    para1 = (f"The weighted evidence score is {draft.score:+.2f}, {maps}. "
             f"{len(bulls)} agents lean bullish and {len(bears)} lean bearish, with agreement {draft.agreement:.0%}.")
    para2 = ("Main conflicts: " + " ".join(c.description for c in draft.conflicts[:3])) if draft.conflicts else \
        "No material conflicts between agents."
    bp = [kp for v in bulls for kp in (findings[v.agent].key_points[:2])][:6]
    br = [r for v in draft.votes for r in findings[v.agent].risks[:2] if findings.get(v.agent)][:6]
    return LLMThesis(headline=f"{ticker}: {decision} ({timing.lower()}). Conviction {draft.conviction:.0f}/100.",
                     thesis=para1 + "\n\n" + para2, bull_points=bp or ["No strong bullish evidence."],
                     bear_points=br or ["No strong bearish evidence."],
                     what_changes_our_mind=["A change in the evidence score's sign at the next refresh",
                                            "Any of the risk agent's invalidation conditions"])


class ConsensusAgent(BaseAgent):
    name, title, finding_cls = "consensus", "Investment Decision & Consensus Agent", ConsensusFinding

    def draft(self, ctx: AgentContext) -> DraftDecision:
        return build_draft(ctx.findings)

    def analyze(self, ctx: AgentContext) -> ConsensusFinding:
        draft: DraftDecision = ctx.draft or build_draft(ctx.findings)
        f = ctx.findings
        risk = f.get("risk")
        review = risk.data.review if risk and risk.data else None
        tech = f.get("technical")
        plan = tech.data.plan if tech and tech.data else None
        price = ctx.shared["quote"].price

        conviction = draft.conviction
        adjustments: list[str] = []
        if review:
            if review.conviction_cap is not None and conviction > review.conviction_cap:
                adjustments.append(f"Conviction capped from {conviction:.0f} to {review.conviction_cap} by the risk agent.")
                conviction = review.conviction_cap
            adjustments += [n for n in review.notes if not n.startswith("No risk objections")]

        decision = draft.decision
        if decision != "HOLD" and conviction < MIN_DIRECTIONAL_CONVICTION:
            adjustments.append(f"Draft {decision} downgraded to HOLD: conviction {conviction:.0f} is below "
                               f"{MIN_DIRECTIONAL_CONVICTION}, so the evidence is too contested for a directional call.")
            decision = "HOLD"
        if decision == "SELL":
            timing, why = "AVOID", "The evidence is net bearish; avoid new long positions (existing holders could reduce)."
        elif review and review.veto:
            timing, why = "AVOID", "Risk veto: " + "; ".join(review.notes)
        elif decision == "HOLD":
            timing = "WAIT FOR BETTER ENTRY"
            why = "The evidence is balanced. Wait for a clearer signal or a pullback into the entry zone."
        elif review and review.force_wait:
            timing, why = "WAIT FOR BETTER ENTRY", " ".join(review.notes)
        elif conviction < MIN_ENTER_NOW_CONVICTION:
            timing, why = "WAIT FOR BETTER ENTRY", (f"Conviction {conviction:.0f} is below {MIN_ENTER_NOW_CONVICTION}; "
                                                     "wait for the evidence to firm up before entering.")
        elif plan and plan.entry_low <= price <= plan.entry_high * 1.005:
            timing, why = "ENTER NOW", f"Price {price:.2f} is inside the entry zone {plan.entry_low:.2f}–{plan.entry_high:.2f}."
        else:
            timing, why = "WAIT FOR BETTER ENTRY", "Price is outside the entry zone."

        catalysts = []
        news = f.get("news")
        if news and news.data:
            catalysts = [c.model_dump() for c in news.data.catalysts[:6]]
        key_risks = []
        if risk and risk.data:
            key_risks += [e["event"] + (f" ({e['date']})" if e.get("date") else "") for e in risk.data.event_risks[:3]]
        for n in ("fundamentals", "technical", "macro", "flow", "analyst", "news"):
            if f.get(n):
                key_risks += f[n].risks[:2]
        key_risks = list(dict.fromkeys(key_risks))[:10]

        # Narrative: the LLM explains the decision; template fallback otherwise.
        evidence = {
            "decision_score": draft.score, "agreement": draft.agreement, "coverage": draft.coverage,
            "votes": [v.model_dump(include={"title", "signal", "confidence", "freshness", "effective_weight", "summary"})
                      for v in draft.votes],
            "conflicts": [c.model_dump() for c in draft.conflicts], "risk_review": adjustments,
            "agent_key_points": {k: v.key_points[:5] for k, v in f.items() if k != "consensus"},
            "agent_risks": {k: v.risks[:4] for k, v in f.items() if k != "consensus"},
            "forecast_vs_analyst": draft.forecast_vs_analyst, "entry_timing_reason": why,
            "trade_plan": plan.model_dump() if plan else None, "catalysts": catalysts,
        }
        name = (ctx.shared.get("profile") or {}).get("longName", ctx.ticker)
        th = ctx.llm.structured(purpose="consensus_thesis", system=SYSTEM_THESIS, schema=LLMThesis, effort="high",
                                user=USER_THESIS.format(ticker=ctx.ticker, name=name, decision=decision,
                                                        conviction=round(conviction), timing=timing,
                                                        evidence=json.dumps(evidence, default=str, indent=1)))
        narrative_src = f"Claude ({ctx.llm.model})" if th else "template (no LLM configured)"
        if th is None:
            th = _template_thesis(ctx.ticker, draft.model_copy(update={"conviction": conviction}), f, timing, decision)

        st = lt = ""
        if tech and tech.data:
            st = f"Technical short-term signal {tech.data.short_term_signal}; momentum {tech.data.momentum}."
            lt = f"Technical long-term signal {tech.data.long_term_signal} ({tech.data.trend})"
        fund = f.get("fundamentals")
        if fund and fund.data:
            lt += f"; fundamentals: {fund.data.valuation_verdict.lower()}."
        return ConsensusFinding(
            agent=self.name, ticker=ctx.ticker, signal=draft.score, confidence=conviction / 100,
            summary=f"{decision}, conviction {conviction:.0f}/100, {timing.lower()}.",
            key_points=[th.headline] + draft.agreements, risks=key_risks,
            sources=[s for k, v in f.items() for s in v.sources[:3]][:25],
            data_as_of=datetime.now(timezone.utc), warnings=[w for v in f.values() for w in v.warnings[:1]][:3],
            data=ConsensusData(
                decision=decision, conviction=int(round(conviction)), score=draft.score, entry_timing=timing,
                entry_rationale=why, entry_zone=[plan.entry_low, plan.entry_high] if plan else None,
                stop_loss=risk.data.stop_loss if risk and risk.data else (plan.stop_loss if plan else None),
                targets=[plan.target_1, plan.target_2] if plan else [], short_term_view=st, long_term_view=lt,
                votes=draft.votes, conflicts=draft.conflicts, agreements=draft.agreements, risk_adjustments=adjustments,
                headline=th.headline, thesis=th.thesis, bull_points=th.bull_points, bear_points=th.bear_points,
                what_changes_our_mind=th.what_changes_our_mind, catalysts=catalysts, key_risks=key_risks,
                forecast_vs_analyst=draft.forecast_vs_analyst, narrative_source=narrative_src, method=METHOD),
        )
