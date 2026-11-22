"""Multi-agent orchestration with LangGraph.

validate ─┬─> forecasting ─┐
          ├─> news ────────┤
          ├─> analyst ─────┤
          ├─> fundamentals ┼─> consensus_draft ─> risk_review ─> consensus_final ─> persist ─> END
          ├─> technical ───┤
          ├─> macro ───────┤
          └─> flow ────────┘
(an invalid ticker routes validate ─> END)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from langgraph.graph import END, START, StateGraph

from ..agents.analyst.agent import AnalystAgent
from ..agents.base import AgentContext, BaseAgent
from ..agents.consensus.agent import ConsensusAgent, build_draft
from ..agents.flow.agent import FlowAgent
from ..agents.forecasting.agent import ForecastingAgent
from ..agents.fundamentals.agent import FundamentalsAgent
from ..agents.macro.agent import MacroAgent
from ..agents.news.agent import NewsAgent
from ..agents.risk.agent import RiskAgent
from ..agents.technical.agent import TechnicalAgent
from ..config import Settings, get_settings
from ..data.base import DataUnavailable, MarketDataProvider
from ..llm.client import LLMClient
from .state import GraphState, audit

RESEARCH_AGENTS: list[BaseAgent] = [ForecastingAgent(), NewsAgent(), AnalystAgent(), FundamentalsAgent(),
                                    TechnicalAgent(), MacroAgent(), FlowAgent()]
TICKER_RE = re.compile(r"^[\^A-Za-z][A-Za-z0-9.\-=^]{0,11}$")


@dataclass
class Runtime:
    provider: MarketDataProvider
    llm: LLMClient
    settings: Settings = field(default_factory=get_settings)
    store: Any = None  # storage.db.Store or None
    ctx: AgentContext | None = None


def resolve_ticker(provider: MarketDataProvider, query: str) -> tuple[str | None, str]:
    q = query.strip()
    if not q:
        return None, "empty query"
    if TICKER_RE.match(q) and " " not in q:
        sym = q.upper()
        try:
            provider.history(sym, "1mo")
            return sym, f"'{q}' validated as ticker {sym}"
        except Exception:
            pass
    matches = [m for m in provider.search(q) if (m.quote_type or "").upper() in ("EQUITY", "ETF", "")]
    for m in matches:
        try:
            provider.history(m.symbol, "1mo")
            return m.symbol, f"'{q}' resolved to {m.symbol} ({m.name}, {m.exchange})"
        except Exception:
            continue
    return None, f"No tradable security found for '{q}'"


def build_graph(rt: Runtime):
    def validate(state: GraphState) -> dict:
        sym, note = resolve_ticker(rt.provider, state["query"])
        if not sym:
            return {"valid": False, "error": note, "audit": [audit("validate", "invalid", note=note)]}
        try:
            prices = rt.provider.history(sym, f"{rt.settings.history_years}y")
            if len(prices) < 120:
                raise DataUnavailable(f"only {len(prices)} sessions of price history")
            quote = rt.provider.quote(sym)
        except Exception as e:
            return {"valid": False, "error": f"{sym}: {e}", "audit": [audit("validate", "invalid", note=str(e))]}
        warnings = []
        try:
            profile = rt.provider.profile(sym)
        except Exception as e:
            profile = {}
            warnings.append(f"profile unavailable: {e}")
        try:
            bench = rt.provider.history("^GSPC", f"{rt.settings.history_years}y")["Close"]
        except Exception:
            bench = None
            warnings.append("S&P 500 benchmark unavailable")
        rt.ctx = AgentContext(ticker=sym, provider=rt.provider, llm=rt.llm, settings=rt.settings,
                              shared={"prices": prices, "quote": quote, "profile": profile, "benchmark": bench})
        company = {"symbol": sym, "name": profile.get("longName") or profile.get("shortName") or sym,
                   "sector": profile.get("sector"), "industry": profile.get("industry"),
                   "exchange": quote.exchange, "currency": quote.currency,
                   "provider": rt.provider.info.name, "synthetic": rt.provider.info.is_synthetic,
                   "provider_notes": rt.provider.info.notes, "llm_enabled": rt.llm.available,
                   "llm_model": rt.llm.model if rt.llm.available else None}
        return {"ticker": sym, "valid": True, "error": None, "company": company, "quote": quote,
                "audit": [audit("validate", "ok", note=note, sessions=len(prices), last_date=str(prices.index[-1].date()),
                                warnings=warnings)]}

    def make_agent_node(agent: BaseAgent) -> Callable[[GraphState], dict]:
        def node(state: GraphState) -> dict:
            f = agent.run(rt.ctx)
            rt.ctx.findings[agent.name] = f
            return {"findings": {agent.name: f},
                    "audit": [audit(agent.name, f.status, signal=round(f.signal, 3), confidence=round(f.confidence, 3),
                                    seconds=f.duration_s, sources=len(f.sources), unavailable=f.unavailable[:5],
                                    warnings=f.warnings[:3])]}
        return node

    def consensus_draft(state: GraphState) -> dict:
        d = build_draft(rt.ctx.findings)
        rt.ctx.draft = d
        return {"draft": d, "audit": [audit("consensus_draft", "ok", decision=d.decision, score=d.score,
                                            conviction=d.conviction, conflicts=len(d.conflicts))]}

    def risk_review(state: GraphState) -> dict:
        f = RiskAgent().run(rt.ctx)
        rt.ctx.findings["risk"] = f
        rv = f.data.review.model_dump() if f.data else {}
        return {"findings": {"risk": f}, "audit": [audit("risk_review", f.status, review=rv)]}

    def consensus_final(state: GraphState) -> dict:
        f = ConsensusAgent().run(rt.ctx)
        rt.ctx.findings["consensus"] = f
        d = f.data
        return {"final": f, "findings": {"consensus": f},
                "audit": [audit("consensus_final", f.status, decision=d.decision if d else None,
                                conviction=d.conviction if d else None, timing=d.entry_timing if d else None,
                                narrative=d.narrative_source if d else None)]}

    def persist(state: GraphState) -> dict:
        llm_audit = [audit("llm", "call" if c.ok else "call_failed", purpose=c.purpose, model=c.model,
                           seconds=round(c.seconds, 2), request_id=c.request_id, input_tokens=c.input_tokens,
                           output_tokens=c.output_tokens, error=c.error) for c in rt.llm.calls]
        if rt.store is None:
            return {"analysis_id": None, "audit": llm_audit}
        aid = rt.store.save_analysis(state["ticker"], state.get("company", {}), rt.ctx.findings,
                                     state.get("audit", []) + llm_audit)
        return {"analysis_id": aid, "audit": llm_audit + [audit("persist", "ok", analysis_id=aid)]}

    g = StateGraph(GraphState)
    g.add_node("validate", validate)
    for a in RESEARCH_AGENTS:
        g.add_node(a.name, make_agent_node(a))
    g.add_node("consensus_draft", consensus_draft)
    g.add_node("risk_review", risk_review)
    g.add_node("consensus_final", consensus_final)
    g.add_node("persist", persist)
    g.add_edge(START, "validate")
    names = [a.name for a in RESEARCH_AGENTS]
    g.add_conditional_edges("validate", lambda s: names if s.get("valid") else END, names + [END])
    for a in RESEARCH_AGENTS:
        g.add_edge(a.name, "consensus_draft")
    g.add_edge("consensus_draft", "risk_review")
    g.add_edge("risk_review", "consensus_final")
    g.add_edge("consensus_final", "persist")
    g.add_edge("persist", END)
    return g.compile()


def stream_analysis(query: str, rt: Runtime) -> Iterator[tuple[str, dict]]:
    """Yield (node_name, update) as each node finishes; the final state is the merge of all updates."""
    graph = build_graph(rt)
    for chunk in graph.stream({"query": query, "findings": {}, "audit": []}, stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update or {}


def run_analysis(query: str, rt: Runtime) -> GraphState:
    graph = build_graph(rt)
    return graph.invoke({"query": query, "findings": {}, "audit": []})
