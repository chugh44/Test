"""Validate each agent on its own before testing the integrated graph."""

import pandas as pd
import pytest

from stockintel.agents.analyst.agent import AnalystAgent
from stockintel.agents.base import SYNTHETIC_WARNING
from stockintel.agents.consensus.agent import ConsensusAgent, build_draft
from stockintel.agents.flow.agent import FlowAgent
from stockintel.agents.forecasting.agent import ForecastingAgent
from stockintel.agents.fundamentals.agent import FundamentalsAgent
from stockintel.agents.macro.agent import MacroAgent
from stockintel.agents.news.agent import NewsAgent, dedupe
from stockintel.agents.news.schema import LLMItemClass, LLMNewsAnalysis
from stockintel.agents.risk.agent import RiskAgent
from stockintel.agents.technical.agent import TechnicalAgent
from stockintel.data.base import NewsItem


def _check(f, name):
    assert f.agent == name
    assert f.status in ("ok", "partial"), f.summary
    assert -1 <= f.signal <= 1 and 0 <= f.confidence <= 1
    assert f.sources, "every finding must cite sources"
    assert SYNTHETIC_WARNING in f.warnings
    f.model_dump_json()  # must serialize to structured JSON


@pytest.mark.parametrize("agent", [NewsAgent(), AnalystAgent(), FundamentalsAgent(), TechnicalAgent(), MacroAgent(), FlowAgent()])
def test_research_agents(ctx, agent):
    f = agent.run(ctx)
    _check(f, agent.name)
    assert f.data is not None


def test_forecasting_agent(ctx):
    f = ForecastingAgent().run(ctx)
    _check(f, "forecasting")
    d = f.data
    assert [h.label for h in d.horizons] == ["1W", "1M", "3M", "6M", "1Y"]
    for h in d.horizons:
        assert h.lo95 <= h.lo80 <= h.bear <= h.base <= h.bull <= h.hi80 <= h.hi95
        assert abs(sum(h.weights.values()) - 1) < 0.01  # weights are rounded to 3 dp
    assert any(m.name == "Random walk (naive)" for m in d.models)
    assert "LSTM" in d.skipped_models and "Prophet" in d.skipped_models  # disabled in test settings


def test_technical_plan_is_coherent(ctx):
    p = TechnicalAgent().run(ctx).data.plan
    assert p.stop_loss < p.entry_low <= p.entry_high < p.target_1 <= p.target_2


def test_analyst_never_fabricates(ctx, monkeypatch):
    monkeypatch.setattr(ctx.provider, "analyst", lambda s: {})
    f = AnalystAgent().run(ctx)
    assert f.status == "unavailable"
    assert all(v is None for v in f.data.targets.values())
    assert "consensus price targets" in f.unavailable


def test_news_dedupe():
    items = [NewsItem("a", "Acme beats quarterly earnings estimates, raises guidance", "Reuters", "u1", None),
             NewsItem("b", "Acme beats quarterly earnings estimates and raises guidance", "Blog", "u2", None),
             NewsItem("c", "Acme faces antitrust probe", "WSJ", "u3", None)]
    kept, dups = dedupe(items)
    assert [k.id for k in kept] == ["a", "c"] and dups == {"a": 1}


def test_news_llm_output_cannot_reference_unknown_items(ctx, monkeypatch):
    fake = LLMNewsAnalysis(items=[LLMItemClass(item_id="made-up", category="m&a", sentiment="positive", impact="high",
                                               status="confirmed", summary="invented")],
                           catalysts=[], risks=[], industry_developments=[], overall_view="x")
    monkeypatch.setattr(ctx.llm, "structured", lambda **kw: fake)
    f = NewsAgent().run(ctx)
    assert all(i.id != "made-up" for i in f.data.items)
    assert all(i.summary != "invented" for i in f.data.items)


def test_consensus_and_risk_review(ctx):
    for a in (ForecastingAgent(), NewsAgent(), AnalystAgent(), FundamentalsAgent(), TechnicalAgent(), MacroAgent(), FlowAgent()):
        ctx.findings[a.name] = a.run(ctx)
    draft = build_draft(ctx.findings)
    assert draft.decision in ("BUY", "HOLD", "SELL") and 0 <= draft.conviction <= 100
    assert len(draft.votes) == 7
    ctx.draft = draft
    ctx.findings["risk"] = r = RiskAgent().run(ctx)
    _check(r, "risk")
    assert r.data.review.notes
    final = ConsensusAgent().run(ctx)
    d = final.data
    assert d.decision == draft.decision  # the LLM/narrative never changes the decision
    if r.data.review.conviction_cap is not None:
        assert d.conviction <= r.data.review.conviction_cap
    if d.decision == "SELL" or r.data.review.veto:
        assert d.entry_timing == "AVOID"
    assert d.narrative_source.startswith("template")


def test_conflict_detection():
    from stockintel.schemas.common import AgentFinding
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    fs = {n: AgentFinding(agent=n, ticker="X", signal=s, confidence=0.7, data_as_of=now)
          for n, s in [("forecasting", 0.6), ("technical", -0.6), ("news", 0.1), ("analyst", 0.2),
                       ("fundamentals", 0.1), ("macro", 0.0), ("flow", 0.0)]}
    d = build_draft(fs)
    assert any(set(c.agents) == {"forecasting", "technical"} for c in d.conflicts)
