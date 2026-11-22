"""End-to-end: LangGraph orchestration, persistence, scenarios and report."""

import pytest

from stockintel.data.synthetic import SyntheticProvider
from stockintel.llm.client import LLMClient
from stockintel.orchestration.graph import Runtime, run_analysis, stream_analysis
from stockintel.report.builder import build_html, build_markdown
from stockintel.scenarios import compute
from stockintel.storage.db import Store


@pytest.fixture(scope="module")
def result(tmp_path_factory, settings):
    store = Store(str(tmp_path_factory.mktemp("db") / "t.db"))
    rt = Runtime(SyntheticProvider(end="2025-06-30"), LLMClient(settings, enabled=False), settings, store)
    nodes = []
    state = {"findings": {}, "audit": []}
    for node, upd in stream_analysis("nvidia", rt):  # company name -> ticker resolution
        nodes.append(node)
        for k, v in upd.items():
            if k == "findings":
                state["findings"].update(v)
            elif k == "audit":
                state["audit"] += v
            else:
                state[k] = v
    return state, nodes, store


def test_graph_runs_all_agents(result):
    state, nodes, _ = result
    assert state["ticker"] == "NVDA"
    assert nodes[0] == "validate" and nodes[-1] == "persist"
    assert set(nodes[1:8]) == {"forecasting", "news", "analyst", "fundamentals", "technical", "macro", "flow"}
    assert nodes[8:11] == ["consensus_draft", "risk_review", "consensus_final"]
    assert set(state["findings"]) == {"forecasting", "news", "analyst", "fundamentals", "technical", "macro", "flow",
                                      "risk", "consensus"}


def test_audit_trail_and_persistence(result):
    state, _, store = result
    assert state["analysis_id"]
    events = {a["node"] for a in state["audit"]}
    assert {"validate", "consensus_draft", "risk_review", "consensus_final"} <= events
    hist = store.history("NVDA")
    assert len(hist) == 1 and hist.iloc[0]["decision"] in ("BUY", "HOLD", "SELL")
    assert len(store.audit_trail(int(state["analysis_id"]))) >= 10
    assert len(store.forecasts("NVDA")) == 5


def test_forecast_accuracy_tracking(result):
    _, _, store = result
    # The run used data ending 2025-06-30; score against a provider whose data runs to the present.
    n = store.score_matured_forecasts(SyntheticProvider())
    assert n >= 4
    acc = store.accuracy_summary("NVDA")
    assert not acc.empty and acc["mape_pct"].notna().all()


def test_invalid_ticker(settings):
    rt = Runtime(SyntheticProvider(), LLMClient(settings, enabled=False), settings, None)
    s = run_analysis("ZZZINVALID", rt)
    assert s["valid"] is False and "No tradable security" in s["error"]
    assert "findings" not in s or not s["findings"]


@pytest.mark.parametrize("strategy", ["Short-term trading", "Swing trading", "Long-term investing"])
@pytest.mark.parametrize("appetite", ["Conservative", "Aggressive"])
def test_scenarios(result, strategy, appetite):
    state, _, _ = result
    s = compute(state["findings"], 5000, "3M", appetite, strategy)
    bull, base, bear = (o.value for o in s.outcomes)
    assert bull >= base >= bear
    assert s.max_downside <= 0
    assert s.entry_range[0] <= s.entry_range[1]
    if strategy != "Long-term investing":
        assert bear >= s.shares * s.stop_loss - 0.05  # stop caps the bear case (values are rounded to cents)


def test_report_has_all_sections(result):
    state, _, _ = result
    md = build_markdown(state, compute(state["findings"], 10000, "1Y", "Moderate", "Long-term investing").to_dict())
    for i in range(1, 13):
        assert f"## {i}." in md
    assert "DEMO DATA" in md and "not financial advice" in md
    assert "<table>" in build_html(md, "t")
