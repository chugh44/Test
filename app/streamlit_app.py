"""StockIntel dashboard. Run with:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
from stockintel import DISCLAIMER  # noqa: E402
from stockintel.analytics.indicators import add_all  # noqa: E402
from stockintel.config import get_settings  # noqa: E402
from stockintel.llm.client import LLMClient  # noqa: E402
from stockintel.orchestration.graph import RESEARCH_AGENTS, Runtime, stream_analysis  # noqa: E402
from stockintel.report.builder import build_html, build_markdown  # noqa: E402
from stockintel.scenarios import STRATEGIES, compute  # noqa: E402
from stockintel.storage.db import Store  # noqa: E402

st.set_page_config(page_title="StockIntel", page_icon="📈", layout="wide")
W = "stretch"
DECISION_COLOR = {"BUY": "green", "HOLD": "orange", "SELL": "red"}
NODE_LABEL = {"validate": "Master agent: validating ticker & loading shared context",
              "forecasting": "Agent 1: price forecasting (backtesting models)", "news": "Agent 2: news intelligence",
              "analyst": "Agent 3: analyst ratings", "fundamentals": "Agent 4: fundamentals",
              "technical": "Agent 5: technical signals", "macro": "Agent 6: market & sector", "flow": "Agent 7: institutional & options",
              "consensus_draft": "Agent 9: drafting consensus & detecting conflicts",
              "risk_review": "Agent 8: risk review of the draft thesis",
              "consensus_final": "Agent 9: final decision & thesis", "persist": "Saving analysis & audit trail"}


@st.cache_resource
def get_store(path: str) -> Store:
    return Store(path)


@st.cache_resource
def get_provider(kind: str):
    if kind == "demo":
        from stockintel.data.synthetic import SyntheticProvider
        return SyntheticProvider()
    from stockintel.data.yfinance_provider import YFinanceProvider
    return YFinanceProvider()


def fmt_big(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "Unavailable"
    for u, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= d:
            return f"{v / d:,.2f}{u}"
    return f"{v:,.0f}"


def pct(v, d=1, scale=100):
    return "Unavailable" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v * scale:+.{d}f}%"


def num(v, d=2):
    return "Unavailable" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:,.{d}f}"


def provenance(f):
    if f is None:
        return
    bits = []
    if f.data_as_of:
        bits.append(f"data as of **{f.data_as_of:%Y-%m-%d %H:%M} UTC**")
    bits.append(f"status `{f.status}` · signal `{f.signal:+.2f}` · confidence `{f.confidence:.2f}` · {f.duration_s or 0:.1f}s")
    st.caption(" · ".join(bits))
    for w in f.warnings:
        st.warning(w)
    with st.expander(f"Sources ({len(f.sources)}) and unavailable data ({len(f.unavailable)})"):
        for s in f.sources:
            label = f"[{s.name}]({s.url})" if s.url else s.name
            pub = f" · published {s.published_at:%Y-%m-%d %H:%M}" if s.published_at else ""
            st.markdown(f"- {label} — *{s.kind}*{' · delayed' if s.delayed else ''}{pub} · retrieved {s.retrieved_at:%H:%M:%S} UTC")
        for u in f.unavailable:
            st.markdown(f"- ⚠️ Unavailable: {u}")


# ------------------------------------------------------------------------------------------- sidebar
settings = get_settings()
with st.sidebar:
    st.title("📈 StockIntel")
    st.caption("Multi-agent equity research & trading decision support")
    query = st.text_input("Ticker or company name", value=st.session_state.get("query", "AAPL"))
    src = st.radio("Data source", ["Live (Yahoo Finance)", "Synthetic demo (offline)"], index=0,
                   help="The synthetic source generates fake data for testing without network access.")
    use_llm = st.toggle("Claude reasoning", value=True,
                        help="Uses Claude for news classification, web search and the thesis narrative. "
                             "Without it, rule-based fallbacks run and are labeled as such.")
    run = st.button("Run multi-agent analysis", type="primary", width="stretch")
    st.divider()
    st.subheader("Investment scenario")
    amount = st.number_input("Investment amount ($)", min_value=100.0, value=10_000.0, step=500.0)
    quick = st.pills("Quick amounts", ["$2,000", "$5,000", "$10,000"], selection_mode="single")
    if quick:
        amount = float(quick.replace("$", "").replace(",", ""))
    horizon = st.selectbox("Horizon", ["1M", "3M", "6M", "1Y"], index=1)
    appetite = st.segmented_control("Risk appetite", ["Conservative", "Moderate", "Aggressive"], default="Moderate")
    strategy = st.selectbox("Strategy", STRATEGIES, index=1)
    st.divider()
    llm_probe = LLMClient(settings, enabled=use_llm)
    st.caption(f"LLM: {'`' + settings.model + '`' if llm_probe.available else 'not configured (set ANTHROPIC_API_KEY)'}")
    st.caption(f"Optional keys: Finnhub {'✅' if settings.finnhub_key else '—'} · NewsAPI {'✅' if settings.newsapi_key else '—'}"
               f" · FRED {'✅' if settings.fred_key else '—'}")

# ------------------------------------------------------------------------------------------- run
if run and query.strip():
    st.session_state["query"] = query
    kind = "demo" if src.startswith("Synthetic") else "live"
    rt = Runtime(get_provider(kind), LLMClient(settings, enabled=use_llm), settings, get_store(settings.db_path))
    state: dict = {"findings": {}, "audit": []}
    t0 = time.time()
    with st.status(f"Running agents for **{query}**…", expanded=True) as status:
        pending = {a.name for a in RESEARCH_AGENTS}
        info = st.empty()
        for node, upd in stream_analysis(query, rt):
            for k, v in upd.items():
                if k == "findings":
                    state["findings"].update(v)
                elif k == "audit":
                    state["audit"] += v
                else:
                    state[k] = v
            f = (upd.get("findings") or {}).get(node if node not in ("risk_review", "consensus_final") else
                                                {"risk_review": "risk", "consensus_final": "consensus"}[node])
            icon = {"ok": "✅", "partial": "🟡", "unavailable": "⚪", "error": "❌"}.get(getattr(f, "status", "ok"), "✅")
            st.write(f"{icon} {NODE_LABEL.get(node, node)}" + (f" — {f.summary[:140]}" if f else ""))
            pending.discard(node)
            if node == "validate":
                if not upd.get("valid"):
                    status.update(label=f"Invalid ticker: {upd.get('error')}", state="error")
                    st.session_state.pop("state", None)
                    break
                info.info("Seven research agents now run in parallel. The forecasting agent backtests every model, "
                          "which takes the longest (~30–60 s).")
            elif node in [a.name for a in RESEARCH_AGENTS] and pending:
                info.info("Waiting for: " + ", ".join(sorted(pending)))
            else:
                info.empty()
        else:
            status.update(label=f"Analysis complete in {time.time() - t0:.0f}s", state="complete", expanded=False)
    if state.get("valid"):
        st.session_state["state"] = state

state = st.session_state.get("state")
if not state:
    st.title("StockIntel: multi-agent stock research")
    st.markdown(
        "Enter a ticker or company name and press **Run multi-agent analysis**. Nine specialised agents will "
        "retrieve and analyse data independently, then a master agent will reconcile their findings into one "
        "auditable recommendation.")
    st.info(DISCLAIMER)
    st.stop()

F = state["findings"]
co, q = state["company"], state["quote"]
cons = F.get("consensus")
C = cons.data if cons else None
fc, nw, an, fu, te, mac, fl, rk = (F.get(k) for k in ("forecasting", "news", "analyst", "fundamentals", "technical",
                                                       "macro", "flow", "risk"))
provider = get_provider("demo" if co.get("synthetic") else "live")
prices = provider.history(state["ticker"], f"{settings.history_years}y")

if co.get("synthetic"):
    st.error("DEMO MODE: every number on this page is synthetic and is not real market data.")

# ------------------------------------------------------------------------------------------- header
h1, h2 = st.columns([3, 2])
with h1:
    st.title(f"{co['name']} ({state['ticker']})")
    st.caption(f"{co.get('sector') or ''} · {co.get('industry') or ''} · {co.get('exchange') or ''} · data: {co['provider']}"
               f" · reasoning: {co.get('llm_model') or 'rule-based fallbacks'}")
with h2:
    if C:
        st.markdown(f"### :{DECISION_COLOR[C.decision]}[{C.decision}] · conviction {C.conviction}/100")
        st.markdown(f"**{C.entry_timing}**: {C.entry_rationale}")
m = st.columns(6)
m[0].metric(f"Price ({q.currency or 'USD'}, delayed)", f"{q.price:,.2f}", f"{q.change_pct:+.2f}%" if q.change_pct is not None else None)
m[1].metric("Market cap", fmt_big(q.market_cap))
m[2].metric("Volume", fmt_big(q.volume), f"avg {fmt_big(q.avg_volume)}", delta_color="off")
if fc and fc.data:
    h3 = next(h for h in fc.data.horizons if h.label == "3M")
    m[3].metric("AI 3M base case", f"{h3.base:,.2f}", f"{h3.expected_return_pct:+.1f}%")
if an and an.data and an.data.targets.get("median"):
    m[4].metric("Analyst median target", f"{an.data.targets['median']:,.2f}", f"{an.data.implied_upside_pct['median']:+.1f}%")
if rk and rk.data:
    m[5].metric("1-month 95% VaR", pct(rk.data.metrics.get("var95_1m")), delta_color="off")
st.caption(f"Quote as of {q.as_of:%Y-%m-%d %H:%M UTC} (delayed ~15 min) · last daily bar {prices.index[-1].date()}")

tabs = st.tabs(["Overview", "Price Forecast", "News", "Analyst Consensus", "Financial Health", "Technical Signals",
                "Institutional Activity", "Risk", "Final Recommendation", "Scenarios", "Agents & Audit", "Report", "History"])

# 1. Overview ---------------------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        if fc and fc.data:
            st.plotly_chart(charts.price_forecast(prices, fc.data, show_levels=False), width=W)
    with c2:
        try:
            prof = provider.profile(state["ticker"]) or {}
        except Exception:
            prof = {}
        st.markdown(f"**Company profile.** {(prof.get('longBusinessSummary') or 'Unavailable')[:900]}")
        st.markdown(f"Employees: {num(prof.get('fullTimeEmployees'), 0)} · Country: {prof.get('country') or 'n/a'} · "
                    f"52-week range: {num(prof.get('fiftyTwoWeekLow'))} – {num(prof.get('fiftyTwoWeekHigh'))}")
        if C:
            st.markdown("**Agent snapshot**")
            st.dataframe(pd.DataFrame([{"Agent": v.title, "Stance": v.stance, "Signal": v.signal, "Confidence": v.confidence}
                                       for v in C.votes]), hide_index=True, width=W)
    if mac and mac.data:
        st.markdown("**Market context:** " + " · ".join(mac.key_points[:3]))

# 2. Forecast ---------------------------------------------------------------------------------------
with tabs[1]:
    if fc and fc.data:
        d = fc.data
        st.markdown("Solid line: **actual history**. Dashed line and shaded bands: **StockIntel AI forecast** "
                    "(median, 80% and 95% intervals). Dots mark bull/base/bear at each horizon.")
        st.plotly_chart(charts.price_forecast(prices, d), width=W)
        st.dataframe(pd.DataFrame([{"Horizon": h.label, "Target date": h.target_date, "Bear (15%)": h.bear, "Base (median)": h.base,
                                    "Bull (85%)": h.bull, "80% low": h.lo80, "80% high": h.hi80, "95% low": h.lo95, "95% high": h.hi95,
                                    "Exp. return %": h.expected_return_pct, "Bull upside %": h.bull_upside_pct,
                                    "Bear downside %": h.bear_downside_pct, "P(up)": h.prob_up,
                                    "Model disagreement (σ of log-ret)": h.model_dispersion} for h in d.horizons]),
                     hide_index=True, width=W)
        st.plotly_chart(charts.model_points(d), width=W)
        st.subheader("Walk-forward backtest (out-of-sample)")
        st.caption(d.method)
        rows = []
        for mdl in d.models:
            for lbl, s in mdl.by_horizon.items():
                rows.append({"Model": mdl.name, "Family": mdl.family, "Horizon": lbl, "n": s.get("n"), "RMSE (log ret)": s.get("rmse"),
                             "MAPE %": s.get("mape"), "Hit rate": s.get("hit_rate"), "80% coverage": s.get("coverage80"),
                             "Skill vs naive": s.get("skill"), "Status": mdl.status})
        bt = pd.DataFrame(rows)
        st.dataframe(bt.pivot_table(index="Model", columns="Horizon", values="Skill vs naive")[list(d.horizons[i].label for i in range(5))],
                     width=W)
        with st.expander("Full backtest metrics"):
            st.dataframe(bt, hide_index=True, width=W)
        st.markdown("**Ensemble weights by horizon:** " + " · ".join(
            f"{h.label}: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(h.weights.items(), key=lambda x: -x[1])) for h in d.horizons))
        st.markdown("**Support / resistance:** " + ", ".join(f"{lv.kind} {lv.price:.2f} ({lv.touches} touches)" for lv in d.levels))
        if d.skipped_models:
            st.caption("Skipped models: " + ", ".join(f"{k} ({v})" for k, v in d.skipped_models.items()))
    provenance(fc)

# 3. News -------------------------------------------------------------------------------------------
with tabs[2]:
    if nw and nw.data:
        d = nw.data
        a, b, c = st.columns(3)
        a.metric("Weighted sentiment", f"{d.sentiment_score:+.2f}")
        b.metric("Positive / negative / neutral", f"{d.counts['positive']} / {d.counts['negative']} / {d.counts['neutral']}")
        c.metric("Classifier", "Claude" if d.classifier.startswith("Claude") else "Rule-based")
        st.markdown(nw.summary)
        for it in d.items:
            icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[it.sentiment]
            title = f"[{it.title}]({it.url})" if it.url else it.title
            tag = "" if it.status == "confirmed" else f" · **{it.status.upper()}**"
            st.markdown(f"{icon} {title}  \n<small>{(it.published_at or '')[:16].replace('T', ' ')} · {it.publisher or ''} "
                        f"(credibility {it.credibility}) · {it.category} · impact **{it.impact}**{tag}"
                        f"{f' · +{it.duplicates} duplicates merged' if it.duplicates else ''}</small>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Upcoming catalysts")
            for e in d.catalysts:
                st.markdown(f"- {e.text}" + (f" — **{e.expected_date}**" if e.expected_date else "")
                            + (f" ([source]({e.source_urls[0]}))" if e.source_urls else f" *({e.kind})*"))
        with c2:
            st.subheader("Risks from news")
            for e in d.risks:
                st.markdown(f"- {e.text}" + (f" ([source]({e.source_urls[0]}))" if e.source_urls else ""))
        if d.industry:
            st.subheader("Industry developments")
            for e in d.industry:
                st.markdown(f"- {e.text}" + (f" ([source]({e.source_urls[0]}))" if e.source_urls else ""))
    provenance(nw)

# 4. Analysts ---------------------------------------------------------------------------------------
with tabs[3]:
    st.info("These are Wall Street analyst estimates. They are separate from StockIntel's own AI forecasts.")
    if an and an.data:
        d = an.data
        cols = st.columns(5)
        for i, k in enumerate(("mean", "median", "high", "low")):
            cols[i].metric(f"{k.title()} target", num(d.targets.get(k)),
                           f"{d.implied_upside_pct[k]:+.1f}%" if d.implied_upside_pct.get(k) is not None else None)
        cols[4].metric("Analysts", d.n_analysts or "n/a",
                       f"score {d.recommendation_score:+.2f}" if d.recommendation_score is not None else None, delta_color="off")
        ai = next((h.base for h in fc.data.horizons if h.label == "1Y"), None) if fc and fc.data else None
        st.plotly_chart(charts.analyst_targets(d, ai), width=W)
        if C and C.forecast_vs_analyst:
            st.markdown(f"**AI vs analysts:** {C.forecast_vs_analyst.get('note')}")
        c1, c2 = st.columns(2)
        with c1:
            if d.recommendation_counts:
                st.bar_chart(pd.Series(d.recommendation_counts), horizontal=True)
            st.markdown("\n".join(f"- {k}" for k in an.key_points))
        with c2:
            if d.revision_history:
                st.plotly_chart(charts.revisions(d.revision_history), width=W)
        if d.changes:
            st.subheader("Recent rating and target changes")
            st.dataframe(pd.DataFrame([c.model_dump() for c in d.changes]), hide_index=True, width=W)
    provenance(an)

# 5. Financials -------------------------------------------------------------------------------------
with tabs[4]:
    if fu and fu.data:
        d, mt, v = fu.data, fu.data.metrics, fu.data.valuation
        st.markdown(f"**{d.valuation_verdict}.** Quality score {d.quality_score:+.2f} · valuation score {d.valuation_score:+.2f}")
        cols = st.columns(6)
        cols[0].metric("Revenue growth YoY", pct(mt.get("revenue_growth_yoy")))
        cols[1].metric("Operating margin", pct(mt.get("operating_margin"), scale=100).lstrip("+"))
        cols[2].metric("FCF", fmt_big(mt.get("free_cash_flow")))
        cols[3].metric("Net debt / EBITDA", num(mt.get("net_debt_to_ebitda"), 1))
        cols[4].metric("Forward P/E", num(v.get("forward_pe"), 1),
                       f"{d.premium_vs_peers['forward_pe']:+.0f}% vs peers" if d.premium_vs_peers.get("forward_pe") is not None else None,
                       delta_color="inverse")
        cols[5].metric("PEG", num(v.get("peg")))
        c1, c2 = st.columns(2)
        with c1:
            if d.annual:
                st.plotly_chart(charts.financials(d.annual, "Annual"), width=W)
        with c2:
            if d.quarterly:
                st.plotly_chart(charts.financials(d.quarterly, "Quarterly"), width=W)
        c1, c2 = st.columns(2)
        with c1:
            if d.peers:
                st.plotly_chart(charts.peer_valuation(d, state["ticker"]), width=W)
                st.dataframe(pd.DataFrame([p.model_dump() for p in d.peers]), hide_index=True, width=W)
            st.caption(f"Peer source: {d.peer_source}")
        with c2:
            for lbl, lst, fn in (("Strengths", d.strengths, st.success), ("Weaknesses", d.weaknesses, st.warning),
                                 ("Red flags", d.red_flags, st.error)):
                if lst:
                    fn(f"**{lbl}:** " + "; ".join(lst))
            st.dataframe(pd.DataFrame({"metric": list(mt) + list(v), "value": list(mt.values()) + list(v.values())}),
                         hide_index=True, width=W)
    provenance(fu)

# 6. Technicals -------------------------------------------------------------------------------------
with tabs[5]:
    if te and te.data:
        d = te.data
        cols = st.columns(4)
        cols[0].metric("Short-term signal", d.short_term_signal, f"{d.short_term_score:+.2f}", delta_color="off")
        cols[1].metric("Long-term signal", d.long_term_signal, f"{d.long_term_score:+.2f}", delta_color="off")
        cols[2].metric("Trend", d.trend)
        cols[3].metric("RSI 14", num(d.indicators.get("rsi14"), 0), d.momentum, delta_color="off")
        st.plotly_chart(charts.technical(prices, add_all(prices), levels=d.levels), width=W)
        if d.plan:
            p = d.plan
            cols = st.columns(5)
            cols[0].metric("Entry zone", f"{p.entry_low:,.2f}–{p.entry_high:,.2f}")
            cols[1].metric("Stop-loss", f"{p.stop_loss:,.2f}")
            cols[2].metric("Target 1", f"{p.target_1:,.2f}", f"R/R {p.risk_reward_t1}", delta_color="off")
            cols[3].metric("Target 2", f"{p.target_2:,.2f}", f"R/R {p.risk_reward_t2}", delta_color="off")
            cols[4].metric("ATR 14", num(d.indicators.get("atr14")))
            st.caption(p.rationale)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Patterns**")
            for pt in d.patterns:
                st.markdown(f"- {pt['name']} — *{pt['bias']}* ({pt['date']}): {pt['detail']}")
            st.markdown("**Trading setups**")
            for s in d.setups or ["No specific setup right now."]:
                st.markdown(f"- {s}")
        with c2:
            st.dataframe(pd.DataFrame({"indicator": list(d.indicators), "value": list(d.indicators.values())}), hide_index=True, width=W)
            st.bar_chart(pd.Series(d.components))
    provenance(te)

# 7. Institutional ----------------------------------------------------------------------------------
with tabs[6]:
    if fl and fl.data:
        d = fl.data
        for lbl in d.data_labels:
            st.caption("ℹ️ " + lbl)
        cols = st.columns(5)
        cols[0].metric("Institutions own", pct(d.ownership.get("institutions_pct"), 1).lstrip("+"))
        cols[1].metric("Insiders own", pct(d.ownership.get("insiders_pct"), 2).lstrip("+"))
        cols[2].metric("Put/call (volume)", num(d.options_summary.get("put_call_volume")))
        cols[3].metric("ATM IV (near)", pct(d.options_summary.get("atm_iv_near"), 0).lstrip("+"),
                       f"RV30 {pct(d.options_summary.get('realized_vol_30d'), 0).lstrip('+')}", delta_color="off")
        cols[4].metric("Max pain (near)", num(d.options_summary.get("max_pain_near")))
        st.markdown(f"**{d.accumulation_verdict}.** " + " ".join(fl.key_points))
        c1, c2 = st.columns(2)
        with c1:
            if d.oi_by_strike:
                st.plotly_chart(charts.options_oi(d.oi_by_strike, q.price), width=W)
            if d.unusual_activity:
                st.markdown("**Unusual options activity**")
                st.dataframe(pd.DataFrame(d.unusual_activity), hide_index=True, width=W)
        with c2:
            if d.iv_term_structure:
                st.plotly_chart(charts.iv_term(d.iv_term_structure), width=W)
            if d.top_institutions:
                st.markdown("**Top institutional holders (13F)**")
                st.dataframe(pd.DataFrame(d.top_institutions), hide_index=True, width=W)
        if d.insider_transactions:
            st.markdown(f"**Insider transactions (last {d.insider_summary.get('window_days')} days):** "
                        f"{d.insider_summary.get('buys')} buys (${fmt_big(d.insider_summary.get('buy_value'))}) · "
                        f"{d.insider_summary.get('sells')} sells (${fmt_big(d.insider_summary.get('sell_value'))})")
            st.dataframe(pd.DataFrame(d.insider_transactions), hide_index=True, width=W)
    provenance(fl)

# 8. Risk -------------------------------------------------------------------------------------------
with tabs[7]:
    if rk and rk.data:
        d, mt = rk.data, rk.data.metrics
        cols = st.columns(6)
        cols[0].metric("Volatility (1y)", pct(mt.get("annual_vol_1y"), 0).lstrip("+"))
        cols[1].metric("Max drawdown (1y)", pct(mt.get("max_drawdown_1y")))
        cols[2].metric("1M VaR / CVaR 95%", f"{pct(mt.get('var95_1m'))} / {pct(mt.get('cvar95_1m'))}")
        cols[3].metric("Beta", num(mt.get("beta")))
        cols[4].metric("Stop", num(d.stop_loss), f"-{d.stop_distance_pct:.1f}%" if d.stop_distance_pct else None, delta_color="off")
        cols[5].metric("Suitable for", d.suitability.title() + "+")
        st.markdown("**Risk review of the draft thesis:** " + " ".join(d.review.notes))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.drawdown(prices["Close"]), width=W)
            st.dataframe(pd.DataFrame(d.downside_scenarios), hide_index=True, width=W)
        with c2:
            st.plotly_chart(charts.return_dist(prices["Close"], mt.get("var95_1m")), width=W)
            st.dataframe(pd.DataFrame([s.model_dump() for s in d.position_sizing]), hide_index=True, width=W)
        st.markdown("**Risk/reward:** " + ", ".join(f"{k.replace('_', ' ')} **{v}**" for k, v in d.risk_reward.items() if v is not None))
        if d.event_risks:
            st.markdown("**Event risks**")
            st.dataframe(pd.DataFrame(d.event_risks), hide_index=True, width=W)
        st.markdown("**Conditions that would invalidate the thesis**\n" + "\n".join(f"- {x}" for x in d.invalidation))
    provenance(rk)

# 9. Final recommendation ---------------------------------------------------------------------------
with tabs[8]:
    if C:
        st.markdown(f"## :{DECISION_COLOR[C.decision]}[{C.decision}], conviction {C.conviction}/100")
        st.progress(C.conviction / 100)
        st.markdown(f"**{C.headline}**")
        st.markdown(C.thesis)
        cols = st.columns(4)
        cols[0].metric("Entry timing", C.entry_timing)
        cols[1].metric("Entry zone", f"{C.entry_zone[0]:,.2f}–{C.entry_zone[1]:,.2f}" if C.entry_zone else "n/a")
        cols[2].metric("Stop-loss", num(C.stop_loss))
        cols[3].metric("Targets", " / ".join(f"{t:,.2f}" for t in C.targets) or "n/a")
        st.caption(C.entry_rationale)
        if C.risk_adjustments:
            st.warning("Risk-agent adjustments: " + " ".join(C.risk_adjustments))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.votes(C.votes), width=W)
            st.markdown(f"Short-term: {C.short_term_view}  \nLong-term: {C.long_term_view}")
        with c2:
            st.markdown("**Bull points**\n" + "\n".join(f"- {x}" for x in C.bull_points))
            st.markdown("**Bear points**\n" + "\n".join(f"- {x}" for x in C.bear_points))
            st.markdown("**What would change our mind**\n" + "\n".join(f"- {x}" for x in C.what_changes_our_mind))
        st.subheader("Agreement and conflicts")
        for a in C.agreements:
            st.markdown(f"- ✅ {a}")
        for c in C.conflicts:
            st.markdown(f"- ⚔️ {c.description} **Resolution:** {c.resolution}")
        if not C.conflicts:
            st.markdown("- No strong conflicts between agents.")
        st.subheader("Key risks and catalysts")
        c1, c2 = st.columns(2)
        c1.markdown("\n".join(f"- {r}" for r in C.key_risks) or "—")
        c2.markdown("\n".join(f"- {x['text']}" + (f" ({x['expected_date']})" if x.get("expected_date") else "") for x in C.catalysts) or "—")
        st.dataframe(pd.DataFrame([v.model_dump() for v in C.votes]), hide_index=True, width=W)
        st.caption(f"Narrative: {C.narrative_source}. Method: {C.method}")
    st.info(DISCLAIMER)

# 10. Scenarios -------------------------------------------------------------------------------------
scen = None
with tabs[9]:
    try:
        scen = compute(F, amount, horizon, appetite or "Moderate", strategy)
        s = scen
        st.markdown(f"**${s.amount:,.0f}** · {s.horizon} · {s.risk_appetite} · {s.strategy} · entry reference {s.entry_price:,.2f} "
                    f"(≈{s.shares:,.2f} shares)")
        cols = st.columns(5)
        cols[0].metric("Expected value (median)", f"${s.expected_value:,.0f}", f"{s.expected_pnl:+,.0f}")
        cols[1].metric("Probability of profit", f"{s.prob_profit:.0%}" if s.prob_profit is not None else "n/a")
        cols[2].metric("Max modeled downside", f"${s.max_downside:,.0f}", f"{s.max_downside_pct:+.1f}%", delta_color="off")
        cols[3].metric("Risk/reward", s.risk_reward if s.risk_reward is not None else "n/a")
        cols[4].metric("Deploy now", f"{s.deploy_now_pct:.0f}%", "stage the rest" if s.deploy_now_pct < 100 else None,
                       delta_color="off")
        st.plotly_chart(charts.scenario_bars([vars(o) for o in s.outcomes], s.amount), width=W)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Suggested entry range:** {s.entry_range[0]:,.2f} – {s.entry_range[1]:,.2f}")
            st.markdown(f"**Stop / risk line:** {num(s.stop_loss)}")
            st.markdown("**Exit targets**\n" + "\n".join(f"- {t['label']}: {t['price']:,.2f}" for t in s.exit_targets))
        with c2:
            st.caption(f"Max downside basis: {s.max_downside_basis}")
            for n in s.notes:
                st.markdown(f"- {n}")
        st.dataframe(pd.DataFrame([vars(o) for o in s.outcomes]), hide_index=True, width=W)
    except ValueError as e:
        st.warning(str(e))

# 11. Agents & audit --------------------------------------------------------------------------------
with tabs[10]:
    st.markdown("Workflow: validate → 7 research agents in parallel → consensus draft → risk review → final decision → persist")
    st.dataframe(pd.DataFrame([{"agent": k, "status": f.status, "signal": f.signal, "confidence": f.confidence,
                                "seconds": f.duration_s, "sources": len(f.sources), "unavailable": len(f.unavailable),
                                "summary": f.summary} for k, f in F.items()]), hide_index=True, width=W)
    st.subheader("Audit trail")
    st.dataframe(pd.DataFrame([{"ts": a["ts"], "node": a["node"], "event": a["event"],
                                "detail": json.dumps(a["detail"], default=str)[:400]} for a in state["audit"]]),
                 hide_index=True, width=W)
    pick = st.selectbox("Inspect an agent's structured JSON output", list(F))
    st.json(json.loads(F[pick].model_dump_json()), expanded=False)

# 12. Report ----------------------------------------------------------------------------------------
with tabs[11]:
    md = build_markdown(state, scen.to_dict() if scen else None)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download Markdown", md, f"{state['ticker']}_report.md", "text/markdown")
    c2.download_button("Download HTML", build_html(md, f"{state['ticker']} research report"), f"{state['ticker']}_report.html",
                       "text/html")
    c3.download_button("Download agent JSON", json.dumps({k: json.loads(v.model_dump_json()) for k, v in F.items()}, indent=1),
                       f"{state['ticker']}_findings.json", "application/json")
    st.markdown(md, unsafe_allow_html=True)

# 13. History & accuracy ----------------------------------------------------------------------------
with tabs[12]:
    store = get_store(settings.db_path)
    if st.button("Score matured forecasts against realized prices"):
        n = store.score_matured_forecasts(provider)
        st.success(f"Scored {n} forecasts.")
    st.subheader("Forecast accuracy (realized)")
    acc = store.accuracy_summary(state["ticker"])
    if acc.empty:
        st.caption("No forecasts have matured yet. Accuracy appears here once target dates pass and you press the button above.")
    else:
        st.dataframe(acc, hide_index=True, width=W)
    st.subheader("Past analyses")
    st.dataframe(store.history(state["ticker"]), hide_index=True, width=W)
    st.subheader("Stored forecasts")
    st.dataframe(store.forecasts(state["ticker"]).head(50), hide_index=True, width=W)
