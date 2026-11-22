"""Investment research report (Markdown + standalone HTML) built only from agent findings."""

from __future__ import annotations

from datetime import datetime, timezone

from .. import DISCLAIMER


def _p(v, d=2, suffix=""):
    return "Unavailable" if v is None else f"{v:,.{d}f}{suffix}"


def _pct(v, d=1, scale=100):
    return "Unavailable" if v is None else f"{v * scale:+.{d}f}%"


def _big(v):
    if v is None:
        return "Unavailable"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.0f}"


def _src(f) -> str:
    if not f:
        return ""
    items = []
    for s in f.sources[:6]:
        label = f"[{s.name}]({s.url})" if s.url else s.name
        ts = f", published {s.published_at:%Y-%m-%d}" if s.published_at else ""
        items.append(f"{label} ({s.kind}{', delayed' if s.delayed else ''}{ts})")
    asof = f"Data as of {f.data_as_of:%Y-%m-%d %H:%M UTC}. " if f.data_as_of else ""
    un = f" Unavailable: {'; '.join(f.unavailable)}." if f.unavailable else ""
    return f"\n\n<sub>{asof}Sources: {'; '.join(items) or 'n/a'}.{un}</sub>\n"


def build_markdown(state: dict, scenario: dict | None = None) -> str:
    f = state.get("findings", {})
    co = state.get("company", {})
    cons = f.get("consensus")
    c = cons.data if cons else None
    t = state.get("ticker", "")
    q = state.get("quote")
    L = []
    L.append(f"# {co.get('name', t)} ({t}) — Investment Research Report")
    L.append(f"*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by the StockIntel multi-agent system · "
             f"data: {co.get('provider')} · reasoning: {co.get('llm_model') or 'rule-based (no LLM)'}*")
    if co.get("synthetic"):
        L.append("\n> **DEMO DATA. Every number in this report is synthetic and is not real market data.**")
    L.append(f"\n> {DISCLAIMER}\n")

    # 1. Executive summary
    L.append("## 1. Executive investment summary")
    if c:
        L.append(f"**Recommendation: {c.decision}** · Conviction **{c.conviction}/100** · Entry: **{c.entry_timing}**\n")
        L.append(f"**{c.headline}**\n\n{c.thesis}")
        L.append(f"\n*Entry timing:* {c.entry_rationale}")
        if c.risk_adjustments:
            L.append("\n*Risk-agent adjustments:* " + " ".join(c.risk_adjustments))
        L.append(f"\n<sub>Narrative written by {c.narrative_source}; the decision and score are computed "
                 f"deterministically ({c.method})</sub>")

    # 2. Price & market overview
    L.append("\n## 2. Current stock price and market overview")
    fc, mac = f.get("forecasting"), f.get("macro")
    prof_line = [f"Sector: {co.get('sector') or 'n/a'}", f"Industry: {co.get('industry') or 'n/a'}"]
    if q:
        L.append(f"Price **{q.price:,.2f} {q.currency or ''}** ({_p(q.change_pct, 2, '%')} on the day; delayed quote as of "
                 f"{q.as_of:%Y-%m-%d %H:%M UTC}). Market cap {_big(q.market_cap)}, volume {_big(q.volume)} "
                 f"(3-month avg {_big(q.avg_volume)}).")
    L.append(" · ".join(prof_line))
    if mac:
        L.append("\n" + "\n".join(f"- {k}" for k in mac.key_points) + _src(mac))

    # 3. Forecasts
    L.append("\n## 3. Price forecasts (StockIntel AI models)")
    if fc and fc.data:
        L.append("| Horizon | Target date | Bear | Base | Bull | 80% interval | Exp. return | P(up) |\n|---|---|---|---|---|---|---|---|")
        for h in fc.data.horizons:
            L.append(f"| {h.label} | {h.target_date} | {h.bear:,.2f} | **{h.base:,.2f}** | {h.bull:,.2f} | "
                     f"{h.lo80:,.2f}–{h.hi80:,.2f} | {h.expected_return_pct:+.1f}% | {h.prob_up:.0%} |")
        L.append(f"\n{fc.data.method}")
        L.append("\n**Backtest (out-of-sample skill vs naive random walk, 3M horizon):** " + ", ".join(
            f"{m.name} {m.by_horizon['3M'].get('skill') if m.by_horizon['3M'].get('skill') is not None else 'n/a'}"
            for m in fc.data.models))
        if fc.warnings:
            L.append("\n" + "\n".join(f"> {w}" for w in fc.warnings))
        L.append(_src(fc))

    # 4. News
    nw = f.get("news")
    L.append("\n## 4. Latest news and sentiment")
    if nw and nw.data:
        d = nw.data
        L.append(f"{nw.summary}\n\nClassifier: {d.classifier}. Weighted sentiment {d.sentiment_score:+.2f}.\n")
        L.append("| Date | Headline | Source | Sentiment | Impact | Status |\n|---|---|---|---|---|---|")
        for it in d.items[:12]:
            hl = f"[{it.title}]({it.url})" if it.url else it.title
            L.append(f"| {(it.published_at or '')[:10]} | {hl} | {it.publisher or ''} ({it.credibility}) | {it.sentiment} | "
                     f"{it.impact} | {it.status} |")
        L.append(_src(nw))
    else:
        L.append("News unavailable.")

    # 5. Analysts
    an = f.get("analyst")
    L.append("\n## 5. Analyst ratings and target revisions (Wall Street, not AI)")
    if an and an.data:
        d = an.data
        L.append(f"Targets: mean {_p(d.targets.get('mean'))}, median {_p(d.targets.get('median'))}, high "
                 f"{_p(d.targets.get('high'))}, low {_p(d.targets.get('low'))}. Implied upside to median: "
                 f"{_p(d.implied_upside_pct.get('median'), 1, '%')}.")
        L.append("\n" + "\n".join(f"- {k}" for k in an.key_points))
        if c and c.forecast_vs_analyst:
            L.append(f"\n**AI forecast vs analyst consensus:** model 1-year base case "
                     f"{c.forecast_vs_analyst.get('ai_1y_base_return_pct')}% vs analyst median "
                     f"{c.forecast_vs_analyst.get('analyst_median_implied_pct')}%. {c.forecast_vs_analyst.get('note')}")
        L.append(_src(an))

    # 6. Fundamentals
    fu = f.get("fundamentals")
    L.append("\n## 6. Financial and fundamental assessment")
    if fu and fu.data:
        d, m, v = fu.data, fu.data.metrics, fu.data.valuation
        L.append(f"{fu.summary}\n")
        L.append(f"- Revenue growth YoY {_pct(m.get('revenue_growth_yoy'))}; operating margin {_pct(m.get('operating_margin'))}; "
                 f"FCF {_big(m.get('free_cash_flow'))} (margin {_pct(m.get('fcf_margin'))})")
        L.append(f"- P/E {_p(v.get('trailing_pe'), 1)}, forward P/E {_p(v.get('forward_pe'), 1)}, PEG {_p(v.get('peg'), 2)}, "
                 f"EV/EBITDA {_p(v.get('ev_ebitda'), 1)}, P/S {_p(v.get('price_to_sales'), 1)}")
        L.append(f"- Peer median ({d.peer_source}): forward P/E {_p(d.peer_median.get('forward_pe'), 1)}, "
                 f"EV/EBITDA {_p(d.peer_median.get('ev_ebitda'), 1)}")
        for lbl, lst in (("Strengths", d.strengths), ("Weaknesses", d.weaknesses), ("Red flags", d.red_flags)):
            if lst:
                L.append(f"- **{lbl}:** " + "; ".join(lst))
        L.append(_src(fu))

    # 7. Technicals
    te = f.get("technical")
    L.append("\n## 7. Technical analysis and trading signals")
    if te and te.data:
        L.append(te.summary + "\n\n" + "\n".join(f"- {k}" for k in te.key_points))
        if te.data.setups:
            L.append("\n**Setups:** " + "; ".join(te.data.setups))
        L.append(_src(te))

    # 8. Institutional
    fl = f.get("flow")
    L.append("\n## 8. Institutional activity and options")
    if fl and fl.data:
        L.append(fl.summary + "\n\n" + "\n".join(f"- {k}" for k in fl.key_points))
        L.append("\n" + "\n".join(f"> {lbl}" for lbl in fl.data.data_labels))
        L.append(_src(fl))

    # 9. Scenarios
    L.append("\n## 9. Bull, base and bear scenarios")
    if fc and fc.data:
        for h in fc.data.horizons:
            if h.label in ("3M", "1Y"):
                L.append(f"- **{h.label}:** bull {h.bull:,.2f} ({h.bull_upside_pct:+.1f}%), base {h.base:,.2f} "
                         f"({h.expected_return_pct:+.1f}%), bear {h.bear:,.2f} ({h.bear_downside_pct:+.1f}%)")
    if scenario:
        s = scenario
        L.append(f"\n**Your scenario:** ${s['amount']:,.0f}, {s['horizon']}, {s['risk_appetite']}, {s['strategy']}")
        L.append("| Case | Price | Portfolio value | P/L |\n|---|---|---|---|")
        for o in s["outcomes"]:
            L.append(f"| {o['name']} | {o['price']:,.2f} | ${o['value']:,.2f} | {o['pnl']:+,.2f} ({o['pnl_pct']:+.1f}%) |")
        L.append(f"\nMax modeled downside: ${s['max_downside']:,.2f} ({s['max_downside_pct']:+.1f}%), based on {s['max_downside_basis']}.")
        for n in s["notes"]:
            L.append(f"- {n}")

    # 10. Risks & catalysts
    rk = f.get("risk")
    L.append("\n## 10. Key risks and upcoming catalysts")
    if c:
        if c.catalysts:
            L.append("**Catalysts:**\n" + "\n".join(
                f"- {x['text']}" + (f" ({x['expected_date']})" if x.get("expected_date") else "")
                + (f" [source]({x['source_urls'][0]})" if x.get("source_urls") else "") for x in c.catalysts))
        L.append("\n**Risks:**\n" + "\n".join(f"- {r}" for r in c.key_risks))
    if rk and rk.data:
        m = rk.data.metrics
        L.append(f"\nVolatility {_pct(m.get('annual_vol_1y'), 0)} (1y), max drawdown {_pct(m.get('max_drawdown_1y'))} (1y) / "
                 f"{_pct(m.get('max_drawdown_5y'))} (5y), 1-month 95% VaR {_pct(m.get('var95_1m'))}, beta {_p(m.get('beta'))}. "
                 f"Suitability: {rk.data.suitability} investors and above.")
        L.append("\n**Thesis invalidation:**\n" + "\n".join(f"- {x}" for x in rk.data.invalidation))

    # 11. Recommendation
    L.append("\n## 11. Final recommendation")
    if c:
        L.append(f"**{c.decision}**, conviction {c.conviction}/100, evidence score {c.score:+.3f}.\n")
        L.append("| Agent | Signal | Confidence | Freshness | Weight | Stance |\n|---|---|---|---|---|---|")
        for v in c.votes:
            L.append(f"| {v.title} | {v.signal:+.2f} | {v.confidence:.2f} | {v.freshness:.2f} | {v.effective_weight:.3f} | {v.stance} |")
        if c.conflicts:
            L.append("\n**Conflicts and resolution:**\n" + "\n".join(f"- {x.description} {x.resolution}" for x in c.conflicts))
        L.append("\n**Bull points:**\n" + "\n".join(f"- {x}" for x in c.bull_points))
        L.append("\n**Bear points:**\n" + "\n".join(f"- {x}" for x in c.bear_points))
        L.append("\n**What would change our mind:**\n" + "\n".join(f"- {x}" for x in c.what_changes_our_mind))

    # 12. Levels
    L.append("\n## 12. Suggested entry, exit and risk-management levels")
    if c:
        ez = c.entry_zone
        L.append(f"- Entry zone: {ez[0]:,.2f} – {ez[1]:,.2f}" if ez else "- Entry zone: unavailable")
        L.append(f"- Stop-loss: {_p(c.stop_loss)}")
        L.append(f"- Targets: {', '.join(f'{x:,.2f}' for x in c.targets) or 'unavailable'}")
    if rk and rk.data:
        L.append(f"- Risk/reward: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in rk.data.risk_reward.items() if v is not None))
        L.append("- Position sizing: " + "; ".join(f"{s.profile}: {s.position_pct_of_portfolio:.1f}% of portfolio"
                                                  + ("" if s.suitable else " (not suitable)") for s in rk.data.position_sizing))
    L.append(f"\n---\n<sub>{DISCLAIMER}</sub>")
    return "\n".join(L)


def build_html(md_text: str, title: str) -> str:
    try:
        import markdown

        body = markdown.markdown(md_text, extensions=["tables"])
    except Exception:  # pragma: no cover
        import html
        body = f"<pre>{html.escape(md_text)}</pre>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Inter,system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1b2430;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}}th,td{{border:1px solid #d6dbe3;padding:.35rem .5rem;text-align:left}}
th{{background:#f2f4f8}}blockquote{{border-left:4px solid #c9a227;margin:0;padding:.25rem .75rem;background:#fbf8ee}}
h1{{font-size:1.7rem}}h2{{border-bottom:1px solid #d6dbe3;padding-bottom:.2rem;margin-top:2rem}}sub{{color:#5b6573}}</style>
</head><body>{body}</body></html>"""
