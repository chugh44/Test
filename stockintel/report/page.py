"""Builds the claude.ai Artifact page for a StockIntel report straight from agent findings.

Generating the page in code means every number on it is exactly what the agents computed; nothing is
retyped. The output follows the Artifact page contract: no doctype/html/head/body tags, its own <title>
and <style>, theme tokens for light and dark, and a 16px+ safe-area side gutter.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone

import pandas as pd

from .. import DISCLAIMER

E = html.escape

CSS = """
/* Layout: a research-terminal report. Decision band first, then levels, forecast, agents, evidence, risk, sources. */
:root{
  --bg:#f5f7fa; --surface:#ffffff; --ink:#132033; --muted:#5a6a80; --line:#dbe2ec;
  --accent:#1d5c8f; --fc:#c9772b; --fc-band:rgba(201,119,43,.16); --fc-band2:rgba(201,119,43,.08);
  --buy:#1f7a4d; --hold:#a06a00; --sell:#b23a3a; --buy-bg:#e4f3ea; --hold-bg:#fbf0d9; --sell-bg:#f8e3e3;
  --warn-bg:#fff4d6; --warn-ink:#6b4b00;
  --display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e141c; --surface:#151e29; --ink:#e4ebf3; --muted:#93a3b8; --line:#26344a; --accent:#6aaee6;
  --fc:#e39a52; --fc-band:rgba(227,154,82,.20); --fc-band2:rgba(227,154,82,.09);
  --buy:#5cc48c; --hold:#e0b044; --sell:#ec7b7b; --buy-bg:#16301f; --hold-bg:#33290f; --sell-bg:#3a1a1a;
  --warn-bg:#3a2f0e; --warn-ink:#f3d58a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e141c; --surface:#151e29; --ink:#e4ebf3; --muted:#93a3b8; --line:#26344a; --accent:#6aaee6;
  --fc:#e39a52; --fc-band:rgba(227,154,82,.20); --fc-band2:rgba(227,154,82,.09);
  --buy:#5cc48c; --hold:#e0b044; --sell:#ec7b7b; --buy-bg:#16301f; --hold-bg:#33290f; --sell-bg:#3a1a1a;
  --warn-bg:#3a2f0e; --warn-ink:#f3d58a; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:24px 48px}
.wrap{max-width:1080px;margin:0 auto;display:grid;gap:22px}
h1,h2,h3{font-family:var(--display);font-weight:600;margin:0;text-wrap:balance;letter-spacing:.005em}
h1{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.1}
h2{font-size:1.25rem;margin-bottom:10px}
h3{font-size:1rem;margin-bottom:6px}
.num,td.n,.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.muted{color:var(--muted)} .small{font-size:.85rem}
.label{font:600 .72rem/1.2 var(--body);text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
header .meta{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:8px;color:var(--muted);font-size:.9rem}
.px{font:600 1.6rem var(--mono);color:var(--ink)}
.chg.up{color:var(--buy)} .chg.down{color:var(--sell)}
.banner{background:var(--warn-bg);color:var(--warn-ink);padding:10px 14px;border-radius:6px;font-weight:600}
.decision{display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:center;padding:18px 20px;border-radius:10px;border:1px solid var(--line);background:var(--surface)}
.decision.BUY{border-left:6px solid var(--buy)} .decision.HOLD{border-left:6px solid var(--hold)} .decision.SELL{border-left:6px solid var(--sell)}
.pill{font:700 1.9rem var(--display);padding:6px 16px;border-radius:8px;letter-spacing:.04em}
.pill.BUY{background:var(--buy-bg);color:var(--buy)} .pill.HOLD{background:var(--hold-bg);color:var(--hold)} .pill.SELL{background:var(--sell-bg);color:var(--sell)}
.meter{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:6px 0 4px;max-width:320px}
.meter>span{display:block;height:100%;background:var(--accent)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(110px,1fr))}
.stat{padding:10px 0;border-top:2px solid var(--line);min-width:0}
.stat .v{font:600 1.15rem var(--mono);font-variant-numeric:tabular-nums;margin-top:2px}
section{min-width:0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;min-width:0}
.two{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font:600 .72rem var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
td.n,th.n{text-align:right;white-space:nowrap}
td.mono{white-space:nowrap}
tr.hl td{background:var(--fc-band2)}
ul{margin:0;padding-left:1.1em} li{margin:3px 0}
.votes .row{display:grid;grid-template-columns:minmax(120px,170px) 1fr 64px;gap:10px;align-items:center;font-size:.88rem;padding:3px 0}
.bar{position:relative;height:14px;background:transparent;border-left:1px solid var(--line);border-right:1px solid var(--line)}
.bar::before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;border-left:1px dashed var(--muted);opacity:.6}
.bar>span{position:absolute;top:2px;bottom:2px;border-radius:2px}
.pos{background:var(--buy)} .neg{background:var(--sell)} .neu{background:var(--muted)}
.tag{display:inline-block;font:600 .7rem var(--body);padding:1px 7px;border-radius:999px;border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.tag.positive{color:var(--buy);border-color:currentColor} .tag.negative{color:var(--sell);border-color:currentColor}
.tag.speculative,.tag.opinion{color:var(--hold);border-color:currentColor}
a{color:var(--accent)} a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
svg text{fill:var(--muted);font:11px var(--mono)}
.sources li{font-size:.82rem;color:var(--muted)}
footer{font-size:.8rem;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
@media (max-width:560px){.decision{grid-template-columns:1fr}.votes .row{grid-template-columns:100px 1fr 52px}}
"""


def _n(v, d=2, pre="", suf=""):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return '<span class="muted">Unavailable</span>'
    return f"{pre}{v:,.{d}f}{suf}"


def _pct(v, d=1, scale=100.0, sign=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '<span class="muted">Unavailable</span>'
    return f"{v * scale:{'+' if sign else ''}.{d}f}%"


def _big(v):
    if v is None:
        return '<span class="muted">Unavailable</span>'
    for u, d in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= d:
            return f"{v / d:,.2f}{u}"
    return f"{v:,.0f}"


def _forecast_svg(prices: pd.Series | None, fc) -> str:
    """Fan chart: ~6 months of actual closes (solid) then the AI forecast median and 80/95% bands (dashed/shaded)."""
    if prices is None or fc is None or not fc.path:
        return ""
    hist = prices.tail(126)
    path = fc.path
    W, H, L, R, T, B = 760, 300, 52, 16, 14, 28
    n_hist, n_fc = len(hist), path[-1]["day"]
    total = n_hist + n_fc
    lo = min(float(hist.min()), min(p["lo95"] for p in path))
    hi = max(float(hist.max()), max(p["hi95"] for p in path))
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    x = lambda i: L + (W - L - R) * i / max(total - 1, 1)  # noqa: E731
    y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    o = n_hist - 1

    def band(a, b, cls):
        top = " ".join(f"{x(o + p['day']):.1f},{y(p[a]):.1f}" for p in path)
        bot = " ".join(f"{x(o + p['day']):.1f},{y(p[b]):.1f}" for p in reversed(path))
        return f'<polygon points="{top} {bot}" style="fill:var({cls})"/>'

    hist_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hist.to_numpy()))
    med = " ".join(f"{x(o + p['day']):.1f},{y(p['base']):.1f}" for p in path)
    ticks = []
    step = (hi - lo) / 4
    for k in range(5):
        v = lo + k * step
        ticks.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" style="stroke:var(--line)"/>'
                     f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    marks = []
    for h in fc.horizons:
        cx = x(o + h.days)
        marks.append(f'<circle cx="{cx:.1f}" cy="{y(h.base):.1f}" r="3.5" style="fill:var(--fc)"/>'
                     f'<text x="{cx:.1f}" y="{H - 8}" text-anchor="middle">{h.label}</text>')
    d0 = str(hist.index[0].date())
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Price history and AI forecast fan chart">'
            + "".join(ticks)
            + band("hi95", "lo95", "--fc-band2") + band("hi80", "lo80", "--fc-band")
            + f'<polyline points="{hist_pts}" style="fill:none;stroke:var(--accent);stroke-width:2"/>'
            + f'<polyline points="{med}" style="fill:none;stroke:var(--fc);stroke-width:2;stroke-dasharray:6 4"/>'
            + f'<line x1="{x(o):.1f}" x2="{x(o):.1f}" y1="{T}" y2="{H - B}" style="stroke:var(--muted);stroke-dasharray:2 3"/>'
            + f'<text x="{L}" y="{H - 8}">{d0}</text><text x="{x(o):.1f}" y="{T + 10}" text-anchor="middle">last close</text>'
            + "".join(marks) + "</svg>"
            + '<p class="small muted">Solid blue: actual closes. Dashed orange: AI forecast median. '
              'Shaded: 80% (darker) and 95% intervals.</p>')


def build_page(state: dict, scenario: dict | None = None, prices: pd.Series | None = None) -> str:
    f = state.get("findings", {})
    co, q = state.get("company", {}), state.get("quote")
    t = state.get("ticker", "")
    cons = f.get("consensus")
    C = cons.data if cons else None
    fc = f["forecasting"].data if f.get("forecasting") else None
    nw = f["news"].data if f.get("news") else None
    an = f["analyst"].data if f.get("analyst") else None
    fu = f["fundamentals"].data if f.get("fundamentals") else None
    te = f["technical"].data if f.get("technical") else None
    mac = f.get("macro")
    fl = f["flow"].data if f.get("flow") else None
    rk = f["risk"].data if f.get("risk") else None
    out: list[str] = []
    a = out.append
    a(f'<title>{E(t)} Research Report</title>\n<meta name="claude:viewport-fit-sides" content="cover">')
    a('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;600&display=swap">')
    a(f"<style>{CSS}</style>")
    a('<div class="wrap">')

    # Header
    chg = q.change_pct if q else None
    a("<header>")
    a(f'<div class="label">StockIntel multi-agent research · {E(co.get("sector") or "")} · {E(co.get("industry") or "")}</div>')
    a(f"<h1>{E(co.get('name', t))} ({E(t)})</h1>")
    if q:
        cls = "up" if (chg or 0) >= 0 else "down"
        a(f'<div class="meta"><span class="px">{q.price:,.2f} {E(q.currency or "")}</span>'
          f'<span class="chg {cls} num">{_n(chg, 2, suf="%", pre="+" if (chg or 0) >= 0 else "")} on the day</span>'
          f"<span>Market cap {_big(q.market_cap)}</span><span>Volume {_big(q.volume)} (3-mo avg {_big(q.avg_volume)})</span>"
          f"<span>Delayed quote, {q.as_of:%Y-%m-%d %H:%M} UTC</span></div>")
    a(f'<div class="meta small"><span>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</span>'
      f'<span>Data: {E(co.get("provider", ""))}</span><span>Reasoning: {E(co.get("llm_model") or "rule-based (no LLM configured)")}</span></div>')
    a("</header>")
    if co.get("synthetic"):
        a('<div class="banner">DEMO DATA: every number on this page is synthetic and is not real market data.</div>')

    # Decision
    if C:
        a(f'<section class="decision {C.decision}"><div class="pill {C.decision}">{C.decision}</div><div>')
        a(f'<div class="label">Conviction</div><div class="num" style="font-size:1.3rem;font-weight:600">{C.conviction}/100</div>'
          f'<div class="meter"><span style="width:{C.conviction}%"></span></div>')
        a(f"<p style='margin:.4rem 0 0'><strong>{E(C.entry_timing)}.</strong> {E(C.entry_rationale)}</p>")
        if C.risk_adjustments:
            a('<ul class="small muted" style="margin-top:6px">' + "".join(f"<li>{E(x)}</li>" for x in C.risk_adjustments) + "</ul>")
        a("</div></section>")
        a('<section class="card"><h2>Thesis</h2>')
        a(f"<p><strong>{E(C.headline)}</strong></p>" + "".join(f"<p>{E(p)}</p>" for p in C.thesis.split("\n\n") if p.strip()))
        a(f'<p class="small muted">Narrative: {E(C.narrative_source)}. The decision and conviction are computed in code from the agents\' signals; the narrative only explains them.</p></section>')

        # Levels
        a('<section><div class="grid">')
        ez = C.entry_zone
        for lbl, v in (("Entry zone", f"{ez[0]:,.2f}–{ez[1]:,.2f}" if ez else None),
                       ("Stop-loss", _n(C.stop_loss)), ("Target 1", _n(C.targets[0]) if C.targets else None),
                       ("Target 2", _n(C.targets[1]) if len(C.targets) > 1 else None),
                       ("Evidence score", f"{C.score:+.3f}")):
            a(f'<div class="stat"><div class="label">{lbl}</div><div class="v">{v or _n(None)}</div></div>')
        a("</div></section>")

    # Forecast
    if fc:
        a('<section class="card"><h2>AI price forecast</h2>')
        a(_forecast_svg(prices, fc))
        a('<div class="scroll"><table><tr><th>Horizon</th><th>Target date</th><th class="n">Bear (15%)</th><th class="n">Base</th>'
          '<th class="n">Bull (85%)</th><th class="n">80% interval</th><th class="n">Exp. return</th><th class="n">P(up)</th></tr>')
        for h in fc.horizons:
            a(f'<tr><td>{h.label}</td><td>{h.target_date}</td><td class="n">{h.bear:,.2f}</td><td class="n"><strong>{h.base:,.2f}</strong></td>'
              f'<td class="n">{h.bull:,.2f}</td><td class="n">{h.lo80:,.2f}–{h.hi80:,.2f}</td>'
              f'<td class="n">{h.expected_return_pct:+.1f}%</td><td class="n">{h.prob_up:.0%}</td></tr>')
        a("</table></div>")
        skill = ", ".join(f"{E(m.name)} {m.by_horizon['3M'].get('skill'):+.2f}" for m in fc.models
                          if m.by_horizon.get("3M", {}).get("skill") is not None)
        a(f'<p class="small muted">Out-of-sample skill vs a naive random walk at 3 months (positive beats the baseline): {skill}. '
          f"{E(fc.method)}</p></section>")

    # Agents
    if C:
        a('<section class="two"><div class="card votes"><h2>Agent votes</h2>')
        for v in C.votes:
            w = abs(v.signal) * 50
            left = 50 if v.signal >= 0 else 50 - w
            cls = "pos" if v.signal > 0.15 else "neg" if v.signal < -0.15 else "neu"
            a(f'<div class="row"><span>{E(v.title)}</span><div class="bar"><span class="{cls}" style="left:{left:.1f}%;width:{max(w, 1):.1f}%;opacity:{0.45 + 0.55 * min(1, v.effective_weight / 0.1):.2f}"></span></div>'
              f'<span class="num small">{v.signal:+.2f}</span></div>')
        a('<p class="small muted">Bar length is the agent\'s signal (bearish left, bullish right). Fainter bars carry less weight '
          "after confidence and data freshness.</p></div>")
        a('<div class="card"><h2>Agreement and conflicts</h2><ul>')
        for x in C.agreements:
            a(f"<li>{E(x)}</li>")
        for c in C.conflicts:
            a(f"<li><strong>Conflict:</strong> {E(c.description)} <span class='muted'>{E(c.resolution)}</span></li>")
        if not C.conflicts:
            a("<li>No strong conflicts between agents.</li>")
        a("</ul></div></section>")

    # Analyst vs AI + fundamentals
    a('<section class="two">')
    if an:
        a('<div class="card"><h2>Wall Street analysts</h2><p class="small muted">Analyst estimates, not StockIntel\'s AI forecasts.</p>')
        a('<div class="grid">')
        for k in ("median", "mean", "low", "high"):
            up = an.implied_upside_pct.get(k)
            a(f'<div class="stat"><div class="label">{k} target</div><div class="v">{_n(an.targets.get(k))}</div>'
              f'<div class="small muted">{_n(up, 1, suf="%", pre="+" if (up or 0) >= 0 else "")}</div></div>')
        a("</div><ul class='small' style='margin-top:8px'>" + "".join(f"<li>{E(k)}</li>" for k in f["analyst"].key_points[1:6]) + "</ul>")
        if C and C.forecast_vs_analyst:
            a(f"<p class='small'><strong>AI vs analysts:</strong> {E(str(C.forecast_vs_analyst.get('note', '')))}</p>")
        a("</div>")
    if fu:
        m, v = fu.metrics, fu.valuation
        a('<div class="card"><h2>Fundamentals</h2>')
        a(f"<p class='small'>{E(fu.valuation_verdict)}. Quality {fu.quality_score:+.2f}, valuation {fu.valuation_score:+.2f}.</p>")
        rows = [("Revenue growth YoY", _pct(m.get("revenue_growth_yoy"))), ("Operating margin", _pct(m.get("operating_margin"), sign=False)),
                ("Free cash flow", _big(m.get("free_cash_flow"))), ("Net debt / EBITDA", _n(m.get("net_debt_to_ebitda"), 1)),
                ("P/E · forward P/E", f"{_n(v.get('trailing_pe'), 1)} · {_n(v.get('forward_pe'), 1)}"),
                ("PEG · EV/EBITDA", f"{_n(v.get('peg'), 2)} · {_n(v.get('ev_ebitda'), 1)}"),
                ("Peer median fwd P/E", _n(fu.peer_median.get("forward_pe"), 1))]
        a("<table>" + "".join(f"<tr><td>{k}</td><td class='n'>{val}</td></tr>" for k, val in rows) + "</table>")
        for lbl, lst in (("Strengths", fu.strengths), ("Weaknesses", fu.weaknesses), ("Red flags", fu.red_flags)):
            if lst:
                a(f"<p class='small'><strong>{lbl}:</strong> {E('; '.join(lst))}</p>")
        a("</div>")
    a("</section>")

    # Technical + market
    a('<section class="two">')
    if te:
        a('<div class="card"><h2>Technical signals</h2>')
        a(f"<p class='small'>{E(f['technical'].summary)}</p><ul class='small'>"
          + "".join(f"<li>{E(k)}</li>" for k in f["technical"].key_points) + "</ul>")
        if te.setups:
            a("<p class='small'><strong>Setups:</strong> " + E("; ".join(te.setups)) + "</p>")
        a("</div>")
    if mac:
        a(f'<div class="card"><h2>Market and sector</h2><p class="small">{E(mac.summary)}</p><ul class="small">'
          + "".join(f"<li>{E(k)}</li>" for k in mac.key_points) + "</ul></div>")
    a("</section>")

    # News
    if nw and nw.items:
        a('<section class="card"><h2>News</h2>')
        a(f"<p class='small muted'>{E(f['news'].summary)} Classifier: {E(nw.classifier)}.</p>")
        a('<div class="scroll"><table><tr><th>Date</th><th>Headline</th><th>Source</th><th>Read</th></tr>')
        for it in nw.items[:10]:
            hl = f'<a href="{E(it.url)}" target="_blank" rel="noopener">{E(it.title)}</a>' if it.url else E(it.title)
            tags = f'<span class="tag {it.sentiment}">{it.sentiment}</span> <span class="tag">{it.impact}</span>' \
                   + ("" if it.status == "confirmed" else f' <span class="tag {it.status}">{it.status}</span>')
            a(f"<tr><td class='mono small'>{E((it.published_at or '')[:10])}</td><td>{hl}</td>"
              f"<td class='small'>{E(it.publisher or '')}</td><td>{tags}</td></tr>")
        a("</table></div></section>")

    # Institutional
    if fl:
        a(f'<section class="card"><h2>Institutional and options activity</h2><p class="small">{E(f["flow"].summary)}</p><ul class="small">'
          + "".join(f"<li>{E(k)}</li>" for k in f["flow"].key_points) + "</ul>"
          + "".join(f"<p class='small muted'>{E(lbl)}</p>" for lbl in fl.data_labels) + "</section>")

    # Scenario + risk
    a('<section class="two">')
    if scenario:
        s = scenario
        a(f'<div class="card"><h2>Your scenario</h2><p class="small muted">${s["amount"]:,.0f} · {E(s["horizon"])} · '
          f'{E(s["risk_appetite"])} · {E(s["strategy"])}</p>')
        a('<table><tr><th>Case</th><th class="n">Price</th><th class="n">Value</th><th class="n">P/L</th></tr>')
        for o in s["outcomes"]:
            a(f"<tr><td>{E(o['name'])}</td><td class='n'>{o['price']:,.2f}</td><td class='n'>${o['value']:,.0f}</td>"
              f"<td class='n'>{o['pnl']:+,.0f} ({o['pnl_pct']:+.1f}%)</td></tr>")
        a("</table>")
        a(f"<p class='small'>Max modeled downside <span class='num'>${s['max_downside']:,.0f} ({s['max_downside_pct']:+.1f}%)</span>, "
          f"based on {E(s['max_downside_basis'])}. Deploy now: {s['deploy_now_pct']:.0f}%.</p>")
        a("<ul class='small'>" + "".join(f"<li>{E(n)}</li>" for n in s["notes"]) + "</ul></div>")
    if rk:
        m = rk.metrics
        a('<div class="card"><h2>Risk</h2><table>')
        for k, val in (("Volatility (1y)", _pct(m.get("annual_vol_1y"), 0, sign=False)), ("Max drawdown 1y · 5y",
                       f"{_pct(m.get('max_drawdown_1y'))} · {_pct(m.get('max_drawdown_5y'))}"),
                       ("1-month 95% VaR · CVaR", f"{_pct(m.get('var95_1m'))} · {_pct(m.get('cvar95_1m'))}"),
                       ("Beta vs S&P 500", _n(m.get("beta"))), ("Suitable for", E(rk.suitability.title()) + " and above")):
            a(f"<tr><td>{k}</td><td class='n'>{val}</td></tr>")
        a("</table><h3 style='margin-top:10px'>What would invalidate the thesis</h3><ul class='small'>"
          + "".join(f"<li>{E(x)}</li>" for x in rk.invalidation) + "</ul></div>")
    a("</section>")

    if C:
        a('<section class="two"><div class="card"><h2>Key risks</h2><ul class="small">'
          + "".join(f"<li>{E(r)}</li>" for r in C.key_risks) + "</ul></div>")
        a('<div class="card"><h2>Catalysts</h2><ul class="small">'
          + ("".join(f"<li>{E(c['text'])}" + (f" ({E(c['expected_date'])})" if c.get("expected_date") else "") + "</li>"
                     for c in C.catalysts) or "<li>No dated catalysts found.</li>") + "</ul></div></section>")

    # Sources
    a('<section class="card"><h2>Sources, freshness and gaps</h2><ul class="sources">')
    for name, fi in f.items():
        if name == "consensus":
            continue
        asof = f"data as of {fi.data_as_of:%Y-%m-%d}" if fi.data_as_of else "no timestamp"
        srcs = "; ".join(sorted({s.name for s in fi.sources}))[:300]
        gaps = f" · Unavailable: {'; '.join(fi.unavailable)}" if fi.unavailable else ""
        a(f"<li><strong>{E(name)}</strong> ({E(fi.status)}, {asof}): {E(srcs)}{E(gaps)}</li>")
    a("</ul></section>")
    a(f"<footer>{E(DISCLAIMER)}</footer></div>")
    return "\n".join(out)
