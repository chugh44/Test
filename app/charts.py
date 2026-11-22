"""Plotly figures for the dashboard. Actual history is always drawn solid; AI forecasts are dashed or shaded."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACTUAL = "#1f3b73"
FORECAST = "#d9822b"
BULL, BEAR = "#2e8b57", "#c0392b"
GRID = "rgba(128,128,128,0.18)"


def _layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def price_forecast(prices: pd.DataFrame, fc, lookback: int = 252, show_levels: bool = True) -> go.Figure:
    hist = prices["Close"].tail(lookback)
    last = hist.index[-1]
    path = pd.DataFrame(fc.path)
    path["date"] = [last + pd.tseries.offsets.BDay(int(d)) for d in path["day"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path["date"], y=path["hi95"], line=dict(width=0), showlegend=False, hoverinfo="skip",
                             legendgroup="fc"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["lo95"], fill="tonexty", fillcolor="rgba(217,130,43,0.12)",
                             line=dict(width=0), name="AI forecast 95% interval", legendgroup="fc"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["hi80"], line=dict(width=0), showlegend=False, hoverinfo="skip",
                             legendgroup="fc"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["lo80"], fill="tonexty", fillcolor="rgba(217,130,43,0.25)",
                             line=dict(width=0), name="AI forecast 80% interval", legendgroup="fc"))
    fig.add_trace(go.Scatter(x=path["date"], y=path["base"], line=dict(color=FORECAST, dash="dash", width=2),
                             name="AI forecast (median)", legendgroup="fc"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist, line=dict(color=ACTUAL, width=2), name="Actual close (history)"))
    for h in fc.horizons:
        d = last + pd.tseries.offsets.BDay(h.days)
        fig.add_trace(go.Scatter(x=[d, d, d], y=[h.bear, h.base, h.bull], mode="markers+text",
                                 marker=dict(color=[BEAR, FORECAST, BULL], size=[7, 9, 7]),
                                 text=["", h.label, ""], textposition="top center", showlegend=False,
                                 hovertext=[f"{h.label} bear {h.bear:.2f}", f"{h.label} base {h.base:.2f}",
                                            f"{h.label} bull {h.bull:.2f}"], hoverinfo="text"))
    fig.add_vline(x=last, line=dict(color="gray", dash="dot"))
    if show_levels:
        for lv in fc.levels:
            fig.add_hline(y=lv.price, line=dict(color=BULL if lv.kind == "support" else BEAR, width=1, dash="dot"),
                          annotation_text=f"{lv.kind} {lv.price:.2f}", annotation_position="top left",
                          annotation_font_size=10)
    return _layout(fig, "", 520)


def model_points(fc) -> go.Figure:
    fig = go.Figure()
    models = sorted({m for h in fc.horizons for m in h.model_point})
    for m in models:
        fig.add_trace(go.Bar(name=m, x=[h.label for h in fc.horizons], y=[h.model_point.get(m) for h in fc.horizons]))
    fig.add_trace(go.Scatter(name="Ensemble base", x=[h.label for h in fc.horizons], y=[h.base for h in fc.horizons],
                             mode="lines+markers", line=dict(color="black", width=3)))
    fig.add_hline(y=fc.last_price, line=dict(color="gray", dash="dot"), annotation_text="last close")
    fig.update_layout(barmode="group")
    return _layout(fig, "Median price by model and horizon", 380)


def technical(prices: pd.DataFrame, ind: pd.DataFrame, lookback: int = 180, levels: list | None = None) -> go.Figure:
    d = ind.tail(lookback)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.13, 0.16, 0.16], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="OHLC",
                                 increasing_line_color=BULL, decreasing_line_color=BEAR), 1, 1)
    for col, color in (("sma20", "#7f8c8d"), ("sma50", "#2980b9"), ("sma200", "#8e44ad"), ("vwap20", "#d4ac0d")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col.upper(), line=dict(color=color, width=1.3)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="Bollinger", line=dict(color="rgba(100,100,100,.4)", width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], showlegend=False, fill="tonexty",
                             fillcolor="rgba(100,100,100,.06)", line=dict(color="rgba(100,100,100,.4)", width=1)), 1, 1)
    for lv in levels or []:
        fig.add_hline(y=lv["price"], line=dict(color=BULL if lv["kind"] == "support" else BEAR, width=1, dash="dot"), row=1, col=1)
    colors = np.where(d["Close"] >= d["Open"], BULL, BEAR)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=colors, name="Volume", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi14"], name="RSI 14", line=dict(color="#34495e")), 3, 1)
    for y in (30, 70):
        fig.add_hline(y=y, line=dict(color="gray", dash="dot", width=1), row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["hist"], name="MACD hist", marker_color=np.where(d["hist"] >= 0, BULL, BEAR)), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#2c3e50", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["signal"], name="Signal", line=dict(color=FORECAST, width=1)), 4, 1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return _layout(fig, "", 760)


def analyst_targets(an, ai_1y_base: float | None) -> go.Figure:
    t, p = an.targets, an.current_price
    fig = go.Figure()
    if t.get("low") and t.get("high"):
        fig.add_trace(go.Scatter(x=[t["low"], t["high"]], y=["Analysts", "Analysts"], mode="lines",
                                 line=dict(color="#95a5a6", width=12), name="Analyst target range"))
    pts = [("Low", t.get("low"), BEAR), ("Median", t.get("median"), ACTUAL), ("Mean", t.get("mean"), "#5d6d7e"),
           ("High", t.get("high"), BULL)]
    for name, v, c in pts:
        if v:
            fig.add_trace(go.Scatter(x=[v], y=["Analysts"], mode="markers+text", text=[name], textposition="top center",
                                     marker=dict(size=13, color=c), name=f"Analyst {name.lower()}"))
    fig.add_vline(x=p, line=dict(color="black", dash="dash"), annotation_text=f"price {p:.2f}")
    if ai_1y_base:
        fig.add_trace(go.Scatter(x=[ai_1y_base], y=["StockIntel AI (1Y base)"], mode="markers+text", text=["AI 1Y base"],
                                 textposition="top center", marker=dict(size=13, color=FORECAST, symbol="diamond"),
                                 name="AI forecast (not an analyst target)"))
    return _layout(fig, "Wall Street price targets vs current price (AI forecast shown separately)", 300)


def revisions(history: list[dict]) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure(go.Bar(x=df["month"], y=df["median_target"], text=df["n"], name="Median new target"))
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    return _layout(fig, "Median published target by month", 300)


def financials(rows: list[dict], title: str) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["period"], y=df["revenue"], name="Revenue", marker_color=ACTUAL), secondary_y=False)
    fig.add_trace(go.Bar(x=df["period"], y=df["net_income"], name="Net income", marker_color="#5dade2"), secondary_y=False)
    if df["free_cash_flow"].notna().any():
        fig.add_trace(go.Bar(x=df["period"], y=df["free_cash_flow"], name="Free cash flow", marker_color=BULL), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["period"], y=df["operating_margin"] * 100, name="Operating margin %", mode="lines+markers",
                             line=dict(color=FORECAST)), secondary_y=True)
    fig.update_layout(barmode="group")
    return _layout(fig, title, 340)


def peer_valuation(fd, ticker: str, key: str = "forward_pe", label: str = "Forward P/E") -> go.Figure:
    rows = [(ticker, fd.valuation.get(key))] + [(p.symbol, getattr(p, key)) for p in fd.peers]
    rows = [(s, v) for s, v in rows if v]
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows],
                           marker_color=[FORECAST if r[0] == ticker else "#aab7c4" for r in rows]))
    if fd.peer_median.get(key):
        fig.add_hline(y=fd.peer_median[key], line=dict(dash="dot"), annotation_text="peer median")
    return _layout(fig, f"{label} vs peers", 300)


def relative_strength(stock: pd.Series, market: pd.Series | None, sector: pd.Series | None, sector_name: str | None) -> go.Figure:
    fig = go.Figure()
    s = stock.tail(252)
    fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name="Stock", line=dict(color=ACTUAL, width=2)))
    for ser, name, color in ((market, "S&P 500", "#7f8c8d"), (sector, sector_name or "Sector", FORECAST)):
        if ser is not None:
            x = ser.reindex(s.index).ffill().dropna()
            if len(x):
                fig.add_trace(go.Scatter(x=x.index, y=x / x.iloc[0] * 100, name=name, line=dict(color=color)))
    return _layout(fig, "1-year performance, rebased to 100", 320)


def options_oi(rows: list[dict], price: float) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strike"], y=df["call_oi"], name="Call OI", marker_color=BULL))
    fig.add_trace(go.Bar(x=df["strike"], y=-df["put_oi"], name="Put OI", marker_color=BEAR))
    fig.add_vline(x=price, line=dict(dash="dash"), annotation_text="price")
    fig.update_layout(barmode="relative")
    return _layout(fig, "Open interest by strike (nearest expiry; puts shown negative)", 320)


def iv_term(term: list[dict]) -> go.Figure:
    df = pd.DataFrame(term)
    fig = go.Figure(go.Scatter(x=df["dte"], y=df["atm_iv"] * 100, mode="lines+markers", text=df["expiration"]))
    fig.update_xaxes(title="days to expiry")
    fig.update_yaxes(title="ATM IV %")
    return _layout(fig, "Implied-volatility term structure", 300)


def drawdown(close: pd.Series) -> go.Figure:
    dd = close / close.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=BEAR), name="Drawdown %"))
    return _layout(fig, "Drawdown from prior peak (%)", 280)


def return_dist(close: pd.Series, var_1m: float | None, horizon: int = 21) -> go.Figure:
    lr = np.log(close)
    r = (np.exp(lr.shift(-horizon) - lr) - 1).dropna() * 100
    fig = go.Figure(go.Histogram(x=r, nbinsx=60, marker_color="#aab7c4", name="Historical 1-month returns"))
    if var_1m is not None:
        fig.add_vline(x=var_1m * 100, line=dict(color=BEAR, dash="dash"), annotation_text="95% VaR")
    return _layout(fig, "Distribution of rolling 1-month returns (%)", 280)


def votes(votes: list) -> go.Figure:
    names = [v.title for v in votes]
    fig = go.Figure(go.Bar(y=names, x=[v.signal for v in votes], orientation="h",
                           marker=dict(color=[BULL if v.signal > 0.15 else BEAR if v.signal < -0.15 else "#95a5a6" for v in votes],
                                       opacity=[0.35 + 0.65 * min(1, v.effective_weight / 0.12) for v in votes]),
                           text=[f"w={v.effective_weight:.3f}" for v in votes], textposition="outside"))
    fig.update_xaxes(range=[-1.1, 1.1], title="signal (bearish ← → bullish); opacity = effective weight")
    return _layout(fig, "Agent votes", 340)


def scenario_bars(outcomes: list[dict], amount: float) -> go.Figure:
    fig = go.Figure(go.Bar(x=[o["name"] for o in outcomes], y=[o["value"] for o in outcomes],
                           marker_color=[BULL, FORECAST, BEAR][: len(outcomes)],
                           text=[f"${o['value']:,.0f} ({o['pnl_pct']:+.1f}%)" for o in outcomes], textposition="outside"))
    fig.add_hline(y=amount, line=dict(dash="dot"), annotation_text="invested")
    return _layout(fig, "Modeled portfolio value at horizon", 320)
