# StockIntel: Multi-Agent Stock Research & Trading Intelligence

You give StockIntel a ticker or company name. Nine specialized agents each retrieve their own data and
analyze it. A LangGraph-orchestrated master agent then reconciles their findings into one auditable
**Buy / Hold / Sell** decision with a conviction score, entry timing, levels and a full research report.

> Decision-support and research output only, **not financial advice**.

## Quick start

```bash
pip install -r requirements.txt            # optional extras: pip install prophet torch
cp .env.example .env                        # add ANTHROPIC_API_KEY to enable Claude reasoning
streamlit run app/streamlit_app.py          # the dashboard
python -m stockintel.cli NVDA               # headless run, writes reports/NVDA_<id>.md/.html
python -m stockintel.cli "apple" --demo     # offline run on SYNTHETIC data
pytest -q                                   # 42 offline tests
```

## The agents

| # | Agent | What it does |
|---|---|---|
| 1 | Forecasting | Runs random walk, drift/GBM, ARIMA, XGBoost, LightGBM, Prophet\* and LSTM\* through a **walk-forward backtest** and combines them in a skill-weighted ensemble. Produces 1W–1Y forecasts, calibrated 80/95% intervals, bull/base/bear cases and support/resistance |
| 2 | News | Pulls news from yfinance (plus optional Finnhub/NewsAPI and Claude web search), removes duplicates, and classifies sentiment, impact and confirmed vs speculative. Identifies catalysts and risks |
| 3 | Analyst | Consensus targets (mean, median, high, low), implied upside, upgrades/downgrades, 90-day revision trend. **Labeled as Wall Street estimates, separate from the AI forecasts** |
| 4 | Fundamentals | Growth, margins, FCF and leverage; P/E, forward P/E, PEG, EV/EBITDA and P/S vs peer median; red flags; whether the valuation is justified |
| 5 | Technical | RSI, MACD, MAs, Bollinger, VWAP, ATR, OBV, patterns and breakouts. Short-term and long-term signals, entry zone, stop and targets |
| 6 | Market & Sector | S&P 500/Nasdaq regime, VIX, yields and curve, sector rotation ranking, relative strength vs market, sector and peers; optional FRED and macro brief |
| 7 | Institutional & Options | 13F holders, Form 4 insider activity, put/call ratios, IV term structure, OI walls, max pain, unusual volume. Labels delayed snapshots vs real-time flow |
| 8 | Risk | Volatility, drawdowns, VaR/CVaR, beta, event risk, R/R, position sizing, suitability, invalidation conditions. **Reviews the draft thesis** and can cap conviction, force a wait or veto |
| 9 | Consensus (master) | Weighs evidence by confidence × freshness × quality, detects and resolves conflicts, decides with risk overlay, writes the thesis |

\* Optional models. They're detected automatically and reported as skipped when not installed.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the workflow graph, the agent JSON contract, the data-source
table and the reliability rules.

## Dashboard sections

Overview · Price Forecast · News · Analyst Consensus · Financial Health · Technical Signals · Institutional
Activity · Risk · Final Recommendation · Scenarios (amount, horizon, risk appetite, strategy) · Agents &
Audit (per-agent JSON, full audit trail) · Report (Markdown/HTML/JSON download) · History (stored analyses
and realized forecast accuracy)

## Configuration

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables Claude for news classification, web search and thesis writing (`ant auth login` also works) |
| `STOCKINTEL_MODEL` | Defaults to `claude-opus-5-5`. Requests enable server-side refusal fallbacks |
| `STOCKINTEL_WEB_SEARCH` | `0` disables Claude web search |
| `FINNHUB_API_KEY`, `NEWSAPI_KEY`, `FRED_API_KEY` | Optional extra news, peers and macro data |
| `STOCKINTEL_BACKTEST_FOLDS`, `STOCKINTEL_ENABLE_LSTM`, `STOCKINTEL_ENABLE_PROPHET` | Forecasting cost/coverage trade-offs |

## Honest limitations

* Yahoo Finance data is free, delayed and unofficial. For production, swap in a licensed provider by
  implementing `MarketDataProvider`.
* Real-time options order flow (sweeps, blocks) needs a paid feed. It's reported as unavailable.
* Daily stock returns are close to a random walk, so the backtest often shows **no model beating the naive
  baseline**. The ensemble then falls back toward "no edge" and says so, rather than presenting a confident
  forecast.
