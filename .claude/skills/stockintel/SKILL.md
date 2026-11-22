---
name: stockintel
description: >-
  StockIntel multi-agent equity research. Runs nine specialized agents (price forecasting with backtested
  models, news, analyst ratings, fundamentals, technicals, market & sector, institutional & options flow,
  risk, consensus) on one stock and returns a Buy/Hold/Sell decision with conviction, entry timing, levels
  and a full research report, then publishes the report as a private claude.ai page. Use when the user says
  "/stockintel NVDA", "analyze <ticker>", "research <company>", "should I buy <stock>", "run StockIntel",
  "stock report on <ticker>", or wants a scenario for an investment amount, horizon, risk appetite or strategy.
argument-hint: "<ticker or company> [amount] [horizon] [risk] [strategy]"
---

# StockIntel

Run from the repository root (the folder containing `stockintel/`).

## 1. Run the analysis

```bash
python -m stockintel.cli "<ticker or company>" \
  --amount 10000 --horizon 3M --risk Moderate --strategy "Swing trading" --out reports
```

- Map the user's words onto the flags: `--horizon` is one of 1M, 3M, 6M, 1Y. `--risk` is Conservative,
  Moderate or Aggressive. `--strategy` is "Short-term trading", "Swing trading" or "Long-term investing".
  If the user gives no value, use the defaults above.
- Add `--no-llm` when no Anthropic credentials are configured. The run says so: news is then classified
  by rules and the thesis comes from a template, both labeled in the report.
- The forecasting agent backtests every model, so a run takes about 1 minute. Use a long Bash timeout
  (300000 ms).
- Progress lines go to stderr. The final JSON on stdout gives `decision`, `conviction`, `entry_timing` and
  the `report` path.
- If it prints `Invalid ticker`, tell the user and ask for the right symbol.
- Use `--demo` only when the user asks for an offline demo. It produces SYNTHETIC data, and you must say so.

## 2. Publish the report page

The run also writes `reports/<TICKER>_page.html`, a finished claude.ai page generated in code from the
agents' findings, so every number on it is exact. Publish that file with the Artifact tool (`icon: "chart"`,
plus a one-sentence `description`). Don't hand-write or edit the figures. Re-running the same ticker rewrites
the same path, so publishing it again in this conversation updates the same page link. If the run printed
DEMO data, the page already carries the banner; say so in chat too.

## 3. Reply in chat

In 3–6 lines, give:
- the decision, conviction and entry timing;
- the one-line thesis;
- the biggest conflict and the biggest risk;
- the page link.

Mention anything the report lists as unavailable that matters, such as news. Not financial advice.

## Related

- `/trader-*` skills cover intraday day-trading workflows (scanner, entry timer, time machine, broker).
- `streamlit run app/streamlit_app.py` is the full interactive dashboard, for running it locally.
