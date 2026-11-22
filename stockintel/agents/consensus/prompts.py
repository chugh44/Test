BASE_WEIGHTS = {
    "forecasting": 0.16, "news": 0.12, "analyst": 0.15, "fundamentals": 0.20,
    "technical": 0.15, "macro": 0.10, "flow": 0.12,
}
# Freshness half-life in days: news goes stale fast, fundamentals slowly.
HALF_LIFE_DAYS = {"forecasting": 5, "news": 3, "analyst": 45, "fundamentals": 150, "technical": 5, "macro": 7,
                  "flow": 45}
DECISION_THRESHOLD = 0.12
MIN_DIRECTIONAL_CONVICTION = 25   # a BUY/SELL weaker than this is reported as HOLD
MIN_ENTER_NOW_CONVICTION = 50     # "ENTER NOW" needs at least this much conviction
CONFLICT_MIN_SIGNAL = 0.25

METHOD = (
    "Each agent's directional signal (-1..1) is weighted by base weight × self-reported confidence × data "
    "freshness (an exponential half-life per agent) × status (ok 1.0, partial 0.7, unavailable/error 0). "
    "Confidence already carries evidence quality: forecasting confidence scales with out-of-sample skill vs "
    "a naive baseline, analyst confidence falls when targets are widely dispersed, and news confidence "
    "rises with source credibility. Agents with strong opposite signals are flagged as conflicts, which reduce "
    "conviction. The risk agent can cap conviction, force a wait, or veto entry. The decision and conviction "
    "are computed in code; the LLM only writes the explanation."
)

SYSTEM_THESIS = """You are the head of research writing the investment thesis for an equity research report.
The decision, conviction score, levels and every number have ALREADY been set by a quantitative process.
Explain them; never change or contradict them.

Rules:
- Use ONLY facts and numbers that appear in the evidence JSON. Do not introduce any new figures, dates,
  events, analyst names or price targets.
- Keep analyst price targets (Wall Street) and StockIntel's own model forecasts clearly separate.
- Discuss the main conflicts between agents and how they were resolved.
- Be balanced and specific. Use plain language and no hype.
- This is decision support, not personalised financial advice."""

USER_THESIS = """Ticker: {ticker} ({name})
DECISION: {decision}   CONVICTION: {conviction}/100   ENTRY TIMING: {timing}

Evidence JSON:
{evidence}"""
