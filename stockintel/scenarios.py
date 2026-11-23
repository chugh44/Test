"""Interactive investment-scenario calculator. It is computed from the agents' findings and adds no new data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

RISK_ORDER = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
STRATEGIES = ("Short-term trading", "Swing trading", "Long-term investing")
HORIZON_FOR_STRATEGY_DEFAULT = {"Short-term trading": "1M", "Swing trading": "3M", "Long-term investing": "1Y"}


@dataclass
class ScenarioOutcome:
    name: str
    price: float
    value: float
    pnl: float
    pnl_pct: float


@dataclass
class ScenarioResult:
    amount: float
    horizon: str
    risk_appetite: str
    strategy: str
    entry_price: float
    shares: float
    outcomes: list[ScenarioOutcome]
    expected_value: float
    expected_pnl: float
    prob_profit: float | None
    max_downside: float
    max_downside_pct: float
    max_downside_basis: str
    entry_range: tuple[float, float]
    exit_targets: list[dict]
    stop_loss: float | None
    risk_reward: float | None
    deploy_now_pct: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute(findings: dict, amount: float, horizon: str, risk_appetite: str, strategy: str) -> ScenarioResult:
    fc, tech, risk, an, cons = (findings.get(k) for k in ("forecasting", "technical", "risk", "analyst", "consensus"))
    if not (fc and fc.data):
        raise ValueError("Forecast unavailable; scenarios cannot be computed.")
    p0 = fc.data.last_price
    h = next(x for x in fc.data.horizons if x.label == horizon)
    plan = tech.data.plan if tech and tech.data else None
    notes: list[str] = []

    if strategy == "Long-term investing":
        stop = None
        if tech and tech.data and tech.data.indicators.get("sma200"):
            stop = round(min(tech.data.indicators["sma200"] * 0.95, p0 * 0.85), 2)
            notes.append("Long-term plan: the risk line is 5% below the 200-day SMA (capped at -15%). "
                         "Treat it as a review trigger, not a resting stop order.")
        entry_lo = min(plan.entry_low if plan else p0, h.lo80 if h.lo80 < p0 else p0 * 0.95)
        entry = (max(entry_lo, p0 * 0.9), p0)
        targets = [{"label": f"AI base case ({horizon})", "price": h.base},
                   {"label": f"AI bull case ({horizon})", "price": h.bull}]
        if an and an.data and an.data.targets.get("median"):
            targets.append({"label": "Analyst median target (Wall Street, not AI)", "price": an.data.targets["median"]})
    else:
        stop = plan.stop_loss if plan else round(p0 * 0.92, 2)
        entry = (plan.entry_low, plan.entry_high) if plan else (p0 * 0.98, p0)
        targets = ([{"label": "Technical target 1", "price": plan.target_1},
                    {"label": "Technical target 2", "price": plan.target_2}] if plan else [])
        targets.append({"label": f"AI bull case ({horizon})", "price": h.bull})
        if strategy == "Short-term trading" and horizon not in ("1M", "1W"):
            notes.append("Short-term trading usually means holding for 1 month or less; consider the 1M horizon.")

    shares = amount / p0
    outcomes = []
    for name, px in (("Bull (85th pct)", h.bull), ("Base (median)", h.base), ("Bear (15th pct)", h.bear)):
        eff = max(px, stop) if (stop and strategy != "Long-term investing") else px
        if eff != px:
            name += " — stopped out"
        val = shares * eff
        outcomes.append(ScenarioOutcome(name, round(eff, 2), round(val, 2), round(val - amount, 2),
                                        round((val / amount - 1) * 100, 2)))
    if stop and strategy != "Long-term investing":
        max_dn_px, basis = stop, "stop-loss exit (gaps through the stop can make the loss larger)"
    else:
        max_dn_px, basis = h.lo95, f"95% lower bound of the {horizon} forecast (no stop)"
    max_dn = max(shares * max(max_dn_px, 0.0) - amount, -amount)
    above = [t["price"] for t in targets if t["price"] > p0 * 1.01]
    t1 = min(above) if above else h.bull
    rr = round((t1 - p0) / (p0 - stop), 2) if stop and p0 > stop else None

    suit = risk.data.suitability if risk and risk.data else "moderate"
    deploy = {"Conservative": 33.0, "Moderate": 60.0, "Aggressive": 100.0}[risk_appetite]
    if RISK_ORDER[risk_appetite] < RISK_ORDER[suit.capitalize()]:
        notes.append(f"Risk mismatch: the risk agent rates this stock suitable for {suit} investors and above, "
                     f"which exceeds your {risk_appetite.lower()} risk appetite.")
        deploy = min(deploy, 25.0)
    if cons and cons.data:
        if cons.data.entry_timing != "ENTER NOW":
            deploy = min(deploy, 33.0)
            notes.append(f"Consensus entry timing is '{cons.data.entry_timing}'; stage entries inside the entry range.")
        if cons.data.decision == "SELL":
            deploy = 0.0
            notes.append("The consensus recommendation is SELL; the scenario is shown for reference only.")
    if risk and risk.data:
        size = next((s for s in risk.data.position_sizing if s.profile == risk_appetite), None)
        if size:
            notes.append(f"Risk-budget sizing for {risk_appetite.lower()} investors: about {size.position_pct_of_portfolio:.1f}% "
                         f"of the total portfolio in this position ({size.note.lower()}).")
    return ScenarioResult(
        amount=amount, horizon=horizon, risk_appetite=risk_appetite, strategy=strategy, entry_price=p0,
        shares=round(shares, 4), outcomes=outcomes, expected_value=round(shares * h.base, 2),
        expected_pnl=round(shares * h.base - amount, 2), prob_profit=h.prob_up, max_downside=round(max_dn, 2),
        max_downside_pct=round(max_dn / amount * 100, 2), max_downside_basis=basis,
        entry_range=(round(entry[0], 2), round(entry[1], 2)), exit_targets=targets, stop_loss=stop, risk_reward=rr,
        deploy_now_pct=deploy, notes=notes)
