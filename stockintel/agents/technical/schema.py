from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class TradePlan(BaseModel):
    entry_low: float
    entry_high: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_t1: float | None
    risk_reward_t2: float | None
    rationale: str


class TechnicalData(BaseModel):
    last_close: float
    indicators: dict[str, float | None]
    components: dict[str, float]            # score contribution per indicator family
    short_term_score: float
    long_term_score: float
    short_term_signal: str                  # BUY / SELL / HOLD
    long_term_signal: str
    trend: str
    momentum: str
    patterns: list[dict] = Field(default_factory=list)
    levels: list[dict] = Field(default_factory=list)
    setups: list[str] = Field(default_factory=list)
    plan: TradePlan | None = None


class TechnicalFinding(AgentFinding):
    data: TechnicalData | None = None
