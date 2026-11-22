from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class PositionSize(BaseModel):
    profile: str
    risk_per_trade_pct: float
    max_position_pct: float
    position_pct_of_portfolio: float     # suggested allocation
    suitable: bool
    note: str


class RiskReview(BaseModel):
    conviction_cap: int | None = None
    force_wait: bool = False
    veto: bool = False
    notes: list[str] = Field(default_factory=list)


class RiskData(BaseModel):
    metrics: dict[str, float | str | None]
    event_risks: list[dict] = Field(default_factory=list)
    stop_loss: float | None
    stop_distance_pct: float | None
    risk_reward: dict[str, float | None]
    position_sizing: list[PositionSize]
    suitability: str                      # conservative | moderate | aggressive
    downside_scenarios: list[dict] = Field(default_factory=list)
    invalidation: list[str] = Field(default_factory=list)
    review: RiskReview = Field(default_factory=RiskReview)


class RiskFinding(AgentFinding):
    data: RiskData | None = None
