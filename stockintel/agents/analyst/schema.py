from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class RatingChange(BaseModel):
    date: str
    firm: str
    action: str
    from_grade: str | None
    to_grade: str | None
    prior_target: float | None
    current_target: float | None
    target_change_pct: float | None
    notable: bool = False


class AnalystData(BaseModel):
    label: str = "Wall Street analyst estimates. These are NOT StockIntel's AI forecasts."
    current_price: float
    targets: dict[str, float | None]          # mean, median, high, low
    implied_upside_pct: dict[str, float | None]
    n_analysts: int | None
    recommendation_counts: dict[str, int] = Field(default_factory=dict)
    recommendation_score: float | None       # -1 (strong sell) .. +1 (strong buy)
    recommendation_trend: float | None       # change in score vs ~3 months ago
    changes: list[RatingChange] = Field(default_factory=list)
    counts_90d: dict[str, int] = Field(default_factory=dict)
    target_revision_90d_pct: float | None = None
    target_dispersion: float | None = None
    revision_history: list[dict] = Field(default_factory=list)  # month, median target, n


class AnalystFinding(AgentFinding):
    data: AnalystData | None = None
