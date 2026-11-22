from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class FlowData(BaseModel):
    ownership: dict[str, float | None]
    top_institutions: list[dict] = Field(default_factory=list)
    institutional_net_change: float | None = None   # holding-weighted avg % change in shares (latest 13F)
    insider_summary: dict[str, float | int | None] = Field(default_factory=dict)
    insider_transactions: list[dict] = Field(default_factory=list)
    options_summary: dict[str, float | str | None] = Field(default_factory=dict)
    iv_term_structure: list[dict] = Field(default_factory=list)
    oi_walls: dict[str, list[dict]] = Field(default_factory=dict)
    unusual_activity: list[dict] = Field(default_factory=list)
    oi_by_strike: list[dict] = Field(default_factory=list)  # nearest expiry, for charting
    accumulation_verdict: str = ""
    data_labels: list[str] = Field(default_factory=list)


class FlowFinding(AgentFinding):
    data: FlowData | None = None
