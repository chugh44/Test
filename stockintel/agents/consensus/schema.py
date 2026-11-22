from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding

EntryTiming = Literal["ENTER NOW", "WAIT FOR BETTER ENTRY", "AVOID"]


class AgentVote(BaseModel):
    agent: str
    title: str
    status: str
    signal: float
    confidence: float
    freshness: float
    base_weight: float
    effective_weight: float
    stance: str
    summary: str


class Conflict(BaseModel):
    agents: list[str]
    description: str
    resolution: str
    severity: float


class DraftDecision(BaseModel):
    decision: Literal["BUY", "HOLD", "SELL"]
    score: float
    conviction: float
    agreement: float
    coverage: float
    votes: list[AgentVote]
    conflicts: list[Conflict]
    agreements: list[str]
    forecast_vs_analyst: dict[str, float | str | None] = Field(default_factory=dict)


# --- LLM narrative schema ---
class LLMThesis(BaseModel):
    headline: str = Field(description="One sentence. Must state the given decision.")
    thesis: str = Field(description="2 short paragraphs explaining the decision using ONLY the provided evidence")
    bull_points: list[str]
    bear_points: list[str]
    what_changes_our_mind: list[str]


class ConsensusData(BaseModel):
    decision: Literal["BUY", "HOLD", "SELL"]
    conviction: int
    score: float
    entry_timing: EntryTiming
    entry_rationale: str
    entry_zone: list[float] | None = None
    stop_loss: float | None = None
    targets: list[float] = Field(default_factory=list)
    short_term_view: str = ""
    long_term_view: str = ""
    votes: list[AgentVote]
    conflicts: list[Conflict]
    agreements: list[str]
    risk_adjustments: list[str] = Field(default_factory=list)
    headline: str
    thesis: str
    bull_points: list[str]
    bear_points: list[str]
    what_changes_our_mind: list[str]
    catalysts: list[dict] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    forecast_vs_analyst: dict[str, float | str | None] = Field(default_factory=dict)
    narrative_source: str = "template"
    method: str = ""


class ConsensusFinding(AgentFinding):
    data: ConsensusData | None = None
