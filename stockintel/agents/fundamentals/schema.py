from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class PeerRow(BaseModel):
    symbol: str
    name: str | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    ev_ebitda: float | None = None
    price_to_sales: float | None = None
    revenue_growth: float | None = None
    operating_margin: float | None = None


class FundamentalsData(BaseModel):
    metrics: dict[str, float | None]          # growth, margins, cash flow, leverage
    valuation: dict[str, float | None]        # P/E, fwd P/E, PEG, EV/EBITDA, P/S, P/B
    peer_median: dict[str, float | None]
    premium_vs_peers: dict[str, float | None] # % premium (+) or discount (-) vs peer median
    peers: list[PeerRow]
    peer_source: str
    annual: list[dict] = Field(default_factory=list)     # period, revenue, net income, fcf, op margin
    quarterly: list[dict] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    quality_score: float                       # -1..1
    valuation_score: float                     # -1 (expensive) .. +1 (cheap)
    valuation_verdict: str
    latest_period: str | None = None


class FundamentalsFinding(AgentFinding):
    data: FundamentalsData | None = None
