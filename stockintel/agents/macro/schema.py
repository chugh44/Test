from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding


class MacroData(BaseModel):
    market: dict[str, float | None]           # S&P/Nasdaq returns, VIX, yields, curve
    regime: str
    sector: str | None
    sector_etf: str | None
    returns: dict[str, dict[str, float | None]]  # {"stock"|"sector"|"market": {"1M": .., ...}}
    relative_strength: dict[str, float | None]    # stock minus market / sector, by period
    sector_ranking: list[dict] = Field(default_factory=list)  # sector ETFs ranked by 3M return
    peers: list[dict] = Field(default_factory=list)
    inflation: dict[str, float | str | None] = Field(default_factory=dict)
    macro_brief: str | None = None
    macro_brief_sources: list[str] = Field(default_factory=list)
    performance_verdict: str = ""


class MacroFinding(AgentFinding):
    data: MacroData | None = None
