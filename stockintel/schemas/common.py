"""Shared output envelope that every agent returns."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


SourceKind = Literal[
    "market_data", "fundamentals", "news", "filing", "analyst", "options", "macro",
    "model", "llm_web_search", "derived", "synthetic",
]
Status = Literal["ok", "partial", "unavailable", "error"]
Stance = Literal["bullish", "neutral", "bearish"]
Decision = Literal["BUY", "HOLD", "SELL"]


class Source(BaseModel):
    name: str
    url: str | None = None
    kind: SourceKind = "market_data"
    retrieved_at: datetime = Field(default_factory=utcnow)
    published_at: datetime | None = None
    delayed: bool = False
    note: str | None = None


class AgentFinding(BaseModel):
    """Common envelope. Each agent subclasses this and narrows ``data`` to its own typed payload."""

    agent: str
    ticker: str
    status: Status = "ok"
    signal: float = Field(0.0, description="Directional view in [-1, 1]; -1 strongly bearish, +1 strongly bullish")
    confidence: float = Field(0.0, description="0-1 trust in the view given data coverage/quality")
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    data_as_of: datetime | None = None
    generated_at: datetime = Field(default_factory=utcnow)
    duration_s: float | None = None
    unavailable: list[str] = Field(default_factory=list, description="Data that could not be retrieved")
    warnings: list[str] = Field(default_factory=list)
    data: Any = None

    @field_validator("signal")
    @classmethod
    def _clip_signal(cls, v: float) -> float:
        return float(max(-1.0, min(1.0, v)))

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        return float(max(0.0, min(1.0, v)))

    @property
    def stance(self) -> Stance:
        if self.signal > 0.15:
            return "bullish"
        if self.signal < -0.15:
            return "bearish"
        return "neutral"


def stance_of(signal: float) -> Stance:
    return "bullish" if signal > 0.15 else "bearish" if signal < -0.15 else "neutral"
