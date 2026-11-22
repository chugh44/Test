from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...schemas.common import AgentFinding

Category = Literal["earnings", "guidance", "product", "partnership", "m&a", "regulatory", "legal", "management",
                   "analyst", "industry", "macro", "capital_return", "other"]


# ---- LLM structured-output schema (kept simple for strict JSON schema) ----
class LLMItemClass(BaseModel):
    item_id: str
    category: Category
    sentiment: Literal["positive", "negative", "neutral"]
    impact: Literal["high", "medium", "low"]
    status: Literal["confirmed", "speculative", "opinion"]
    summary: str = Field(description="One factual sentence; no information beyond the headline/summary given")
    duplicate_of: str | None = Field(None, description="item_id of an earlier item reporting the same event")


class LLMEvent(BaseModel):
    text: str
    expected_date: str | None = None
    item_ids: list[str]


class LLMNewsAnalysis(BaseModel):
    items: list[LLMItemClass]
    catalysts: list[LLMEvent]
    risks: list[LLMEvent]
    industry_developments: list[LLMEvent]
    overall_view: str


# ---- Agent output ----
class NewsItemOut(BaseModel):
    id: str
    title: str
    publisher: str | None
    url: str | None
    published_at: str | None
    origin: str
    credibility: Literal["high", "medium", "low"]
    category: str
    sentiment: str
    impact: str
    status: str
    summary: str
    duplicates: int = 0


class EventOut(BaseModel):
    text: str
    expected_date: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    kind: str = "news"


class NewsData(BaseModel):
    items: list[NewsItemOut]
    catalysts: list[EventOut]
    risks: list[EventOut]
    industry: list[EventOut]
    sentiment_score: float
    counts: dict[str, int]
    classifier: str
    web_search_used: bool


class NewsFinding(AgentFinding):
    data: NewsData | None = None
