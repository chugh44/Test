"""LangGraph state definition and reducers."""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class GraphState(TypedDict, total=False):
    query: str
    ticker: str
    valid: bool
    error: str | None
    company: dict[str, Any]
    quote: Any
    findings: Annotated[dict[str, Any], merge_dicts]   # agent name -> AgentFinding (parallel-safe merge)
    audit: Annotated[list[dict], operator.add]          # append-only audit trail
    draft: Any
    final: Any
    analysis_id: int | None


def audit(node: str, event: str, **detail) -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(), "node": node, "event": event, "detail": detail}
