"""Base class and execution context shared by all agents."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings
from ..data.base import MarketDataProvider
from ..llm.client import LLMClient
from ..schemas.common import AgentFinding, Source

log = logging.getLogger(__name__)

SYNTHETIC_WARNING = "DEMO DATA: this finding was computed from synthetic data, not real market data."


@dataclass
class AgentContext:
    ticker: str
    provider: MarketDataProvider
    llm: LLMClient
    settings: Settings = field(default_factory=get_settings)
    shared: dict[str, Any] = field(default_factory=dict)       # prices, profile, quote, benchmarks
    findings: dict[str, AgentFinding] = field(default_factory=dict)  # visible to downstream agents
    draft: Any = None  # the consensus draft, read by the risk agent

    def source(self, name: str, kind: str = "market_data", **kw) -> Source:
        if self.provider.info.is_synthetic:
            kind, kw["note"] = "synthetic", "synthetic demo data"
        return Source(name=name, kind=kind, delayed=kw.pop("delayed", True), **kw)


class BaseAgent:
    name: str = "base"
    title: str = "Base agent"
    finding_cls: type[AgentFinding] = AgentFinding

    def analyze(self, ctx: AgentContext) -> AgentFinding:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, ctx: AgentContext) -> AgentFinding:
        t0 = time.time()
        try:
            f = self.analyze(ctx)
        except Exception as e:
            log.exception("agent %s failed", self.name)
            f = self.finding_cls(agent=self.name, ticker=ctx.ticker, status="error", confidence=0.0,
                                 summary=f"{self.title} failed: {e}",
                                 warnings=[traceback.format_exc(limit=2)[-500:]])
        f.duration_s = round(time.time() - t0, 2)
        if ctx.provider.info.is_synthetic and SYNTHETIC_WARNING not in f.warnings:
            f.warnings.insert(0, SYNTHETIC_WARNING)
        return f
