"""Headless entry point: python -m stockintel.cli AAPL [--demo] [--no-llm] [--out reports/]"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import get_settings
from .llm.client import LLMClient
from .orchestration.graph import Runtime, stream_analysis
from .report.builder import build_html, build_markdown
from .scenarios import compute
from .storage.db import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="StockIntel multi-agent stock research")
    ap.add_argument("query", help="ticker or company name")
    ap.add_argument("--demo", action="store_true", help="use SYNTHETIC data (offline demo)")
    ap.add_argument("--no-llm", action="store_true", help="disable Claude reasoning (rule-based fallbacks)")
    ap.add_argument("--out", default="reports", help="directory for the Markdown/HTML report")
    ap.add_argument("--amount", type=float, default=10_000)
    ap.add_argument("--horizon", default="3M", choices=["1M", "3M", "6M", "1Y"])
    ap.add_argument("--risk", default="Moderate", choices=["Conservative", "Moderate", "Aggressive"])
    ap.add_argument("--strategy", default="Swing trading",
                    choices=["Short-term trading", "Swing trading", "Long-term investing"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if a.demo:
        from .data.synthetic import SyntheticProvider
        provider = SyntheticProvider()
    else:
        from .data.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()
    rt = Runtime(provider, LLMClient(settings, enabled=not a.no_llm), settings, Store(settings.db_path))
    state: dict = {"findings": {}, "audit": []}
    for node, upd in stream_analysis(a.query, rt):
        for k, v in upd.items():
            if k == "findings":
                state["findings"].update(v)
            elif k == "audit":
                state["audit"] += v
            else:
                state[k] = v
        for name, f in (upd.get("findings") or {}).items():
            print(f"  [{name:12s}] {f.status:11s} signal {f.signal:+.2f}  conf {f.confidence:.2f}  {f.duration_s or 0:5.1f}s",
                  file=sys.stderr)
    if not state.get("valid"):
        print(f"Invalid ticker: {state.get('error')}", file=sys.stderr)
        return 2
    scen = compute(state["findings"], a.amount, a.horizon, a.risk, a.strategy).to_dict()
    md = build_markdown(state, scen)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"{state['ticker']}_{state.get('analysis_id') or 'run'}")
    with open(base + ".md", "w") as fh:
        fh.write(md)
    with open(base + ".html", "w") as fh:
        fh.write(build_html(md, f"{state['ticker']} research report"))
    c = state["final"].data
    print(json.dumps({"ticker": state["ticker"], "decision": c.decision, "conviction": c.conviction,
                      "entry_timing": c.entry_timing, "score": c.score, "report": base + ".md",
                      "analysis_id": state.get("analysis_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
