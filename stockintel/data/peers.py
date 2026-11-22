"""Sector ETF mapping and a curated peer map. Optional Finnhub peers override the static map."""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

SECTOR_ETF = {
    "Technology": "XLK", "Information Technology": "XLK", "Communication Services": "XLC",
    "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY", "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP", "Healthcare": "XLV", "Health Care": "XLV", "Financial Services": "XLF",
    "Financials": "XLF", "Industrials": "XLI", "Energy": "XLE", "Basic Materials": "XLB", "Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
}

PEERS = {
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "DELL"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL", "CRM"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL", "SNAP"], "GOOG": ["META", "MSFT", "AMZN", "AAPL"],
    "AMZN": ["WMT", "MSFT", "GOOGL", "SHOP", "COST"], "META": ["GOOGL", "SNAP", "PINS", "MSFT"],
    "NVDA": ["AMD", "AVGO", "INTC", "QCOM", "TSM"], "AMD": ["NVDA", "INTC", "AVGO", "QCOM"],
    "INTC": ["AMD", "NVDA", "TXN", "QCOM"], "AVGO": ["NVDA", "QCOM", "MRVL", "TXN"],
    "TSLA": ["GM", "F", "RIVN", "TM", "NIO"], "NFLX": ["DIS", "WBD", "PARA", "ROKU"],
    "JPM": ["BAC", "WFC", "C", "GS", "MS"], "BAC": ["JPM", "WFC", "C"], "GS": ["MS", "JPM", "C"],
    "XOM": ["CVX", "COP", "SHEL", "BP"], "CVX": ["XOM", "COP", "SHEL"],
    "JNJ": ["PFE", "MRK", "ABBV", "LLY"], "LLY": ["NVO", "MRK", "PFE", "ABBV"], "PFE": ["MRK", "JNJ", "BMY"],
    "WMT": ["COST", "TGT", "AMZN", "KR"], "KO": ["PEP", "MDLZ", "KDP"], "PEP": ["KO", "MDLZ", "KDP"],
    "V": ["MA", "AXP", "PYPL"], "MA": ["V", "AXP", "PYPL"], "CRM": ["MSFT", "ORCL", "NOW", "ADBE"],
    "ORCL": ["MSFT", "SAP", "CRM", "IBM"], "ADBE": ["CRM", "MSFT", "INTU"], "DIS": ["NFLX", "CMCSA", "WBD"],
    "BA": ["AIR.PA", "LMT", "RTX", "GE"], "COST": ["WMT", "TGT", "BJ"], "HD": ["LOW", "FND"],
}


def get_peers(symbol: str, finnhub_key: str | None = None, limit: int = 5) -> tuple[list[str], str]:
    """Return (peers, source_label)."""
    sym = symbol.upper()
    if finnhub_key:
        try:
            r = requests.get("https://finnhub.io/api/v1/stock/peers", params={"symbol": sym, "token": finnhub_key},
                             timeout=10)
            if r.ok and isinstance(r.json(), list):
                peers = [p for p in r.json() if p and p != sym][:limit]
                if peers:
                    return peers, "Finnhub peers API"
        except Exception as e:  # pragma: no cover - network
            log.info("finnhub peers failed: %s", e)
    if sym in PEERS:
        return PEERS[sym][:limit], "curated peer map"
    return [], "no peer list available"
