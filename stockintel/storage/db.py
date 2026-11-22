"""SQLite persistence: analyses, per-agent findings, forecasts (for accuracy tracking) and the audit trail."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    created_at TEXT NOT NULL,
    company_json TEXT,
    decision TEXT,
    conviction INTEGER,
    entry_timing TEXT,
    score REAL,
    synthetic INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_findings (
    analysis_id INTEGER REFERENCES analyses(id),
    agent TEXT,
    status TEXT,
    signal REAL,
    confidence REAL,
    duration_s REAL,
    finding_json TEXT
);
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER REFERENCES analyses(id),
    ticker TEXT,
    made_on TEXT,          -- date of last price used (no look-ahead)
    horizon TEXT,
    horizon_days INTEGER,
    target_date TEXT,
    base_price REAL,
    pred_base REAL, pred_lo80 REAL, pred_hi80 REAL,
    realized REAL,         -- filled once target_date has passed
    abs_pct_error REAL,
    in_80 INTEGER,
    synthetic INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audit_log (
    analysis_id INTEGER REFERENCES analyses(id),
    ts TEXT, node TEXT, event TEXT, detail_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_fc_ticker ON forecasts(ticker, target_date);
"""


class Store:
    def __init__(self, path: str = "stockintel.db"):
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def save_analysis(self, ticker: str, company: dict, findings: dict, audit: list[dict]) -> int:
        cons = findings.get("consensus")
        d = cons.data if cons is not None else None
        synth = int(bool(company.get("synthetic")))
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO analyses (ticker, created_at, company_json, decision, conviction, entry_timing, score, synthetic)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (ticker, datetime.now(timezone.utc).isoformat(), json.dumps(company, default=str),
                 d.decision if d else None, d.conviction if d else None, d.entry_timing if d else None,
                 d.score if d else None, synth))
            aid = cur.lastrowid
            for name, f in findings.items():
                c.execute("INSERT INTO agent_findings VALUES (?,?,?,?,?,?,?)",
                          (aid, name, f.status, f.signal, f.confidence, f.duration_s, f.model_dump_json()))
            fc = findings.get("forecasting")
            if fc is not None and fc.data:
                for h in fc.data.horizons:
                    c.execute("INSERT INTO forecasts (analysis_id, ticker, made_on, horizon, horizon_days, target_date,"
                              " base_price, pred_base, pred_lo80, pred_hi80, synthetic) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                              (aid, ticker, fc.data.last_date, h.label, h.days, h.target_date, fc.data.last_price,
                               h.base, h.lo80, h.hi80, synth))
            for a in audit:
                c.execute("INSERT INTO audit_log VALUES (?,?,?,?,?)",
                          (aid, a.get("ts"), a.get("node"), a.get("event"), json.dumps(a.get("detail"), default=str)))
        return aid

    # ----------------------------------------------------------------- reads
    def history(self, ticker: str | None = None, limit: int = 50) -> pd.DataFrame:
        q = "SELECT id, ticker, created_at, decision, conviction, entry_timing, score, synthetic FROM analyses"
        args: tuple = ()
        if ticker:
            q += " WHERE ticker = ?"
            args = (ticker,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            return pd.read_sql_query(q, c, params=args + (limit,))

    def audit_trail(self, analysis_id: int) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT ts, node, event, detail_json FROM audit_log WHERE analysis_id = ? ORDER BY ts",
                                     c, params=(analysis_id,))

    def forecasts(self, ticker: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM forecasts", ()
        if ticker:
            q, args = q + " WHERE ticker = ?", (ticker,)
        with self._conn() as c:
            return pd.read_sql_query(q + " ORDER BY made_on DESC", c, params=args)

    # ----------------------------------------------------------------- accuracy tracking
    def score_matured_forecasts(self, provider, today: str | None = None) -> int:
        """Fill realized prices for forecasts whose target date has passed. Returns the number scored."""
        today = today or datetime.now(timezone.utc).date().isoformat()
        with self._conn() as c:
            rows = c.execute("SELECT id, ticker, target_date, pred_base, pred_lo80, pred_hi80 FROM forecasts "
                             "WHERE realized IS NULL AND target_date <= ? AND synthetic = ?",
                             (today, int(provider.info.is_synthetic))).fetchall()
        n = 0
        by_ticker: dict[str, list] = {}
        for r in rows:
            by_ticker.setdefault(r["ticker"], []).append(r)
        for t, rs in by_ticker.items():
            try:
                close = provider.history(t, "2y")["Close"]
            except Exception:
                continue
            with self._lock, self._conn() as c:
                for r in rs:
                    s = close[close.index >= pd.Timestamp(r["target_date"])]
                    if s.empty:
                        continue
                    real = float(s.iloc[0])
                    c.execute("UPDATE forecasts SET realized=?, abs_pct_error=?, in_80=? WHERE id=?",
                              (real, abs(r["pred_base"] / real - 1) * 100,
                               int(r["pred_lo80"] <= real <= r["pred_hi80"]), r["id"]))
                    n += 1
        return n

    def accuracy_summary(self, ticker: str | None = None) -> pd.DataFrame:
        q = ("SELECT horizon, COUNT(*) AS n, AVG(abs_pct_error) AS mape_pct, AVG(in_80) AS coverage_80 "
             "FROM forecasts WHERE realized IS NOT NULL")
        args: tuple = ()
        if ticker:
            q, args = q + " AND ticker = ?", (ticker,)
        with self._conn() as c:
            return pd.read_sql_query(q + " GROUP BY horizon ORDER BY MIN(horizon_days)", c, params=args)
