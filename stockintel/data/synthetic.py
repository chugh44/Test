"""Deterministic SYNTHETIC market data for tests and offline demos.

None of this is real. Every agent that consumes this provider adds a "DEMO DATA" warning to its finding,
and the UI shows a banner. Never use this provider for actual research.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .base import DataUnavailable, MarketDataProvider, NewsItem, OptionChain, ProviderInfo, Quote, TickerMatch

_NAMES = {
    "AAPL": ("Apple Inc. (synthetic)", "Technology", "Consumer Electronics"),
    "MSFT": ("Microsoft Corp. (synthetic)", "Technology", "Software—Infrastructure"),
    "NVDA": ("NVIDIA Corp. (synthetic)", "Technology", "Semiconductors"),
    "AMD": ("Advanced Micro Devices (synthetic)", "Technology", "Semiconductors"),
    "JPM": ("JPMorgan Chase (synthetic)", "Financial Services", "Banks—Diversified"),
    "XOM": ("Exxon Mobil (synthetic)", "Energy", "Oil & Gas Integrated"),
}
_ALIASES = {"apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "jpmorgan": "JPM", "exxon": "XOM"}


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _bdays(n: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = (end or pd.Timestamp.today()).normalize()
    return pd.bdate_range(end=end, periods=n)


class SyntheticProvider(MarketDataProvider):
    def __init__(self, end: str | None = None, n_days: int = 1260):
        self.end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        self.n_days = n_days
        self.info = ProviderInfo(name="SYNTHETIC DEMO DATA", is_synthetic=True,
                                 notes=["All values are randomly generated. Not real market data."])

    # ------------------------------------------------------------------ helpers
    def _valid(self, symbol: str) -> None:
        s = symbol.upper()
        if "INVALID" in s or not s or len(s.lstrip("^")) > 6 or s.lower() in _ALIASES:
            raise DataUnavailable(f"unknown symbol {symbol}")

    def _market_returns(self) -> np.ndarray:
        rng = np.random.default_rng(_seed("market"))
        return rng.normal(0.0004, 0.011, self.n_days)

    # ------------------------------------------------------------------ interface
    def search(self, query: str) -> list[TickerMatch]:
        q = query.strip()
        sym = _ALIASES.get(q.lower(), q.upper())
        try:
            self._valid(sym)
        except DataUnavailable:
            return []
        name = _NAMES.get(sym, (f"{sym} Corp. (synthetic)",))[0]
        return [TickerMatch(sym, name, "DEMO", "EQUITY")]

    def history(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        self._valid(symbol)
        sym = symbol.upper()
        rng = np.random.default_rng(_seed("px", sym))
        mkt = self._market_returns()
        if sym.startswith("^"):
            base = {"^GSPC": 4500.0, "^IXIC": 14000.0, "^VIX": 18.0, "^TNX": 4.2, "^IRX": 5.0}.get(sym, 1000.0)
            if sym in ("^VIX", "^TNX", "^IRX"):  # mean-reverting levels, not prices
                x = np.empty(self.n_days)
                x[0] = base
                for i in range(1, self.n_days):
                    x[i] = x[i - 1] + 0.03 * (base - x[i - 1]) + rng.normal(0, base * 0.02)
                close = np.abs(x)
            else:
                beta = 1.0 if sym == "^GSPC" else 1.2
                close = base * np.exp(np.cumsum(beta * mkt + rng.normal(0, 0.002, self.n_days)))
        else:
            beta = 0.8 + (rng.random() * 0.8)
            drift = rng.normal(0.0002, 0.0003)
            vol = 0.008 + rng.random() * 0.012
            r = drift + beta * (mkt - 0.0004) + rng.normal(0, vol, self.n_days)
            close = (50 + rng.random() * 250) * np.exp(np.cumsum(r))
        idx = _bdays(self.n_days, self.end)
        spread = np.abs(rng.normal(0, 0.008, self.n_days))
        open_ = close * (1 + rng.normal(0, 0.004, self.n_days))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        vol_ = rng.lognormal(16, 0.35, self.n_days)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol_}, index=idx)
        years = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5d": 5}.get(period)
        return df.tail(years) if years else df

    def quote(self, symbol: str) -> Quote:
        h = self.history(symbol)
        p, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
        shares = 1e9 * (1 + _seed("sh", symbol) % 10)
        return Quote(p, prev, (p / prev - 1) * 100, float(h["Volume"].iloc[-1]), float(h["Volume"].tail(63).mean()),
                     p * shares, "USD", datetime.now(timezone.utc), delayed=True, exchange="DEMO")

    def profile(self, symbol: str) -> dict:
        self._valid(symbol)
        sym = symbol.upper()
        name, sector, industry = _NAMES.get(sym, (f"{sym} Corp. (synthetic)", "Technology", "Software—Application"))
        rng = np.random.default_rng(_seed("prof", sym))
        q = self.quote(sym)
        eps = q.price / rng.uniform(15, 40)
        return {
            "symbol": sym, "longName": name, "sector": sector, "industry": industry, "country": "United States",
            "fullTimeEmployees": int(rng.integers(5_000, 200_000)), "website": "https://example.com",
            "longBusinessSummary": f"{name} is a SYNTHETIC company used for offline demonstrations.",
            "marketCap": q.market_cap, "currentPrice": q.price, "currency": "USD",
            "trailingPE": q.price / eps, "forwardPE": q.price / (eps * rng.uniform(1.02, 1.25)),
            "pegRatio": rng.uniform(0.8, 3.0), "enterpriseToEbitda": rng.uniform(8, 30),
            "priceToSalesTrailing12Months": rng.uniform(1.5, 12), "priceToBook": rng.uniform(2, 25),
            "trailingEps": eps, "forwardEps": eps * 1.12,
            "revenueGrowth": rng.uniform(-0.05, 0.3), "earningsGrowth": rng.uniform(-0.1, 0.4),
            "grossMargins": rng.uniform(0.3, 0.75), "operatingMargins": rng.uniform(0.1, 0.4),
            "profitMargins": rng.uniform(0.05, 0.3), "returnOnEquity": rng.uniform(0.05, 0.6),
            "debtToEquity": rng.uniform(10, 180), "currentRatio": rng.uniform(0.8, 3.0),
            "freeCashflow": q.market_cap * rng.uniform(0.01, 0.05), "totalCash": q.market_cap * 0.05,
            "totalDebt": q.market_cap * 0.04, "beta": rng.uniform(0.7, 1.6),
            "fiftyTwoWeekHigh": float(self.history(sym)["High"].tail(252).max()),
            "fiftyTwoWeekLow": float(self.history(sym)["Low"].tail(252).min()),
            "dividendYield": rng.uniform(0, 0.03), "sharesOutstanding": q.market_cap / q.price,
            "shortPercentOfFloat": rng.uniform(0.005, 0.06),
        }

    def financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        self._valid(symbol)
        rng = np.random.default_rng(_seed("fin", symbol.upper()))
        base_rev = rng.uniform(5e9, 100e9)

        def frame(n: int, freq_days: int, scale: float) -> dict[str, pd.DataFrame]:
            dates = [self.end - timedelta(days=freq_days * i + 30) for i in range(n)]
            g = rng.uniform(0.0, 0.08 if freq_days < 100 else 0.2)
            rev = np.array([base_rev * scale / (1 + g) ** i for i in range(n)])
            gm, om = rng.uniform(0.35, 0.7), rng.uniform(0.12, 0.35)
            inc = pd.DataFrame({d: {
                "Total Revenue": r, "Gross Profit": r * gm, "Operating Income": r * om,
                "Net Income": r * om * 0.8, "EBITDA": r * (om + 0.05), "Diluted EPS": r * om * 0.8 / 1e9,
            } for d, r in zip(dates, rev)})
            bal = pd.DataFrame({d: {
                "Total Assets": r * 2.2, "Total Debt": r * 0.5, "Stockholders Equity": r * 0.9,
                "Cash And Cash Equivalents": r * 0.25, "Current Assets": r * 0.8, "Current Liabilities": r * 0.5,
            } for d, r in zip(dates, rev)})
            cf = pd.DataFrame({d: {
                "Operating Cash Flow": r * om * 1.1, "Capital Expenditure": -r * 0.06,
                "Free Cash Flow": r * (om * 1.1 - 0.06), "Repurchase Of Capital Stock": -r * 0.05,
            } for d, r in zip(dates, rev)})
            return {"income": inc, "balance": bal, "cashflow": cf}

        a, q = frame(4, 365, 1.0), frame(5, 91, 0.25)
        return {"income_annual": a["income"], "income_quarterly": q["income"], "balance_annual": a["balance"],
                "balance_quarterly": q["balance"], "cashflow_annual": a["cashflow"], "cashflow_quarterly": q["cashflow"]}

    def analyst(self, symbol: str) -> dict:
        self._valid(symbol)
        rng = np.random.default_rng(_seed("an", symbol.upper()))
        p = self.quote(symbol).price
        targets = p * rng.uniform(0.8, 1.45, 25)
        firms = ["Demo Securities", "Example Capital", "Sample & Co", "Placeholder Partners", "Mock Research",
                 "Test Bank", "Synthetic Advisors", "Fictional Markets"]
        rows = []
        for i in range(14):
            d = self.end - timedelta(days=int(rng.integers(1, 200)))
            prior = p * rng.uniform(0.8, 1.3)
            cur = prior * rng.uniform(0.9, 1.2)
            act = rng.choice(["main", "up", "down", "init", "reit"], p=[0.45, 0.2, 0.12, 0.08, 0.15])
            rows.append({"GradeDate": d, "Firm": firms[i % len(firms)],
                         "ToGrade": rng.choice(["Buy", "Overweight", "Neutral", "Outperform", "Underweight"]),
                         "FromGrade": rng.choice(["Buy", "Neutral", "Hold", ""]), "Action": act,
                         "priceTargetAction": "Raises" if cur > prior else "Lowers",
                         "currentPriceTarget": round(cur, 2), "priorPriceTarget": round(prior, 2)})
        ud = pd.DataFrame(rows).set_index("GradeDate").sort_index(ascending=False)
        recs = pd.DataFrame([{"period": f"-{m}m", "strongBuy": int(rng.integers(3, 12)), "buy": int(rng.integers(5, 20)),
                              "hold": int(rng.integers(3, 15)), "sell": int(rng.integers(0, 3)),
                              "strongSell": int(rng.integers(0, 2))} for m in range(4)])
        recs.loc[0, "period"] = "0m"
        return {"price_targets": {"current": p, "low": float(targets.min()), "high": float(targets.max()),
                                  "mean": float(targets.mean()), "median": float(np.median(targets))},
                "recommendations": recs, "upgrades_downgrades": ud}

    def news(self, symbol: str, limit: int = 30) -> list[NewsItem]:
        self._valid(symbol)
        sym = symbol.upper()
        rng = np.random.default_rng(_seed("news", sym))
        templates = [
            "{s} beats quarterly earnings estimates, raises full-year guidance",
            "{s} announces strategic partnership to expand AI offerings",
            "{s} faces regulatory probe over market practices, sources say",
            "{s} shares slip as analysts flag slowing demand",
            "{s} launches new flagship product line",
            "{s} reportedly in talks to acquire smaller rival",
            "{s} CFO to step down at year end",
            "{s} beats quarterly earnings estimates and raises full-year guidance",  # near-duplicate on purpose
            "{s} settles patent lawsuit",
            "Sector peers rally as industry demand outlook improves",
        ]
        out = []
        for i, t in enumerate(templates[:limit]):
            out.append(NewsItem(id=f"demo-{sym}-{i}", title=t.format(s=sym), publisher="Demo Wire",
                                url=f"https://example.com/demo/{sym.lower()}/{i}",
                                published_at=datetime.now(timezone.utc) - timedelta(hours=int(rng.integers(2, 400))),
                                summary="Synthetic headline for offline demonstration.", origin="synthetic"))
        return out

    def holders(self, symbol: str) -> dict:
        self._valid(symbol)
        rng = np.random.default_rng(_seed("hold", symbol.upper()))
        inst = pd.DataFrame([{"Date Reported": self.end - timedelta(days=60), "Holder": f"Demo Fund {i}",
                              "pctHeld": rng.uniform(0.005, 0.08), "Shares": rng.uniform(1e7, 1e9),
                              "Value": rng.uniform(1e9, 1e11), "pctChange": rng.normal(0.01, 0.05)} for i in range(10)])
        p = self.quote(symbol).price
        ins = pd.DataFrame([{"Start Date": self.end - timedelta(days=int(rng.integers(1, 180))),
                             "Insider": f"Demo Insider {i}", "Position": rng.choice(["Director", "CEO", "CFO", "Officer"]),
                             "Transaction": "", "Text": rng.choice(["Sale at price", "Purchase at price", "Stock Award(Grant)"],
                                                                    p=[0.6, 0.15, 0.25]),
                             "Shares": float(rng.integers(1_000, 200_000)), "Value": 0.0} for i in range(12)])
        ins["Value"] = ins["Shares"] * p
        return {"major_holders": {"insidersPercentHeld": rng.uniform(0.001, 0.1),
                                  "institutionsPercentHeld": rng.uniform(0.5, 0.85),
                                  "institutionsFloatPercentHeld": rng.uniform(0.55, 0.9),
                                  "institutionsCount": float(rng.integers(1500, 6000))},
                "institutional_holders": inst, "mutualfund_holders": inst.head(5).copy(),
                "insider_transactions": ins}

    def options(self, symbol: str, max_expirations: int = 6) -> list[OptionChain]:
        self._valid(symbol)
        if symbol.startswith("^"):
            raise DataUnavailable("no options for index in demo")
        rng = np.random.default_rng(_seed("opt", symbol.upper()))
        p = self.quote(symbol).price
        chains = []
        for k in range(max_expirations):
            exp = (self.end + timedelta(days=7 * (k + 1) + 21 * k)).strftime("%Y-%m-%d")
            strikes = np.round(np.linspace(p * 0.7, p * 1.3, 25), 0)
            base_iv = rng.uniform(0.22, 0.4) + 0.01 * k

            def side(put: bool):
                m = np.abs(strikes / p - 1)
                oi = (rng.lognormal(8, 1, len(strikes)) * np.exp(-6 * m)).round()
                vol = (oi * rng.uniform(0.05, 0.6, len(strikes))).round()
                if not put and k == 0:
                    vol[len(strikes) // 2 + 3] = oi[len(strikes) // 2 + 3] * 3 + 500  # an "unusual" print
                iv = base_iv + 0.25 * m + (0.04 if put else 0) * (strikes < p)
                return pd.DataFrame({"strike": strikes, "openInterest": oi, "volume": vol, "impliedVolatility": iv,
                                     "lastPrice": np.maximum(0.05, (p - strikes if not put else strikes - p).clip(0) + p * iv * 0.05)})
            chains.append(OptionChain(exp, side(False), side(True)))
        return chains

    def calendar(self, symbol: str) -> dict:
        self._valid(symbol)
        d = int(_seed("cal", symbol.upper()) % 60) + 3
        return {"earnings_date": datetime.now(timezone.utc) + timedelta(days=d)}
