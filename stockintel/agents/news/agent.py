"""Agent 2: Latest News & Market Intelligence."""

from __future__ import annotations

import difflib
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from ...data.base import NewsItem
from ...data.news_sources import finnhub_news, newsapi_news
from ..base import AgentContext, BaseAgent
from .prompts import SYSTEM_CLASSIFY, SYSTEM_SEARCH, USER_CLASSIFY, USER_SEARCH
from .schema import EventOut, LLMNewsAnalysis, NewsData, NewsFinding, NewsItemOut

HIGH_CRED = {"reuters", "bloomberg", "wsj", "wall street journal", "financial times", "ft.com", "cnbc", "barron",
             "marketwatch", "sec.gov", "businesswire", "business wire", "prnewswire", "pr newswire",
             "globenewswire", "associated press", "apnews", "the information", "nikkei"}
MED_CRED = {"yahoo", "investing.com", "seeking alpha", "motley fool", "benzinga", "zacks", "thestreet", "forbes",
            "fortune", "techcrunch", "the verge", "axios", "business insider", "investor's business daily", "fierce"}

POS = ["beat", "beats", "raises", "raised", "record", "surge", "soar", "jumps", "upgrade", "approval", "approved",
       "partnership", "wins", "launch", "expands", "buyback", "dividend increase", "strong demand", "outperform",
       "settles", "rally"]
NEG = ["miss", "misses", "cut", "cuts", "lowers", "probe", "investigation", "lawsuit", "sues", "recall", "downgrade",
       "slump", "plunge", "falls", "slip", "slips", "fine", "antitrust", "layoff", "resign", "step down", "delay",
       "warning", "weak", "ban", "halt", "slowing"]
SPEC = ["reportedly", "sources say", "people familiar", "rumor", "in talks", "considering", "could ", "may ",
        "weighs", "exploring", "said to"]
CATS = {
    "earnings": ["earnings", "quarter", "eps", "revenue", "results"], "guidance": ["guidance", "outlook", "forecast"],
    "product": ["launch", "product", "unveil", "release"], "partnership": ["partner", "deal with", "collaborat"],
    "m&a": ["acquire", "acquisition", "merger", "buyout", "takeover"],
    "regulatory": ["regulat", "probe", "antitrust", "ftc", "sec ", "doj", "approval", "fda"],
    "legal": ["lawsuit", "sues", "court", "patent", "settle", "verdict"],
    "management": ["ceo", "cfo", "chief", "step down", "resign", "appoint", "board"],
    "analyst": ["upgrade", "downgrade", "price target", "analyst"], "capital_return": ["buyback", "dividend"],
    "industry": ["sector", "industry", "peers"],
}
HIGH_IMPACT = {"earnings", "guidance", "m&a", "regulatory", "legal", "management"}


def _cred(publisher: str | None, url: str | None) -> str:
    s = f"{publisher or ''} {urlparse(url).netloc if url else ''}".lower()
    if any(k in s for k in HIGH_CRED):
        return "high"
    if any(k in s for k in MED_CRED):
        return "medium"
    return "low"


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def dedupe(items: list[NewsItem], threshold: float = 0.82) -> tuple[list[NewsItem], dict[str, int]]:
    """Cluster near-duplicate headlines. Keep the most credible item, earliest first on ties."""
    rank = {"high": 0, "medium": 1, "low": 2}
    seen_urls: set[str] = set()
    kept: list[NewsItem] = []
    dup_count: dict[str, int] = {}
    for it in sorted(items, key=lambda i: (rank[_cred(i.publisher, i.url)],
                                           i.published_at or datetime.max.replace(tzinfo=timezone.utc))):
        if it.url and it.url in seen_urls:
            continue
        match = next((k for k in kept if difflib.SequenceMatcher(None, _norm(k.title), _norm(it.title)).ratio()
                      >= threshold), None)
        if match:
            dup_count[match.id] = dup_count.get(match.id, 0) + 1
            continue
        kept.append(it)
        if it.url:
            seen_urls.add(it.url)
    return kept, dup_count


def rule_classify(it: NewsItem) -> dict:
    t = f" {it.title.lower()} {(it.summary or '').lower()[:300]} "
    pos = sum(w in t for w in POS)
    neg = sum(w in t for w in NEG)
    cat = next((c for c, kws in CATS.items() if any(k in t for k in kws)), "other")
    status = "speculative" if any(w in t for w in SPEC) else "confirmed"
    sentiment = "positive" if pos > neg else "negative" if neg > pos else "neutral"
    return {"category": cat, "sentiment": sentiment, "status": status,
            "impact": "high" if cat in HIGH_IMPACT else "medium" if cat != "other" else "low",
            "summary": it.title}


class NewsAgent(BaseAgent):
    name, title, finding_cls = "news", "Latest News & Market Intelligence Agent", NewsFinding

    def analyze(self, ctx: AgentContext) -> NewsFinding:
        prof = ctx.shared.get("profile") or {}
        name = prof.get("longName") or ctx.ticker
        today = datetime.now(timezone.utc).date().isoformat()
        unavailable, warnings = [], []

        items: list[NewsItem] = []
        try:
            items += ctx.provider.news(ctx.ticker)
        except Exception as e:
            unavailable.append(f"{ctx.provider.info.name} news ({e})")
        items += finnhub_news(ctx.ticker, ctx.settings.finnhub_key)
        items += newsapi_news(f'"{name}" OR {ctx.ticker}', ctx.settings.newsapi_key)
        if not ctx.settings.finnhub_key:
            unavailable.append("Finnhub company news (no FINNHUB_API_KEY)")

        web_used = False
        if not ctx.provider.info.is_synthetic:
            ws = ctx.llm.web_search(purpose="news_search", system=SYSTEM_SEARCH,
                                    prompt=USER_SEARCH.format(name=name, ticker=ctx.ticker, today=today), max_uses=5)
            if ws:
                web_used = True
                for i, r in enumerate(ws.results):
                    pub = None
                    if r.page_age:
                        try:
                            pub = datetime.fromisoformat(r.page_age.replace("Z", "+00:00"))
                            if pub.tzinfo is None:
                                pub = pub.replace(tzinfo=timezone.utc)
                        except ValueError:
                            pub = None
                    items.append(NewsItem(id=f"web-{i}", title=r.title, publisher=urlparse(r.url).netloc,
                                          url=r.url, published_at=pub, origin="web_search"))
        if not items:
            return NewsFinding(agent=self.name, ticker=ctx.ticker, status="unavailable", confidence=0.0,
                               summary="No news could be retrieved.", unavailable=unavailable)

        items, dups = dedupe(items)
        # Provider feeds mix in general market stories. Without an LLM to judge relevance, keep only items
        # that name the company or the ticker.
        if not ctx.llm.available:
            names = {ctx.ticker.lower()} | {w.lower().strip(",.") for w in name.split()[:1] if len(w) > 2}
            rel = [i for i in items if any(n in f"{i.title} {i.summary or ''}".lower() for n in names)]
            if len(rel) < len(items):
                warnings.append(f"{len(items) - len(rel)} feed items that do not mention {ctx.ticker} were excluded.")
            items = rel
        items = sorted(items, key=lambda i: i.published_at or datetime.min.replace(tzinfo=timezone.utc),
                       reverse=True)[:30]
        by_id = {i.id: i for i in items}

        cls: dict[str, dict] = {}
        catalysts, risks, industry = [], [], []
        classifier = "rule-based keyword classifier (no LLM configured)"
        overall = ""
        listing = "\n".join(
            f"[{i.id}] {i.published_at.date() if i.published_at else 'n/d'} | {i.publisher or 'unknown'} | {i.title}"
            + (f" — {i.summary[:280]}" if i.summary else "") for i in items)
        res = ctx.llm.structured(
            purpose="news_classify", system=SYSTEM_CLASSIFY, schema=LLMNewsAnalysis, effort="low",
            user=USER_CLASSIFY.format(name=name, ticker=ctx.ticker, sector=prof.get("sector"),
                                      industry=prof.get("industry"), today=today, items=listing))
        if res:
            classifier = f"Claude ({ctx.llm.model}) structured classification"
            overall = res.overall_view
            for c in res.items:
                if c.item_id in by_id:  # ignore any reference to an item that was not retrieved
                    if c.duplicate_of and c.duplicate_of in by_id and c.duplicate_of != c.item_id:
                        dups[c.duplicate_of] = dups.get(c.duplicate_of, 0) + 1
                        cls[c.item_id] = {"_dup": True}
                        continue
                    cls[c.item_id] = c.model_dump()

            def ev(lst, kind):
                out = []
                for e in lst:
                    urls = [by_id[i].url for i in e.item_ids if i in by_id and by_id[i].url]
                    if urls:  # keep only events backed by a retrieved item
                        out.append(EventOut(text=e.text, expected_date=e.expected_date, source_urls=urls, kind=kind))
                return out
            catalysts, risks, industry = ev(res.catalysts, "news"), ev(res.risks, "news"), ev(res.industry_developments, "news")
        else:
            if ctx.llm.available:
                warnings.append("LLM classification failed; used the rule-based classifier.")

        out_items: list[NewsItemOut] = []
        num = den = 0.0
        now = datetime.now(timezone.utc)
        for it in items:
            c = cls.get(it.id) or rule_classify(it)
            if c.get("_dup"):
                continue
            cred = _cred(it.publisher, it.url)
            out_items.append(NewsItemOut(
                id=it.id, title=it.title, publisher=it.publisher, url=it.url,
                published_at=it.published_at.isoformat() if it.published_at else None, origin=it.origin,
                credibility=cred, category=c["category"], sentiment=c["sentiment"], impact=c["impact"],
                status=c["status"], summary=c.get("summary") or it.title, duplicates=dups.get(it.id, 0)))
            age_d = (now - it.published_at).total_seconds() / 86400 if it.published_at else 7
            w = {"high": 3, "medium": 2, "low": 1}[c["impact"]] * math.exp(-max(age_d, 0) / 7) \
                * {"high": 1.0, "medium": 0.8, "low": 0.5}[cred] * (0.5 if c["status"] != "confirmed" else 1.0)
            num += w * {"positive": 1, "negative": -1, "neutral": 0}[c["sentiment"]]
            den += w
        score = num / den if den else 0.0

        if not res:  # rule-based catalyst and risk extraction
            for o in out_items:
                if o.sentiment == "negative" and o.impact == "high":
                    risks.append(EventOut(text=o.title, source_urls=[o.url] if o.url else []))
        try:
            ed = ctx.provider.calendar(ctx.ticker).get("earnings_date")
            if ed and ed.date() >= datetime.now(timezone.utc).date():
                catalysts.insert(0, EventOut(text="Next earnings report", expected_date=ed.date().isoformat(),
                                             kind="calendar", source_urls=[]))
        except Exception:
            unavailable.append("earnings calendar")

        counts = {s: sum(o.sentiment == s for o in out_items) for s in ("positive", "negative", "neutral")}
        n_cred = sum(o.credibility == "high" for o in out_items)
        confidence = min(0.8, 0.2 + 0.03 * len(out_items) + 0.04 * n_cred) * (1.0 if res else 0.75)
        latest = max((i.published_at for i in items if i.published_at), default=None)
        top = sorted(out_items, key=lambda o: ({"high": 0, "medium": 1, "low": 2}[o.impact], o.published_at or ""))[:5]
        return NewsFinding(
            agent=self.name, ticker=ctx.ticker, status="ok" if out_items else "partial",
            signal=float(max(-1, min(1, score * 1.2))), confidence=confidence,
            summary=overall or (f"{len(out_items)} unique items: {counts['positive']} positive, "
                                f"{counts['negative']} negative, {counts['neutral']} neutral "
                                f"(impact- and recency-weighted sentiment {score:+.2f})."),
            key_points=[f"[{o.sentiment}/{o.impact}{'' if o.status == 'confirmed' else ', ' + o.status}] {o.title}"
                        for o in top],
            risks=[r.text for r in risks[:5]],
            sources=[ctx.source(o.publisher or o.origin, "llm_web_search" if o.origin == "web_search" else "news",
                                url=o.url, delayed=False,
                                published_at=datetime.fromisoformat(o.published_at) if o.published_at else None)
                     for o in out_items[:20]],
            data_as_of=latest, unavailable=unavailable, warnings=warnings,
            data=NewsData(items=out_items, catalysts=catalysts, risks=risks, industry=industry,
                          sentiment_score=round(score, 3), counts=counts, classifier=classifier,
                          web_search_used=web_used),
        )
