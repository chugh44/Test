"""The institutional and options flow agent is rule-based. These labels are shown with its output."""

LABELS = [
    "Institutional holdings come from quarterly 13F filings, which are reported up to 45 days after quarter end.",
    "Insider transactions come from Form 4 filings (reported within 2 business days).",
    "Options data is a delayed end-of-day/intraday snapshot of listed chains. It is NOT real-time order flow;"
    " sweeps, blocks and trade-side aggressor data need a paid feed and are not included.",
]
UNUSUAL_VOL_OI = 2.0       # volume at least 2× open interest
UNUSUAL_MIN_VOLUME = 500
