"""The fundamentals agent is rule-based, so every metric traces back to a provider field. These thresholds
are its "prompt": the explicit criteria behind strengths, weaknesses and red flags."""

THRESHOLDS = {
    "strong_growth": 0.15,       # revenue growth above this counts as a strength
    "weak_growth": 0.0,          # declining revenue is a weakness
    "high_op_margin": 0.25,
    "low_op_margin": 0.05,
    "margin_compression_pp": 3.0,
    "max_net_debt_ebitda": 3.0,
    "min_current_ratio": 1.0,
    "min_fcf_conversion": 0.5,
    "share_dilution": 0.03,
}
