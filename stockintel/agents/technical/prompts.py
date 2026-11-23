"""The technical agent is rule-based. These are its signal thresholds."""

SIGNAL_THRESHOLD = 0.25   # |score| above this produces BUY or SELL; otherwise HOLD
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOP_ATR_BUFFER = 0.5     # stop goes this many ATRs below support
MIN_STOP_ATR = 1.0        # but never closer than 1 ATR to the price
MAX_STOP_ATR = 3.5
MAX_STOP_PCT = 0.5        # and never more than 50% below the price (or below zero)
