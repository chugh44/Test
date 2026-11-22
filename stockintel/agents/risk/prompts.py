"""Risk-agent policy parameters. These are the explicit rules it uses to review the draft thesis."""

PROFILES = {
    #               risk per trade, max position, max annual vol for suitability
    "Conservative": (0.005, 0.05, 0.25),
    "Moderate": (0.01, 0.10, 0.45),
    "Aggressive": (0.02, 0.20, 9.99),
}
MIN_RR = 1.5                 # below this, a BUY becomes "wait for better entry"
EARNINGS_WAIT_DAYS = 10      # earnings this close forces wait and caps conviction
EARNINGS_CAP = 60
HIGH_VOL = 0.60              # annualized vol above this caps conviction
HIGH_VOL_CAP = 55
EXTENDED_ATR = 1.0           # price more than 1 ATR above the entry zone counts as extended
MAX_STOP_PCT = 0.20          # a stop further than 20% away means no sensible stop exists; veto the entry
