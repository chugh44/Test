"""The forecasting agent is fully quantitative and makes no LLM calls. This text documents its method in reports."""

METHOD = (
    "Each model forecasts the distribution of h-day log returns. Models are scored in a walk-forward "
    "backtest: at every cutoff they see only data up to that date. Ensemble weights are 1/RMSE², and only "
    "models whose backtest error is within 5% of the naive random walk are eligible. Interval width is "
    "calibrated by the realized standardized errors. Bull and bear cases are the 85th and 15th percentiles; "
    "the base case is the median."
)
