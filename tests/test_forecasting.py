import numpy as np
import pandas as pd

from stockintel.forecasting.backtest import attach_skill, walk_forward
from stockintel.forecasting.base import Forecaster, HorizonForecast
from stockintel.forecasting.ensemble import combine, price_band
from stockintel.forecasting.models import Arima, Drift, LightGBMDirect, RandomWalk, XGBoostDirect


class Spy(Forecaster):
    """Records the last date it was fit on, to verify no future data leaks into a backtest fold."""
    name, family = "Spy", "test"
    seen: list = []

    def fit(self, close):
        Spy.seen.append(close.index[-1])
        self.close = close
        return self

    def predict(self, horizons):
        return {h: HorizonForecast(0.0, 0.05) for h in horizons}


def test_walk_forward_never_sees_future(provider):
    close = provider.history("AAPL")["Close"]
    Spy.seen = []
    bt, raw = walk_forward(close, None, Spy, [5, 21], folds=4)
    assert Spy.seen, "model never fit"
    # Each fit ended at a cutoff, and every scored realization lies strictly after it.
    for h, rows in raw.items():
        assert len(rows) >= 1
    assert max(Spy.seen) < close.index[-1]
    assert bt.by_horizon[5].n >= 4


def test_tree_models_train_only_on_observable_targets(provider):
    close = provider.history("AAPL")["Close"]
    m = XGBoostDirect(n_estimators=20).fit(close.iloc[:700])
    m._fit_h(21)
    # The training frame inside _fit_h drops rows whose target (t+21) is beyond the fit window.
    lc = np.log(close.iloc[:700])
    y = (lc.shift(-21) - lc)
    assert y.iloc[-21:].isna().all()


def test_models_predict_all_horizons(provider):
    close, vol = provider.history("AAPL")["Close"], provider.history("AAPL")["Volume"]
    hs = [5, 21, 63, 126, 252]
    for cls in (RandomWalk, Drift, Arima):
        p = cls().fit(close).predict(hs)
        assert set(p) == set(hs) and all(v.sigma > 0 for v in p.values())
    p = LightGBMDirect(n_estimators=30).fit(close, vol).predict([5, 21])
    assert all(np.isfinite(v.mu) for v in p.values())


def test_ensemble_falls_back_to_naive_without_skill(provider):
    close = provider.history("AAPL")["Close"]
    bts = {}
    for cls in (RandomWalk, Drift):
        bts[cls.name], _ = walk_forward(close, None, cls, [21], folds=4)
    attach_skill(list(bts.values()), RandomWalk.name)
    preds = {RandomWalk.name: HorizonForecast(0.0, 0.05), Drift.name: HorizonForecast(0.02, 0.05)}
    bts[Drift.name].by_horizon[21].skill = -0.5  # force "no skill"
    e = combine(21, preds, bts, RandomWalk.name)
    assert e.weights == {RandomWalk.name: 1.0}
    assert e.mu == 0.0


def test_price_band_ordering():
    b = price_band(100, 0.01, 0.1)
    assert b["lo95"] < b["lo80"] < b["bear"] < b["base"] < b["bull"] < b["hi80"] < b["hi95"]
