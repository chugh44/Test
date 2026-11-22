"""Forecasting models: random-walk baseline, drift (GBM), ARIMA, XGBoost, LightGBM, Prophet (optional), LSTM (optional).

Every model sees only the ``close`` series passed to ``fit``. The backtester passes data truncated at the
cutoff, so no model ever sees future prices.
"""

from __future__ import annotations

import importlib.util
import logging
import warnings

import numpy as np
import pandas as pd

from ..analytics.indicators import rsi
from .base import Forecaster, HorizonForecast, sigma_for

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------- baselines
class RandomWalk(Forecaster):
    """Zero-drift random walk. The benchmark every other model must beat."""
    name, family = "Random walk (naive)", "baseline"

    def fit(self, close):
        self.close = close
        return self

    def predict(self, horizons):
        return {h: HorizonForecast(0.0, sigma_for(self.close, h)) for h in horizons}


class Drift(Forecaster):
    """Geometric Brownian motion with a shrunk historical drift."""
    name, family = "Drift / GBM", "statistical"

    def __init__(self, lookback: int = 756, shrink: float = 0.5):
        self.lookback, self.shrink = lookback, shrink

    def fit(self, close):
        self.close = close
        r = np.log(close).diff().dropna().tail(self.lookback)
        self.mu_d = float(r.mean()) * self.shrink
        return self

    def predict(self, horizons):
        return {h: HorizonForecast(self.mu_d * h, sigma_for(self.close, h)) for h in horizons}


# --------------------------------------------------------------------------------------- ARIMA
class Arima(Forecaster):
    name, family = "ARIMA", "statistical"

    def __init__(self, lookback: int = 750, orders=((0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 2))):
        self.lookback, self.orders = lookback, orders

    def fit(self, close):
        from statsmodels.tsa.arima.model import ARIMA

        self.close = close
        y = np.log(close.tail(self.lookback).to_numpy())
        best, best_aic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in self.orders:
                try:
                    res = ARIMA(y, order=order, trend="t").fit()
                    if res.aic < best_aic:
                        best, best_aic, self.order = res, res.aic, order
                except Exception:
                    continue
        if best is None:
            raise RuntimeError("ARIMA failed to fit")
        self.res = best
        return self

    def predict(self, horizons):
        H = max(horizons)
        fc = self.res.get_forecast(H)
        mean = np.asarray(fc.predicted_mean)
        se = np.asarray(fc.se_mean)
        last = float(np.log(self.close.iloc[-1]))
        out = {}
        for h in horizons:
            # ARIMA's linear trend over-extrapolates at long horizons, so dampen the drift beyond about a quarter.
            mu = (mean[h - 1] - last) * (1.0 if h <= 63 else 63 / h + (1 - 63 / h) * 0.5)
            out[h] = HorizonForecast(float(mu), float(max(se[h - 1], sigma_for(self.close, h) * 0.8)))
        return out


# --------------------------------------------------------------------------------------- tree models
def make_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    lc = np.log(close)
    f = pd.DataFrame(index=close.index)
    for n in (1, 5, 21, 63, 126, 252):
        f[f"ret_{n}"] = lc.diff(n)
    r = lc.diff()
    f["vol_21"] = r.rolling(21).std()
    f["vol_63"] = r.rolling(63).std()
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["rsi14"] = rsi(close) / 100
    for n in (50, 200):
        f[f"dist_sma{n}"] = close / close.rolling(n).mean() - 1
    f["dd_252"] = close / close.rolling(252).max() - 1
    if volume is not None:
        v = volume.reindex(close.index)
        f["vol_z"] = (v / v.rolling(63).mean()).apply(np.log)
    return f


class _TreeDirect(Forecaster):
    """Direct multi-horizon regression: one model per horizon predicts the forward h-day log return.

    Training rows are restricted to dates t with t + h <= the last date in the fitted data, so every
    training target was observable at fit time.
    """
    family = "machine learning"
    lib = ""

    def __init__(self, n_estimators: int = 250, max_depth: int = 3, learning_rate: float = 0.03):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    def _model(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, close, volume: pd.Series | None = None):
        self.close = close
        self.X = make_features(close, volume)
        self.models = {}
        self.resid_sd = {}
        return self

    def _fit_h(self, h):
        lc = np.log(self.close)
        y = (lc.shift(-h) - lc)
        data = self.X.assign(_y=y).dropna()
        if len(data) < 150:
            raise RuntimeError(f"not enough rows for h={h}")
        split = int(len(data) * 0.8)
        tr, va = data.iloc[:split], data.iloc[split:]
        # Purge: drop the last h training rows so their targets don't overlap the validation window.
        tr = tr.iloc[:-h] if len(tr) > h + 50 else tr
        m = self._model()
        m.fit(tr.drop(columns="_y"), tr["_y"])
        resid = va["_y"] - m.predict(va.drop(columns="_y"))
        m_full = self._model()
        m_full.fit(data.drop(columns="_y"), data["_y"])
        self.models[h] = m_full
        self.resid_sd[h] = float(max(resid.std(), sigma_for(self.close, h) * 0.6))

    def predict(self, horizons):
        x = self.X.iloc[[-1]]
        out = {}
        for h in horizons:
            if h not in self.models:
                self._fit_h(h)
            mu = float(self.models[h].predict(x)[0])
            out[h] = HorizonForecast(mu, self.resid_sd[h])
        return out


class XGBoostDirect(_TreeDirect):
    name, lib = "XGBoost", "xgboost"

    def _model(self):
        from xgboost import XGBRegressor
        return XGBRegressor(**self.params, subsample=0.8, colsample_bytree=0.8, n_jobs=2, verbosity=0)


class LightGBMDirect(_TreeDirect):
    name, lib = "LightGBM", "lightgbm"

    def _model(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**self.params, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             min_child_samples=30, n_jobs=2, verbose=-1)


# --------------------------------------------------------------------------------------- Prophet
class ProphetModel(Forecaster):
    name, family = "Prophet", "statistical"

    def fit(self, close):
        import logging as _l

        for n in ("cmdstanpy", "prophet"):
            _l.getLogger(n).disabled = True
        from prophet import Prophet

        self.close = close
        df = pd.DataFrame({"ds": close.index, "y": np.log(close.to_numpy())}).tail(1000)
        self.m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
                         changepoint_prior_scale=0.05, interval_width=0.8, uncertainty_samples=200)
        self.m.fit(df)
        return self

    def predict(self, horizons):
        H = max(horizons)
        future = pd.DataFrame({"ds": pd.bdate_range(self.close.index[-1] + pd.Timedelta(days=1), periods=H)})
        fc = self.m.predict(future)
        last = float(np.log(self.close.iloc[-1]))
        out = {}
        for h in horizons:
            row = fc.iloc[h - 1]
            sig = (row["yhat_upper"] - row["yhat_lower"]) / (2 * 1.2816)
            out[h] = HorizonForecast(float(row["yhat"] - last), float(max(sig, sigma_for(self.close, h) * 0.8)))
        return out


# --------------------------------------------------------------------------------------- LSTM
class LSTMModel(Forecaster):
    """A small LSTM over 60-day windows of standardized daily returns. It predicts all horizons at once."""
    name, family = "LSTM", "deep learning"

    def __init__(self, window: int = 60, epochs: int = 25, hidden: int = 24, seed: int = 7):
        self.window, self.epochs, self.hidden, self.seed = window, epochs, hidden, seed

    def fit(self, close, horizons: list[int] | None = None):
        self.close = close
        self.horizons = horizons or [5, 21, 63, 126, 252]
        return self

    def _train(self, horizons):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        lc = np.log(self.close.to_numpy())
        r = np.diff(lc)
        self.mu_r, self.sd_r = r.mean(), r.std() + 1e-9
        z = (r - self.mu_r) / self.sd_r
        W = self.window
        horizons = sorted(horizons)
        while len(horizons) > 1 and len(r) - max(horizons) + 1 - W < 200:
            horizons = horizons[:-1]  # drop horizons the training window cannot support
        H = max(horizons)
        X, Y = [], []
        for t in range(W, len(r) - H + 1):  # targets end on or before the last observed date
            X.append(z[t - W:t])
            Y.append([(lc[t + h] - lc[t]) / (self.sd_r * np.sqrt(h)) for h in horizons])
        if len(X) < 200:
            raise RuntimeError("not enough data for LSTM")
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)
        split = int(len(X) * 0.8)

        class Net(nn.Module):
            def __init__(s, hid, k):
                super().__init__()
                s.lstm = nn.LSTM(1, hid, batch_first=True)
                s.head = nn.Linear(hid, k)

            def forward(s, x):
                o, _ = s.lstm(x)
                return s.head(o[:, -1])

        net = Net(self.hidden, len(horizons))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            perm = torch.randperm(split)
            for i in range(0, split, 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss = loss_fn(net(X[idx]), Y[idx])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            resid = (net(X[split:]) - Y[split:]).numpy()
            self.last_pred = net(torch.tensor(z[-W:], dtype=torch.float32).view(1, W, 1)).numpy()[0]
        self.resid_sd = resid.std(axis=0)
        self.trained_h = list(horizons)

    def predict(self, horizons):
        if getattr(self, "trained_h", None) is None:
            self._train(self.horizons)
        out = {}
        for h in horizons:
            if h not in self.trained_h:
                raise RuntimeError(f"LSTM: not enough history for {h}-day horizon")
            k = self.trained_h.index(h)
            scale = self.sd_r * np.sqrt(h)
            out[h] = HorizonForecast(float(self.last_pred[k] * scale),
                                     float(max(self.resid_sd[k] * scale, sigma_for(self.close, h) * 0.6)))
        return out


# --------------------------------------------------------------------------------------- registry
def available_models(enable_prophet: bool = True, enable_lstm: bool = True) -> tuple[list[type[Forecaster]], dict[str, str]]:
    """Return the model classes that can run here, plus {name: reason} for skipped ones."""
    models: list[type[Forecaster]] = [RandomWalk, Drift, Arima]
    skipped: dict[str, str] = {}
    for cls in (XGBoostDirect, LightGBMDirect):
        if importlib.util.find_spec(cls.lib):
            models.append(cls)
        else:
            skipped[cls.name] = f"{cls.lib} not installed"
    if not enable_prophet:
        skipped["Prophet"] = "disabled in settings"
    elif importlib.util.find_spec("prophet"):
        models.append(ProphetModel)
    else:
        skipped["Prophet"] = "prophet not installed"
    if not enable_lstm:
        skipped["LSTM"] = "disabled in settings"
    elif importlib.util.find_spec("torch"):
        models.append(LSTMModel)
    else:
        skipped["LSTM"] = "torch not installed"
    return models, skipped


def fit_model(cls: type[Forecaster], close: pd.Series, volume: pd.Series | None, horizons: list[int]) -> Forecaster:
    m = cls()
    if isinstance(m, _TreeDirect):
        return m.fit(close, volume)
    if isinstance(m, LSTMModel):
        return m.fit(close, horizons)
    return m.fit(close)
