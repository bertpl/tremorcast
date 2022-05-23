from __future__ import annotations

import numpy as np

from src.base.forecasting.models.time_series.ts_model import TimeSeriesModel


class TimeSeriesModelNaiveExponentialDecay(TimeSeriesModel):
    """
    Naive timeseries forecast model that generates predictions that start at the last value and decay exponentially
    towards the mean with a fixed time constant.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, tau: float = 1, **kwargs):
        super().__init__("naive-exp-decay", **kwargs)
        self.mean = 0.0
        self.tau = tau

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def min_hist(self) -> int:
        return 1  # just need the last sample

    def fit(self, x: np.ndarray):
        self.mean = x.mean()

    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:
        return np.full(hor, self.mean) + (x_hist[-1] - self.mean) * np.exp(-np.arange(1, hor + 1) / self.tau)
