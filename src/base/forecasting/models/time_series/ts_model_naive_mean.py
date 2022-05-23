from __future__ import annotations

import numpy as np

from src.base.forecasting.models.time_series.ts_model import TimeSeriesModel


class TimeSeriesModelNaiveMean(TimeSeriesModel):
    """Naive timeseries forecast model that always predicts the mean of the training dataset."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__("naive-mean", **kwargs)
        self.mean = 0.0

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def min_hist(self) -> int:
        return 0

    def fit(self, x: np.ndarray):
        self.mean = x.mean()

    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:
        return np.full(hor, self.mean)
