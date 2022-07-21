from __future__ import annotations

import numpy as np

from src.base.forecasting.models.time_series.ts_model import TimeSeriesModel


class TimeSeriesModelNaiveConstant(TimeSeriesModel):
    """Naive timeseries forecast model that always predicts the latest value will stay constant in the future."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__("naive-constant", **kwargs)

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def min_hist(self) -> int:
        return 1  # just need the last sample

    def fit(self, x: np.ndarray):
        pass  # nothing needs to be done here

    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:
        last_value = x_hist[-1]
        return np.full(hor, last_value)
