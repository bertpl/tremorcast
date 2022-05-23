from __future__ import annotations

import numpy as np

from src.base.forecasting.models.time_series.ts_model import TimeSeriesModelNormalized

from .helpers.arma import arma_fit, arma_fit_robust, arma_predict


class TimeSeriesModelARMA(TimeSeriesModelNormalized):
    """ARMA time-series model."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, p: int = 1, q: int = 0, wd: float = 0.0, fit_robustness: int = 0, **kwargs):
        """
        :param p: (int, default=1) order of AR-submodel
        :param q: (int, default=0) order of MA-submodel
        :param wd: (float, default=0) L2 regularization on weights (a & b)
        :param fit_robustness: (int, default=0) set to any value >0 to increase fitting robustness at the cost of speed
        """
        if wd == 0:
            super().__init__(f"arma-{p}-{q}", **kwargs)
        else:
            super().__init__(f"arma-{p}-{q}-reg", **kwargs)

        self.p = p
        self.q = q
        self.wd = wd
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.fit_robustness = fit_robustness
        self.fitting_cost = 0.0

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def min_hist(self) -> int:
        return self.p + self.q

    def _fit_normalized(self, x_norm: np.ndarray):
        self.a, self.b, self.fitting_cost = arma_fit_robust(
            x_norm, self.p, self.q, self.wd, silent=True, robustness=self.fit_robustness
        )

    def _predict_normalized(self, x_hist_norm: np.ndarray, hor: int) -> np.ndarray:
        return arma_predict(self.a, self.b, x_hist_norm, hor)
