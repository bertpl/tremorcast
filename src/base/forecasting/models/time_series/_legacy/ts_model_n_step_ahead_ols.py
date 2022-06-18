from typing import Callable

import numpy as np
from scipy.linalg import lstsq

from .ts_model_n_step_ahead_linear import TimeSeriesModelMultiStepLinear


class TimeSeriesModelMultiStepOLS(TimeSeriesModelMultiStepLinear):
    """Linear auto-regressive n-step-ahead predictor using OLS."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, signal_name: str, p: int, n: int):
        super().__init__(
            model_type="n-step-ols",
            signal_name=signal_name,
            p=p,
            n=n,
            avoid_training_nans=False,  # we will train a separate regressor for each target
        )

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def _fit_tabulated(self, x: np.ndarray, y: np.ndarray):

        print(f"Training n-step-ols with (p, n)=({self.p}, {self.n}).")

        for i in range(self.n):
            # perform OLS for each target separately

            # --- relevant subset of the data -------------
            x_sub = x.copy()
            y_sub = y[:, i : i + 1].copy()

            # --- remove NaN ------------------------------
            x_sub, y_sub = self._remove_nan(x_sub, y_sub)

            # --- compute OLS -----------------------------
            c, *_ = lstsq(x_sub, y_sub)
            self.C[:, i] = c.flatten()
