import numpy as np
from scipy.linalg import lstsq

from .tabular_regressor_linear import TabularRegressorLinear


# =================================================================================================
#  Base Class
# =================================================================================================
class TabularRegressorOLS(TabularRegressorLinear):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, n_inputs: int, n_outputs: int, **kwargs):
        super().__init__("ols", n_inputs, n_outputs, **kwargs)

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray):

        print(f"Training {self.n_inputs} -> {self.n_outputs} OLS model.")

        for i in range(self.n_outputs):
            # perform OLS for each target separately

            # --- relevant subset of the data -------------
            x_sub = x.copy()
            y_sub = y[:, i : i + 1].copy()

            # --- remove NaN ------------------------------
            x_sub, y_sub = self._remove_nan(x_sub, y_sub)

            # --- compute OLS -----------------------------
            c, *_ = lstsq(x_sub, y_sub)
            self.C[:, i] = c.flatten()
