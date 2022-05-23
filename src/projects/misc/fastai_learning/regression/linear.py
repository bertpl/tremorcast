import numpy as np
from scipy.linalg import lstsq

from ._base_class import Regressor


class LinearRegressor(Regressor):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, n_features: int, n_targets: int, name: str = None):
        super().__init__(name or "lin", n_features, n_targets)
        self.C = np.zeros((self.n_features, self.n_targets))

    # -------------------------------------------------------------------------
    #  Train
    # -------------------------------------------------------------------------
    def train(self, x: np.ndarray, y: np.ndarray):

        for i in range(self.n_targets):
            # perform OLS for each target separately

            # --- relevant subset of the data -------------
            x_sub = x.copy()
            y_sub = y[:, i : i + 1].copy()

            # --- compute OLS -----------------------------
            c, *_ = lstsq(x_sub, y_sub)
            self.C[:, i] = c.flatten()

    # -------------------------------------------------------------------------
    #  Predict
    # -------------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.C
