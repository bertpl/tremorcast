import numpy as np

from src.base.forecasting.models.tabular.tabular_regressor import TabularRegressor


# =================================================================================================
#  Linear Tabular Model
# =================================================================================================
class TabularRegressorLinear(TabularRegressor):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, n_inputs: int, n_outputs: int, **kwargs):
        super().__init__(name, n_inputs, n_outputs, **kwargs)
        self.C = np.zeros((self.n_inputs, self.n_outputs))

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.C
