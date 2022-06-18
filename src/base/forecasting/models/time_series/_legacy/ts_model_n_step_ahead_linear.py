import numpy as np

from .ts_model_n_step_ahead import TimeSeriesModelMultiStepRegression


class TimeSeriesModelMultiStepLinear(TimeSeriesModelMultiStepRegression):
    """
    This class implements a LINEAR auto-regressive model predicting n steps ahead:

        y = x*C

    The training method to determine C needs to be implemented by the child class.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, model_type: str, signal_name: str, p: int, n: int, avoid_training_nans: bool, cv: dict = None):

        super().__init__(
            model_type=model_type, signal_name=signal_name, p=p, n=n, avoid_training_nans=avoid_training_nans, cv=cv
        )

        self.C = np.zeros((p, n))  # linear regression matrix

    # -------------------------------------------------------------------------
    #  Predict
    # -------------------------------------------------------------------------
    def _predict_tabulated(self, x: np.ndarray) -> np.ndarray:
        return x @ self.C
