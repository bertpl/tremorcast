from __future__ import annotations

from src.base.forecasting.models.tabular.legacy.tabular_regressor_multi_mlp import TabularRegressorMultiMLP

from .ts_model_regression_multi import TimeSeriesModelRegressionMulti


# =================================================================================================
#  TimeSeries model based on Multi-Layer-Perceptron - 1 MLP per predicted output
# =================================================================================================
class TimeSeriesModelRegressionMultiMLP(TimeSeriesModelRegressionMulti):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        signal_name: str,
        p: int,
        n: int,
    ):
        super().__init__(signal_name, TabularRegressorMultiMLP(n_inputs=p, n_outputs=n), avoid_training_nans=True)
