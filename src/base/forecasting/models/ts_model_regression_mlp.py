from typing import List, Union

from .tabular_regressor_mlp import TabularRegressorMLP
from .ts_model_regression import TimeSeriesModelRegression


# =================================================================================================
#  TimeSeries model based on Multi-Layer-Perceptron
# =================================================================================================
class TimeSeriesModelRegressionMLP(TimeSeriesModelRegression):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        signal_name: str,
        p: int,
        n: int,
        lr_max: Union[str, float] = "minimum",  # "valley", "intermediate", "minimum", "aggressive" or fixed lr_max.
        n_hidden_layers: int = 1,
        layer_width: int = 50,
        n_epochs: int = 100,
        wd: float = 1e-2,
        activation: str = "elu",  # one of "elu", "relu", "selu"
        input_selection: List[int] = None,
    ):

        mlp = TabularRegressorMLP(
            n_inputs=p,
            n_outputs=n,
            lr_max=lr_max,
            n_hidden_layers=n_hidden_layers,
            layer_width=layer_width,
            n_epochs=n_epochs,
            wd=wd,
            activation=activation,
            input_selection=input_selection,
        )

        super().__init__(signal_name, mlp, avoid_training_nans=True)
