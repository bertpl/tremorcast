import math
from typing import List, Union

from .ts_model_n_step_ahead_nn import TimeSeriesModelMultiStepNeural


class TimeSeriesModelMultiStepNeuralMLP(TimeSeriesModelMultiStepNeural):
    """
    Linear auto-regressive n-step-ahead predictor using neural networks.

    This class uses a very traditional densely connected feed-forward neural network (Multi-Layer Perceptron = MLP)
      to find a regression model between the p past values and n future values of the time series.

    Parameters:
      - n_hidden_layers:  number of layers between input & output layer             (default: 1)
      - layer_width: number of neurons in each hidden layer                         (default: 50)
      - activation: one of ['elu', 'relu', 'selu']                                  (default: 'elu')
      - wd: weight decay >= 0                                                       (default: 1e-2)
      - lr_max_method:  one of ['valley', 'intermediate', 'minimum', 'aggressive']  (default: 'minimum')
      - n_epochs: >0                                                                (default: 100)
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        signal_name: str,
        p: int,
        n: int,
        n_hidden_layers: int = 1,
        layer_width: int = 50,
        lr_max_method: str = "minimum",  # "valley", "intermediate", "minimum", "aggressive"
        n_epochs: int = 100,
        wd: float = 1e-2,
        activation: str = "elu",  # one of "elu", "relu", "selu"
        cv: dict = None,
    ):
        super().__init__(
            model_type="n-step-nn-bottleneck",
            signal_name=signal_name,
            p=p,
            n=n,
            lr_max_method=lr_max_method,
            n_epochs=n_epochs,
            wd=wd,
            activation=activation,
            cv=cv,
        )

        # init attributes for parameters
        self.n_hidden_layers = 0  # type: int
        self.layer_width = 0  # type: int

        # set parameter values
        self.set_param("n_hidden_layers", n_hidden_layers)
        self.set_param("layer_width", layer_width)

    # -------------------------------------------------------------------------
    #  Model structure
    # -------------------------------------------------------------------------
    def _get_layer_sizes(self) -> List[int]:
        """
        Determines layer sizes based on attributes .p, .n, .n_hidden_layers.

        Layer sizes are returned as lists of integers.
        """

        # each layer same width
        return [self.layer_width] * self.n_hidden_layers

    # -------------------------------------------------------------------------
    #  Model complexity
    # -------------------------------------------------------------------------
    def _model_complexity(self) -> Union[float, tuple]:
        # roughly speaking the number of free parameters in the model
        return self.p, self.n_hidden_layers, self.layer_width, -self.wd
