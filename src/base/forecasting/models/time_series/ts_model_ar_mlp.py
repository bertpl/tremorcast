import math
from typing import Optional, Union

import torch
from torch import Tensor

from src.base.forecasting.models.tabular import Activation, FeatureSelector, LrMaxCriterion
from src.base.forecasting.models.tabular.tabular_regressor_mlp import HashableLoss, TabularRegressorMLP
from src.base.forecasting.models.time_series.ts_model_ar import TimeSeriesModelAutoRegressive


# =================================================================================================
#  Time Series Model
# =================================================================================================
class TimeSeriesModelAutoRegressiveMLP(TimeSeriesModelAutoRegressive):
    def __init__(
        self,
        p: int,
        n: int,
        n_hidden_layers: int = 1,
        layer_width: int = 50,
        activation: Activation = Activation.ELU,
        lr_max: Union[LrMaxCriterion, float] = LrMaxCriterion.MINIMUM,
        n_epochs: int = 100,
        wd: float = 1e-2,
        dropout: float = 0.0,
        n_seeds: int = 1,
        loss: HashableLoss = None,
        batch_size: int = 64,
        feature_selector: Optional[FeatureSelector] = None,
        **kwargs,
    ):
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.activation = activation
        self.lr_max = lr_max
        self.n_epochs = n_epochs
        self.wd = wd
        self.dropout = dropout
        self.n_seeds = n_seeds
        self.loss = loss
        self.batch_size = batch_size
        self.feature_selector = feature_selector

        super().__init__(
            p,
            n,
            TabularRegressorMLP(
                n_hidden_layers,
                layer_width,
                activation,
                lr_max,
                n_epochs,
                wd,
                dropout,
                n_seeds,
                loss,
                batch_size,
                feature_selector,
            ),
            **kwargs,
        )


# =================================================================================================
#  Custom loss functions
# =================================================================================================
class MSELoss(HashableLoss):
    """Replicates built-in MSE"""

    def __init__(self):
        super().__init__(eq_values=[])

    def __str__(self):
        return "MSE"

    def compute_loss(self, y: Tensor, yhat: Tensor) -> Tensor:

        delta = yhat.subtract(y)  # type: Tensor
        mse = torch.mean(torch.flatten(delta).pow(2))

        return mse


class LogLogAUCLoss(HashableLoss):
    """Implements a loss-function consistent with the AreaUnderCurveLogLog time series metric."""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__(eq_values=[epsilon])
        self.epsilon = epsilon

    def __str__(self):
        return "LogLogAuc"

    def compute_loss(self, y: Tensor, yhat: Tensor) -> Tensor:

        # --- init ----------------------------------------
        n_targets = y.shape[1]

        # --- compute rmse curve a.f.o. lead time ---------
        delta = yhat.subtract(y)
        rmse_curve = torch.sqrt(torch.mean(delta.pow(2), dim=0))

        # --- compute log-log auc -------------------------
        log_rmse_curve = torch.log2(rmse_curve.add(self.epsilon))
        weights = torch.Tensor([math.log2(i + 2) - math.log2(i + 1) for i in range(n_targets)]).unsqueeze(0)

        loglog_auc = torch.sum(log_rmse_curve * weights)

        # --- return --------------------------------------
        return loglog_auc
