from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .helpers import MLP, Activation, FeatureSelector, LrMaxCriterion
from .helpers.mlp_base import HashableLoss
from .tabular_regressor_wrapper import TabularRegressorWrapper


class TabularRegressorMLP(TabularRegressorWrapper):
    def __init__(
        self,
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
        """
        Regressor that uses a single MLP to forecast all outputs, with automatic feature
        & target scaling + optional feature selection.
        """

        super().__init__(
            name="mlp",
            model=MLP(
                n_hidden_layers=n_hidden_layers,
                layer_width=layer_width,
                activation=activation,
                lr_max=lr_max,
                n_epochs=n_epochs,
                wd=wd,
                dropout=dropout,
                n_seeds=n_seeds,
                loss=loss,
                batch_size=batch_size,
            ),
            feature_selector=feature_selector,
            **kwargs,
        )

        # set parameters
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

    def fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> TabularRegressorMLP:
        super().fit(x, y, **fit_params)
        return self

    @property
    def last_lr_max_value(self):
        return self.model.last_lr_max_value
