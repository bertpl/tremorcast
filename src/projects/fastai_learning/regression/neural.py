from typing import List, Optional, Union

import fastai
import numpy as np
import pandas as pd
from fastai.callback.schedule import minimum, valley
from fastai.metrics import rmse
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import TabularLearner
from fastai.tabular.model import TabularModel
from fastai.torch_core import Tensor
from torch.nn import ELU, SELU, ReLU

from src.tools.matplotlib import plot_style_matplotlib_default
from src.tools.progress import add_tqdm_callback, remove_tqdm_callback

from ._base_class import Regressor


class NeuralRegressor(Regressor):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        *,
        name: str = None,
        n_hidden_layers: int = 1,
        layer_width: int = 25,
        n_epochs: int = 100,
        wd: float = 0.01,
        lr_max: Union[str, float] = "valley",  # <lr_max>, "valley" or "minimum"
        activation: str = "elu",  # one of ["elu", "relu", "selu"]
    ):
        super().__init__(name or "nn", n_features, n_targets)

        # network structure
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.activation = activation

        # training settings
        self.n_epochs = n_epochs
        self.wd = wd  # weight decay
        self.lr_max = lr_max  # lr_max-finding-method or actual lr_max value

        # regressor
        self._nn = None  # type: Optional[TabularLearner]

        # internal
        self._feature_names = [f"x{i}" for i in range(self.n_features)]
        self._target_names = [f"y{i}" for i in range(self.n_targets)]

    # -------------------------------------------------------------------------
    #  Train
    # -------------------------------------------------------------------------
    def train(self, x: np.ndarray, y: np.ndarray):

        # --- general settings ----------------------------
        fastai.torch_core.set_seed(1, reproducible=True)
        fastai.torch_core.set_num_threads(32)

        # --- prepare data --------------------------------

        # convert data to df
        df = pd.DataFrame(data=np.concatenate([x, y], axis=1), columns=self._feature_names + self._target_names)

        # construct DataLoader
        data = TabularDataLoaders.from_df(df, cont_names=self._feature_names, y_names=self._target_names, valid_idx=[0])

        # --- init neural network -------------------------
        model = TabularModel(
            emb_szs=[],
            n_cont=self.n_features,
            out_sz=self.n_targets,
            layers=self._get_layer_sizes(),
            act_cls=self._get_activation(),
        )

        # construct learner
        learner = TabularLearner(data, model, metrics=rmse, wd=self.wd)
        self._nn = learner

        # --- train ---------------------------------------
        plot_style_matplotlib_default()

        # determine lr_max
        if self.lr_max == "valley":
            lr_max = learner.lr_find(start_lr=1e-6, end_lr=1e3, suggest_funcs=[valley], show_plot=False)
            lr_max = lr_max.valley
        elif self.lr_max == "minimum":
            lr_max = learner.lr_find(start_lr=1e-6, end_lr=1e3, suggest_funcs=[minimum], show_plot=False)
            lr_max = lr_max.minimum
        elif isinstance(self.lr_max, float):
            lr_max = self.lr_max
        else:
            raise NotImplementedError(f"lr_max='{self.lr_max}' not supported.")

        # fit one cycle
        add_tqdm_callback(learner)
        self._nn.fit_one_cycle(self.n_epochs, lr_max=lr_max)
        remove_tqdm_callback(learner)

    def _get_layer_sizes(self) -> List[int]:
        return [self.layer_width] * self.n_hidden_layers

    def _get_activation(self):
        if self.activation == "relu":
            return ReLU(inplace=True)
        elif self.activation == "elu":
            return ELU(inplace=True)
        else:
            return SELU(inplace=True)

    # -------------------------------------------------------------------------
    #  Predict
    # -------------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:

        # construct DataLoader with features
        #  (see: https://docs.fast.ai/tutorial.tabular.html)
        df_features = pd.DataFrame(data=x, columns=self._feature_names)
        dl = self._nn.dls.test_dl(df_features)

        # generate predictions
        pred, _ = self._nn.get_preds(dl=dl)

        # return as numpy array
        return np.array(pred)
