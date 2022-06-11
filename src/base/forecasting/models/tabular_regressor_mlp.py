import math
from typing import List, Union

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

from src.base.forecasting.models.tabular_regressor import TabularRegressor
from src.tools.progress import add_tqdm_callback, remove_tqdm_callback


# =================================================================================================
#  Base Class
# =================================================================================================
class TabularRegressorMLP(TabularRegressor):
    """Tabular model implementing an MLP (Multi-Layer-Perceptron)."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        lr_max: Union[str, float] = "minimum",  # "valley", "intermediate", "minimum", "aggressive" or fixed lr_max.
        n_hidden_layers: int = 1,
        layer_width: int = 50,
        n_epochs: int = 100,
        wd: float = 1e-2,
        activation: str = "elu",  # one of "elu", "relu", "selu"
        input_selection_indices: List[int] = None,
        input_selection_first_n: int = None,
        input_selection_last_n: int = None,
        **kwargs,
    ):

        super().__init__("mlp", n_inputs, n_outputs, **kwargs)

        # hyper-parameters
        self.lr_max = lr_max
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.n_epochs = n_epochs
        self.wd = wd
        self.activation = activation

        # input selection
        self.input_selection_indices = input_selection_indices  # list if indexes of selected inputs
        self.input_selection_first_n = input_selection_first_n
        self.input_selection_last_n = input_selection_last_n

    def get_selected_input_count(self) -> int:
        return len(self.get_selected_inputs())

    def get_selected_inputs(self) -> List[int]:
        selected_indices = self.input_selection_indices or list(range(self.n_inputs))
        if self.input_selection_last_n is not None:
            selected_indices = [idx for idx in selected_indices if idx >= (self.n_inputs - self.input_selection_last_n)]
        if self.input_selection_first_n is not None:
            selected_indices = [idx for idx in selected_indices if idx < self.input_selection_first_n]
        return selected_indices

    def select_inputs(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.get_selected_inputs()].copy()

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray):

        # --- general settings ----------------------------
        fastai.torch_core.set_seed(1, reproducible=True)

        # --- prepare data --------------------------------

        # input selection
        x_subset = self.select_inputs(x)

        # convert data to df
        self.set_feature_and_target_names()
        df = pd.DataFrame(data=np.concatenate([x_subset, y], axis=1), columns=self._feature_names + self._target_names)

        # construct DataLoader
        data = TabularDataLoaders.from_df(df, cont_names=self._feature_names, y_names=self._target_names, valid_idx=[0])

        # --- init neural network -------------------------
        model = TabularModel(
            emb_szs=[],
            n_cont=self.get_selected_input_count(),
            out_sz=self.n_outputs,
            layers=self._get_layer_sizes(),
            act_cls=self._get_activation(),
        )

        # construct learner
        learner = TabularLearner(data, model, metrics=rmse, wd=self.wd)
        self._nn = learner

        # --- determine lr_max ----------------------------
        lr_max = self._determine_lr_max(learner)

        # --- train ---------------------------------------
        add_tqdm_callback(learner, enabled=self.show_progress)
        learner.fit_one_cycle(self.n_epochs, lr_max=lr_max)
        remove_tqdm_callback(learner)

    def predict(self, x: np.ndarray) -> np.ndarray:

        # input selection
        x_subset = self.select_inputs(x)

        # construct DataLoader with features
        #  (see: https://docs.fast.ai/tutorial.tabular.html)
        self.set_feature_and_target_names()
        df_features = pd.DataFrame(data=x_subset, columns=self._feature_names)
        dl = self._nn.dls.test_dl(df_features)

        # generate predictions
        pred, _ = self._nn.get_preds(dl=dl)

        # return as numpy array
        return np.array(pred)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def set_feature_and_target_names(self):
        self._feature_names = [f"x{i}" for i in self.get_selected_inputs()]
        self._target_names = [f"y{i}" for i in range(self.n_outputs)]

    def _get_layer_sizes(self) -> List[int]:
        return [self.layer_width] * self.n_hidden_layers

    def _get_activation(self):
        """Returns activation function to be used as class instance; e.g. SELU()"""
        if self.activation == "relu":
            return ReLU(inplace=True)
        elif self.activation == "elu":
            return ELU(inplace=True)
        elif self.activation == "selu":
            return SELU(inplace=True)
        else:
            raise NotImplementedError(f"Activation function '{self.activation}' is not implemented.")

    def _determine_lr_max(self, learner: TabularLearner) -> float:

        if isinstance(self.lr_max, (float, int)):
            # FIXED lr_max
            return float(self.lr_max)

        elif isinstance(self.lr_max, str):
            # use lr_find

            try:

                lr_max = learner.lr_find(start_lr=1e-6, end_lr=1e3, suggest_funcs=[valley, minimum], show_plot=False)

                if self.lr_max == "valley":
                    lr_max = lr_max.valley
                elif self.lr_max == "intermediate":
                    lr_max = math.sqrt(lr_max.valley * lr_max.minimum)
                elif self.lr_max == "minimum":
                    lr_max = lr_max.minimum
                elif self.lr_max == "aggressive":
                    lr_max = 2 * lr_max.minimum
                else:
                    raise NotImplementedError(f"lr_max method='{self.lr_max}' not implemented.")

                if self.show_progress:
                    print(f"  lr_max: {lr_max:.3e}  [{self.lr_max.upper()}]")

            except Exception as e:

                print(e)

                print("------====== lr_find failed --> falling back to lr_max=1e-3 ======------")
                lr_max = 1e-3

                if self.show_progress:
                    print(f"lr_max: {lr_max:.3e}  [{self.lr_max.upper()}; fallback]")

            return lr_max

        else:
            # not supported

            raise NotImplementedError(f"unsupported type for lr_max parameters: '{type(self.lr_max)}'")
