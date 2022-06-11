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

        # other
        self.last_lr_max = None

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
        data_loader = TabularDataLoaders.from_df(
            df, cont_names=self._feature_names, y_names=self._target_names, valid_idx=[0]
        )

        # --- determine lr_max ----------------------------
        self._initialize_model(data_loader)
        lr_max = self._determine_lr_max(self._nn)

        # --- train ---------------------------------------
        self._learn_one_cycle(self._nn, self.n_epochs, lr_max, self.show_progress)
        while self._learner_weights_are_nan():
            print(f"WARNING: training was unstable; reducing lr_max from {lr_max} to {lr_max/10}.")
            lr_max = lr_max / 10
            self._initialize_model(data_loader)
            self._learn_one_cycle(self._nn, self.n_epochs, lr_max, self.show_progress)

    def _initialize_model(self, data_loader):
        """Initializes self._nn with new TabularLearner with newly initialized weights"""
        model = TabularModel(
            emb_szs=[],
            n_cont=self.get_selected_input_count(),
            out_sz=self.n_outputs,
            layers=self._get_layer_sizes(),
            act_cls=self._get_activation(),
        )

        self._nn = TabularLearner(data_loader, model, metrics=rmse, wd=self.wd)

    def _learn_one_cycle(self, learner: TabularLearner, n_epochs: int, lr_max: float, show_progress: bool):
        add_tqdm_callback(learner, enabled=show_progress)
        learner.fit_one_cycle(n_epochs, lr_max=lr_max)
        remove_tqdm_callback(learner)

        self.last_lr_max = lr_max

    def _learner_weights_are_nan(self) -> bool:
        all_params = [p.T.detach().numpy() for p in self._nn.parameters()]
        return any([any(np.isnan(arr.flatten())) or any(np.isinf(arr.flatten())) for arr in all_params])

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
        pred = np.array(pred)

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

        # --- determine lr_max ----------------------------
        if isinstance(self.lr_max, (float, int)):
            # FIXED lr_max
            lr_max = float(self.lr_max)

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

            except Exception as e:

                print(e)

                print("------====== lr_find failed --> falling back to lr_max=1e-3 ======------")
                lr_max = 1e-3

        else:
            # not supported

            raise NotImplementedError(f"unsupported type for lr_max parameters: '{type(self.lr_max)}'")

        # --- store & return lr_max -----------------------
        if self.show_progress:
            print(f"lr_max: {lr_max:.3e}  [self.lr_max='{self.lr_max}']")

        self.last_lr_max = lr_max
        return lr_max
