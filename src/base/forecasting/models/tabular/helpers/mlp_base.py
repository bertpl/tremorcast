"""Simple base fastai-based MLP regressor, acting as an sklearn estimator without using our own wrappers."""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastai.callback.schedule import minimum, valley
from fastai.metrics import rmse
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import TabularLearner
from fastai.tabular.model import TabularModel
from sklearn.base import BaseEstimator, RegressorMixin
from torch.nn import ELU, SELU, ReLU

from src.tools.math import remove_nan_rows, set_all_random_seeds
from src.tools.progress import add_tqdm_callback, remove_tqdm_callback


# =================================================================================================
#  Enums
# =================================================================================================
class Activation(Enum):
    ELU = auto()
    RELU = auto()
    SELU = auto()

    def get_activation(self):
        """Returns instance of activation function to be used; e.g. SELU()"""
        if self == Activation.RELU:
            return ReLU(inplace=True)
        elif self == Activation.ELU:
            return ELU(inplace=True)
        elif self == Activation.SELU:
            return SELU(inplace=True)
        else:
            raise NotImplementedError(f"Activation function for '{self}' is not implemented.")


class LrMaxCriterion(Enum):
    VALLEY = auto()
    INTERMEDIATE = auto()
    MINIMUM = auto()
    AGGRESSIVE = auto()


# =================================================================================================
#  MLP class
# =================================================================================================
class MLP(BaseEstimator, RegressorMixin):

    # -------------------------------------------------------------------------
    #  CONSTRUCTOR
    # -------------------------------------------------------------------------
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
        show_progress: bool = True,
    ):

        # set parameters
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.activation = activation
        self.lr_max = lr_max
        self.n_epochs = n_epochs
        self.wd = wd
        self.dropout = dropout
        self.n_seeds = n_seeds
        self.show_progress = show_progress

        # other
        self.seed = None  # type: Optional[int]
        self.last_lr_max_value = None  # type: Optional[float]

        # internal
        self._n_features = None  # type; Optional[int]
        self._n_targets = None  # type; Optional[int]
        self._feature_names = None  # type: Optional[List[str]]
        self._target_names = None  # type: Optional[List[str]]

        self._nn = None  # type: Optional[TabularLearner]

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> MLP:

        # --- init ----------------------------------------
        if y.ndim == 1:
            y = y.reshape((y.size, 1))
        x, y = remove_nan_rows(x, y)
        self._learn_dimensions(x, y)

        # --- train ---------------------------------------
        self._fit_best_of_n_seeds(x, y)

        # --- return --------------------------------------
        return self

    def _learn_dimensions(self, x: np.ndarray, y: np.ndarray):
        self._n_features = x.shape[1]
        self._n_targets = y.shape[1]
        self._feature_names = [f"x{i}" for i in range(self._n_features)]
        self._target_names = [f"y{i}" for i in range(self._n_targets)]

    def _fit_best_of_n_seeds(self, x: np.ndarray, y: np.ndarray):

        # --- init ----------------------------------------
        all_solutions = []  # type: List[Tuple[float, float, TabularLearner]]
        all_rmses = []  # type: List[float]

        # --- train for n seeds ---------------------------
        for seed in range(self.n_seeds):

            self.seed = seed

            self._fit_one_cycle_until_convergence(x, y)
            final_rmse = self.training_losses()[-1]

            all_solutions.append((final_rmse, self.last_lr_max_value, self._nn))
            all_rmses.append(final_rmse)

        # --- select best solution -----------------------------------------
        best_rmse = None  # type: Optional[float]
        best_nn = None  # type: Optional[TabularLearner]

        for rmse, last_lr_max_value, nn in all_solutions:
            if (best_rmse is None) or (rmse < best_rmse):
                best_rmse = rmse
                best_nn = nn
                self.last_lr_max_value = last_lr_max_value

        if self.show_progress:
            if self.n_seeds > 1:
                print(
                    f"  Picked best of {self.n_seeds} runs --> loss = {best_rmse:.3e} = min("
                    + "["
                    + ", ".join([f"{x:.3e}" for x in all_rmses])
                    + "])"
                )

        self._nn = best_nn

    def _fit_one_cycle_until_convergence(self, x: np.ndarray, y: np.ndarray):
        """Runs _learn_one_cycle with decreasing values of lr_max until we converge."""

        # --- determine lr_max ----------------------------
        lr_max = self._determine_lr_max(x, y)

        # --- fit until convergence -----------------------
        self.last_lr_max_value = None
        while (self.last_lr_max_value is None) or self._learner_weights_are_nan():

            self._fit_one_cycle(x, y, lr_max)

            if self._learner_weights_are_nan():
                lr_max = lr_max / 10
                if self.show_progress:
                    print(f"WARNING: training was unstable; reducing lr_max to {lr_max}.")

    def _determine_lr_max(self, x: np.ndarray, y: np.ndarray) -> float:

        # --- initialize nn -------------------------------
        self._initialize_nn(x, y)

        # --- determine lr_max ----------------------------
        if isinstance(self.lr_max, (float, int)):
            # FIXED lr_max
            lr_max = float(self.lr_max)

        elif isinstance(self.lr_max, LrMaxCriterion):
            # use lr_find

            try:

                lr_max = self._nn.lr_find(start_lr=1e-6, end_lr=1e3, suggest_funcs=[valley, minimum], show_plot=False)

                if self.lr_max == LrMaxCriterion.VALLEY:
                    lr_max = lr_max.valley
                elif self.lr_max == LrMaxCriterion.INTERMEDIATE:
                    lr_max = math.sqrt(lr_max.valley * lr_max.minimum)
                elif self.lr_max == LrMaxCriterion.MINIMUM:
                    lr_max = lr_max.minimum
                elif self.lr_max == LrMaxCriterion.AGGRESSIVE:
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
        return lr_max

    def _initialize_nn(self, x: np.ndarray, y: np.ndarray):

        # --- set all seeds -------------------------------
        set_all_random_seeds(self.seed)

        # --- convert to df -------------------------------
        df = pd.DataFrame(data=np.concatenate([x, y], axis=1), columns=self._feature_names + self._target_names)

        # --- construct dataloader ------------------------
        #  (set first sample as validation set, to avoid annoying errors)
        data_loader = TabularDataLoaders.from_df(
            df, cont_names=self._feature_names, y_names=self._target_names, valid_idx=[0]
        )

        # --- tabular model -------------------------------
        model = TabularModel(
            emb_szs=[],
            n_cont=self._n_features,
            out_sz=self._n_targets,
            layers=[self.layer_width] * self.n_hidden_layers,
            ps=[self.dropout] * self.n_hidden_layers,
            act_cls=self.activation.get_activation(),
        )

        self._nn = TabularLearner(data_loader, model, metrics=rmse, wd=self.wd)

    def _fit_one_cycle(self, x: np.ndarray, y: np.ndarray, lr_max: float):

        # --- initialize nn -------------------------------
        self._initialize_nn(x, y)

        # --- determine progress msg ----------------------
        extra_msg = f"seed={self.seed}".ljust(8) if self.n_seeds > 1 else ""
        extra_msg = extra_msg + f"| lr_max={lr_max:.2e}"

        # --- actual fitting ------------------------------
        add_tqdm_callback(self._nn, enabled=self.show_progress, extra_msg=extra_msg)
        self._nn.fit_one_cycle(self.n_epochs, lr_max=lr_max)
        remove_tqdm_callback(self._nn)

        # --- remember lr_max -----------------------------
        self.last_lr_max_value = lr_max

    def _learner_weights_are_nan(self) -> bool:
        all_params = [p.T.detach().numpy() for p in self._nn.parameters()]
        return any([any(np.isnan(arr.flatten())) or any(np.isinf(arr.flatten())) for arr in all_params])

    # -------------------------------------------------------------------------
    #  Metrics
    # -------------------------------------------------------------------------
    def training_losses(self) -> np.ndarray:
        return np.array(self._nn.recorder.losses)

    # -------------------------------------------------------------------------
    #  Predict
    # -------------------------------------------------------------------------
    def predict(self, x: np.ndarray, **predict_params) -> np.ndarray:

        # construct DataLoader with features
        #  (see: https://docs.fast.ai/tutorial.tabular.html)
        df_features = pd.DataFrame(data=x, columns=self._feature_names)
        dl = self._nn.dls.test_dl(df_features)

        # generate predictions
        pred, _ = self._nn.get_preds(dl=dl)

        # return as numpy array
        return np.array(pred)

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------
