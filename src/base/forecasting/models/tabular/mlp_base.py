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
        self.n_seeds = n_seeds
        self.show_progress = show_progress

        # other
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

        # --- determine lr_max ----------------------------
        lr_max = self._determine_lr_max(x, y)

        # --- train ---------------------------------------
        self._train_best_of_n_seeds(x, y, lr_max)

        # --- return --------------------------------------
        return self

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
    def _learn_dimensions(self, x: np.ndarray, y: np.ndarray):
        self._n_features = x.shape[1]
        self._n_targets = y.shape[1]
        self._feature_names = [f"x{i}" for i in range(self._n_features)]
        self._target_names = [f"y{i}" for i in range(self._n_targets)]

    def _get_layer_sizes(self) -> List[int]:
        return [self.layer_width] * self.n_hidden_layers

    def _initialize_nn(self, x: np.ndarray, y: np.ndarray, seed: int):

        # --- set seeds -----------------------------------
        set_all_random_seeds(seed)

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
            layers=self._get_layer_sizes(),
            act_cls=self.activation.get_activation(),
        )

        self._nn = TabularLearner(data_loader, model, metrics=rmse, wd=self.wd)

    def _determine_lr_max(self, x: np.ndarray, y: np.ndarray) -> float:

        # --- determine lr_max ----------------------------
        if isinstance(self.lr_max, (float, int)):
            # FIXED lr_max
            lr_max = float(self.lr_max)

        elif isinstance(self.lr_max, LrMaxCriterion):
            # use lr_find

            try:

                self._initialize_nn(x, y, seed=1_234)
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
        if self.show_progress:
            print(f"lr_max: {lr_max:.3e}  [self.lr_max='{self.lr_max}']")

        return lr_max

    def _train_best_of_n_seeds(self, x: np.ndarray, y: np.ndarray, lr_max: float):

        # --- init ----------------------------------------
        all_solutions = []  # type: List[Tuple[float, TabularLearner]]
        all_rmses = []  # type: List[float]

        # --- train for n seeds ---------------------------
        for seed in range(self.n_seeds):

            self._learn_one_cycle_until_convergence(x, y, lr_max, seed)
            final_rmse = self.training_losses()[-1]

            all_solutions.append((final_rmse, self._nn))
            all_rmses.append(final_rmse)

        # --- select best solution -----------------------------------------
        best_rmse = None  # type: Optional[float]
        best_nn = None  # type: Optional[TabularLearner]

        for rmse, nn in all_solutions:
            if (best_rmse is None) or (rmse < best_rmse):
                best_rmse = rmse
                best_nn = nn

        if self.show_progress:
            if self.n_seeds > 1:
                print(f"  Picked best nn of {self.n_seeds}: loss = {best_rmse} out of {all_rmses}")

        self._nn = best_nn

    def _learn_one_cycle_until_convergence(self, x: np.ndarray, y: np.ndarray, lr_max: float, seed: int):
        """Runs _learn_one_cycle with decreasing values of lr_max until we converge."""

        self.last_lr_max_value = None
        while (self.last_lr_max_value is None) or self._learner_weights_are_nan():

            self._initialize_nn(x, y, seed)
            self._learn_one_cycle(lr_max, seed)

            if self._learner_weights_are_nan():
                print(f"WARNING: training was unstable; reducing lr_max from {lr_max} to {lr_max / 10}.")
                lr_max = lr_max / 10

    def _learn_one_cycle(self, lr_max: float, seed: int):

        add_tqdm_callback(self._nn, enabled=self.show_progress, extra_msg=f"seed={seed}")
        self._nn.fit_one_cycle(self.n_epochs, lr_max=lr_max)
        remove_tqdm_callback(self._nn)

        self.last_lr_max_value = lr_max

    def _learner_weights_are_nan(self) -> bool:
        all_params = [p.T.detach().numpy() for p in self._nn.parameters()]
        return any([any(np.isnan(arr.flatten())) or any(np.isinf(arr.flatten())) for arr in all_params])
