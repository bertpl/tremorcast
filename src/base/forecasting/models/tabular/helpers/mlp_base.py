"""Simple base fastai-based MLP regressor, acting as an sklearn estimator without using our own wrappers."""
from __future__ import annotations

import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fastai.callback.schedule import minimum, valley
from fastai.metrics import rmse
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import TabularLearner
from fastai.tabular.model import TabularModel
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor
from torch.nn import ELU, GELU, SELU, ReLU

from src.tools.math import remove_nan_rows, set_all_random_seeds
from src.tools.progress import add_tqdm_callback, remove_tqdm_callback


# =================================================================================================
#  Enums
# =================================================================================================
class Activation(Enum):
    ELU = auto()
    RELU = auto()
    SELU = auto()
    GELU = auto()

    def __str__(self):
        return self.name

    def get_activation(self):
        """Returns instance of activation function to be used; e.g. SELU()"""
        if self == Activation.RELU:
            return ReLU(inplace=True)
        elif self == Activation.ELU:
            return ELU(inplace=True)
        elif self == Activation.SELU:
            return SELU(inplace=True)
        elif self == Activation.GELU:
            return GELU()
        else:
            raise NotImplementedError(f"Activation function for '{self}' is not implemented.")


class LrMaxCriterion(Enum):
    VALLEY = auto()
    INTERMEDIATE = auto()
    MINIMUM = auto()
    AGGRESSIVE = auto()

    def __str__(self):
        return self.name


# =================================================================================================
#  MLP class
# =================================================================================================
class MLP(BaseEstimator, RegressorMixin):

    MAX_VALUE = 1e10
    MIN_VALUE = -1e10

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
        loss: HashableLoss = None,
        batch_size: int = 64,
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
        self.loss = loss
        self.batch_size = batch_size

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

    def _determine_lr_max(self, x: np.ndarray, y: np.ndarray) -> float:

        # --- initialize nn -------------------------------
        self._initialize_nn(x, y)

        # --- determine lr_max ----------------------------
        if isinstance(self.lr_max, (float, int)):
            # FIXED lr_max
            lr_max = float(self.lr_max)

        elif isinstance(self.lr_max, LrMaxCriterion):
            # use lr_find

            lr_max = None
            for start_lr, stop_div in [(1e-9, True), (1e-8, True), (1e-7, True), (1e-6, True), (1e-6, False)]:

                try:

                    lr_max = self._nn.lr_find(
                        start_lr=start_lr,
                        end_lr=1e3,
                        num_it=200,
                        suggest_funcs=(valley, minimum),
                        show_plot=False,
                        stop_div=stop_div,
                    )

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

                    # we found a value for lr_max --> stop trying
                    break

                except:

                    continue

            if lr_max is None:
                print("------====== lr_find failed --> falling back to lr_max=1e-9 ======------")
                lr_max = 1e-9

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
            df, cont_names=self._feature_names, y_names=self._target_names, valid_idx=[0], bs=self.batch_size
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

        # --- tabular learner with appropriate loss -------
        self._nn = TabularLearner(data_loader, model, loss_func=self.loss, metrics=rmse, wd=self.wd)

    def _fit_one_cycle(self, x: np.ndarray, y: np.ndarray, lr_max: float):

        # --- initialize nn -------------------------------
        self._initialize_nn(x, y)

        # --- actual fitting ------------------------------
        add_tqdm_callback(self._nn, enabled=False)  # in this config it just removes the default progress reporter
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

        # convert to numpy array & do some error checking
        pred = np.array(pred)

        pred[pred == np.inf] = self.MAX_VALUE
        pred[pred == np.nan] = self.MAX_VALUE
        pred[pred == -np.inf] = -self.MAX_VALUE
        pred[pred > self.MAX_VALUE] = self.MAX_VALUE
        pred[pred < self.MIN_VALUE] = self.MIN_VALUE

        # return
        return pred


# =================================================================================================
#  Custom loss functions
# =================================================================================================
class HashableLoss:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, eq_values: Iterable = ()):
        # eq_values should contain hashable values that are used in eq and hash
        self.__eq_values = tuple(eq_values)

    # -------------------------------------------------------------------------
    #  Eq & Hash
    # -------------------------------------------------------------------------
    def __eq__(self, other):
        """Equal if of same type and identical __eq_values (set in constructor)"""
        return (type(self) == type(other)) and (self.__eq_values == other.__eq_values)

    def __hash__(self):
        """hash implementation consistent with __eq__"""
        return hash((self.__class__, self.__eq_values))

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        return self.compute_loss(y, yhat).float()  # fastai metric Recorder expects float, not double

    # -------------------------------------------------------------------------
    #  Actual loss function
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_loss(self, y: Tensor, yhat: Tensor) -> Tensor:
        # y & yhat will have [batch_size x output_size] as dimensions
        # (so e.g. [64 x 192] if we have a batch size of 64 & 192 targets)
        # The implementation should perform all computations on Tensors, since pytorch needs this to compute gradients
        pass


class WeightedMSELoss(HashableLoss):
    def __init__(self, target_weights: np.ndarray):
        super().__init__(eq_values=tuple(target_weights))
        self.target_weights = torch.tensor(target_weights).unsqueeze(0)  # type: Tensor # of shape (1, n_targets)

    def compute_loss(self, y: Tensor, yhat: Tensor) -> Tensor:
        batch_size = y.shape[0]

        delta = torch.subtract(y, yhat)
        weighted_delta = delta * self.target_weights.tile((batch_size, 1))
        loss = torch.sum(torch.pow(torch.flatten(weighted_delta), 2))

        return loss
