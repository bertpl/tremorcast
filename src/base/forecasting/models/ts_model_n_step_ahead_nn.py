import math
from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

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

from .ts_model_n_step_ahead import TimeSeriesModelMultiStepRegression


class TimeSeriesModelMultiStepNeural(TimeSeriesModelMultiStepRegression):
    """
    Linear auto-regressive n-step-ahead predictor using neural networks with arbitrary structure as defined by child
    class.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        model_type: str,
        signal_name: str,
        p: int,
        n: int,
        lr_max_method: str = "minimum",  # "valley", "intermediate", "minimum", "aggressive"
        n_epochs: int = 500,
        wd: float = 1e-2,
        activation: str = "elu",  # one of "elu", "relu", "selu"
        cv: dict = None,
    ):
        super().__init__(
            model_type=model_type,
            signal_name=signal_name,
            p=p,
            n=n,
            avoid_training_nans=True,  # remove rows with at least 1 NaN in either features or targets
            cv=cv,
        )

        # init attributes for parameters
        self.lr_max_method = ""
        self.n_epochs = n_epochs
        self.wd = wd
        self.activation = activation

        # regressor
        self._nn = None  # type: Optional[TabularLearner]

        # set parameter values
        self.set_param("lr_max_method", lr_max_method)
        self.set_param("n_epochs", n_epochs)
        self.set_param("wd", wd)
        self.set_param("activation", activation)

        # internal
        self._feature_names = [f"x{i}" for i in range(p)]
        self._target_names = [f"y{i}" for i in range(n)]

    # -------------------------------------------------------------------------
    #  Parameters
    # -------------------------------------------------------------------------
    def set_param(self, param_name: str, param_value: Any):
        super().set_param(param_name, param_value)
        self._feature_names = [f"x{i}" for i in range(self.p)]
        self._target_names = [f"y{i}" for i in range(self.n)]

    # -------------------------------------------------------------------------
    #  Model structure
    # -------------------------------------------------------------------------
    @abstractmethod
    def _get_layer_sizes(self) -> List[int]:
        """
        Determines layer sizes as list of integers.
        """
        raise NotImplementedError()

    def _get_activation(self):
        """Returns activation function to be used as class instance; e.g. SELU()"""
        if self.activation == "relu":
            return ReLU(inplace=True)
        elif self.activation == "elu":
            return ELU(inplace=True)
        else:
            return SELU(inplace=True)

    # -------------------------------------------------------------------------
    #  Train
    # -------------------------------------------------------------------------
    def _fit_tabulated(self, x: np.ndarray, y: np.ndarray):

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
            emb_szs=[], n_cont=self.p, out_sz=self.n, layers=self._get_layer_sizes(), act_cls=self._get_activation()
        )

        # construct learner
        learner = TabularLearner(data, model, metrics=rmse, wd=self.wd)
        self._nn = learner

        # --- train ---------------------------------------
        plot_style_matplotlib_default()

        # determine optimal learning rates
        try:

            lr_max = learner.lr_find(start_lr=1e-6, end_lr=1e3, suggest_funcs=[valley, minimum], show_plot=False)

            if self.lr_max_method == "valley":
                lr_max = lr_max.valley
            elif self.lr_max_method == "intermediate":
                lr_max = math.sqrt(lr_max.valley * lr_max.minimum)
            elif self.lr_max_method == "minimum":
                lr_max = lr_max.minimum
            elif self.lr_max_method == "aggressive":
                lr_max = 2 * lr_max.minimum
            else:
                raise NotImplementedError(f"lr_max_method='{self.lr_max_method}' not implemented.")

            if self.show_progress:
                print(f"  lr_max: {lr_max:.3e}  [{self.lr_max_method.upper()}]")

        except Exception as e:

            print(e)

            print("------====== lr_find failed --> falling back to lr_max=1e-3 ======------")
            lr_max = 1e-3

            if self.show_progress:
                print(f"lr_max: {lr_max:.3e}  [{self.lr_max_method.upper()}; fallback]")

        # add tqdm callback
        add_tqdm_callback(learner, enabled=self.show_progress)
        learner.fit_one_cycle(self.n_epochs, lr_max=lr_max)
        remove_tqdm_callback(learner)

    # -------------------------------------------------------------------------
    #  Predict
    # -------------------------------------------------------------------------
    def _predict_tabulated(self, x: np.ndarray) -> np.ndarray:

        # construct DataLoader with features
        #  (see: https://docs.fast.ai/tutorial.tabular.html)
        df_features = pd.DataFrame(data=x, columns=self._feature_names)
        dl = self._nn.dls.test_dl(df_features)

        # generate predictions
        pred, _ = self._nn.get_preds(dl=dl)

        # return as numpy array
        return np.array(pred)

    # -------------------------------------------------------------------------
    #  Training metrics etc...
    # -------------------------------------------------------------------------
    def get_training_losses(self) -> List[float]:
        return [float(x) for x in self._nn.recorder.losses]

    def get_learning_rates(self) -> List[float]:
        return [float(x) for x in self._nn.recorder.lrs]
