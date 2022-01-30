import math
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.base.forecasting.evaluation.cross_validation import (
    ParamSelectionMethod,
    param_grid_dict_to_list,
    select_params,
    split_cv_data,
)

from .helpers import build_toeplitz
from .ts_model import TimeSeriesForecastModelAutoScaled


# =================================================================================================
#  N-Step-Ahead regression - Base Class
# =================================================================================================
class TimeSeriesModelMultiStepRegression(TimeSeriesForecastModelAutoScaled):
    """
    This class implements an n-step-ahead timeseries regression model.
       n-step-ahead: we forecast samples 0, ... n-1
       regression:   we use the past p samples as features for the model.

    These choices make it such that at the core the problem reduces to a tabulated regression model
      with p features an n targets.

    The child class needs to implement the tabulated regression problem (fit & predict).
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self, model_type: str, signal_name: str, p: int, n: int, avoid_training_nans: bool = False, cv: dict = None
    ):
        """
        Constructor of the multi-step regression model class.
        :param model_type: (str) model type string
        :param signal_name: (str) name of the signal to be predicted.
        :param p (int): number of lags to use for forecasting (= number of tabulated features)
        :param n (int): how many steps ahead we're predicting (= number of tabulated targets)
        :param avoid_training_nans: (bool, default=False) When True, the tabulated data set provided to
                                     the fit_tabulated method will not contain NaNs, i.e. those rows where either
                                     features or targets contain NaNs will be removed.  This can be handy for those
                                     child classes that have 1 global fit-method.  For those child classes that fit
                                     separate sub-models for each target, it is more convenient to set this flag to
                                     False and keep NaNs in the data set, in order to avoid filtering out rows that
                                     could still contain useful data for some sub-models.
        :param cv: (dict, optional): dictionary with cross-validation settings if CV is desired.
                    'n_splits': (int, default=10) number of splits in k-fold cross-validation
                    'randomize': (bool, default=True) if true, data is randomized before splitting in train & val. sets
                    'param_grid': definition of parameter grid.
                        SYNTAX 1:  dict mapping param_name -> values_list
                        SYNTAX 2:  list of dicts mapping param_name -> value
                    'loss': loss function to be used to evaluate model performance
                              should be a callable  f(y_pred, y_actual) -> float
        """
        self.p = p
        self.n = n
        self._avoid_training_nans = avoid_training_nans
        super().__init__(model_type, signal_name)

        # cross-validation
        self.cv_settings = cv
        self.cv_results = dict()

    # -------------------------------------------------------------------------
    #  Set parameters
    # -------------------------------------------------------------------------
    def set_param(self, param_name: str, param_value: Any):
        """Sets parameter value; default value is simply attribute assignment to self; can be overridden / extended
        in child classes."""
        setattr(self, param_name, param_value)

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def _fit(self, scaled_training_data: pd.DataFrame):

        # --- build tabulated dataset ---------------------
        ts = scaled_training_data[self.signal_name].to_numpy()
        x = self.__build_features(ts)
        y = self.__build_targets(ts)

        # --- remove NaN rows, if needed ------------------
        if self._avoid_training_nans:
            x, y = self._remove_nan(x, y)

        # --- cross-validation ----------------------------
        self._cross_validation(x, y)

        # --- fit model on tabulated dataset --------------
        self._fit_tabulated(x, y)

    def _predict(self, scaled_history: pd.DataFrame, n_samples: int) -> np.ndarray:

        # --- convert to numpy array ----------------------
        history = scaled_history[self.signal_name].to_numpy()
        history = history.reshape((1, len(history)))

        # --- predict -------------------------------------
        n_iterations = math.ceil(n_samples / self.n)
        predictions = np.zeros(n_iterations * self.n)

        for i in range(n_iterations):

            x = np.fliplr(history[:, -self.p :])
            y = self._predict_tabulated(x)

            predictions[i * self.n : (i + 1) * self.n] = y.flatten()
            history = np.concatenate([history, y], axis=1)

        # --- return --------------------------------------
        return predictions[0:n_samples].flatten()

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    def _cross_validation(self, x: np.ndarray, y: np.ndarray):
        """
        Perform cross-validation based on cv settings set in constructor, if set.

        Optimal hyper-parameter values will be set to the model and detailed results
          of cross-validation will be set in self.cv_results.

        :param x: (2d numpy array) tabulated features
        :param y: (2d numpy array) tabulated targets
        """

        # --- extract parameters --------------------------
        if not self.cv_settings:
            return

        n_splits = self.cv_settings.get("n_splits", 5)
        randomize = self.cv_settings.get("randomize", True)
        param_grid = self.cv_settings["param_grid"]
        loss_function = self.cv_settings["loss"]
        selection_method = self.cv_settings.get("selection_method", ParamSelectionMethod.BALANCED)

        # --- process parameter grid ----------------------
        if isinstance(param_grid, dict):
            param_grid = param_grid_dict_to_list(param_grid)  # type: List[Dict[str, Any]]

        # --- cross-validation ----------------------------
        all_cv_results = []
        for params_dict in param_grid:

            # set parameter values
            for param_name, param_value in params_dict.items():
                self.set_param(param_name, param_value)

            # go over all splits
            training_losses = []
            validation_losses = []
            for i_split in range(n_splits):

                # split in training & validation data
                x_train, x_val = split_cv_data(x, n_splits, i_split, randomize)
                y_train, y_val = split_cv_data(y, n_splits, i_split, randomize)

                # train
                self._fit_tabulated(x_train, y_train)

                # evaluate
                training_losses.append(loss_function(y_train, self._predict_tabulated(x_train)))
                validation_losses.append(loss_function(y_val, self._predict_tabulated(x_val)))

            # summarize results
            all_cv_results.append(
                dict(
                    params=params_dict,
                    complexity=self._model_complexity(),
                    training_losses=dict(
                        all=training_losses,
                        mean=np.mean(training_losses),
                        std=np.std(training_losses),
                    ),
                    validation_losses=dict(
                        all=validation_losses,
                        mean=np.mean(validation_losses),
                        std=np.std(validation_losses),
                    ),
                )
            )

        # --- optimal parameter selection -----------------
        best_params_by_method = {method: select_params(all_cv_results, method) for method in ParamSelectionMethod}

        selected_params = best_params_by_method[selection_method]
        for param_name, param_value in selected_params["params"].items():
            self.set_param(param_name, param_value)

        # --- remember all results ------------------------
        self.cv_results = dict(
            all=all_cv_results, best_params_by_method=best_params_by_method, selected_params=selected_params
        )

    def _model_complexity(self) -> float:
        """
        Should return an indicator for model complexity, based on the values of the model hyperparameters.
        This should allow cross-validation to choose the simplest well-performing model.
        """
        return 1

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def _fit_tabulated(self, x: np.ndarray, y: np.ndarray):
        """
        Fits the regression model f() such that y = f(x).
        :param x: (np.ndarray) (k x p) array containing the tabulated features.
        :param y: (np.ndarray) (k x n) array containing the tabulated targets
        """
        pass

    @abstractmethod
    def _predict_tabulated(self, x: np.ndarray) -> np.ndarray:
        """
        Produces a prediction y_hat = f(x).
        :param x: (np.ndarray) (k x p) array containing the tabulated features.
        :return: (k x n) array containing the predictions.
        """
        pass

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _remove_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        rows_without_nan = [not (any(x_row) or any(y_row)) for x_row, y_row in zip(np.isnan(x), np.isnan(y))]

        x = x[rows_without_nan]
        y = y[rows_without_nan]

        return x, y

    def __build_features(self, ts: np.ndarray) -> np.ndarray:
        # build Toeplitz matrix containing samples 1,...,p in the past,
        #  so first build Toeplitz of width p+1, containing samples 0,...,p in the past and omit column 0.
        return build_toeplitz(ts, window_size=self.p + 1, forward=False)[:, 1:]

    def __build_targets(self, ts: np.ndarray) -> np.ndarray:
        # build Toeplitz matrix containing samples 0,...,n-1 in the future
        return build_toeplitz(ts, window_size=self.n, forward=True)


# =================================================================================================
#  Loss functions for cross-validation
# =================================================================================================
def loss_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def loss_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred).flatten())
