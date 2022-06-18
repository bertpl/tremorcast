import math
import random
from abc import abstractmethod
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from src.base.forecasting.evaluation.cross_validation import (
    ParamSelectionMethod,
    param_grid_dict_to_list,
    select_params,
    split_cv_data,
)
from src.base.forecasting.models.time_series.helpers import build_toeplitz
from src.base.forecasting.models.time_series.ts_model import TimeSeriesForecastModelAutoScaled
from src.tools.progress import ProgressTimer


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
                    'n_processes: (int, default=1) number of parallel processes to use
                    'randomize': (bool, default=True) if True, data is randomized before splitting in train & val. sets
                    'randomize_runs': (bool, default=False) if True, training runs are executed randomly, to improve ETA estimates
                    'param_grid': definition of parameter grid.
                        SYNTAX 1:  dict mapping param_name -> values_list
                        SYNTAX 2:  list of dicts mapping param_name -> value
                    'loss': loss function to be used to evaluate model performance
                              should be a callable  f(y_pred, y_actual) -> float
                    'selection_method': ParamSelectionMethod     (default = BALANCED)
        """
        self.show_progress = True
        self.p = p
        self.n = n
        self._avoid_training_nans = avoid_training_nans
        super().__init__(model_type, signal_name)

        # cross-validation
        self.cv_settings = cv
        self.cv_results = dict()
        self._cv_init()

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

        # --- cross-validation ----------------------------
        self._cross_validation(scaled_training_data)

        # --- fit model on tabulated dataset --------------
        x, y = self._build_tabulated_data(scaled_training_data)
        self._fit_tabulated(x, y)

    def _build_tabulated_data(self, scaled_training_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        # --- convert to tabular data ---------------------
        ts = scaled_training_data[self.signal_name].to_numpy()
        x = self.__build_features(ts)
        y = self.__build_targets(ts)

        # --- remove NaN rows, if needed ------------------
        if self._avoid_training_nans:
            x, y = self._remove_nan(x, y)

        # --- return --------------------------------------
        return x, y

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
    def _cross_validation(self, scaled_training_data: pd.DataFrame):
        """
        Perform cross-validation based on cv settings set in constructor, if set.

        Optimal hyper-parameter values will be set to the model and detailed results
          of cross-validation will be set in self.cv_results.

        :param scaled_training_data: pd.DataFrame with scaled signals in the columns
        """

        # --- extract parameters --------------------------
        if not self.cv_settings:
            return

        n_splits = self.cv_settings["n_splits"]
        n_param_sets = self._cv_n_param_sets()
        randomize = self.cv_settings["randomize"]
        randomize_runs = self.cv_settings["randomize_runs"]
        loss_function = self.cv_settings["loss"]
        n_processes = self.cv_settings["n_processes"]

        # --- cross-validation loop -----------------------
        timer = ProgressTimer(total=n_param_sets * n_splits)

        # determine and training run order
        if randomize_runs:
            cv_scope = [
                (i_param_set, i_split)
                for i_split in range(n_splits)
                for i_param_set in random.sample(range(n_param_sets), n_param_sets)
            ]
        else:
            cv_scope = [(i_param_set, i_split) for i_split in range(n_splits) for i_param_set in range(n_param_sets)]

        if n_processes == 1:
            # --- SINGLE PROCESS ---

            for i_run, (i_param_set, i_split) in enumerate(cv_scope):  # type int, int

                # display progress
                print(
                    f"Cross-Validation: training run {i_run+1}/{len(cv_scope)}:"
                    f" param set {i_param_set+1}/{self._cv_n_param_sets()}, "
                    f"cv-split {i_split+1}/{n_splits}.  [ETA={timer.eta_str()} --> {timer.estimated_end_time_str()}]"
                )

                # set parameter values
                self._cv_activate_param_set(i_param_set)

                # convert to tabulated data & split in (training_data, validation_data)
                x, y = self._build_tabulated_data(scaled_training_data)
                x_train, x_val = split_cv_data(x, n_splits, i_split, randomize)
                y_train, y_val = split_cv_data(y, n_splits, i_split, randomize)

                # train
                self._fit_tabulated(x_train, y_train)

                # evaluate
                training_loss = loss_function(y_train, self._predict_tabulated(x_train))
                validation_loss = loss_function(y_val, self._predict_tabulated(x_val))

                # store result
                self._cv_set_result(i_param_set, i_split, training_loss, validation_loss)

                # update timer
                timer.iter_done(1)

        else:
            # --- MULTI_PROCESSING ---

            self.show_progress = False

            # yields dicts to be passed to
            def training_iterable() -> dict:
                for i_run, (i_param_set, i_split) in enumerate(cv_scope):

                    yield dict(
                        i_run=i_run,
                        i_param_set=i_param_set,
                        i_split=i_split,
                        model=self,
                        params=self._cv_get_param_set(i_param_set),
                        scaled_training_data=scaled_training_data,
                        randomize=randomize,
                        n_splits=n_splits,
                        loss_function=loss_function,
                    )

            with Pool(processes=n_processes) as pool:

                for i_run, i_param_set, i_split, training_loss, validation_loss in pool.imap_unordered(
                    cv_train, training_iterable()
                ):

                    # display progress
                    print(
                        f"Cross-Validation: training run {i_run + 1}/{len(cv_scope)}:"
                        f" param set {i_param_set + 1}/{self._cv_n_param_sets()}, "
                        f"cv-split {i_split + 1}/{n_splits}.  [ETA={timer.eta_str()} --> {timer.estimated_end_time_str()}]"
                    )

                    # store result
                    self._cv_set_result(i_param_set, i_split, training_loss, validation_loss)

                    # update timer
                    timer.iter_done(1)

            self.show_progress = True

        # --- finalize ------------------------------------
        self._cv_finalize_results()

    def _cv_init(self):
        """
        Initializes data structures based on self.cv_settings.  This method is called from the constructor
          after setting the self.cv_settings attribute.

        This makes sure that...
          - param_grid in cv_settings is a list of dicts   (and not a dict of param ranges)
          - n_splits, randomize, randomize_runs, selection_method are set
          - cv_results is initialized.
        """

        if self.cv_settings is not None:

            # --- process param_grid --------------------------
            if isinstance(self.cv_settings["param_grid"], dict):

                # convert dict of param-ranges to list of param-sets
                self.cv_settings["param_grid"] = param_grid_dict_to_list(
                    self.cv_settings["param_grid"]
                )  # type: List[Dict[str, Any]]

            # --- defaults for optional settings ----------
            self.cv_settings["n_splits"] = self.cv_settings.get("n_splits", 5)
            self.cv_settings["n_processes"] = self.cv_settings.get("n_processes", 1)
            self.cv_settings["randomize"] = self.cv_settings.get("randomize", True)
            self.cv_settings["randomize_runs"] = self.cv_settings.get("randomize_runs", False)
            self.cv_settings["selection_method"] = self.cv_settings.get(
                "selection_method", ParamSelectionMethod.BALANCED
            )

            # --- initialize cv_results -------------------
            self.cv_results = dict(
                all=[
                    dict(
                        params=params_dict,
                        complexity=np.nan,  # will be set by _cv_finalize_results
                        training_losses=dict(
                            all=[np.nan] * self.cv_settings["n_splits"],  # info added by _cv_set_result
                            mean=np.nan,  # will be set by _cv_finalize_results
                            std=np.nan,  # will be set by _cv_finalize_results
                        ),
                        validation_losses=dict(
                            all=[np.nan] * self.cv_settings["n_splits"],  # info added by _cv_set_result
                            mean=np.nan,  # will be set by _cv_finalize_results
                            std=np.nan,  # will be set by _cv_finalize_results
                        ),
                    )
                    for params_dict in self._cv_param_grid()
                ],
                best_params_by_method=dict(),  # will be set by _cv_finalize_results
                selected_params=dict(),  # will be set by _cv_finalize_results
            )

    def _cv_param_grid(self) -> List[dict]:
        if self.cv_settings is not None:
            return self.cv_settings["param_grid"]
        else:
            return []

    def _cv_n_param_sets(self) -> int:
        return len(self.cv_settings["param_grid"])

    def _cv_get_param_set(self, i_param_set: int) -> dict:
        return self.cv_settings["param_grid"][i_param_set]

    def _cv_get_result(self, i_param_set: int) -> dict:
        return self.cv_results["all"][i_param_set]

    def _cv_activate_param_set(self, i_param_set: int):
        """Activates the i-th set of parameters."""
        param_set = self._cv_get_param_set(i_param_set)
        for param_name, param_value in param_set.items():
            self.set_param(param_name, param_value)

    def _cv_set_result(self, i_param_set: int, i_split: int, training_loss: float, validation_loss: float):
        """Stores the training & validation loss for the i_split-th split of the i_param_set-th set of parameters."""
        self.cv_results["all"][i_param_set]["training_losses"]["all"][i_split] = training_loss
        self.cv_results["all"][i_param_set]["validation_losses"]["all"][i_split] = validation_loss

    def _cv_finalize_results(self):
        """
        Performs following tasks:
          - Computes mean & std of train & validation losses of all cv results
          - Determines model complexity for all parameter values
          - Determines optimal parameter values
        """

        # --- Compute stats -------------------------------
        for i_param_set in range(self._cv_n_param_sets()):

            # training mean & std
            self.cv_results["all"][i_param_set]["training_losses"]["mean"] = np.mean(
                self.cv_results["all"][i_param_set]["training_losses"]["all"]
            )
            self.cv_results["all"][i_param_set]["training_losses"]["std"] = np.std(
                self.cv_results["all"][i_param_set]["training_losses"]["all"]
            )

            # validation mean & std
            self.cv_results["all"][i_param_set]["validation_losses"]["mean"] = np.mean(
                self.cv_results["all"][i_param_set]["validation_losses"]["all"]
            )
            self.cv_results["all"][i_param_set]["validation_losses"]["std"] = np.std(
                self.cv_results["all"][i_param_set]["validation_losses"]["all"]
            )

        # --- Compute model complexities ------------------
        for i_param_set in range(self._cv_n_param_sets()):
            self._cv_activate_param_set(i_param_set)
            self.cv_results["all"][i_param_set]["complexity"] = self._model_complexity()

        # --- Select optimal parameters -------------------
        best_params_by_method = {
            method: select_params(self.cv_results["all"], method) for method in ParamSelectionMethod
        }
        selected_params = best_params_by_method[self.cv_settings["selection_method"]]

        print("-------------------------------------------------------")
        print(f"Selected parameters: {selected_params}")
        print("-------------------------------------------------------")

        for param_name, param_value in selected_params["params"].items():
            self.set_param(param_name, param_value)

        self.cv_results["best_params_by_method"] = best_params_by_method
        self.cv_results["selected_params"] = select_params

    def _model_complexity(self) -> Union[float, tuple]:
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
    return float(np.sqrt(np.mean(np.power(y_true - y_pred, 2))))


def loss_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred).flatten()))


# =================================================================================================
#  Parallel processing entry-point for cross-validation
# =================================================================================================
def cv_train(data_dict: dict) -> Tuple[int, int, int, float, float]:

    # extract values from data_dict
    i_run = data_dict["i_run"]
    i_split = data_dict["i_split"]
    i_param_set = data_dict["i_param_set"]
    model = data_dict["model"]  # type: TimeSeriesModelMultiStepRegression
    params = data_dict["params"]
    scaled_training_data = data_dict["scaled_training_data"]
    loss_function = data_dict["loss_function"]
    randomize = data_dict["randomize"]
    n_splits = data_dict["n_splits"]

    # set parameters
    for param_name, value in params.items():
        model.set_param(param_name, value)

    # create in training & validation sets
    x, y = model._build_tabulated_data(scaled_training_data)
    x_train, x_val = split_cv_data(x, n_splits, i_split, randomize)
    y_train, y_val = split_cv_data(y, n_splits, i_split, randomize)

    # train model
    model._fit_tabulated(x_train, y_train)

    # evaluate
    training_loss = loss_function(y_train, model._predict_tabulated(x_train))
    validation_loss = loss_function(y_val, model._predict_tabulated(x_val))

    # return
    return i_run, i_param_set, i_split, training_loss, validation_loss
