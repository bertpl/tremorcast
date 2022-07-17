from __future__ import annotations

import datetime
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from joblib import parallel_backend
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold

from src.base.forecasting.evaluation.cross_validation import (
    CV_METADATA_PARAM,
    CVMetaData,
    CVResult,
    CVResults,
    materialize_param_grid,
)
from src.base.forecasting.evaluation.metrics.tabular_metrics import TabularMetric
from src.tools.datetime import estimate_eta, format_datetime, format_timedelta
from src.tools.math import remove_nan_rows
from src.tools.progress import ProgressTimer


# =================================================================================================
#  Base Class
# =================================================================================================
class TabularRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible regression class that can e.g. be used inside a GridSearchCV.

    The main purpose of this class is to provide its own cross-validation wrapper around GridSearchCV with
        following additional features:
            - progress indication with time estimates
            - randomization of grid search to remove bias of time estimates
            - convenience class for fetching & inspecting cross-validation results

    Other benefits are the ability to wrap regressors of other packages (fast.ai) in a sklearn-compatible container.

    NOTE: sklearn's GridSearchCV performs the following tasks at the beginning of each experiment
           - params = estimator.get_params()
           - update some parameters in 'params' corresponding to grid search
           - create new_estimator as new instance of estimator's class, providing **params to the constructor
           - check via new_estimator.get_params() to see if params are correctly set

          our own grid search then sets optimal parameters on this instance using the set_params() method

          AS A RESULT, the following requirements need to hold for child classes:
            - hyper-parameters need to be passable to the constructor
            - hyper-parameters need to be stored in identically named attributes
                     (because that's how get_params() gets its parameter values)
            - constructors of child classes should accept **kwargs to be passed on to superclass constructor
                     (because that's how we manage to sneak in additional parameters such as CV_METADATA_PARAM)
            - set_params() needs to behave consistently
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, remove_nans_before_fit: bool = True, **kwargs):
        self.name = name
        self.remove_nans_before_fit = remove_nans_before_fit
        self._is_fitted = False

        # other hyper-parameters
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

        # internal
        self._cv = TabularCrossValidation(self)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params[CV_METADATA_PARAM] = None  # make sure CV_METADATA_PARAM is recognized as a valid parameter
        return params

    def get_tunable_param_names(self) -> Set[str]:
        """Returns subset of parameters that are actually tunable, i.e. excluding e.g. CV_METADATA_PARAM"""
        param_names = set(self.get_params().keys())
        return param_names.difference([CV_METADATA_PARAM])

    def set_params(self, **params) -> TabularRegressor:
        return super().set_params(**params)

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> TabularRegressor:
        """Fit model based on (m, n_inputs) array x and (m, n_outputs) array y."""

        # fix flattened matrices
        if y.ndim == 1:
            y = y.reshape((y.size, 1))

        timer = ProgressTimer()
        if self.cv_active():
            self.cv.pre_fit_progress(self.get_cv_metadata())

        if self.remove_nans_before_fit:
            x, y = remove_nan_rows(x, y)

        self._fit(x, y, **fit_params)
        self._is_fitted = True

        if self.cv_active():
            self.cv.post_fit_progress(self.get_cv_metadata(), timer.sec_elapsed())

        return self

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def cv(self):
        """Return TabularCrossValidation object that can perform grid search CV on this model."""
        return self._cv

    def get_cv_metadata(self) -> Optional[CVMetaData]:
        return getattr(self, CV_METADATA_PARAM, None)

    def cv_active(self) -> bool:
        """True if this instance is being used inside a CV grid search"""
        return self.get_cv_metadata() is not None

    @property
    def show_progress(self) -> bool:
        return not self.cv_active()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _remove_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        rows_without_nan = [not (any(x_row) or any(y_row)) for x_row, y_row in zip(np.isnan(x), np.isnan(y))]

        x = x[rows_without_nan]
        y = y[rows_without_nan]

        return x, y

    # -------------------------------------------------------------------------
    #  Abstract Methods
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray, **fit_params):
        """Fit model based on (m, n_inputs) array x and (m, n_outputs) array y."""
        raise NotImplementedError()

    def predict(self, x: np.ndarray, **predict_params) -> np.ndarray:
        """Return (m, n_outputs) array y based on (m, n_inputs) array x."""
        raise NotImplementedError()


# =================================================================================================
#  GridSearch Cross-Validation
# =================================================================================================
class TabularCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, regressor: TabularRegressor):
        self.regressor = regressor
        self.results = None  # type: Optional[CVResults]

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param_grid: Union[dict, List[dict]],
        metric: TabularMetric,
        n_splits: int = 10,
        shuffle_data: bool = False,
        n_jobs: int = -1,
    ):
        """Use the sklearn class GridSearchCV to perform cross-validated grid-search over parameters."""

        # --- remove NaNs ---------------------------------
        x, y = remove_nan_rows(x, y)

        # --- perform grid search CV ----------------------
        param_set_list = materialize_param_grid(param_grid, shuffle=True, add_meta_info=True)

        print("-" * 80)
        print(f" Grid Search over {len(param_set_list)} candidates using {n_splits}-fold Cross-Validation.")
        print("-" * 80)

        with parallel_backend("multiprocessing"):
            # using the 'multiprocessing' backend instead of the standard 'loky' backend, makes it such that
            # output still appears in Jupyter notebooks.
            # https://stackoverflow.com/questions/55955330/printed-output-not-displayed-when-using-joblib-in-jupyter-notebook

            grid_search = GridSearchCV(
                estimator=self.regressor,
                param_grid=param_set_list,
                scoring=metric.get_sklearn_scorer(),
                n_jobs=n_jobs,
                cv=KFold(n_splits=n_splits, shuffle=shuffle_data),
                verbose=0,
                refit=False,  # we refit ourselves on the 'regressor' instance of this class
                return_train_score=True,
            )

            timer = ProgressTimer()

            grid_search.fit(x, y)  # run actual grid-search

            print("-" * 80)
            print(f"Total computation time: {format_timedelta(timer.sec_elapsed())}.")

        # --- extract results -----------------------------
        # the param_sets in param_set_list always have their param_values inside a list of just 1 element;
        #   this is purely for GridSearchCV, which expects it that way, but CVResults does not expect this.
        param_set_list = [
            {param_name: param_values[0] for param_name, param_values in param_set.items()}
            for param_set in param_set_list
        ]

        self.results = CVResults(metric, param_set_list, n_splits)

        # populate results object
        for i_param_set, param_set in enumerate(param_set_list):
            del param_set[CV_METADATA_PARAM]  # internal metadata; not needed

            # extract metrics & fit times
            train_metrics = [
                metric.score_to_metric(grid_search.cv_results_[f"split{i}_train_score"][i_param_set])
                for i in range(n_splits)
            ]
            val_metrics = [
                metric.score_to_metric(grid_search.cv_results_[f"split{i}_test_score"][i_param_set])
                for i in range(n_splits)
            ]

            # GridSearchCV does not return individual fit times; so we just duplicate the mean n_splits times
            fit_times = [float(grid_search.cv_results_["mean_fit_time"][i_param_set])] * n_splits

            cv_result = CVResult(metric, param_set, n_splits)

            cv_result.train_metrics.all = train_metrics
            cv_result.train_metrics.compute_overall()

            cv_result.val_metrics.all = val_metrics
            cv_result.val_metrics.compute_overall()

            cv_result.fit_times.all = fit_times
            cv_result.fit_times.compute_overall()

            self.results.all_results[i_param_set] = cv_result

        self.results.update_best_result()

        # --- show result ---------------------------------
        self.results.show_optimal_results()

        # --- transfer params to regressor & refit --------
        tunable_param_names = set(self.regressor.get_tunable_param_names())
        params_to_be_set = {
            param_name: param_value
            for param_name, param_value in grid_search.best_params_.items()
            if param_name in tunable_param_names
        }
        self.regressor.set_params(**params_to_be_set)

        # refit 'regressor' on full data after having transferred optimal parameters
        self.regressor.fit(x, y)
        print("-" * 80)

    # -------------------------------------------------------------------------
    #  Progress reporting
    # -------------------------------------------------------------------------
    @staticmethod
    def pre_fit_progress(cv_metadata: CVMetaData):
        print(
            f"[{format_datetime(datetime.datetime.now())}] "
            + f"[{cv_metadata.i_param_set+1: >4}/{cv_metadata.n_param_sets: <4}] START ".ljust(120, ".")
        )

    @staticmethod
    def post_fit_progress(cv_metadata: CVMetaData, time_elapsed: float):

        eta_dt, eta_secs = estimate_eta(
            start_time=cv_metadata.start_time,
            work_fraction_done=(cv_metadata.i_param_set + 0.5) / cv_metadata.n_param_sets,
        )

        print(
            f"[{format_datetime(datetime.datetime.now())}] "
            + (
                f"[{cv_metadata.i_param_set + 1: >4}/{cv_metadata.n_param_sets: <4}] END ".ljust(25, ".")
                + f" [fit: {format_timedelta(time_elapsed): <6}] "
            ).ljust(60, ".")
            + f" [eta: {format_timedelta(eta_secs).ljust(8)} -->   {format_datetime(eta_dt)}]".rjust(60, ".")
        )
