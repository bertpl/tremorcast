from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

CV_METADATA_PARAM = "cv_metadata"


# =================================================================================================
#  Base Class
# =================================================================================================
class TabularRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible regression class that can e.g. be used inside a GridSearchCV.

    Requisites for child classes:
      - hyper-parameters need to be passable to the constructor
      - hyper-parameters need to be stored in identically named attributes
      - constructors of child classes should accept **kwargs to be passed on to superclass constructor
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, n_inputs: int, n_outputs: int, **kwargs):
        self.name = name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._is_fitted = False

        self.show_progress = False

        # other hyper-parameters
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

        # contains results of cross-validation when grid_search_cv has been called.
        self._grid_search_cv = TabularCrossValidation(self)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params[CV_METADATA_PARAM] = None
        return params

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit model based on (m, n_inputs) array x and (m, n_outputs) array y."""
        self.cv.pre_fit_progress()  # only when CV is active

        self._fit(x, y)
        self._is_fitted = True

        self.cv.post_fit_progress()  # only when CV is active

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    # -------------------------------------------------------------------------
    #  Predict
    # -------------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return (m, n_outputs) array y based on (m, n_inputs) array x."""
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def cv(self):
        """Return TabularCrossValidation object that can perform grid search CV on this model."""
        return self._grid_search_cv

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray):
        """Fit model based on (m, n_inputs) array x and (m, n_outputs) array y."""
        raise NotImplementedError()

    @staticmethod
    def _remove_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        rows_without_nan = [not (any(x_row) or any(y_row)) for x_row, y_row in zip(np.isnan(x), np.isnan(y))]

        x = x[rows_without_nan]
        y = y[rows_without_nan]

        return x, y


# =================================================================================================
#  GridSearch Cross-Validation
# =================================================================================================
class ScoreMetric(Enum):
    MSE = auto()
    MAE = auto()

    def get_scorer(self):
        if self == ScoreMetric.MSE:
            return make_scorer(mean_squared_error, greater_is_better=False)
        elif self == ScoreMetric.MAE:
            return make_scorer(mean_absolute_error, greater_is_better=False)
        else:
            raise NotImplementedError(f"get_scorer() not implemented for '{self}'.")

    @staticmethod
    def score_to_metric(score: float):
        # for all current metrics, the score function is the negative of the metric,
        #   to make sure higher is better.
        return -score

    @staticmethod
    def metric_to_score(metric: float):
        return -metric


@dataclass
class CVMetaData:
    start_time: datetime.datetime
    i_param_set: int
    n_param_sets: int


@dataclass
class CVResults:

    # --- nested class ------------------------------------
    @dataclass
    class CVResult:

        params: dict

        train_metrics: List[float]
        train_metric_mean: float
        train_metric_std: float

        val_metrics: List[float]
        val_metric_mean: float
        val_metric_std: float

        fit_time_mean: float
        fit_time_std: float

    # --- members -----------------------------------------
    score_metric: ScoreMetric
    best_result: Optional[CVResult]
    all_results: List[CVResult]

    # --- helper functions --------------------------------
    def update_best_result(self):
        """Updates the best_result member based on all_results."""
        for cv_result in self.all_results:
            if (self.best_result is None) or (
                self.score_metric.metric_to_score(cv_result.val_metric_mean)
                > self.score_metric.metric_to_score(self.best_result.val_metric_mean)
            ):

                self.best_result = cv_result

    def all_param_values(self) -> Dict:
        """Returns dictionary with all param_names & corresponding values in this cv_result."""
        return {
            param_name: sorted({cv_result.params[param_name] for cv_result in self.all_results})
            for param_name in sorted(
                {param_name for cv_result in self.all_results for param_name in cv_result.params.keys()}
            )
        }

    def filter(self, param_filter: dict = None) -> CVResults:

        # --- create new subset of all_results ------------
        filtered_results = self.all_results.copy()
        for param_name, param_value in param_filter.items():
            filtered_results = [
                cv_result for cv_result in filtered_results if cv_result.params.get(param_name) == param_value
            ]

        # --- create new CVResults object -----------------
        new_cv_results = CVResults(score_metric=self.score_metric, best_result=None, all_results=filtered_results)
        new_cv_results.update_best_result()

        # --- return --------------------------------------
        return new_cv_results

    def sweep_by_filter(self, param_name: str, param_filter: dict = None) -> Tuple[List, List, List, List, List]:
        """
        Returns a 1D sweep across the results, where...
          - param_name determines the x-value of the sweep  (independent variable)
          - param_filter, when specified, determines which subset of results to consider
        If multiple results are obtain for the same value of param_name, we take the best one.
        :param param_filter:
        :param param_name:
        :return: (param_values, train_metric_mean, train_metric_std, val_metric_mean, val_metric_std)-tuple
        """

        # --- get all parameter values --------------------
        param_values = self.all_param_values()[param_name]

        # --- fetch results -------------------------------
        train_metric_mean = []
        train_metric_std = []
        val_metric_mean = []
        val_metric_std = []
        for param_value in param_values:

            # filter results by provided filter + this specific param value
            param_filter[param_name] = param_value
            filtered_results = self.filter(param_filter)

            # extract info from best result
            train_metric_mean.append(filtered_results.best_result.train_metric_mean)
            train_metric_std.append(filtered_results.best_result.train_metric_std)
            val_metric_mean.append(filtered_results.best_result.val_metric_mean)
            val_metric_std.append(filtered_results.best_result.val_metric_std)

        # --- return result -------------------------------
        return param_values, train_metric_mean, train_metric_std, val_metric_mean, val_metric_std


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
        score_metric: ScoreMetric,
        n_jobs: int = 1,
        n_splits: int = 5,
        shuffle_data: bool = False,
    ):
        """Use the sklearn class GridSearchCV to perform cross-validated grid-search over parameters."""

        # --- perform grid search CV ----------------------
        param_set_list = self._materialize_param_grid(param_grid)

        print("-" * 80)
        print(
            f" Grid Search over {len(param_set_list)} parameter candidates " f"using {n_splits}-fold Cross-Validation."
        )
        print("-" * 80)

        grid_search = GridSearchCV(
            estimator=self.regressor,
            param_grid=param_set_list,
            scoring=score_metric.get_scorer(),
            n_jobs=n_jobs,
            cv=KFold(n_splits=n_splits, shuffle=shuffle_data),
            verbose=0,
            refit=False,  # we refit ourselves on the 'regressor' instance of this class
            return_train_score=True,
        )
        grid_search.fit(x, y)

        # --- transfer params to regressor & refit --------
        print("-" * 80)
        print("Grid search optimal parameters:")
        for param_name, param_value in grid_search.best_params_.items():
            if param_name == CV_METADATA_PARAM:
                setattr(self.regressor, CV_METADATA_PARAM, None)
            else:
                print(f"  {param_name}: {param_value}")
                setattr(self.regressor, param_name, param_value)
        print("-" * 80)

        self.regressor.fit(x, y)

        # --- extract results -----------------------------
        self.results = CVResults(score_metric, None, [])

        for i, param_set in enumerate(param_set_list):
            del param_set[CV_METADATA_PARAM]  # internal metadata; not needed

            train_metrics = [
                score_metric.score_to_metric(grid_search.cv_results_[f"split{i}_train_score"]) for i in range(n_splits)
            ]
            val_metrics = [
                score_metric.score_to_metric(grid_search.cv_results_[f"split{i}_test_score"]) for i in range(n_splits)
            ]

            self.results.all_results.append(
                CVResults.CVResult(
                    params=param_set,
                    train_metrics=train_metrics,
                    train_metric_mean=float(np.mean(train_metrics)),
                    train_metric_std=float(np.std(train_metrics)),
                    val_metrics=val_metrics,
                    val_metric_mean=float(np.mean(val_metrics)),
                    val_metric_std=float(np.std(val_metrics)),
                    fit_time_mean=grid_search.cv_results_["mean_fit_time"][i],
                    fit_time_std=grid_search.cv_results_["std_fit_time"][i],
                )
            )

        self.results.update_best_result()

    # -------------------------------------------------------------------------
    #  Progress reporting
    # -------------------------------------------------------------------------
    def pre_fit_progress(self):
        if (
            hasattr(self.regressor, CV_METADATA_PARAM)
            and (cv_metadata := getattr(self.regressor, CV_METADATA_PARAM)) is not None
        ):
            print(
                f"[{self.format_datetime(datetime.datetime.now())}] "
                + f"[param set {cv_metadata.i_param_set+1}/{cv_metadata.n_param_sets}] START ".ljust(120, ".")
            )

    def post_fit_progress(self):
        if (
            hasattr(self.regressor, CV_METADATA_PARAM)
            and (cv_metadata := getattr(self.regressor, CV_METADATA_PARAM)) is not None
        ):

            eta_secs, eta_dt = self.compute_eta(
                cv_metadata.i_param_set, cv_metadata.n_param_sets, cv_metadata.start_time
            )

            print(
                f"[{self.format_datetime(datetime.datetime.now())}] "
                + f"[param set {cv_metadata.i_param_set+1}/{cv_metadata.n_param_sets}] END ".ljust(60, ".")
                + f"[{self.format_timedelta(eta_secs).ljust(8)} -->   {self.format_datetime(eta_dt)}]".rjust(60, ".")
            )

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _materialize_param_grid(param_grid: Union[Dict, List[Dict]], shuffle: bool = True) -> List[Dict]:
        """Materializes a GridSearchCV param_grid into a list of single-param-set dicts with cv-meta-info attached"""

        # --- argument handling ---------------------------
        if isinstance(param_grid, dict):
            param_grid = [param_grid]

        # --- materialize ---------------------------------
        param_set_list = []
        for one_grid in param_grid:
            param_set_list.extend(
                [
                    {param_name: [param_value] for param_name, param_value in zip(one_grid.keys(), param_values)}
                    for param_values in product(*one_grid.values())
                ]
            )

        # --- shuffle -------------------------------------
        if shuffle:
            # mainly intended to make sure slow-fitting vs fast-fitting parameter sets are evenly spread out,
            #  so our time estimates are more reliable and less skewed.
            random.shuffle(param_set_list)

        # --- add meta-info -------------------------------
        now = datetime.datetime.now()
        n_param_sets = len(param_set_list)
        for i_param_set, param_set in enumerate(param_set_list):
            param_set[CV_METADATA_PARAM] = [CVMetaData(now, i_param_set, n_param_sets)]

        # --- return --------------------------------------
        return param_set_list

    @staticmethod
    def compute_eta(
        i_param_set: int, n_param_sets: int, start_time: datetime.datetime
    ) -> Tuple[float, datetime.datetime]:
        """Return ETA in seconds_to_go and estimated finishing datetime."""

        iters_done = i_param_set + 0.5  # on avg. half of current parameter set will be done
        iters_todo = n_param_sets - (i_param_set + 0.5)  # on avg. half still to do

        sec_elapsed = (datetime.datetime.now() - start_time).total_seconds()
        sec_togo = (sec_elapsed / iters_done) * iters_todo

        return sec_togo, datetime.datetime.now() + datetime.timedelta(seconds=sec_togo)

    @staticmethod
    def format_datetime(dt: datetime.datetime) -> str:
        return dt.strftime("%a - %Y-%m-%d - %H:%M:%S")

    @classmethod
    def format_timedelta(cls, total_seconds: float) -> str:
        if total_seconds:
            d, h, m, s = cls._split_sec(total_seconds)
            if d + h + m == 0:
                if s < 10:
                    return f"{s:.2f}s"
                else:
                    return f"{s:.1f}s"
            elif d + h == 0:
                return f"{m}m{s:.0f}s"
            elif d == 0:
                return f"{h}h{m}m{s:.0f}s"
            else:
                return f"{d}d{h}h{m}m{s:.0f}s"
        else:
            return "???"

    @staticmethod
    def _split_sec(sec: float) -> Tuple[int, int, int, float]:
        """split total seconds in (days_int, hours_int, minutes_int, secs_float)"""
        d = int(sec // (24 * 60 * 60))
        sec -= d * 24 * 60 * 60

        h = int(sec // (60 * 60))
        sec -= h * 60 * 60

        m = int(sec // 60)
        sec -= m * 60

        return d, h, m, sec
