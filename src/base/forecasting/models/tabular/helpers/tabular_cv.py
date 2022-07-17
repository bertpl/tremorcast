from __future__ import annotations

import datetime
from typing import List, Optional, Union

import numpy as np
from joblib import parallel_backend
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
#  GridSearch Cross-Validation
# =================================================================================================
class TabularCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, regressor: "TabularRegressor"):
        from src.base.forecasting.models.tabular.tabular_regressor import TabularRegressor

        self.regressor = regressor  # type: TabularRegressor
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
        param_set_list = materialize_param_grid(
            param_grid, shuffle=True, add_meta_info=True, encapsulate_param_values_in_list=True
        )

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
            cv_result.val_metrics.all = val_metrics
            cv_result.fit_times.all = fit_times

            cv_result.update_stats()

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
