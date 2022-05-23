# =================================================================================================
#  OVERVIEW OF CLASSES:
#
#  TimeSeriesCrossValidation    -->  like the GridSearchCV of sklearn, but better & for time series ;-)
#  -------------------------
#    .results = {
#       TimeSeriesMetric -> TimeSeriesCVResults
#    }
#
#
#  TimeSeriesCVResults          --> cross-validation results over all parameters sets, for 1 metric
#  -------------------
#    .all_results = List[TimeSeriesCVResult]
#    .best_result = TimeSeriesCVResults
#
#
#  TimeSeriesCVResult           --> cross-validation results for 1 parameter set
#  ------------------
#    .train_metric = TimeSeriesCVMetricResult
#    .val_metrics = TimeSeriesCVMetricResult
#
#
#  TimeSeriesCVMetricResult     --> validation results over all splits for 1 parameter set (either train or test)
#  ------------------------
#    .all = List[float]     (list of metric values for each split)
#    .overall = float       (overall metric over all splits; not necessarily the mean)
#    .metric_curves = List[np.ndarray]      (timeseries-specific addition; metric curves as a function of lead time)
#
# =================================================================================================
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.base import clone

from src.base.forecasting.evaluation.cross_validation import CVResult, CVResults, TimeSeriesCVSplitter
from src.base.forecasting.evaluation.cross_validation.cv_results import CVMetricResult
from src.base.forecasting.evaluation.metrics import TimeSeriesMetric
from src.base.optimization import GridSearch, InformedSearch, LineSearch, ParallelOptimizer, RandomSearch, Scheduler


# =================================================================================================
#  CVResult(s) sub-classes
# =================================================================================================
class TimeSeriesCVMetricResult(CVMetricResult):
    """
    This class adds functionality to the CVMetricResult parent class, in that it also
    keeps track of the metric curve of each split, to facilitate computation of the actual
    metric & the overall metric.
    """

    # -------------------------------------------------------------------------
    #  Override parent class behavior
    # -------------------------------------------------------------------------
    def __init__(self, metric: TimeSeriesMetric, n_splits: int):
        super().__init__(metric, n_splits)
        self._metric = metric  # type: TimeSeriesMetric  # override parent class type hint
        self.metric_curves = [None] * n_splits  # type: List[Optional[np.ndarray]]

    def compute_overall(self):
        # compute overall metric, based on the overall metric curve
        self.overall = self._metric.compute(self.overall_metric_curve())

    # -------------------------------------------------------------------------
    #  Custom methods
    # -------------------------------------------------------------------------
    def set_result(self, i_split: int, metric_curve: np.ndarray):
        self.metric_curves[i_split] = metric_curve
        self.all[i_split] = self._metric.compute(metric_curve)

    def overall_metric_curve(self) -> np.ndarray:
        return self._metric.aggregate_metric_curves(self.metric_curves)


class TimeSeriesCVResult(CVResult):
    """
    Like the regular CVResult class, but the train_metrics & val_metrics fields are now
    of type TimeSeriesCVMetricResult
    """

    def __init__(self, metric: TimeSeriesMetric, params: dict, n_splits: int):
        super().__init__(metric, params, n_splits)

        self.metric = metric  # type: TimeSeriesMetric  # override parent class type hint

        # train & val metrics are of TimeSeries-specific class, for the additional meta-info
        #   which allows us to compute more reliable / accurate overall values of the metrics
        self.train_metrics = TimeSeriesCVMetricResult(metric, n_splits)
        self.val_metrics = TimeSeriesCVMetricResult(metric, n_splits)

    def __float__(self) -> float:
        """Used in ParallelOptimizer to determine which result is better; it assumes lower is better, hence the -sign"""
        return -self.metric.metric_to_score(self.val_metrics.overall)


class TimeSeriesCVResults(CVResults):
    """
    Like the regular CVResults class, but the individual results are now of type TimeSeriesCVResult.
    """

    metric: TimeSeriesMetric
    all_results: List[TimeSeriesCVResult]
    best_result: TimeSeriesCVResult

    def __init__(self, metric: TimeSeriesMetric, param_sets: List[dict], n_splits: int):
        super().__init__(metric, param_sets, n_splits)
        self.metric = metric  # type: TimeSeriesMetric  # override parent class type hint

    def _init_results(self, param_sets: List[dict]) -> List[TimeSeriesCVResult]:
        return [TimeSeriesCVResult(self.metric, param_set, self.n_splits) for param_set in param_sets]


# =================================================================================================
#  Cross-Validation
# =================================================================================================
class TimeSeriesCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, ts_model: "TimeSeriesModel"):
        from src.base.forecasting.models.time_series.ts_model import TimeSeriesModel

        self.ts_model = ts_model  # type: TimeSeriesModel
        self.results = None  # type: Optional[TimeSeriesCVResults]

    # -------------------------------------------------------------------------
    #  General Cross-Validation
    # -------------------------------------------------------------------------
    def _cross_validate(
        self,
        scheduler: Scheduler,
        x: np.ndarray,
        param_names: List[str],
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
        n_jobs: int = -1,
        param_validator: Callable[[Dict], bool] = None,
    ):

        # --- init ----------------------------------------
        optimizer = ParallelOptimizer(n_workers=n_jobs)
        context = CrossValidateContext(
            model=clone(self.ts_model),
            x=x,
            ts_cv_splitter=ts_cv_splitter,
            metric=metric,
            hor=hor,
            param_names=param_names,
            param_validator=param_validator,
        )

        # --- run optimization ----------------------------
        optimizer.optimize(objective=cross_validate_ts_model, scheduler=scheduler, fixed_param=context)

        # --- extract results -----------------------------
        self.results = TimeSeriesCVResults(metric=metric, param_sets=[], n_splits=ts_cv_splitter.n_splits)
        self.results.all_results = list(optimizer.results.values())
        self.results.update_best_result()

        # --- set optimal parameters ----------------------
        optimal_params = self.results.best_result.params
        self.ts_model.set_params(**optimal_params)

        # --- retrain if needed ---------------------------
        if retrain:
            self.ts_model.fit(x)

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        x: np.ndarray,
        param_grid: dict,
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
        n_jobs: int = -1,
        param_validator: Callable[[Dict], bool] = None,
    ):

        scheduler = GridSearch(param_grid=param_grid, shuffle=True)
        self._cross_validate(
            scheduler, x, list(param_grid.keys()), metric, ts_cv_splitter, hor, retrain, n_jobs, param_validator
        )

    def random_search(
        self,
        x: np.ndarray,
        param_grid: dict,
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
        n_jobs: int = -1,
        param_validator: Callable[[Dict], bool] = None,
        max_iter: int = None,
        max_seconds: float = None,
    ):

        scheduler = RandomSearch(param_grid=param_grid, max_iter=max_iter, max_seconds=max_seconds)
        self._cross_validate(
            scheduler, x, list(param_grid.keys()), metric, ts_cv_splitter, hor, retrain, n_jobs, param_validator
        )

    def line_search(
        self,
        x: np.ndarray,
        param_grid: dict,
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
        n_jobs: int = -1,
        param_validator: Callable[[Dict], bool] = None,
        max_iter: int = None,
        max_seconds: float = None,
        closest_first: bool = True,
        init_iters: int = None,
        hint: dict = None,
    ):

        scheduler = LineSearch(
            param_grid=param_grid,
            max_iter=max_iter,
            max_seconds=max_seconds,
            closest_first=closest_first,
            init_iters=init_iters,
            hint=hint,
        )
        self._cross_validate(
            scheduler, x, list(param_grid.keys()), metric, ts_cv_splitter, hor, retrain, n_jobs, param_validator
        )

    def informed_search(
        self,
        x: np.ndarray,
        param_grid: dict,
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
        n_jobs: int = -1,
        max_iter: int = None,
        max_seconds: float = None,
        min_focus: float = 0.0,
        max_focus: float = 5.0,
        focus_exponent: float = 1.0,
    ):

        scheduler = InformedSearch(
            param_grid=param_grid,
            max_iter=max_iter,
            max_seconds=max_seconds,
            min_focus=min_focus,
            max_focus=max_focus,
            focus_exponent=focus_exponent,
        )
        self._cross_validate(scheduler, x, list(param_grid.keys()), metric, ts_cv_splitter, hor, retrain, n_jobs)

    # -------------------------------------------------------------------------
    #  Static helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def cross_validate_ts_model(
        model: "TimeSeriesModel",
        x: np.ndarray,
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
    ) -> TimeSeriesCVResult:

        param_set = {param_name: param_value for param_name, param_value in model.get_params().items()}
        context = CrossValidateContext(
            model=clone(model),
            x=x,
            ts_cv_splitter=ts_cv_splitter,
            metric=metric,
            hor=hor,
            param_names=list(param_set.keys()),
        )

        cv_result = cross_validate_ts_model(context, *param_set.values())

        if retrain:
            model.fit(x)

        return cv_result


# =================================================================================================
#  Objective function for ParallelOptimization
# =================================================================================================
@dataclasses.dataclass(frozen=True)
class CrossValidateContext:
    """Contains all variables that stay the same across all parameter sets, but that we need to pass to the
    workers in order to perform cross-validation for this set of parameters"""

    model: "TimeSeriesModel"
    x: np.ndarray
    ts_cv_splitter: TimeSeriesCVSplitter
    metric: TimeSeriesMetric
    hor: int
    param_names: List[str]
    param_validator: Callable[[Dict], bool] = None


def cross_validate_ts_model(context: CrossValidateContext, *args) -> Optional[TimeSeriesCVResult]:

    # --- unpack info -------------------------------------
    param_set = {param_name: value for param_name, value in zip(context.param_names, args)}
    cv_splits = context.ts_cv_splitter.get_splits(context.x.size)
    n_splits = len(cv_splits)  # determines k of k-fold cross-validation

    # --- parameter validation -------------------------------
    if (context.param_validator is not None) and (not context.param_validator(param_set)):
        # if a validator is set and if validation of the current parameters fails -> return None
        return None

    # --- main loop ------------------------------------------
    cv_result = TimeSeriesCVResult(context.metric, param_set, n_splits)
    for i_split, (n_train, n_val) in enumerate(cv_splits):

        # --- create new model ---
        model = clone(context.model)  # type: "TimeSeriesModel"

        # --- set parameters ---
        # copied from sklearn cross-validation code
        if param_set:
            cloned_param_set = dict()
            for k, v in param_set.items():
                cloned_param_set[k] = clone(v, safe=False)
            model.set_params(**cloned_param_set)

        # --- x_train, x_val ---
        x_train = context.x[:n_train]
        x_val = context.x[n_train : n_train + n_val]

        # --- fit ---
        model.fit(x_train)

        # --- evaluate ---
        metric_curve_train = evaluate_ts_model(
            model,
            x_hist=x_train[: model.min_hist()],
            x_val=x_train[model.min_hist() :],
            metric=context.metric,
            hor=context.hor,
        )
        metric_curve_val = evaluate_ts_model(model, x_hist=x_train, x_val=x_val, metric=context.metric, hor=context.hor)

        # --- add to cv result ---
        cv_result.train_metrics.set_result(i_split, metric_curve_train)
        cv_result.val_metrics.set_result(i_split, metric_curve_val)
        cv_result.fit_times.all[i_split] = 0.0  # TODO

    # --- finalize & return -------------------------------
    cv_result.update_stats()
    return cv_result


def evaluate_ts_model(
    model: "TimeSeriesModel",
    x_hist: np.ndarray,
    x_val: np.ndarray,
    metric: TimeSeriesMetric,
    hor: int,
    silent: bool = True,
) -> np.ndarray:
    """
    Evaluate a fitted model on a provided time series (x_hist, x_val), resulting in a np.ndarry representing a metric curve.
    :param model: fitted time series model
    :param x_hist: array of past time series values
    :param x_val: array of future time series values for which we want batch predictions and make comparisons
    :param metric: TimeSeriesMetric to be used to construct metric curves
    :param hor: prediction horizon as an integer
    :param silent: if True no output or progress bars
    :return: (np.ndarray) with metric curve
    """

    # --- obtain batch predictions ------------------------
    x_all = np.concatenate([x_hist, x_val])
    first_sample = x_hist.size
    predictions = model.batch_predict(x_all, first_sample, hor, overlap_end=False, stride=1, silent=silent)

    # --- convert to metric curve & return ----------------
    return metric.metric_curve(prediction_results=[(x_all[i : i + pred.size], pred) for i, pred in predictions])
