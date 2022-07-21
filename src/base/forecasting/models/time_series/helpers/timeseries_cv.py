from __future__ import annotations

import itertools
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from tqdm.auto import tqdm

from src.base.forecasting.evaluation.cross_validation import (
    CVResult,
    CVResults,
    TimeSeriesCVSplitter,
    materialize_param_grid,
)
from src.base.forecasting.evaluation.cross_validation.cv_results import CVMetricResult
from src.base.forecasting.evaluation.metrics import TabularMetric, TimeSeriesMetric, compute_metric_vs_lead_time


# =================================================================================================
#  Misc helper classes
# =================================================================================================
class ValidationPredictions:
    # class that encapsulates a set of validation predictions, i.e. a list of
    # (actual, pred)-tuples, each of which are 1d np arrays of equal length.

    # -------------------------------------------------------------------------
    #  Basic list-like interface
    # -------------------------------------------------------------------------
    def __init__(self, predictions: List[Tuple[np.ndarray, np.ndarray]]):
        # takes a list of (actual, pred)-tuples
        self._predictions = predictions

    def items(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for actual, pred in self._predictions:
            yield actual, pred

    def __add__(self, other: ValidationPredictions) -> ValidationPredictions:
        return ValidationPredictions(self._predictions + list(other.items()))

    def __len__(self):
        return len(self._predictions)

    # -------------------------------------------------------------------------
    #  Validation
    # -------------------------------------------------------------------------
    def get_metric_value(self, metric: TimeSeriesMetric) -> float:
        return metric.compute(self._predictions)

    def get_metric_curve(self, tabular_metric: TabularMetric) -> np.ndarray:
        return compute_metric_vs_lead_time(self._predictions, tabular_metric)


# =================================================================================================
#  CVResult(s) sub-classes
# =================================================================================================
class TimeSeriesCVMetricResult(CVMetricResult):
    """
    This class adds functionality to the CVMetricResult parent class, in that it also
    keeps track of the ValidationPredictions of each split, to facilitate computation of the actual
    metric & the overall metric.
    """

    # -------------------------------------------------------------------------
    #  Override parent class behavior
    # -------------------------------------------------------------------------
    def __init__(self, metric: TimeSeriesMetric, n_splits: int):
        super().__init__(metric, n_splits)
        self._metric = metric  # type: TimeSeriesMetric  # override parent class type hint
        self.val_preds = [None] * n_splits  # type: List[Optional[ValidationPredictions]]

    def compute_overall(self):
        # compute overall metric, based on the overall ValidationPredictions data
        self.overall = self.overall_val_preds().get_metric_value(self._metric)

    # -------------------------------------------------------------------------
    #  Custom methods
    # -------------------------------------------------------------------------
    def set_result(self, i_split: int, val_preds: ValidationPredictions):
        self.val_preds[i_split] = val_preds
        self.all[i_split] = val_preds.get_metric_value(self._metric)

    def overall_val_preds(self) -> Optional[ValidationPredictions]:
        # returns 1 ValidationPredictions object containing all validation predictions of all n splits.
        overall_val_preds = None
        for val_pred in [vp for vp in self.val_preds if vp is not None]:
            overall_val_preds = val_pred if overall_val_preds is None else overall_val_preds + val_pred
        return overall_val_preds


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


class TimeSeriesCVResults(CVResults):
    """
    Like the regular CVResults class, but the individual results are now of type TimeSeriesCVResult.
    """

    metric: TimeSeriesMetric
    all_results: List[TimeSeriesCVResult]

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
        self.results = dict()  # type: Dict[TimeSeriesMetric, TimeSeriesCVResults]

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        x: np.ndarray,
        param_grid: Union[dict, List[dict]],
        metric: Union[TimeSeriesMetric, List[TimeSeriesMetric]],
        ts_cv_splitter: TimeSeriesCVSplitter,
        hor: int,
        retrain: bool = True,
        n_jobs: int = -1,
    ):

        # --- argument handling ---------------------------
        if isinstance(metric, list):
            metrics = metric
        else:
            metrics = [metric]

        # --- init ----------------------------------------
        param_sets = materialize_param_grid(
            param_grid=param_grid, shuffle=False, add_meta_info=True, encapsulate_param_values_in_list=False
        )
        cv_splits = ts_cv_splitter.get_splits(x.size)

        all_experiments = [
            (i_param_set, i_split, param_set, n_train, n_val)
            for (i_param_set, param_set), (i_split, (n_train, n_val)) in itertools.product(
                enumerate(param_sets), enumerate(cv_splits)
            )
        ]
        random.shuffle(all_experiments)

        # --- initiate joblib -----------------------------
        pre_dispatch = 2 * n_jobs  # same as GridSearchCV default

        # using the 'multiprocessing' backend instead of the standard 'loky' backend, makes it such that
        # output still appears in Jupyter notebooks.
        # https://stackoverflow.com/questions/55955330/printed-output-not-displayed-when-using-joblib-in-jupyter-notebook
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, backend="multiprocessing")

        # --- main cross-validation loop ------------------
        base_ts_model = clone(self.ts_model)
        with parallel:

            tqdm_desc = f"Grid Search over {len(param_sets)} candidates using {len(cv_splits)}-fold Cross-Validation"

            out = parallel(
                delayed(fit_and_evaluate)(
                    clone(base_ts_model), param_set, x[:n_train], x[n_train : n_train + n_val], metrics, hor
                )
                for i_param_set, i_split, param_set, n_train, n_val in tqdm(
                    all_experiments, desc=tqdm_desc, file=sys.stdout
                )
            )  # type: Iterable[Tuple[ValidationPredictions, ValidationPredictions]]

        # --- construct CVResults objects -----------------
        # initialize results
        self.results = dict()

        # reorder results in dict
        results_dict = {
            (i_param_set, i_split): (train_sims, val_sims)
            for (i_param_set, i_split, param_set, n_train, n_val), (train_sims, val_sims) in list(
                zip(all_experiments, out)
            )
        }

        # process all results
        progress = tqdm(
            total=len(metrics) * len(all_experiments),
            desc=f"Processing cross-validation results".ljust(len(tqdm_desc)),
            file=sys.stdout,
        )

        for metric in metrics:

            # create new cv_results object
            cv_results = TimeSeriesCVResults(metric=metric, param_sets=param_sets, n_splits=len(cv_splits))

            # add all info
            for i_param_set in range(len(param_sets)):

                for i_split in range(len(cv_splits)):

                    # extract results
                    train_sims, val_sims = results_dict[i_param_set, i_split]

                    # update CVResult
                    cv_results.all_results[i_param_set].train_metrics.set_result(i_split, train_sims)
                    cv_results.all_results[i_param_set].val_metrics.set_result(i_split, val_sims)
                    cv_results.all_results[i_param_set].fit_times.all[i_split] = 0.0  # not implemented yet

                    # update progress bar
                    progress.update()

                # finalize results for this parameter set
                cv_results.all_results[i_param_set].update_stats()

            progress.close()

            # update best result for this metric
            cv_results.update_best_result()

            # assign to internal dict
            self.results[metric] = cv_results

        # --- set optimal parameters ----------------------
        # select optimal parameters of 1st metric that was provided
        optimal_params = self.results[metrics[0]].best_result.params
        self.ts_model.set_params(optimal_params)

        # --- retrain if needed ---------------------------
        if retrain:
            self.ts_model.fit(x)


# =================================================================================================
#  Helper functions
# =================================================================================================
def fit_and_evaluate(
    model: "TimeSeriesModel",
    param_set: dict,
    x_train: np.ndarray,
    x_val: np.ndarray,
    hor: int,
) -> Tuple[ValidationPredictions, ValidationPredictions]:
    """
    Fits a time series model to provided training data & evaluates on both training & validation data.
    :param model: time series model to be used to fit & validate using the provided parameters
    :param param_set: dictionary with parameters to be set to the provided model
    :param x_train: np.ndarray with training data
    :param x_val: np.ndarray with validation data, which is assumed to chronologically directly follow the training data
    :param hor: horizon to be used when generating evaluation predictions
    :return: tuple of ValidationPredictions  (1 for training data, 1 for validation data)
    """

    # --- parameter handling ------------------------------
    cloned_param_set = dict()
    for k, v in param_set.items():
        cloned_param_set[k] = clone(v, safe=False)

    model.set_params(**cloned_param_set)

    # --- fit model ---------------------------------------
    model.fit(x_train)

    # --- evaluate ----------------------------------------
    train_sims = evaluate(model, x_hist=x_train[: model.min_hist], x_val=x_train[model.min_hist :], hor=hor)
    val_sims = evaluate(model, x_hist=x_train, x_val=x_val, hor=hor)

    # --- return ------------------------------------------
    return train_sims, val_sims


def evaluate(model: "TimeSeriesModel", x_hist: np.ndarray, x_val: np.ndarray, hor: int) -> ValidationPredictions:
    """
    Evaluate a fitted model on a provided time series (x_hist, x_val), resulting in a ValidationPredictions object.
    :param model: fitted time series model
    :param x_hist: array of past time series values
    :param x_val: array of future time series values for which we want batch predictions and make comparisons
    :param hor: prediction horizon as an integer
    :return: ValidationPredictions object
    """

    # --- obtain batch predictions ------------------------
    x_all = np.concatenate([x_hist, x_val])
    first_sample = x_hist.size
    predictions = model.batch_predict(x_all, first_sample, hor, overlap_end=False, stride=1)

    # --- return as ValidationPredictions object ----------
    return ValidationPredictions([(x_all[i : i + pred.size], pred) for i, pred in predictions])
