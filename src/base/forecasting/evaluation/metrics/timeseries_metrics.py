from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

from src.base.forecasting.evaluation.helpers import compute_maximum_reliable_lead_time

from .base_metric import BaseMetric
from .tabular_metrics import TabularMetric


# =================================================================================================
#  Abstract base class
# =================================================================================================
class TimeSeriesMetric(BaseMetric):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, eq_values=Iterable):
        super().__init__(eq_values=list(eq_values) + [name])
        self.name = name

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute(self, prediction_results: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Computes the metric based on the provided prediction results.
        :param prediction_results: prediction results as a list of (actual, forecast)-tuples
                                    where 'actual' & 'forecast' are 1d numpy arrays of identical size
        :return: computed metric as a float
        """
        pass

    # -------------------------------------------------------------------------
    #  Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def unweighted_error(cls, tabular_metric: TabularMetric) -> UnweightedError:
        return UnweightedError(tabular_metric)

    @classmethod
    def max_reliable_lead_time(cls, tabular_metric: TabularMetric, threshold: float) -> MaxReliableLeadTime:
        return MaxReliableLeadTime(tabular_metric, threshold)


# =================================================================================================
#  Metric - UNWEIGHTED TABULAR METRIC
# =================================================================================================
class UnweightedError(TimeSeriesMetric):
    def __init__(self, tabular_metric: TabularMetric):
        super().__init__("unweighted_metric", eq_values=[tabular_metric])

        self.tabular_metric = tabular_metric

    def compute(self, prediction_results: List[Tuple[np.ndarray, np.ndarray]]) -> float:

        return self.tabular_metric.compute(
            np.array(
                [
                    forecast - actual
                    for actual_array, forecast_array in prediction_results
                    for actual, forecast in zip(actual_array, forecast_array)
                ]
            )
        )

    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.tabular_metric.metric_to_score(metric)

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.tabular_metric.score_to_metric(score)


# =================================================================================================
#  Metric - MAX RELIABLE LEAD TIME
# =================================================================================================
class MaxReliableLeadTime(TimeSeriesMetric):
    def __init__(self, tabular_metric: TabularMetric, threshold: float):
        super().__init__(f"max_reliable_lead_time", eq_values=[tabular_metric])

        self.tabular_metric = tabular_metric
        self.score_threshold = self.tabular_metric.metric_to_score(threshold)

    def compute(self, prediction_results: List[Tuple[np.ndarray, np.ndarray]]) -> float:

        # compute score vs lead time
        score_vs_lead_time = self.tabular_metric.metric_to_score(
            np.array(
                [
                    self.tabular_metric.compute(errors)
                    for errors in self._compute_errors_vs_lead_time(prediction_results)
                ]
            )
        )

        # compute max reliable lead time
        return compute_maximum_reliable_lead_time(score_vs_lead_time, threshold=self.score_threshold)

    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return metric

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return score

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_errors_vs_lead_time(prediction_results: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:

        # --- init ----------------------------------------
        errors_dict = defaultdict(list)  # type: Dict[int, List[float]]

        # --- compute errors by lead time -----------------
        for actual, forecast in prediction_results:
            errors = forecast - actual
            for i, error in enumerate(errors):
                errors_dict[i].append(error)

        # --- return in right format ----------------------
        return [np.array(errors_dict[lead_time]) for lead_time in sorted(errors_dict.keys())]
