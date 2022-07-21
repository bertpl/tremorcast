from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

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
    def max_accurate_lead_time(cls, tabular_metric: TabularMetric, threshold: float) -> MaxAccurateLeadTime:
        return MaxAccurateLeadTime(tabular_metric, threshold)


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
#  Metric - MAX ACCURATE LEAD TIME
# =================================================================================================
class MaxAccurateLeadTime(TimeSeriesMetric):
    def __init__(self, tabular_metric: TabularMetric, threshold: float):
        super().__init__(f"max_accurate_lead_time", eq_values=[tabular_metric])

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

        # compute max accurate lead time
        return compute_max_accurate_lead_time(score_vs_lead_time, threshold=self.score_threshold)

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


# =================================================================================================
#  Helpers
# =================================================================================================
def compute_max_accurate_lead_time(score_curve: np.ndarray, threshold: float) -> float:
    """
    Computes 'Maximum Accurate Lead Time', expressed in # of samples, by evaluating for how many samples
    the error curve does not exceed the threshold.

    The result is returned as a float, by interpolation between the first sample exceeding the threshold and the
    sample before.

    :param score_curve: (1D numpy array) containing the score curve, where the first value represents the score of
                             forecasting 1 sample ahead.  Scores are within range [0,1] and higher is always better.
    :param threshold: (float >= 0)
    :return: Computed metric expressed in number of samples, value between 1 and len(score_curve)+1, except for the
                following corner cases:
                  1) if score_curve[0] < threshold: a value between 0 and 1 is returned
                  2) if all(score_curve > threshold): np.inf is returned
    """

    # --- corner case 2 -----------------------------------
    if all(score_curve > threshold):
        return np.inf

    # --- corner case 1 -----------------------------------
    if score_curve[0] <= threshold:
        # return value in [0, 1]
        return abs(threshold) / (abs(threshold) + abs(score_curve[0] - threshold))

    # --- regular case ------------------------------------

    # we are guaranteed to find one (as we're not in corner case 2) + i_first will not be 0 (which is corner case 1)
    i_first = next(i for i, score in enumerate(score_curve) if score <= threshold)

    # interpolate between score curve values i_first-1 and i_first to find intersection point with threshold
    return 1 + np.interp(x=threshold, xp=[score_curve[i_first], score_curve[i_first - 1]], fp=[i_first, i_first - 1])
