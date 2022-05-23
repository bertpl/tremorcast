from __future__ import annotations

import math
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
    def __init__(self, name: str, tabular_metric: TabularMetric, eq_values=Iterable):
        """
        :param name: (str) name of time series metric
        :param tabular_metric: (TabularMetric) tabular metric to evaluate per-sample errors
                                  to be used for constructing error_curves.  Normally this would be a tabular metric
                                  with greater_metric_is_better() == False, i.e. interpretable as an 'error'.
        :param eq_values: iterable with values to be used in the __eq__ & __hash__ implementations of parent class.
        """
        super().__init__(eq_values=list(eq_values) + [name])
        self.name = name
        self.tabular_metric = tabular_metric

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute(self, metric_curve: np.ndarray) -> float:
        """
        Computes the metric based on the provided prediction results.
        :param metric_curve: (np.ndarray, flat) metric as a function of lead time,
                                                     as computed with embedded tabular metric
        :return: computed metric as a float
        """
        pass

    def metric_curve(self, prediction_results: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        :param prediction_results: prediction results as a list of (actual, forecast)-tuples
                                    where 'actual' & 'forecast' are 1d numpy arrays of identical size
        """
        return compute_metric_vs_lead_time(prediction_results, self.tabular_metric)

    def aggregate_metric_curves(self, curves: List[np.ndarray]) -> np.ndarray:
        """Aggregates metric curves, resulting in 1 curve with length = max curve lengths"""
        return np.array(
            [
                self.tabular_metric.aggregate(metric_values=[curve[i] for curve in curves if i < len(curve)])
                for i in range(max(len(curve) for curve in curves))
            ]
        )

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
        super().__init__("unweighted_metric", tabular_metric, eq_values=[tabular_metric])

    def compute(self, metric_curve: np.ndarray) -> float:
        return self.tabular_metric.aggregate(list(metric_curve))

    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.tabular_metric.metric_to_score(metric)

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.tabular_metric.score_to_metric(score)


# =================================================================================================
#  Metric - MAX ACCURATE LEAD TIME
# =================================================================================================
class MaxAccurateLeadTime(TimeSeriesMetric):
    def __init__(self, tabular_metric: TabularMetric, metric_threshold: float):
        super().__init__("max_accurate_lead_time", tabular_metric, eq_values=[tabular_metric, metric_threshold])

        self.score_threshold = self.tabular_metric.metric_to_score(metric_threshold)

    def compute(self, metric_curve: np.ndarray) -> float:
        return compute_max_accurate_lead_time(
            score_curve=self.tabular_metric.metric_to_score(metric_curve), threshold=self.score_threshold
        )

    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return metric

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return score


# =================================================================================================
#  Metric - Area-Under-Curve - LogLog
# =================================================================================================
class AreaUnderCurveLogLog(TimeSeriesMetric):
    def __init__(self, tabular_metric: TabularMetric):
        super().__init__("area_under_curve_log_log", tabular_metric, eq_values=[tabular_metric])

    def compute(self, metric_curve: np.ndarray) -> float:
        # compute area under the metric curve when plotted on a log-log axis (base-2)
        weights = [math.log2(i + 2) - math.log2(i + 1) for i in range(metric_curve.size)]
        return sum([math.log2(metric_value) * weights[i] for i, metric_value in enumerate(metric_curve)])

    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.tabular_metric.metric_to_score(metric)

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.tabular_metric.score_to_metric(score)


# =================================================================================================
#  Helpers
# =================================================================================================
def compute_max_accurate_lead_time(score_curve: np.ndarray, threshold: float) -> float:
    """
    Computes 'Maximum Accurate Lead Time', expressed in # of samples, by evaluating for how many samples
    the score curve stays above the threshold.

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


def compute_deltas_vs_lead_time(prediction_results: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:

    # --- init ----------------------------------------
    deltas_dict = defaultdict(list)  # type: Dict[int, List[float]]

    # --- compute deltas by lead time -----------------
    for actual, forecast in prediction_results:
        deltas = forecast - actual
        for i, delta in enumerate(deltas):
            deltas_dict[i].append(delta)

    # --- return in right format ----------------------
    return [np.array(deltas_dict[lead_time]) for lead_time in sorted(deltas_dict.keys())]


def compute_metric_vs_lead_time(
    prediction_results: List[Tuple[np.ndarray, np.ndarray]], tabular_metric: TabularMetric
) -> np.ndarray:
    return np.array([tabular_metric.compute(deltas) for deltas in compute_deltas_vs_lead_time(prediction_results)])
