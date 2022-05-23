from abc import ABC, abstractmethod
from typing import Iterable, List, Union

import numpy as np


class BaseMetric(ABC):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, eq_values: Iterable = ()):
        # eq_values should contain hashable values that are used in eq and hash
        self.__eq_values = tuple(eq_values)

    # -------------------------------------------------------------------------
    #  Eq & Hash
    # -------------------------------------------------------------------------
    def __eq__(self, other):
        """Equal if of same type and identical __eq_values (set in constructor)"""
        return (type(self) == type(other)) and (self.__eq_values == other.__eq_values)

    def __hash__(self):
        """hash implementation consistent with __eq__"""
        return hash((self.__class__, self.__eq_values))

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert metric into a score >= 0 where higher values are better."""
        pass

    @abstractmethod
    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert score back to metric."""
        pass

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def greater_metric_is_better(self) -> bool:
        """Override in child class if this rudimentary implementation won't work for your metric"""
        return self.metric_to_score(0.2) > self.metric_to_score(0.1)

    def aggregate(self, metric_values: List[float]) -> float:
        # aggregates multiple metric values into 1; to be overridden by child classes if mean() is not appropriate
        return float(np.mean(metric_values))


class ModelFitTime(BaseMetric):
    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # shorter fit time is better; by computing 1/metric we have a proper score that is still >0
        return 1 / metric

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1 / score
