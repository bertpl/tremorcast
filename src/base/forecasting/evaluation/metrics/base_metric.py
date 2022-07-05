from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class BaseMetric(ABC):
    @abstractmethod
    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert metric into a score >= 0 where higher values are better."""
        pass

    @abstractmethod
    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert score back to metric."""
        pass

    def greater_metric_is_better(self) -> bool:
        """Override in child class if this rudimentary implementation won't work for your metric"""
        return self.metric_to_score(0.2) > self.metric_to_score(0.1)
