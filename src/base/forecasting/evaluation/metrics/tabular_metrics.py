from abc import abstractmethod
from functools import partial
from typing import Callable, List, Union

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

from .base_metric import BaseMetric


# =================================================================================================
#  Abstract base class
# =================================================================================================
class TabularMetric(BaseMetric):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str):
        super().__init__(eq_values=[name])
        self.name = name

    # -------------------------------------------------------------------------
    #  Methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute(self, errors: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_sklearn_scorer(self) -> Callable:
        """Return scorer function that can be used in e.g. scikit-learns GridSearchCV"""
        pass

    # -------------------------------------------------------------------------
    #  Other
    # -------------------------------------------------------------------------
    def metric_to_score(self, metric: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # override this method if this assumption does not hold up; so far it does for all child classes.
        return -metric  # consistent with sklearn implementation

    def score_to_metric(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # override this method if this assumption does not hold up; so far it does for all child classes.
        return -score  # consistent with sklearn implementation

    # -------------------------------------------------------------------------
    #  Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def mae(cls):
        return MAE()

    @classmethod
    def rmse(cls):
        return RMSE()


# =================================================================================================
#  Metric - RMSE
# =================================================================================================
class RMSE(TabularMetric):
    def __init__(self):
        super().__init__("RMSE")

    def compute(self, errors: np.ndarray) -> float:
        errors = errors.flatten()
        return float(np.sqrt(np.mean(errors**2)))

    def get_sklearn_scorer(self) -> Callable:
        return make_scorer(partial(mean_squared_error, squared=False), greater_is_better=False)

    def aggregate(self, metric_values: List[float]) -> float:
        return self.compute(np.array(metric_values))  # use RMSE to aggregate RMSE values


# =================================================================================================
#  Metric - MAE
# =================================================================================================
class MAE(TabularMetric):
    def __init__(self):
        super().__init__("MAE")

    def compute(self, errors: np.ndarray) -> float:
        errors = errors.flatten()
        return float(np.mean(errors))

    def get_sklearn_scorer(self) -> Callable:
        return make_scorer(mean_absolute_error, greater_is_better=False)
