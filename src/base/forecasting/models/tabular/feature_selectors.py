from __future__ import annotations

import sys
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.tools.math import exp_spaced_indices_fixed_max

LARGE_INTEGER = sys.maxsize


# =================================================================================================
#  Base Class
# =================================================================================================
class FeatureSelector(TransformerMixin, BaseEstimator):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, selected_indices: List[int]):
        self.selected_indices = selected_indices

    @property
    def n_features(self) -> int:
        return len(self.selected_indices)

    @property
    def first_index(self) -> int:
        return min(self.selected_indices)

    @property
    def last_index(self) -> int:
        return max(self.selected_indices)

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray = None, **fit_params):
        # only keep indices that fit within in # of columns in x
        self.selected_indices = [i for i in self.selected_indices if i < x.shape[1]]
        return self

    def transform(self, x: np.ndarray, **transform_params) -> np.ndarray:
        return x[:, self.selected_indices]

    # -------------------------------------------------------------------------
    #  Misc
    # -------------------------------------------------------------------------
    def __str__(self):
        if (self.last_index - self.first_index) == (self.n_features - 1):
            if self.first_index == 0:
                return f"first {self.n_features}"
            else:
                return f"all in [{self.first_index}, {self.last_index}]"
        else:
            return f"{self.n_features} in [{self.first_index}, {self.last_index}]"

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    def _get_feature_indices(self, n_features_total: int):
        """
        Returns indices of the features that are selected by this feature selector, based on the total number of
        features presented to this transformers input.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    #  Sorting
    # -------------------------------------------------------------------------
    def __gt__(self, other) -> bool:
        return isinstance(other, self.__class__) and (self.n_features, tuple(self.selected_indices)) > (
            other.n_features,
            tuple(other.selected_indices),
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and (self.selected_indices == other.selected_indices)

    def __hash__(self):
        return hash(tuple(self.selected_indices))

    # -------------------------------------------------------------------------
    #  Factor methods
    # -------------------------------------------------------------------------
    @classmethod
    def first(cls, n: int) -> FeatureSelector:
        return FeatureSelector(list(range(n)))

    @classmethod
    def range(cls, first_index: int, last_index: int) -> FeatureSelector:
        return FeatureSelector(list(range(first_index, last_index + 1)))

    @classmethod
    def exp_spaced(cls, first_index: int, last_index: int, n_features: int) -> FeatureSelector:
        return FeatureSelector(
            [first_index + i for i in exp_spaced_indices_fixed_max(n_features, max_index=last_index - first_index)]
        )
