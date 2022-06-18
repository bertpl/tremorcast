from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.tools.math import exp_spaced_indices_fixed_max


# =================================================================================================
#  Base Class
# =================================================================================================
class FeatureSelector(TransformerMixin, BaseEstimator):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self):
        self._selected_indices = []  # type: List[int]

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray):
        self._selected_indices = self._get_feature_indices(n_features_total=x.shape[1])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x[:, self._selected_indices]

    # -------------------------------------------------------------------------
    #  Misc
    # -------------------------------------------------------------------------
    def __str__(self):
        return f"FeatureSelector: {str(self._selected_indices)}"

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    def _get_feature_indices(self, n_features_total: int):
        """
        Returns indices of the features that are selected by this feature selector, based on the total number of
        features presented to this transformers input.
        """
        raise NotImplementedError


# =================================================================================================
#  Child Classes
# =================================================================================================
class FeatureSelector_All(FeatureSelector):
    def _get_feature_indices(self, n_features_total: int):
        return list(range(n_features_total))

    def __str__(self):
        return "FeatureSelector: all"


class FeatureSelector_ExponentialSpacing(FeatureSelector):
    def __init__(self, first_index=None, last_index=None, n_selected_features: int = None, reverse: bool = False):
        super().__init__()

        self.n_selected_features = n_selected_features  # None = all features; should be >0
        self.first_index = first_index  # first index to select  (will always be selected)
        self.last_index = last_index  # last index to select  (will always be selected if n_selected_features > 1)
        self.reverse = reverse  # if True, exponential spacing works top-down instead of bottom-up

    def _get_feature_indices(self, n_features_total: int):

        # --- pre-processing -------------------------------

        # defaults in case of None
        first_index = self.first_index if self.first_index is not None else 0
        last_index = self.last_index if self.last_index is not None else n_features_total - 1
        n_selected_features = (
            self.n_selected_features if self.n_selected_features is not None else last_index - first_index + 1
        )

        # last_index should be in [first_index, n_features_total-1]
        last_index = max(min(last_index, n_features_total - 1), first_index)

        # n_selected_features should be positive and fit in [first_index, last_index]
        n_selected_features = max(1, min(n_selected_features, last_index - first_index + 1))

        # --- compute indices -----------------------------
        if n_selected_features == 1:
            indices = [first_index]
        else:
            indices = [
                first_index + i for i in exp_spaced_indices_fixed_max(n_selected_features, last_index - first_index)
            ]

        # --- post-processing -----------------------------
        if self.reverse:
            indices = [last_index - (i - first_index) for i in reversed(indices)]

        # --- return --------------------------------------
        return indices

    def __str__(self):
        return f"FeatureSelector: {self.n_selected_features} in [{self.first_index}, {self.last_index}]"
