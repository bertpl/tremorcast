from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from src.base.forecasting.evaluation.cross_validation import CV_METADATA_PARAM, CVMetaData
from src.tools.math import remove_nan_rows
from src.tools.progress import ProgressTimer


# =================================================================================================
#  Base Class
# =================================================================================================
class TabularRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible regression class that can e.g. be used inside a GridSearchCV.

    The main purpose of this class is to provide its own cross-validation wrapper around GridSearchCV with
        following additional features:
            - progress indication with time estimates
            - randomization of grid search to remove bias of time estimates
            - convenience class for fetching & inspecting cross-validation results

    Other benefits are the ability to wrap regressors of other packages (fast.ai) in a sklearn-compatible container.

    NOTE: sklearn's GridSearchCV performs the following tasks at the beginning of each experiment
           - params = estimator.get_params()
           - update some parameters in 'params' corresponding to grid search
           - create new_estimator as new instance of estimator's class, providing **params to the constructor
           - check via new_estimator.get_params() to see if params are correctly set

          our own grid search then sets optimal parameters on this instance using the set_params() method

          AS A RESULT, the following requirements need to hold for child classes:
            - hyper-parameters need to be passable to the constructor
            - hyper-parameters need to be stored in identically named attributes
                     (because that's how get_params() gets its parameter values)
            - constructors of child classes should accept **kwargs to be passed on to superclass constructor
                     (because that's how we manage to sneak in additional parameters such as CV_METADATA_PARAM)
            - set_params() needs to behave consistently with how the constructor handles parameters
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, remove_nans_before_fit: bool = True, **kwargs):
        self.name = name
        self.remove_nans_before_fit = remove_nans_before_fit
        self._is_fitted = False

        # other hyper-parameters
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

        # internal
        from .helpers import TabularCrossValidation

        self._cv = TabularCrossValidation(self)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params[CV_METADATA_PARAM] = None  # make sure CV_METADATA_PARAM is recognized as a valid parameter
        return params

    def get_tunable_param_names(self) -> Set[str]:
        """Returns subset of parameters that are actually tunable, i.e. excluding e.g. CV_METADATA_PARAM"""
        param_names = set(self.get_params().keys())
        return param_names.difference([CV_METADATA_PARAM])

    def set_params(self, **params) -> TabularRegressor:
        return super().set_params(**params)

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> TabularRegressor:
        """Fit model based on (m, n_inputs) array x and (m, n_outputs) array y."""

        # fix flattened matrices
        if y.ndim == 1:
            y = y.reshape((y.size, 1))

        timer = ProgressTimer()
        if self.cv_active():
            self.cv.pre_fit_progress(self.get_cv_metadata())

        if self.remove_nans_before_fit:
            x, y = remove_nan_rows(x, y)

        self._fit(x, y, **fit_params)
        self._is_fitted = True

        if self.cv_active():
            self.cv.post_fit_progress(self.get_cv_metadata(), timer.sec_elapsed())

        return self

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def cv(self):
        """Return TabularCrossValidation object that can perform grid search CV on this model."""
        return self._cv

    def get_cv_metadata(self) -> Optional[CVMetaData]:
        return getattr(self, CV_METADATA_PARAM, None)

    def cv_active(self) -> bool:
        """True if this instance is being used inside a CV grid search"""
        return self.get_cv_metadata() is not None

    @property
    def show_progress(self) -> bool:
        return not self.cv_active()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _remove_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        rows_without_nan = [not (any(x_row) or any(y_row)) for x_row, y_row in zip(np.isnan(x), np.isnan(y))]

        x = x[rows_without_nan]
        y = y[rows_without_nan]

        return x, y

    # -------------------------------------------------------------------------
    #  Abstract Methods
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray, **fit_params):
        """Fit model based on (m, n_inputs) array x and (m, n_outputs) array y."""
        raise NotImplementedError()

    def predict(self, x: np.ndarray, **predict_params) -> np.ndarray:
        """Return (m, n_outputs) array y based on (m, n_inputs) array x."""
        raise NotImplementedError()
