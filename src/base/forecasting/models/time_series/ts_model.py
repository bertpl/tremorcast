from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from src.base.forecasting.evaluation.cross_validation import CV_METADATA_PARAM, CVMetaData


# =================================================================================================
#  BASE CLASS
# =================================================================================================
class TimeSeriesModel(ABC, BaseEstimator):
    """
    Abstract class implementing an sklearn-like fit/predict interface for time series forecasting.

    The following requirements need to hold for child classes:
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
    def __init__(self, name: str, **kwargs):
        """
        Constructor of TimeSeriesForecastModel class.
        :param name: (str) type/name of the model
        """
        self.name = name

        # other hyper-parameters
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

        # internal
        from .helpers import TimeSeriesCrossValidation

        self._cv = TimeSeriesCrossValidation(self)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params[CV_METADATA_PARAM] = None  # make sure CV_METADATA_PARAM is recognized as a valid parameter
        return params

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def cv(self):
        """Return TimeSeriesCrossValidation object that can perform grid search CV on this model."""
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
    #  SIMULATION
    # -------------------------------------------------------------------------
    def batch_predict(
        self,
        x: np.ndarray,
        first_sample: int,
        hor: int,
        overlap_end: bool = False,
        stride: int = 1,
        silent: bool = True,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Performs a batch of predictions on a single dataset with fixed horizon and regularly spaced initial samples.

        :param x: (np.ndarray, 1D) containing the entire time series for which we want to perform batch predictions.
        :param first_sample: (int) The sample at which the first prediction will start
        :param hor: (int) number of samples to be forecast each time.
        :param overlap_end: (bool, default=False) if False, the horizon is shortened for the last predictions, to not
                              extend beyond the provided dataset.
        :param stride: (int) number of samples between each prediction.
        :param silent: (bool) if True no output or progress bars
        :return: list of (initial_sample, forecast)-tuples, of type (int, np.ndarray)
        """

        forecasts = []  # type: List[Tuple[int, np.ndarray]]

        for i in tqdm(
            range(first_sample, x.size, stride),
            desc=f"Evaluating model '{self.name}'".ljust(60),
            file=sys.stdout,
            disable=silent,
            leave=False,
        ):

            forecasts.append((i, self.predict(x_hist=x[0:i], hor=hor if overlap_end else min(hor, x.size - i))))

        return forecasts

    # -------------------------------------------------------------------------
    #  INTERFACE - Fit & Predict
    # -------------------------------------------------------------------------
    @abstractmethod
    def min_hist(self) -> int:
        """Return minimal number of samples of history needed to start making forecasts."""
        pass

    @abstractmethod
    def fit(self, x: np.ndarray):
        """
        Train forecast model on provided time series.

        :param x: (np.ndarray, 1D) containing training data.
        """
        pass

    @abstractmethod
    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:
        """
        Predict the next n samples given the time series history (=most recent samples).
        :param x_hist: (np.ndarray, 1D) most recent data of the time series
        :param hor: (int) number of samples we need to predict.
        :return: 1D numpy array of length 'hor' with forecasts.
        """
        pass


# =================================================================================================
#  Base model with automatic normalization
# =================================================================================================
class TimeSeriesModelNormalized(TimeSeriesModel):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, **kwargs):
        """
        Constructor of TimeSeriesForecastModelNormalized class.
        :param name: (str) type/name of the model
        """
        super().__init__(name, **kwargs)
        self._mean = 0.0
        self._std = 1.0

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray):
        self._mean = x.mean()
        self._std = x.std()
        self._fit_normalized((x - self._mean) / self._std)

    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:
        return self._mean + (
            self._std * self._predict_normalized(x_hist_norm=(x_hist - self._mean) / self._std, hor=hor)
        )

    # -------------------------------------------------------------------------
    #  Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def _fit_normalized(self, x_norm: np.ndarray):
        """
        Train forecast model on normalized data.
        :param x_norm: (np.ndarray, 1D) containing normalized training data.
        """
        pass

    def _predict_normalized(self, x_hist_norm: np.ndarray, hor: int) -> np.ndarray:
        """
        Predict the next n samples given the normalized time series history (=most recent samples).
        :param x_hist: (np.ndarray, 1D) most recent data of the time series
        :param hor: (int) number of samples we need to predict.
        :return: 1D numpy array of length 'hor' with forecasts.
        """
        pass
