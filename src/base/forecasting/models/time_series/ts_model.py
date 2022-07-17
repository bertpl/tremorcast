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
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, min_hist: int):
        """
        Constructor of TimeSeriesForecastModel class.
        :param name: (str) type/name of the model
        :param min_hist: (int) minimum history in samples for the model to work properly, used when
                               generating validation results using validate()
        """
        self.name = name
        self.min_hist = min_hist

        # internal
        from .helpers import TimeSeriesCrossValidation

        self._cv = TimeSeriesCrossValidation(self)

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
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Performs a batch of predictions on a single dataset with fixed horizon and regularly spaced initial samples.

        :param x: (np.ndarray, 1D) containing the entire time series for which we want to perform batch predictions.
        :param first_sample: (int) The sample at which the first prediction will start
        :param hor: (int) number of samples to be forecast each time.
        :param overlap_end: (bool, default=False) if False, the horizon is shortened for the last predictions, to not
                              extend beyond the provided dataset.
        :param stride: (int) number of samples between each prediction.
        :return: list of (initial_sample, forecast)-tuples, of type (int, np.ndarray)
        """

        forecasts = []  # type: List[Tuple[int, np.ndarray]]

        for i in tqdm(
            range(first_sample, x.size, stride),
            desc=f"Evaluating model '{self.name}'".ljust(60),
            file=sys.stdout,
        ):

            forecasts.append((i, self.predict(x_hist=x[0:i], hor=hor if overlap_end else min(hor, x.size - i))))

        return forecasts

    # -------------------------------------------------------------------------
    #  INTERFACE - Fit & Predict
    # -------------------------------------------------------------------------
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
        Predict the next n samples given the time series history (=most recent samples).  Child classes should
        be able to deal with histories that are shorter than they would ideally need.
        :param x_hist: (np.ndarray, 1D) most recent data of the time series
        :param hor: (int) number of samples we need to predict.
        :return: 1D numpy array of length 'hor' with forecasts.
        """
        pass
