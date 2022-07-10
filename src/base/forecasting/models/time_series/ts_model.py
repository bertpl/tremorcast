from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from src.base.forecasting.evaluation.cross_validation import (
    CV_METADATA_PARAM,
    CVMetaData,
    CVResult,
    CVResults,
    materialize_param_grid,
)
from src.base.forecasting.evaluation.metrics import TimeSeriesMetric


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
    def __init__(self, name: str):
        """
        Constructor of TimeSeriesForecastModel class.
        :param name: (str) type/name of the model
        """
        self.name = name

        # internal
        self._cv = TimeSeriesCrossValidation(self)

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


# =================================================================================================
#  Cross-Validation
# =================================================================================================
@dataclass
class TimeSeriesCVSplitter:

    min_samples_train: int
    min_samples_validate: int
    n_splits: int = 10

    def get_splits(self, n_samples_tot: int) -> List[Tuple[int, int]]:
        # returns a list of length n_splits containing (n_samples_train, n_samples_val)-tuples
        # consistent with requirements and available data

        # total number of samples that we can use in the validation sets
        n_samples_validation_tot = n_samples_tot - self.min_samples_train

        if n_samples_validation_tot >= self.n_splits * self.min_samples_validate:
            # validation sets do not overlap and are >= min_samples_validate

            val_set_sizes = np.diff(np.round(np.linspace(0, n_samples_tot - self.min_samples_train, self.n_splits + 1)))
            return [
                (int(self.min_samples_train + sum(val_set_sizes[:i])), int(val_set_sizes[i]))
                for i in range(self.n_splits)
            ]

        else:
            # validation sets overlap and are == min_samples_validate

            n_train_samples = np.round(
                np.linspace(self.min_samples_train, n_samples_tot - self.min_samples_validate, self.n_splits)
            )

            if len(set(n_train_samples)) < self.n_splits:
                raise ValueError(f"Cannot generate {self.n_splits} unique splits for given settings; not enough data.")

            return [(int(n_train), self.min_samples_validate) for n_train in n_train_samples]


class TimeSeriesCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, ts_model: TimeSeriesModel):
        self.ts_model = ts_model
        self.results = None  # type: Optional[CVResults]

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        x: np.ndarray,
        param_grid: Union[dict, List[dict]],
        metric: TimeSeriesMetric,
        ts_cv_splitter: TimeSeriesCVSplitter,
        n_jobs: int = -1,
    ):
        pass
