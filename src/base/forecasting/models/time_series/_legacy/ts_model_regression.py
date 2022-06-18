from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.base.forecasting.models.tabular.tabular_regressor import CVResults, ScoreMetric, TabularRegressor
from src.base.forecasting.models.time_series.helpers import build_toeplitz
from src.base.forecasting.models.time_series.ts_model import TimeSeriesForecastModelAutoScaled


# =================================================================================================
#  TimeSeries model based on TabularRegressor
# =================================================================================================
class TimeSeriesModelRegression(TimeSeriesForecastModelAutoScaled):
    """
    This class implements a n-step-ahead timeseries regression model.
        n-step-ahead: we forecast samples 0, ... n-1
        regression:   we use the past p samples as features for the model.

    Hence, the core problem becomes a p -> n regression problem.  The user needs to provide
     a suitable TabularRegressor to the constructor.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, signal_name: str, regressor: TabularRegressor, avoid_training_nans: bool = False):
        super().__init__(model_type=f"regression_{regressor.name}", signal_name=signal_name)

        self.regressor = regressor  # type: TabularRegressor

        self.p = regressor.n_inputs
        self.n = regressor.n_outputs

        self._avoid_training_nans = avoid_training_nans

        self._cv = TimeSeriesRegressionCrossValidation(self)

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def _fit(self, scaled_training_data: pd.DataFrame):

        # --- construct tabulated dataset x,y -------------
        x, y = self.build_tabulated_data(scaled_training_data)

        # --- fit model -----------------------------------
        self.regressor.fit(x, y)

    def _predict(self, scaled_history: pd.DataFrame, n_samples: int) -> np.ndarray:

        # --- convert to numpy array ----------------------
        history = scaled_history[self.signal_name].to_numpy()
        history = history.reshape((1, len(history)))

        # --- predict -------------------------------------
        n_iterations = math.ceil(n_samples / self.n)
        predictions = np.zeros(n_iterations * self.n)

        for i in range(n_iterations):

            x = np.fliplr(history[:, -self.p :])
            y = self.regressor.predict(x)

            predictions[i * self.n : (i + 1) * self.n] = y.flatten()
            history = np.concatenate([history, y], axis=1)

        # --- return --------------------------------------
        return predictions[0:n_samples].flatten()

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def cv(self) -> TimeSeriesRegressionCrossValidation:
        return self._cv

    # -------------------------------------------------------------------------
    #  Dataset handling
    # -------------------------------------------------------------------------
    def build_tabulated_data(self, scaled_training_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        # --- convert to tabular data ---------------------
        ts = scaled_training_data[self.signal_name].to_numpy()
        x = self.__build_features(ts)
        y = self.__build_targets(ts)

        # --- remove NaN rows, if needed ------------------
        if self._avoid_training_nans:
            # Remove any row that has at least 1 NaN in x or y from the dataset.
            # This should avoid confusing the regressor that we train on this dataset.
            # However, some regressors might want to have all data, especially if e.g.
            # only part of a y-row has NaNs with no NaNs in the corresponding x-row.
            x, y = self._remove_nan(x, y)

        # --- return --------------------------------------
        return x, y

    @staticmethod
    def _remove_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        rows_without_nan = [not (any(x_row) or any(y_row)) for x_row, y_row in zip(np.isnan(x), np.isnan(y))]

        x = x[rows_without_nan]
        y = y[rows_without_nan]

        return x, y

    def __build_features(self, ts: np.ndarray) -> np.ndarray:
        # build Toeplitz matrix containing samples 1,...,p in the past,
        #  so first build Toeplitz of width p+1, containing samples 0,...,p in the past and omit column 0.
        return build_toeplitz(ts, window_size=self.p + 1, forward=False)[:, 1:]

    def __build_targets(self, ts: np.ndarray) -> np.ndarray:
        # build Toeplitz matrix containing samples 0,...,n-1 in the future
        return build_toeplitz(ts, window_size=self.n, forward=True)


# =================================================================================================
#  Cross-Validation
# =================================================================================================
class TimeSeriesRegressionCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, ts_model: TimeSeriesModelRegression):
        self.ts_model = ts_model
        self.results = None  # type: Optional[CVResults]

    # -------------------------------------------------------------------------
    #  Parameters
    # -------------------------------------------------------------------------
    def get_param_names(self) -> List[str]:
        """Parameters that can be tuned using cross-validation."""
        param_names = self.ts_model.regressor.cv.get_param_names()

        # These cannot be tuned by cross-validation because they influence the evaluation criterion.
        for p_name in ["n_inputs", "n_outputs"]:
            if p_name in param_names:
                param_names.remove(p_name)

        return param_names

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        training_data: pd.DataFrame,
        param_grid: Union[Dict, List[Dict]],
        score_metric: ScoreMetric,
        n_splits: int = 5,
        n_jobs: int = -1,
    ):

        # --- learn scaling -------------------------------
        self.ts_model.fit_scaling(training_data)
        scaled_training_data = self.ts_model.scale_df(training_data)

        # --- construct training data ---------------------
        x_train, y_train = self.ts_model.build_tabulated_data(scaled_training_data)

        # --- tabular regressor cross-validation ----------
        self.ts_model.regressor.cv.grid_search(
            x=x_train,
            y=y_train,
            param_grid=param_grid,
            score_metric=score_metric,
            n_splits=n_splits,
            n_jobs=n_jobs,
            shuffle_data=False,
        )

        # --- extract results -----------------------------
        self.results = self.ts_model.regressor.cv.results
