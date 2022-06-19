"""Base class for auto-regressive modeling"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.base.forecasting.models.tabular.tabular_regressor import CVResults, ScoreMetric, TabularRegressor
from src.tools.math import remove_nan_rows

from .helpers import build_toeplitz
from .ts_model import TimeSeriesForecastModel


# =================================================================================================
#  TimeSeries model based on TabularRegressor
# =================================================================================================
class TimeSeriesModelAutoRegressive(TimeSeriesForecastModel):
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
    def __init__(
        self, signal_name: str, p: int, n: int, regressor: TabularRegressor, avoid_training_nans: bool = False
    ):
        """
        Constructor of an auto-regressive model using a user-provided tabular regressor.
        :param signal_name: (str) signal name to forecast
        :param p: (int) number of past samples to use as features in auto-regression
        :param n: (int) number of future samples forecast by the tabular regressor
        :param regressor: (TabularRegressor) regressor model.
        :param avoid_training_nans: (bool) set to True if the regressor cannot cope well with NaNs and these need
                                            to be removed (i.e. any row containing at least 1 NaN) from the dataset.
        """
        super().__init__(model_type=f"ar_{regressor.name}", signal_name=signal_name)

        self.regressor = regressor  # type: TabularRegressor

        self.p = p
        self.n = n
        self._avoid_training_nans = avoid_training_nans

        self._cv = TimeSeriesAutoRegressiveCrossValidation(self)

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def fit(self, training_data: pd.DataFrame):

        # --- construct tabulated dataset x,y -------------
        x, y = self.build_tabulated_data(training_data)

        # --- fit model -----------------------------------
        self.regressor.fit(x, y)

    def predict(self, df_history: pd.DataFrame, n_samples: int) -> np.ndarray:

        # --- convert to numpy array ----------------------
        history = df_history[self.signal_name].to_numpy()
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
    def cv(self) -> TimeSeriesAutoRegressiveCrossValidation:
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
            x, y = remove_nan_rows(x, y)

        # --- return --------------------------------------
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
class TimeSeriesAutoRegressiveCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, ts_model: TimeSeriesModelAutoRegressive):
        self.ts_model = ts_model
        self.results = None  # type: Optional[CVResults]

    # -------------------------------------------------------------------------
    #  Parameters
    # -------------------------------------------------------------------------
    def get_tunable_params_names(self) -> Set[str]:
        return self.ts_model.regressor.get_tunable_param_names()

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        training_data: pd.DataFrame,
        param_grid: Union[Dict, List[Dict]],
        score_metric: ScoreMetric,
        n_splits: int = 10,
        n_jobs: int = -1,
    ):

        # --- construct training data ---------------------
        x_train, y_train = self.ts_model.build_tabulated_data(training_data)

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
