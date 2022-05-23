"""Base class for auto-regressive modeling"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from src.base.forecasting.evaluation.cross_validation import CV_METADATA_PARAM, CVResults
from src.base.forecasting.evaluation.metrics import TabularMetric
from src.base.forecasting.models.tabular.tabular_regressor import TabularRegressor
from src.tools.math import remove_nan_rows

from .helpers import build_toeplitz
from .ts_model import TimeSeriesModel


# =================================================================================================
#  TimeSeries model based on TabularRegressor
# =================================================================================================
class TimeSeriesModelAutoRegressive(TimeSeriesModel):
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
    def __init__(self, p: int, n: int, regressor: TabularRegressor, avoid_training_nans: bool = False, **kwargs):
        """
        Constructor of an auto-regressive model using a user-provided tabular regressor.
        :param p: (int) number of past samples to use as features in auto-regression
        :param n: (int) number of future samples forecast by the tabular regressor
        :param regressor: (TabularRegressor) regressor model.
        :param avoid_training_nans: (bool) set to True if the regressor cannot cope well with NaNs and these need
                                            to be removed (i.e. any row containing at least 1 NaN) from the dataset.
        """
        super().__init__(name=f"ar_{regressor.name}", **kwargs)

        self.regressor = regressor  # type: TabularRegressor

        self.p = p
        self.n = n
        self.avoid_training_nans = avoid_training_nans

        self._tabular_cv = TimeSeriesTabularCrossValidation(self)

    # -------------------------------------------------------------------------
    #  Parameter handling
    # -------------------------------------------------------------------------
    def get_mapped_params(self) -> Set[str]:
        """Returns parameters that appear in both this and the embedded model and hence should be mapped."""
        wrapper_params = set(list(self.get_params().keys()) + ["show_progress"])
        model_params = set(self.regressor.get_params().keys())
        # return wrapper_params.intersection(model_params).difference([CV_METADATA_PARAM])
        return wrapper_params.intersection(model_params)

    def set_params(self, **params) -> TimeSeriesModelAutoRegressive:
        """This implementation makes sure any params set to this object are mapped - if needed - to the submodel."""
        super().set_params(**params)
        for param in self.get_mapped_params():
            if param in params:
                self.regressor.set_params(**{param: params[param]})
        return self

    # -------------------------------------------------------------------------
    #  Fit & Predict
    # -------------------------------------------------------------------------
    def min_hist(self) -> int:
        return self.p

    def fit(self, x: np.ndarray):

        # --- construct tabulated dataset x,y -------------
        x_tabular, y_tabular = self.build_tabulated_data(x)

        # --- fit model -----------------------------------
        self.regressor.fit(x_tabular, y_tabular)

    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:

        # this will return [(i, prediction)]
        all_preds = self.batch_predict(x_hist, first_sample=x_hist.size, hor=hor, overlap_end=True)

        # return prediction
        return all_preds[0][1]

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
        In this implementation we batch together calls to our tabular regressor, to minimize overhead.  For certain
        types of regressors (such as fast.ai models) this can dramatically increase total computation speed.
        """

        # --- init ----------------------------------------

        # number of calls to our tabular regressor
        n_iterations = math.ceil(hor / self.n)

        # each row of predictions represents 1 prediction we need to make
        # we will iteratively predict n samples forward for each of these rows and pre-populate this array
        # with the history of p samples needed to start each of these predictions.
        predictions = np.concatenate(
            [x[i_start - self.p : i_start].reshape((1, self.p)) for i_start in range(first_sample, x.size + 1, stride)],
            axis=0,
        )

        # --- iteratively predict 'hor' ahead -------------
        for i in tqdm(
            range(n_iterations),
            desc=f"Batch prediction for model {self.name} [hor={hor}, batch_size={predictions.shape[0]}]",
            file=sys.stdout,
            disable=silent,
            leave=False,
        ):

            # Previous p samples for each prediction, needed to predict an additional n steps ahead.
            # We need to flip left-right, because the tabular regressor is trained with [lag_1, lag_2, ..., lag_p]
            #  as features.
            x_hist = np.fliplr(predictions[:, -self.p :])

            # call regressor.predict, leading to n new samples
            new_preds = self.regressor.predict(x_hist)

            # add n new samples to values we already have
            predictions = np.concatenate([predictions, new_preds], axis=1)

        predictions = predictions[:, self.p :]  # remove the pre-populated values such that only the predictions remain

        # --- return results ------------------------------
        return [
            (
                i_sample,
                predictions[i_pred, :] if overlap_end else predictions[i_pred, 0 : min(hor, x.size - i_sample)],
            )
            for i_pred, i_sample in enumerate(range(first_sample, x.size + 1, stride))
        ]

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def tabular_cv(self) -> TimeSeriesTabularCrossValidation:
        return self._tabular_cv

    # -------------------------------------------------------------------------
    #  Dataset handling
    # -------------------------------------------------------------------------
    def build_tabulated_data(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # --- convert to tabular data ---------------------
        x_tabular = self.__build_features(x)
        y_tabular = self.__build_targets(x)

        # --- remove NaN rows, if needed ------------------
        if self.avoid_training_nans:
            # Remove any row that has at least 1 NaN in x or y from the dataset.
            # This should avoid confusing the regressor that we train on this dataset.
            # However, some regressors might want to have all data, especially if e.g.
            # only part of a y-row has NaNs with no NaNs in the corresponding x-row.
            x_tabular, y_tabular = remove_nan_rows(x_tabular, y_tabular)

        # --- return --------------------------------------
        return x_tabular, y_tabular

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
class TimeSeriesTabularCrossValidation:

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
        x: np.ndarray,
        param_grid: Union[Dict, List[Dict]],
        score_metric: TabularMetric,
        n_splits: int = 10,
        n_jobs: int = -1,
    ):

        # --- construct training data ---------------------
        x_tabular, y_tabular = self.ts_model.build_tabulated_data(x)

        # --- tabular regressor cross-validation ----------
        self.ts_model.regressor.cv.grid_search(
            x=x_tabular,
            y=y_tabular,
            param_grid=param_grid,
            score_metric=score_metric,
            n_splits=n_splits,
            n_jobs=n_jobs,
            shuffle_data=False,
        )

        # --- extract results -----------------------------
        self.results = self.ts_model.regressor.cv.results
