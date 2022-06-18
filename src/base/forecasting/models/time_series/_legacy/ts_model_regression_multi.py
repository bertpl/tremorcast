from __future__ import annotations

from typing import Dict, List, Union

import pandas as pd

from src.base.forecasting.models import ScoreMetric
from src.base.forecasting.models.tabular.legacy.tabular_regressor_multi import TabularRegressorMulti

from .ts_model_regression import TimeSeriesModelRegression


# =================================================================================================
#  TimeSeries model based on Multi-Regression model
# =================================================================================================
class TimeSeriesModelRegressionMulti(TimeSeriesModelRegression):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, signal_name: str, regressor: TabularRegressorMulti, avoid_training_nans: bool = False):
        super().__init__(signal_name, regressor, avoid_training_nans)
        self.regressor = regressor  # type: TabularRegressorMulti
        self._sub_cv = SubModelCrossValidation(self)

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def sub_cv(self) -> SubModelCrossValidation:
        return self._sub_cv


# =================================================================================================
#  Cross-Validation
# =================================================================================================
class SubModelCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, ts_model: TimeSeriesModelRegressionMulti):
        self.ts_model = ts_model
        self.results = None

    # -------------------------------------------------------------------------
    #  Parameters
    # -------------------------------------------------------------------------
    def get_param_names(self) -> List[str]:
        """Parameters that can be tuned using cross-validation."""
        return self.ts_model.regressor.sub_models[0].cv.get_param_names()

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
        i_sub_models: List[int] = None,
    ):

        # --- argument handling ---------------------------
        i_sub_models = i_sub_models or list(range(self.ts_model.regressor.n_sub_models))

        # --- learn scaling -------------------------------
        self.ts_model.fit_scaling(training_data)
        scaled_training_data = self.ts_model.scale_df(training_data)

        # --- construct training data ---------------------
        x_train, y_train = self.ts_model.build_tabulated_data(scaled_training_data)

        # --- tabular regressor cross-validation ----------
        self.ts_model.regressor.sub_cv.grid_search(
            x=x_train,
            y=y_train,
            param_grid=param_grid,
            score_metric=score_metric,
            n_splits=n_splits,
            n_jobs=n_jobs,
            shuffle_data=False,
            i_sub_models=i_sub_models,
        )

        # --- extract results -----------------------------
        self.results = self.ts_model.regressor.sub_cv.results
