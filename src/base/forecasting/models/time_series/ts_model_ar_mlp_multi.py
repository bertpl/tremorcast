from __future__ import annotations

from typing import Dict, List, Set, Union

import pandas as pd

from src.base.forecasting.models.tabular import CVResults, TabularMetric
from src.base.forecasting.models.tabular.tabular_regressor_mlp_multi import TabularRegressorMLPMulti

from .ts_model_ar import TimeSeriesModelAutoRegressive


# =================================================================================================
#  Time Series Model
# =================================================================================================
class TimeSeriesModelAutoRegressiveMLPMulti(TimeSeriesModelAutoRegressive):
    def __init__(self, signal_name: str, p: int, n: int, **kwargs):

        # regressor
        regressor = TabularRegressorMLPMulti(n_targets=n, **kwargs)
        super().__init__(signal_name, p, n, regressor)
        self.regressor = regressor  # type: TabularRegressorMLPMulti

        # cv
        self._sub_cv = TimeSeriesAutoRegressiveSubModelCrossValidation(self)

    @property
    def sub_cv(self) -> TimeSeriesAutoRegressiveSubModelCrossValidation:
        return self._sub_cv


# =================================================================================================
#  Sub-model cross-validation
# =================================================================================================
class TimeSeriesAutoRegressiveSubModelCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, ts_model: TimeSeriesModelAutoRegressiveMLPMulti):
        self.ts_model = ts_model
        self.results = dict()  # type: Dict[int, CVResults]

    # -------------------------------------------------------------------------
    #  Parameters
    # -------------------------------------------------------------------------
    def get_tunable_param_names(self) -> Set[str]:
        return self.ts_model.regressor.sub_models[0].get_tunable_param_names()

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
        i_sub_models: List[int] = None,
    ):

        # --- argument handling ---------------------------
        i_sub_models = i_sub_models or list(range(self.ts_model.regressor.n_sub_models))

        # --- construct training data ---------------------
        x_train, y_train = self.ts_model.build_tabulated_data(training_data)

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
