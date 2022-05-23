from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from src.base.forecasting.evaluation.cross_validation import CVResults

from .helpers import FeatureSelector
from .tabular_regressor import TabularRegressor
from .tabular_regressor_mlp import TabularRegressorMLP


class TabularRegressorMLPMulti(TabularRegressor):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n_targets: int,
        feature_selector: FeatureSelector = None,
        **kwargs,
    ):
        """
        Regressor that uses a separate MLP for each target.  Dedicated CV functionality is provided, to
        be able to perform separate grid search runs for all (or a subset of) MLP sub-models.
        """

        super().__init__(name="mlp-multi", remove_nans_before_fit=False, **kwargs)

        self.sub_models = [TabularRegressorMLP(feature_selector=feature_selector) for _ in range(n_targets)]

        self.n_targets = n_targets
        self.feature_selector = feature_selector

        self._sub_cv = SubModelCrossValidation(self)

    @property
    def n_sub_models(self) -> int:
        return self.n_targets

    # -------------------------------------------------------------------------
    #  Hyper-parameters
    # -------------------------------------------------------------------------
    def set_sub_params(self, i_sub_models: List[int] = None, **params):
        """Sets the provided keywords arguments as parameters in each sub-model"""
        if i_sub_models is None:
            for m in self.sub_models:
                m.set_params(**params)
        else:
            for i in i_sub_models:
                self.sub_models[i].set_params(**params)

    @property
    def sub_cv(self) -> SubModelCrossValidation:
        return self._sub_cv

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray, **fit_params) -> TabularRegressorMLPMulti:

        # --- fit sub-models -------------------------------
        for i, sub_model in enumerate(self.sub_models):
            sub_model.fit(x, y[:, [i]], **fit_params)

        # --- return ---------------------------------------
        return self


# =================================================================================================
#  Cross Validation
# =================================================================================================
class SubModelCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, regressor: TabularRegressorMLPMulti):
        self.regressor = regressor
        self.results = dict()  # type: Dict[int, CVResults]

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param_grid: Union[dict, List[dict]],
        score_metric: ScoreMetric,
        n_splits: int = 10,
        shuffle_data: bool = False,
        n_jobs: int = -1,
        i_sub_models: List[int] = None,  # indices of sub-models for which to perform CV
    ):

        # --- argument handling ---------------------------
        i_sub_models = i_sub_models or list(range(self.regressor.n_targets))

        # --- grid search model by model ------------------
        for i_sub_model in i_sub_models:

            # prep regressor & data
            sub_model = self.regressor.sub_models[i_sub_model]
            y_sub = y[:, [i_sub_model]]

            # run actual grid search
            print(f"=== PERFORMING GRID SEARCH CV FOR SUBMODEL {i_sub_model} ===")
            sub_model.cv.grid_search(x, y_sub, param_grid, score_metric, n_splits, shuffle_data, n_jobs)

        # --- collect results -----------------------------
        self.results = {
            i_sub_model: self.regressor.sub_models[i_sub_model].cv.results for i_sub_model in i_sub_models
        }  # type: Dict[int, CVResults]

        # --- show results --------------------------------
        self.show_results()

    # -------------------------------------------------------------------------
    #  Show results
    # -------------------------------------------------------------------------
    def show_results(self):

        # init
        all_results = self.get_best_hyper_params()
        col_names = sorted(list(all_results.values())[0].keys())
        col_width = 20

        # show
        print("-" * 120)
        print("Sub-model grid-search cv results:")
        print(" " * col_width + "".join([cn.ljust(col_width) for cn in col_names]))

        for i, result in all_results.items():
            print(
                f"{i}".ljust(col_width)
                + "".join(
                    [
                        f"{value}".ljust(col_width)
                        if (len(f"{value}") < 15) or not isinstance(value, float)
                        else f"{value:.5f}".ljust(col_width)
                        for value in [result.get(cn, "/") for cn in col_names]
                    ]
                )
            )
        print("-" * 120)

    def get_best_hyper_params(self) -> Dict[int, Dict[str, Any]]:
        """Return optimal hyper-parameter values for each submodel, resulting from cross-validation."""

        dct = dict()

        param_names = sorted(list(self.results.values())[0].all_param_values().keys())
        for i, result in self.results.items():

            best_params = {pn: result.best_result.params.get(pn, "/") for pn in param_names}
            best_params["loss"] = result.best_result.val_metric_mean
            best_params["lr_max_value"] = self.regressor.sub_models[i].last_lr_max_value

            dct[i] = best_params

        return dct
