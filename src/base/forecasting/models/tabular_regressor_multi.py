from __future__ import annotations

from typing import List, Union

import numpy as np

from .tabular_regressor import ScoreMetric, TabularRegressor


# =================================================================================================
#  Tabular Model - Composed of multiple single-output models
# =================================================================================================
class TabularRegressorMulti(TabularRegressor):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, regressors: List[TabularRegressor], **kwargs):

        # --- error checking ------------------------------
        assert len(regressors) > 0, "should have at least 1 regressor"
        assert len({r.n_inputs for r in regressors}) == 1, "n_input cannot differ between regressors"
        assert len({type(r) for r in regressors}) == 1, "regressors should all be of the same type"

        # --- determine n_inputs, n_outputs ---------------
        n_inputs = regressors[0].n_inputs
        n_outputs = sum([r.n_outputs for r in regressors])

        # --- superclass constructor ----------------------
        super().__init__(name, n_inputs, n_outputs, **kwargs)

        # --- sub-model mgmt ------------------------------
        self.regressors = regressors  # type: List[TabularRegressor]
        self.output_mapping = [
            list(np.arange(i_ref - r.n_outputs, i_ref))
            for i_ref, r in zip(np.cumsum([r.n_outputs for r in regressors]), regressors)
        ]  # type: List[List[int]]
        self._sub_cv = SubModelCrossValidation(self)

    @property
    def n_sub_models(self) -> int:
        return len(self.regressors)

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray):

        # --- pre-fit -------------------------------------
        self._pre_fit_hook()

        # --- fit -----------------------------------------
        for i, regressor in enumerate(self.regressors):
            # just fit each regressor with the correct subset of outputs
            regressor.fit(x, y[:, self.output_mapping[i]])

        # --- post-fit ------------------------------------
        self._post_fit_hook()

    def predict(self, x: np.ndarray) -> np.ndarray:
        # perform predictions with each regressor and concatenate outputs
        return np.concatenate([r.predict(x) for r in self.regressors], axis=1)

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def sub_cv(self) -> SubModelCrossValidation:
        return self._sub_cv

    # -------------------------------------------------------------------------
    #  Hyper-parameters
    # -------------------------------------------------------------------------
    def set_sub_params(self, **kwargs):
        """Sets the provided keywords arguments as parameters in each sub-model"""
        for r in self.regressors:
            r.set_params(**kwargs)

    # -------------------------------------------------------------------------
    #  Hooks
    # -------------------------------------------------------------------------
    def _pre_fit_hook(self):
        """Called right before fitting."""
        pass

    def _post_fit_hook(self):
        """Called right after fitting."""
        pass


# =================================================================================================
#  Cross Validation
# =================================================================================================
class SubModelCrossValidation:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, regressor: TabularRegressorMulti):
        self.regressor = regressor
        self.results = None

    # -------------------------------------------------------------------------
    #  Grid Search
    # -------------------------------------------------------------------------
    def grid_search(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param_grid: Union[dict, List[dict]],
        score_metric: ScoreMetric,
        n_splits: int = 5,
        shuffle_data: bool = False,
        n_jobs: int = -1,
        i_sub_models: List[int] = None,  # indices of sub-models for which to perform CV
    ):

        # --- argument handling ---------------------------
        i_sub_models = i_sub_models or list(range(self.regressor.n_sub_models))

        # --- grid search model by model ------------------
        for i_sub_model in i_sub_models:

            # prep regressor & data
            regressor = self.regressor.regressors[i_sub_model]
            y_sub = y[:, self.regressor.output_mapping[i_sub_model]]

            # run actual grid search
            print(f"=== PERFORMING GRID SEARCH CV FOR SUBMODEL {i_sub_model} ===")
            regressor.cv.grid_search(x, y_sub, param_grid, score_metric, n_splits, shuffle_data, n_jobs)

        # --- collect results -----------------------------
        self.results = {i_sub_model: self.regressor.regressors[i_sub_model].cv.results for i_sub_model in i_sub_models}
