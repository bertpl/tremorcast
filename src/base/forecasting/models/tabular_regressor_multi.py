from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from .tabular_regressor import CVResults, ScoreMetric, TabularRegressor


# =================================================================================================
#  Tabular Model - Composed of multiple single-output models
# =================================================================================================
class TabularRegressorMulti(TabularRegressor):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, name: str, sub_models: List[TabularRegressor], **kwargs):

        # --- error checking ------------------------------
        assert len(sub_models) > 0, "should have at least 1 regressor"
        assert len({r.n_inputs for r in sub_models}) == 1, "n_input cannot differ between regressors"
        assert len({type(r) for r in sub_models}) == 1, "regressors should all be of the same type"

        # --- determine n_inputs, n_outputs ---------------
        n_inputs = sub_models[0].n_inputs
        n_outputs = sum([m.n_outputs for m in sub_models])

        # --- superclass constructor ----------------------
        super().__init__(name, n_inputs, n_outputs, **kwargs)

        # --- sub-model mgmt ------------------------------
        self.sub_models = sub_models  # type: List[TabularRegressor]
        self.output_mapping = [
            list(np.arange(i_ref - m.n_outputs, i_ref))
            for i_ref, m in zip(np.cumsum([r.n_outputs for r in sub_models]), sub_models)
        ]  # type: List[List[int]]
        self._sub_cv = SubModelCrossValidation(self)

    @property
    def n_sub_models(self) -> int:
        return len(self.sub_models)

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray):

        # --- pre-fit -------------------------------------
        self._pre_fit_hook()

        # --- fit -----------------------------------------
        for i, regressor in enumerate(self.sub_models):
            # just fit each regressor with the correct subset of outputs
            regressor.fit(x, y[:, self.output_mapping[i]])

        # --- post-fit ------------------------------------
        self._post_fit_hook()

    def predict(self, x: np.ndarray) -> np.ndarray:
        # perform predictions with each regressor and concatenate outputs
        return np.concatenate([r.predict(x) for r in self.sub_models], axis=1)

    # -------------------------------------------------------------------------
    #  Cross-Validation
    # -------------------------------------------------------------------------
    @property
    def sub_cv(self) -> SubModelCrossValidation:
        return self._sub_cv

    # -------------------------------------------------------------------------
    #  Hyper-parameters
    # -------------------------------------------------------------------------
    def set_sub_params(self, i_sub_models: List[int] = None, **kwargs):
        """Sets the provided keywords arguments as parameters in each sub-model"""
        if i_sub_models is None:
            for m in self.sub_models:
                m.set_params(**kwargs)
        else:
            for i in i_sub_models:
                self.sub_models[i].set_params(**kwargs)

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
            sub_model = self.regressor.sub_models[i_sub_model]
            y_sub = y[:, self.regressor.output_mapping[i_sub_model]]

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
            if hasattr(self.regressor.sub_models[i], "last_lr_max"):
                best_params["lr_max_value"] = self.regressor.sub_models[i].last_lr_max

            dct[i] = best_params

        return dct
