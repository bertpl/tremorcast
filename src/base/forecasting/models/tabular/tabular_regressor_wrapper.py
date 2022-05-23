from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Set

import numpy as np
from sklearn import pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

from .helpers import FeatureSelector
from .tabular_regressor import TabularRegressor


# =================================================================================================
#  Feature / Target scaling
# =================================================================================================
class ScalingType(Enum):
    MEAN_STD = auto()
    ROBUST = auto()

    def get_scaler(self):
        if self == ScalingType.MEAN_STD:
            return StandardScaler()
        elif self == ScalingType.ROBUST:
            return RobustScaler()
        else:
            raise NotImplementedError(f"no scaler implemented for {self}")


# =================================================================================================
#  Wrapper
# =================================================================================================
class TabularRegressorWrapper(TabularRegressor):

    # -------------------------------------------------------------------------
    #  CONSTRUCTOR
    # -------------------------------------------------------------------------
    def __init__(
        self,
        model: BaseEstimator,
        feature_scaler: Optional[ScalingType] = ScalingType.MEAN_STD,
        target_scaler: Optional[ScalingType] = ScalingType.MEAN_STD,
        feature_selector: Optional[FeatureSelector] = None,
        name: str = "wrapper",
        **kwargs,
    ):
        """Wraps around another estimator in order to provide automatic scaling & feature selection."""

        # --- superclass constructor ----------------------
        super().__init__(name=name, remove_nans_before_fit=True, **kwargs)

        # --- set parameters ------------------------------
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.feature_selector = feature_selector

        # --- pipeline ------------------------------------
        self._pipeline = None

    # -------------------------------------------------------------------------
    #  PARAMETER MAPPING
    # -------------------------------------------------------------------------
    def get_mapped_params(self) -> Set[str]:
        """Returns parameters that appear in both this and the embedded model and hence should be mapped."""
        wrapper_params = set(list(self.get_params().keys()) + ["show_progress"])
        model_params = set(self.model.get_params().keys())
        return wrapper_params.intersection(model_params)

    def set_params(self, **params) -> TabularRegressorWrapper:
        """This implementation makes sure any params set to this object are mapped - if needed - to the submodel."""
        super().set_params(**params)
        for param in self.get_mapped_params():
            if param in params:
                self.model.set_params(**{param: params[param]})
        return self

    # -------------------------------------------------------------------------
    #  FIT & PREDICT
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray, **fit_params):
        self._build_pipeline()
        self._pipeline.fit(x, y, **fit_params)

    def predict(self, x: np.ndarray, **predict_params) -> np.ndarray:
        return self._pipeline.predict(x, **predict_params)

    # -------------------------------------------------------------------------
    #  INTERNAL
    # -------------------------------------------------------------------------
    def _build_pipeline(self):
        """Sets the _pipeline attribute based on the other parameters of the model."""

        model = self.model

        if self.feature_selector or self.feature_scaler:
            pipeline_steps = []

            if self.feature_selector:
                pipeline_steps.append(("feature_selector", self.feature_selector))

            if self.feature_scaler:
                pipeline_steps.append(("feature_scaling", self.feature_scaler.get_scaler()))

            pipeline_steps.append(("model", self.model))

            model = pipeline.Pipeline(steps=pipeline_steps)

        if self.target_scaler:
            model = TransformedTargetRegressor(
                regressor=model,
                transformer=self.target_scaler.get_scaler(),
            )

        self._pipeline = model
