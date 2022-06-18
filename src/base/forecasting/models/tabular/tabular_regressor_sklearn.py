from enum import Enum, auto
from typing import Optional

import numpy as np
from sklearn import pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

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
#  SKlearn wrapper
# =================================================================================================
class TabularRegressorSklearn(TabularRegressor):

    # -------------------------------------------------------------------------
    #  CONSTRUCTOR
    # -------------------------------------------------------------------------
    def __init__(
        self,
        model: BaseEstimator,
        feature_scaler: Optional[ScalingType] = ScalingType.MEAN_STD,
        target_scaler: Optional[ScalingType] = ScalingType.MEAN_STD,
        remove_nans_before_fit: bool = True,
        name: str = "sklearn_wrapper",
        **kwargs,
    ):

        # --- superclass constructor ----------------------
        super().__init__(name, remove_nans_before_fit, **kwargs)

        # --- set parameters ------------------------------
        self.model = model  # originally provided model, which we need to save to attribute with same name
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        # --- build pipeline ------------------------------
        if feature_scaler:
            model = pipeline.Pipeline(steps=[("scaling", feature_scaler.get_scaler()), ("model", model)])

        if target_scaler:
            model = TransformedTargetRegressor(
                regressor=model,
                transformer=target_scaler.get_scaler(),
            )

        self._pipeline = model

    # -------------------------------------------------------------------------
    #  FIT & PREDICT
    # -------------------------------------------------------------------------
    def _fit(self, x: np.ndarray, y: np.ndarray, **fit_params):
        self._pipeline.fit(x, y, **fit_params)

    def predict(self, x: np.ndarray, **predict_params) -> np.ndarray:
        return self._pipeline.predict(x, **predict_params)
