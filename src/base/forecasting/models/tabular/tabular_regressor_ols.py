from __future__ import annotations

from typing import Optional

from sklearn.linear_model import Ridge

from .helpers import FeatureSelector
from .tabular_regressor_wrapper import TabularRegressorWrapper


class TabularRegressorOLS(TabularRegressorWrapper):
    def __init__(self, feature_selector: Optional[FeatureSelector] = None, alpha: float = 0.0, **kwargs):
        self.alpha = alpha
        super().__init__(
            name="ols",
            model=Ridge(fit_intercept=False, alpha=alpha),
            feature_selector=feature_selector,
            **kwargs,
        )
