from typing import Optional

from sklearn.linear_model import LinearRegression

from .feature_selectors import FeatureSelector
from .tabular_regressor_sklearn import TabularRegressorSklearn


class TabularRegressorOLS(TabularRegressorSklearn):
    def __init__(self, feature_selector: Optional[FeatureSelector] = None, **kwargs):

        super().__init__(
            name="ols",
            model=LinearRegression(
                fit_intercept=False,
            ),
            remove_nans_before_fit=True,
            feature_selector=feature_selector,
            **kwargs
        )
