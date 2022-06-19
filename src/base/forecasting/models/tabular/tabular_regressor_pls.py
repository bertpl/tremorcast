from typing import Optional

from sklearn.cross_decomposition import PLSRegression

from .feature_selectors import FeatureSelector
from .tabular_regressor_wrapper import TabularRegressorWrapper


class TabularRegressorPLS(TabularRegressorWrapper):
    def __init__(
        self,
        n_components: int,
        max_iter: int = 500,
        tol: float = 1e-6,
        feature_selector: Optional[FeatureSelector] = None,
        **kwargs,
    ):

        super().__init__(
            name="pls",
            model=PLSRegression(n_components=n_components, max_iter=max_iter, tol=tol, scale=False),
            feature_selector=feature_selector,
            **kwargs,
        )

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
