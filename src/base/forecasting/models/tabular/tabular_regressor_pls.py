from typing import Optional

from sklearn.cross_decomposition import PLSRegression

from .helpers import FeatureSelector
from .tabular_regressor_wrapper import TabularRegressorWrapper


class TabularRegressorPLS(TabularRegressorWrapper):
    def __init__(
        self,
        n_components: int,
        max_iter: int = 1_000,
        tol: float = 1e-6,
        feature_selector: Optional[FeatureSelector] = None,
        **kwargs,
    ):
        """
        NOTE: n_components should be <= min(n_samples, n_features), so does not depend on n_targets (!!!)

              See: https://scikit-learn.org/stable/modules/cross_decomposition.html#plsregression
        """

        super().__init__(
            name="pls",
            model=PLSRegression(n_components=n_components, max_iter=max_iter, tol=tol, scale=False),
            feature_selector=feature_selector,
            **kwargs,
        )

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
