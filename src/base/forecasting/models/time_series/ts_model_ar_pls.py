from src.base.forecasting.models.tabular.tabular_regressor_pls import TabularRegressorPLS

from .ts_model_ar import TimeSeriesModelAutoRegressive


class TimeSeriesModelAutoRegressivePLS(TimeSeriesModelAutoRegressive):
    def __init__(self, p: int, n: int, n_components: int, **kwargs):
        self.n_components = n_components
        super().__init__(p, n, TabularRegressorPLS(n_components), **kwargs)
