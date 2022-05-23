from src.base.forecasting.models.tabular.tabular_regressor_ols import TabularRegressorOLS

from .ts_model_ar import TimeSeriesModelAutoRegressive


class TimeSeriesModelAutoRegressiveOLS(TimeSeriesModelAutoRegressive):
    def __init__(self, p: int, n: int, alpha: float = 0.0, **kwargs):
        self.alpha = alpha
        super().__init__(p, n, TabularRegressorOLS(alpha=alpha), **kwargs)
