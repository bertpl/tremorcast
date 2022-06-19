from src.base.forecasting.models.tabular.tabular_regressor_ols import TabularRegressorOLS

from .ts_model_ar import TimeSeriesModelAutoRegressive


class TimeSeriesModelAutoRegressiveOLS(TimeSeriesModelAutoRegressive):
    def __init__(self, signal_name: str, p: int, n: int):
        super().__init__(signal_name, p, n, TabularRegressorOLS())
