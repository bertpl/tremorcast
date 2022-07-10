from src.base.forecasting.models.tabular.tabular_regressor_mlp import TabularRegressorMLP

from .ts_model_ar import TimeSeriesModelAutoRegressive


class TimeSeriesModelAutoRegressiveMLP(TimeSeriesModelAutoRegressive):
    def __init__(self, p: int, n: int, **kwargs):
        super().__init__(p, n, TabularRegressorMLP(**kwargs))
