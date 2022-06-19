from src.base.forecasting.models.tabular.tabular_regressor_pls import TabularRegressorPLS

from .ts_model_ar import TimeSeriesModelAutoRegressive


class TimeSeriesModelAutoRegressivePLS(TimeSeriesModelAutoRegressive):
    def __init__(self, signal_name: str, p: int, n: int, n_components: int):
        super().__init__(signal_name, p, n, TabularRegressorPLS(n_components))
