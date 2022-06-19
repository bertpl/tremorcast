from src.base.forecasting.models.tabular.tabular_regressor_mlp_multi import TabularRegressorMLPMulti

from .ts_model_ar import TimeSeriesModelAutoRegressive


class TimeSeriesModelAutoRegressiveMLPMulti(TimeSeriesModelAutoRegressive):
    def __init__(self, signal_name: str, p: int, n: int, **kwargs):
        super().__init__(signal_name, p, n, TabularRegressorMLPMulti(n_targets=n, **kwargs))
