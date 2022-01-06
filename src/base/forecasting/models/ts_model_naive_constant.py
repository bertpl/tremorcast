import numpy as np
import pandas as pd

from .ts_model import TimeSeriesForecastModel


class TimeSeriesModelNaiveConstant(TimeSeriesForecastModel):
    """Naive timeseries forecast model that always predicts the latest value will stay constant in the future."""

    def __init__(self, signal_name: str):
        super().__init__(signal_name)

    def fit(self, training_data: pd.DataFrame):
        pass  # nothing needs to be done here

    def predict(self, history: pd.DataFrame, n_samples: int) -> np.ndarray:
        last_value = history.at[history.index[-1], self.signal_name]
        return np.full(n_samples, last_value)
