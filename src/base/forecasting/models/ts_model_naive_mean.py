import numpy as np
import pandas as pd

from .ts_model import TimeSeriesForecastModel


class TimeSeriesModelNaiveMean(TimeSeriesForecastModel):
    """Naive timeseries forecast model that always predicts the mean of the training dataset."""

    def __init__(self, signal_name: str):
        super().__init__("naive-mean", signal_name)
        self.mean = 0.0

    def fit(self, training_data: pd.DataFrame):
        self.mean = training_data[self.signal_name].mean()

    def predict(self, history: pd.DataFrame, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.mean)
