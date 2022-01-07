from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TimeSeriesForecastModel(ABC):
    """
    Abstract class implementing an sklearn-like fit/predict interface for time series forecasting.
    """

    def __init__(self, model_type: str, signal_name: str):
        """
        Constructor of TimeSeriesForecastModel class.
        :param model_type: (str) type of model
        :param signal_name: (str) name of the signal we're forecasting.
        """
        self.model_type = model_type
        self.signal_name = signal_name

    @abstractmethod
    def fit(self, training_data: pd.DataFrame):
        """
        Train forecast model on provided data.

        Training data dataframe structure:
          - index: datetimes of samples
          - columns: each column represents 1 signal (assuming 1 column name = self.signel_name)

        :param training_data: (pd.DataFrame) containing training data.
        """
        pass

    @abstractmethod
    def predict(self, history: pd.DataFrame, n_samples: int) -> np.ndarray:
        """
        Predict the next n samples given the time series history (=most recent x samples).
        :param history: (pd.DataFrame) most recent data of the time series (+ possibly other time series used as inputs)
        :param n_samples: (int) number of samples we need to predict.
        :return: 1D numpy array of length n_samples with forecasts.
        """
        pass
