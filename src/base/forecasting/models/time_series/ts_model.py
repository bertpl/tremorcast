import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =================================================================================================
#  BASE CLASS
# =================================================================================================
class TimeSeriesForecastModel(ABC):
    """
    Abstract class implementing an sklearn-like fit/predict interface for time series forecasting.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, model_type: str, signal_name: str):
        """
        Constructor of TimeSeriesForecastModel class.
        :param model_type: (str) type of model
        :param signal_name: (str) name of the signal we're forecasting.
        """
        self.model_type = model_type
        self.signal_name = signal_name

    # -------------------------------------------------------------------------
    #  SIMULATION
    # -------------------------------------------------------------------------
    def batch_predict(
        self,
        data: pd.DataFrame,
        retrain_model: bool,
        first_sample: int,
        horizon: int,
        overlap_end: bool = False,
        stride: int = 1,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Performs a batch of predictions on a single dataset with fixed horizon and regularly spaced initial samples.
        Optionally the model can be retrained on the relevant history before each forecast.

        :param data: pd.DataFrame, containing at least a column with a name equal to self.signal_name.
        :param retrain_model: (bool) if True the model is retrained before each prediction.
        :param first_sample: (int) The sample at which the first prediction will start
        :param horizon: (int) number of samples to be forecast each time.
        :param overlap_end: (bool, default=False) if False, the horizon is shortened for the last predictions, to not
                              extend beyond the provided dataset.
        :param stride: (int) number of samples between each prediction.
        :return: list of (initial_sample, forecast)-tuples, of type (int, np.ndarray)
        """

        forecasts = []  # type: List[Tuple[int, np.ndarray]]

        for i in tqdm(
            range(first_sample, len(data), stride),
            desc=f"Evaluating model '{self.model_type}' (retrain={retrain_model})".ljust(60),
            file=sys.stdout,
        ):

            # construct history
            history = data.iloc[0:i]

            # retrain if needed
            if retrain_model:
                self.fit(history)

            # create forecast
            if overlap_end:
                n_samples = horizon
            else:
                n_samples = min(horizon, len(data) - i)

            forecast = self.predict(history, n_samples)

            forecasts.append((i, forecast))

        return forecasts

    # -------------------------------------------------------------------------
    #  ABSTRACT INTERFACE
    # -------------------------------------------------------------------------
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
