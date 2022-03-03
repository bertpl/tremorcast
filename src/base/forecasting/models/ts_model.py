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


# =================================================================================================
#  AUTO-SCALED VERSION
# =================================================================================================
class TimeSeriesForecastModelAutoScaled(TimeSeriesForecastModel):
    """
    Time series forecast model that auto-scales all signals.
    """

    # -------------------------------------------------------------------------
    #  CONSTRUCTOR
    # -------------------------------------------------------------------------
    def __init__(self, model_type: str, signal_name: str):
        super().__init__(model_type, signal_name)

        # dict mapping signal_name -> (mean, std)
        self._scaling = dict()  # type: Dict[str, Tuple[float, float]]

    # -------------------------------------------------------------------------
    #  PUBLIC INTERFACE
    # -------------------------------------------------------------------------
    def fit(self, training_data: pd.DataFrame):
        """
        Train forecast model and learn the scaling of the data.

        Training data dataframe structure:
          - index: datetimes of samples
          - columns: each column represents 1 signal (assuming 1 column name = self.signal_name)

        :param training_data: (pd.DataFrame) containing training data.
        """

        # --- learn and apply scaling --------------------
        self._scaling = self._extract_scaling(training_data)
        scaled_training_data = self._scale_df(training_data)

        # --- fit -----------------------------------------
        self._fit(scaled_training_data)

    def predict(self, history: pd.DataFrame, n_samples: int) -> np.ndarray:

        # --- apply scaling -------------------------------
        scaled_history = self._scale_df(history)

        # --- predict -------------------------------------
        forecast = self._predict(scaled_history, n_samples)

        # --- return unscaled forecast --------------------
        return self._unscale_np(forecast, self.signal_name)

    # -------------------------------------------------------------------------
    #  INTERNAL
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_scaling(data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Return dictionary mapping col_names -> (mean, std)"""
        return {signal_name: (data[signal_name].mean(), data[signal_name].std()) for signal_name in data.columns}

    def _scale_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to those columns for which we can find a scaling in this objects scaling dict"""
        scaled_data = data.copy()

        for signal_name in data.columns:
            if signal_name in self._scaling.keys():
                mean, std = self._scaling[signal_name]
                scaled_data[signal_name] = (scaled_data[signal_name] - mean) / std

        return scaled_data

    def _unscale_np(self, data: np.ndarray, signal_name: str) -> np.ndarray:
        mean, std = self._scaling.get(signal_name, (0.0, 1.0))
        return mean + std * data

    # -------------------------------------------------------------------------
    #  METHODS TO BE IMPLEMENTED BY CHILD CLASSES
    # -------------------------------------------------------------------------
    @abstractmethod
    def _fit(self, scaled_training_data: pd.DataFrame):
        pass

    @abstractmethod
    def _predict(self, scaled_history: pd.DataFrame, n_samples: int) -> np.ndarray:
        pass
