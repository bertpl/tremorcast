from typing import List, Tuple

import numpy as np
import pandas as pd
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries

from .ts_model import TimeSeriesForecastModel


class TimeSeriesModelDarts(TimeSeriesForecastModel):
    """
    Implements a special case of TimeSeriesForecastModel (auto-scaled version) that wraps around a darts forecast model,
    where the batch_predict method makes use of the historical_forecasts method.  This might be more
    efficient, but especially allows us to easily perform such simulations for darts models that do not
    have a 'series' argument in their predict method, such as e.g. ARIMA models.

    NOTE: the predict method is implemented but returns a NotImplementedException if not overridden by
          a child class.

    """

    def __init__(self, model_type: str, signal_name: str, darts_model: ForecastingModel, fit_kwargs: dict = None):
        super().__init__(model_type, signal_name)
        self.darts_model = darts_model  # type: ForecastingModel
        self._fit_kwargs = fit_kwargs or dict()

    def batch_predict(
        self,
        data: pd.DataFrame,
        retrain_model: bool,
        first_sample: int,
        horizon: int,
        overlap_end: bool = False,
        stride: int = 1,
    ) -> List[Tuple[int, np.ndarray]]:

        # --- create joint TimeSeries ---------------------
        series = TimeSeries.from_series(data[self.signal_name])

        # --- historical_forecasts ------------------------
        ts_scaled_forecasts = self.darts_model.historical_forecasts(
            series=series,
            start=first_sample,
            forecast_horizon=horizon,
            stride=stride,
            retrain=retrain_model,
            last_points_only=False,
            overlap_end=overlap_end,
            verbose=True,
        )  # type: List[TimeSeries]

        # --- return in appropriate format ----------------
        return [
            (i, time_series.data_array().to_numpy().flatten())
            for i, time_series in zip(range(first_sample, len(data), stride), ts_scaled_forecasts)
        ]

    def fit(self, training_data: pd.DataFrame):
        ts_train = TimeSeries.from_series(training_data[self.signal_name])
        print("Training...   ", end="")
        self.darts_model.fit(ts_train, **self._fit_kwargs)
        print("Done.")

    def predict(self, history: pd.DataFrame, n_samples: int) -> np.ndarray:
        raise NotImplementedError(
            "predict method not implemented; possible the darts model does not support"
            "predictions from arbitrary past time series.  Use the batch_predict method instead."
        )
