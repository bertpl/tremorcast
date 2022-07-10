from abc import ABC
from typing import List, Tuple

import numpy as np
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries

from src.base.forecasting.models.time_series.ts_model import TimeSeriesModel


class TimeSeriesModelDarts(TimeSeriesModel, ABC):
    """
    Implements a special case of TimeSeriesForecastModel (auto-scaled version) that wraps around a darts forecast model,
    where the batch_predict method makes use of the historical_forecasts method.  This might be more
    efficient, but especially allows us to easily perform such simulations for darts models that do not
    have a 'series' argument in their predict method, such as e.g. ARIMA models.

    NOTE: the predict method is implemented but returns a NotImplementedException if not overridden by
          a child class.

    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, model_type: str, darts_model: ForecastingModel, fit_kwargs: dict = None):
        super().__init__(model_type)
        self.darts_model = darts_model  # type: ForecastingModel
        self.fit_kwargs = fit_kwargs or dict()

    # -------------------------------------------------------------------------
    #  Fit / Predict
    # -------------------------------------------------------------------------
    def fit(self, x: np.ndarray):
        ts_train = TimeSeries.from_values(x.reshape((x.size(), 1)))
        print("Training...   ", end="")
        self.darts_model.fit(ts_train, **self.fit_kwargs)
        print("Done.")

    def predict(self, x_hist: np.ndarray, hor: int) -> np.ndarray:
        pass  # optional to be implemented by child classes, but won't always be possible due to Darts limitations

    def batch_predict(
        self,
        x: np.ndarray,
        first_sample: int,
        hor: int,
        overlap_end: bool = False,
        stride: int = 1,
    ) -> List[Tuple[int, np.ndarray]]:

        # --- create joint TimeSeries ---------------------
        series = TimeSeries.from_values(x.reshape((x.size(), 1)))

        # --- historical_forecasts ------------------------
        ts_forecasts = self.darts_model.historical_forecasts(
            series=series,
            start=first_sample,
            forecast_horizon=hor,
            stride=stride,
            retrain=False,
            last_points_only=False,
            overlap_end=overlap_end,
            verbose=True,
        )  # type: List[TimeSeries]

        # --- return in appropriate format ----------------
        return [
            (i, time_series.data_array().to_numpy().flatten())
            for i, time_series in zip(range(first_sample, x.size(), stride), ts_forecasts)
        ]
