import datetime

import numpy as np
import pandas as pd

from src.base.forecasting.models.ts_model_naive_mean import TimeSeriesModelNaiveMean


def test_ts_model_naive_mean():

    # --- arrange -----------------------------------------
    training_data = pd.DataFrame(
        columns=["signal_a", "signal_b"],
        index=[datetime.datetime(2022, 12, 6, 0, 0), datetime.datetime(2022, 12, 7, 0, 0)],
        data=[[1, 10], [2, 20]],
    )

    history = pd.DataFrame(
        columns=["signal_a", "signal_c"],
        index=[datetime.datetime(2022, 12, 9, 0, 0), datetime.datetime(2022, 12, 10, 0, 0)],
        data=[[3, 30], [4, 40]],
    )

    expected_forecast = np.array([1.5, 1.5, 1.5])

    # --- act ---------------------------------------------
    model = TimeSeriesModelNaiveMean("signal_a")
    model.fit(training_data)
    forecast = model.predict(history, 3)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(forecast, expected_forecast)
