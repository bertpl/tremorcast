import numpy as np

from src.base.forecasting.models.time_series import TimeSeriesModelNaiveConstant


def test_ts_model_naive_constant():

    # --- arrange -----------------------------------------
    training_data = np.array([1, 2, 1, 2])
    history = np.array([3, 4, 5, 6])

    expected_forecast = np.array([6, 6, 6])

    # --- act ---------------------------------------------
    model = TimeSeriesModelNaiveConstant()
    model.fit(training_data)
    forecast = model.predict(history, 3)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(forecast, expected_forecast)
