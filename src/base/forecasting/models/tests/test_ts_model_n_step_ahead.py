import numpy as np
import pytest

from src.base.forecasting.models.ts_model_n_step_ahead import TimeSeriesModelMultiStepRegression


@pytest.mark.parametrize(
    "x_in, y_in, expected_x_out, expected_y_out",
    [
        # no rows with NaN
        (np.array([[0, 1], [2, 3]]), np.array([[0], [20]]), np.array([[0, 1], [2, 3]]), np.array([[0], [20]])),
        # 1 row with NaN
        (np.array([[np.nan, 1], [2, 3]]), np.array([[0], [20]]), np.array([[2, 3]]), np.array([[20]])),
        # all rows with NaN
        (np.array([[0, 1], [2, 3]]), np.array([[0], [np.nan]]), np.array([[0, 1]]), np.array([[0]])),
        (np.array([[0, np.nan], [2, 3]]), np.array([[0], [np.nan]]), np.zeros((0, 2)), np.zeros((0, 1))),
    ],
)
def test_remove_nan(x_in: np.ndarray, y_in: np.ndarray, expected_x_out: np.ndarray, expected_y_out: np.ndarray):

    # --- act ---------------------------------------------
    x_out, y_out = TimeSeriesModelMultiStepRegression._remove_nan(x_in, y_in)

    # --- assert ------------------------------------------
    np.testing.assert_array_almost_equal(x_out, expected_x_out)
    np.testing.assert_array_almost_equal(y_out, expected_y_out)
