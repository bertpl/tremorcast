import numpy as np
import pytest

from src.base.forecasting.models.helpers.toeplitz import build_toeplitz


@pytest.mark.parametrize(
    "time_series, window_size, forward, expected_result",
    [
        (
            np.array([0, 1, 2, 3]),
            3,
            False,
            np.array([[0, np.nan, np.nan], [1, 0, np.nan], [2, 1, 0], [3, 2, 1]]),
        ),
        (
            np.array([0, 1, 2, 3]),
            4,
            False,
            np.array([[0, np.nan, np.nan, np.nan], [1, 0, np.nan, np.nan], [2, 1, 0, np.nan], [3, 2, 1, 0]]),
        ),
        (
            np.array([0, 1, 2, 3]),
            3,
            True,
            np.array([[0, 1, 2], [1, 2, 3], [2, 3, np.nan], [3, np.nan, np.nan]]),
        ),
        (
            np.array([0, 1, 2, 3]),
            1,
            True,
            np.array([[0], [1], [2], [3]]),
        ),
        (
            np.array([0, 1, 2, 3]),
            1,
            False,
            np.array([[0], [1], [2], [3]]),
        ),
    ],
)
def test_moving_window_aggregation_build_toeplitz(
    time_series: np.ndarray, window_size: int, forward: bool, expected_result: np.ndarray
):

    # --- act ---------------------------------------------
    toeplitz = build_toeplitz(time_series, window_size, forward)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(toeplitz, expected_result)
