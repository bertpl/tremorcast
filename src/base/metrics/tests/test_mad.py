import numpy as np

from src.base.metrics.mad import compute_mad_curve


def test_compute_mad_curve():

    # --- arrange -----------------------------------------
    observations = np.array([0, 1, 2, 2, 2])

    forecasts = [
        (0, np.array([0, 0, 0])),  # abs. deviations: [0, 1, 2]
        (1, np.array([1, 1, 1])),  # abs. deviations: [0, 1, 1]
        (2, np.array([2, 2, 2])),  # abs. deviations: [0, 0, 0]
        (3, np.array([2, 3])),  # abs. deviations: [0, 1]
    ]

    expected_mad_curve = np.array([0, 3 / 4, 3 / 3])

    # --- act ---------------------------------------------
    mad_curve = compute_mad_curve(observations, forecasts)

    # --- assert ------------------------------------------
    np.testing.assert_array_almost_equal(expected_mad_curve, mad_curve)
