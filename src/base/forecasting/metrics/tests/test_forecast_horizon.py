from typing import List

import numpy as np
import pytest

from src.base.forecasting.metrics.forecast_horizon import compute_maximum_reliable_forecast_horizon


@pytest.mark.parametrize(
    "mad_curve, threshold, expected_result",
    [
        ([10, 20, 30, 40, 50], 15, 1.5),  # regular case
        ([10, 20, 11, 12, 13], 15, 1.5),  # non-monotonously increasing mad curve
        ([10, 20, 30, 40, 50], 5, 0.5),  # corner case: threshold low
        ([10, 20, 30, 40, 50], 60, np.inf),  # corner case: threshold high
    ],
)
def test_compute_maximum_reliable_forecast_horizon(mad_curve: List[float], threshold: float, expected_result: float):

    # --- act ---------------------------------------------
    result = compute_maximum_reliable_forecast_horizon(np.array(mad_curve), threshold)

    # --- assert ------------------------------------------
    assert result == expected_result
