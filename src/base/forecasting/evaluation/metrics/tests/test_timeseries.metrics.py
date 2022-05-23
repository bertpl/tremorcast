from typing import List

import numpy as np
import pytest

from src.base.forecasting.evaluation.metrics.timeseries_metrics import compute_max_accurate_lead_time


@pytest.mark.parametrize(
    "score_curve, threshold, expected_result",
    [
        # positive scores
        ([0.6, 0.4, 0.3, 0.2, 0.1], 0.5, 1.5),  # regular case
        ([0.6, 0.4, 0.55, 0.54, 0.53], 0.5, 1.5),  # non-monotonously decreasing score curve
        ([0.6, 0.5, 0.4, 0.3, 0.2], 0.8, 0.8 / (0.8 + 0.2)),  # corner case: threshold high
        ([0.6, 0.5, 0.4, 0.3, 0.2], 0.1, np.inf),  # corner case: threshold low
        # negative scores
        ([-0.2, -0.4, -0.6, -0.8, -1.0], -0.5, 2.5),  # regular case
        ([-0.2, -0.4, -0.6, -0.35, -1.0], -0.5, 2.5),  # non-monotonously decreasing score curve
        ([-0.2, -0.4, -0.6, -0.8, -1.0], -0.1, 0.1 / (0.1 + 0.1)),  # corner case: threshold high
        ([-0.2, -0.4, -0.6, -0.8, -1.0], -2.0, np.inf),  # corner case: threshold low
    ],
)
def test_compute_maximum_reliable_lead_time(score_curve: List[float], threshold: float, expected_result: float):

    # --- act ---------------------------------------------
    result = compute_max_accurate_lead_time(np.array(score_curve), threshold)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(result, expected_result)
