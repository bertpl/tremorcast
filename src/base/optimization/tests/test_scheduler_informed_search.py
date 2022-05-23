import numpy as np
import pytest

from src.base.optimization.scheduler_informed_search import AdaptiveSampler


# =================================================================================================
#  Test AdaptiveSampler
# =================================================================================================
@pytest.mark.parametrize(
    "values_count, best_results, focus, expected_pmf",
    [
        ([0, 0, 0], [np.nan, np.nan, np.nan], 0.5, [1 / 3, 1 / 3, 1 / 3]),
        ([1, 0, 0], [np.nan, np.nan, np.nan], 0.5, [0, 1 / 2, 1 / 2]),
        ([1, 1, 0], [np.nan, np.nan, np.nan], 0.5, [0, 0, 1]),
        ([1, 1, 1], [np.nan, np.nan, np.nan], 0.5, [1 / 3, 1 / 3, 1 / 3]),
        ([1, 1, 1], [10.0, np.nan, np.nan], 0.5, [1 / 3, 1 / 3, 1 / 3]),
        ([1, 1, 1], [10.0, 10.0, np.nan], 0.5, [1 / 3, 1 / 3, 1 / 3]),
        ([1, 1, 1], [10.0, 10.0, 10.0], 0.5, [1 / 3, 1 / 3, 1 / 3]),
        ([1, 1, 1], [10.0, 20.0, 10.0], 0.0, [1 / 3, 1 / 3, 1 / 3]),
        ([1, 1, 1], [10.0, 20.0, 10.0], 0.1, [1.0, np.exp(-1), 1.0] / np.sum([1.0, np.exp(-1), 1.0])),
        ([1, 1, 1], [10.0, 20.0, 10.0], 0.2, [1.0, np.exp(-2), 1.0] / np.sum([1.0, np.exp(-2), 1.0])),
        ([1, 1, 1], [10.0, 20.0, 13.0], 0.1, [1.0, np.exp(-1), np.exp(-0.3)] / np.sum([1.0, np.exp(-1), np.exp(-0.3)])),
        ([1, 1, 1], [10.0, 20.0, 10.0], 100, [0.5, 0.0, 0.5]),
        ([1, 1, 1], [10.0, 20.0, 15.0], 100, [1.0, 0.0, 0.0]),
    ],
)
def test_adaptive_sampler(values_count: list, best_results: list, focus: float, expected_pmf: list):

    # --- arrange -----------------------------------------
    sampler = AdaptiveSampler(values=[0, 1, 2])

    sampler.values_count = np.array(values_count)
    sampler.best_results = np.array(best_results)

    # --- act ---------------------------------------------
    pmf = sampler.get_sampling_pmf(focus)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(pmf, np.array(expected_pmf), decimal=3)
