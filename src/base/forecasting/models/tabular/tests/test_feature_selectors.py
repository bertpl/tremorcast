from typing import List

import numpy as np
import pytest

from src.base.forecasting.models.tabular.feature_selectors import (
    FeatureSelector,
    FeatureSelector_All,
    FeatureSelector_ExponentialSpacing,
)


def test_feature_selector_base_class():

    # --- arrange -----------------------------------------
    x = np.array(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35],
        ]
    )

    class MyFeatureSelector(FeatureSelector):
        def _get_feature_indices(self, n_features_total: int):
            return [0, 2, 3]

    my_feature_selector = MyFeatureSelector()

    # --- act ---------------------------------------------
    y = my_feature_selector.fit_transform(x)

    # --- assert ------------------------------------------
    np.testing.assert_equal(
        y,
        np.array(
            [
                [10, 12, 13],
                [20, 22, 23],
                [30, 32, 33],
            ]
        ),
    )


@pytest.mark.parametrize("n_features_total, n_rows", [(1, 100), (5, 100), (20, 100), (1, 1), (100, 1)])
def test_feature_selector_all(n_features_total: int, n_rows: int):

    # --- arrange -----------------------------------------
    x = np.random.random((n_rows, n_features_total))

    feature_selector = FeatureSelector_All()

    # --- act ---------------------------------------------
    y = feature_selector.fit_transform(x)

    # --- assert ------------------------------------------
    np.testing.assert_equal(x, y)


@pytest.mark.parametrize(
    "n_features_total, first_index, last_index, n_selected_features, reverse, expected_indices",
    [
        (10, None, None, None, False, list(range(10))),
        (10, 2, None, None, False, list(range(2, 10))),
        (10, None, 7, None, False, list(range(8))),
        (10, 3, 7, None, False, list(range(3, 8))),
        (10, None, None, 5, False, [0, 1, 3, 6, 9]),
        (10, None, None, 5, True, [0, 3, 6, 8, 9]),
        (10, 2, 4, 8, False, [2, 3, 4]),
        (10, 7, 4, None, False, [7]),
        (10, 7, 4, 10, False, [7]),
    ],
)
def test_feature_selector_exponential(
    n_features_total: int,
    first_index: int,
    last_index: int,
    n_selected_features: int,
    reverse: bool,
    expected_indices: List[int],
):

    # --- arrange -----------------------------------------
    x = np.zeros((10, n_features_total))
    selector = FeatureSelector_ExponentialSpacing(first_index, last_index, n_selected_features, reverse)

    # --- act ---------------------------------------------
    selector.fit(x)

    # --- assert ------------------------------------------
    assert selector._selected_indices == expected_indices
