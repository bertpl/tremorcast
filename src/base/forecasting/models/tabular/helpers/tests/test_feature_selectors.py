import numpy as np

from src.base.forecasting.models.tabular.helpers.feature_selectors import FeatureSelector


def test_feature_selector():

    # --- arrange -----------------------------------------
    x = np.array(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35],
        ]
    )

    my_feature_selector = FeatureSelector(selected_indices=[0, 2, 3])

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
