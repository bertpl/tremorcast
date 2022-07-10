from typing import Optional

import numpy as np
import pytest

from src.base.forecasting.models.tabular import TabularRegressor
from src.base.forecasting.models.time_series.ts_model_ar import TimeSeriesModelAutoRegressive


# =================================================================================================
#  Mocks / Dummies
# =================================================================================================
class DummyTabularRegressor(TabularRegressor):
    def __init__(self):
        super().__init__("dummy")
        self.n_targets = None  # type: Optional[int]

    def _fit(self, x: np.ndarray, y: np.ndarray, **fit_params):
        self.n_targets = y.shape[1]

    def predict(self, x: np.ndarray, **predict_params) -> np.ndarray:
        # predictions use first feature and increment by 1, 2, 3, ... to construct targets
        return np.array([[x[i_row, 0] + j + 1 for j in range(self.n_targets)] for i_row in range(x.shape[0])])


class DummyTimeSeriesModel(TimeSeriesModelAutoRegressive):
    def __init__(self, n: int):
        super().__init__(p=1, n=n, regressor=DummyTabularRegressor())


# =================================================================================================
#  Test
# =================================================================================================
@pytest.mark.parametrize("n", [1, 2, 5])
def test_ts_model_ar_predict(n: int):

    # --- arrange -----------------------------------------
    ts_model = DummyTimeSeriesModel(n)

    x_hist = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    expected_result = np.array([91, 92, 93, 94, 95, 96, 97, 98, 99, 100])

    # --- act ---------------------------------------------
    ts_model.fit(x_hist)
    pred = ts_model.predict(x_hist, hor=10)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(expected_result, pred)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_ts_model_ar_batch_predict(n: int):

    # --- arrange -----------------------------------------
    ts_model = DummyTimeSeriesModel(n)

    first_sample = 3
    hor = 10
    overlap_end = False
    stride = 3

    x_hist = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    expected_result = [
        (3, np.array([21, 22, 23, 24, 25, 26, 27])),
        (6, np.array([51, 52, 53, 54])),
        (9, np.array([81])),
    ]

    # --- act ---------------------------------------------
    ts_model.fit(x_hist)
    pred = ts_model.batch_predict(x_hist, first_sample, hor, overlap_end, stride)

    # --- assert ------------------------------------------
    for (i_actual, pred_actual), (i_expected, pred_expected) in zip(pred, expected_result):
        assert i_actual == i_expected
        np.testing.assert_almost_equal(pred_actual, pred_expected)
