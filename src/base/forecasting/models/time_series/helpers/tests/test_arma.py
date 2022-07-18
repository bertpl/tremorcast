from typing import List

import numpy as np
import pytest

from src.base.forecasting.models.time_series.helpers.arma import (
    arma_compute_e_hist,
    arma_compute_initial_e,
    arma_predict,
    arma_predict_with_e_hist,
)


@pytest.mark.parametrize(
    "x_init, a, b, expected_e_init",
    [
        (1.0, [1.0], [], 0.0),
        (1.0, [1.0, 0.0], [0.0], 0.0),
        (1.0, [0.8, 0.1], [0.0], 0.1),
        (1.0, [0.8, 0.1], [1.0], 0.05),
        (1.0, [0.8, 0.1], [-1.0], 0.0),
    ],
)
def test_arma_compute_initial_e(x_init: float, a: List[float], b: List[float], expected_e_init: float):

    # --- arrange -----------------------------------------
    a = np.array(a)
    b = np.array(b)

    # --- act ---------------------------------------------
    e_init = arma_compute_initial_e(a, b, x_init)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(e_init, expected_e_init, decimal=3)


def test_arma_compute_e_hist():

    # --- arrange -----------------------------------------
    a = np.array([0.8, 0.1])
    b = np.array([1.0])

    x_hist = np.array([1.0, 1.0, 1.1, 1.0])

    # e_init = ~0.05
    expected_e_hist = np.array([0.05, 0.05, 0.15, -0.13])

    # --- act ---------------------------------------------
    e_hist = arma_compute_e_hist(a, b, x_hist)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(e_hist, expected_e_hist, decimal=3)


def test_arma_predict_with_e_hist():

    # --- arrange -----------------------------------------
    a = np.array([0.8, 0.1])
    b = np.array([1.0])

    x_hist = np.array([1.0, 1.0, 1.1, 1.0])
    e_hist = np.array([0.05, 0.05, 0.15, -0.13])
    hor = 3

    # hand-calculated result:
    expected_x_pred = np.array([0.78, 0.724, 0.6572])

    # --- act ---------------------------------------------
    x_pred = arma_predict_with_e_hist(a, b, x_hist, e_hist, hor)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(x_pred, expected_x_pred, decimal=5)


def test_arma_predict_arma():

    # --- arrange -----------------------------------------
    a = np.array([0.8, 0.1])
    b = np.array([1.0])

    x_hist = np.array([1.0, 1.0, 1.1, 1.0])
    hor = 3

    # hand-calculated result:
    expected_x_pred = np.array([0.78, 0.724, 0.6572])

    # --- act ---------------------------------------------
    x_pred = arma_predict(a, b, x_hist, hor)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(x_pred, expected_x_pred, decimal=3)


def test_arma_predict_ar():

    # --- arrange -----------------------------------------
    a = np.array([0.8, 0.1])
    b = np.array([])

    x_hist = np.array([1.0, 1.0])
    hor = 3

    # hand-calculated result:
    expected_x_pred = np.array([0.9, 0.82, 0.746])

    # --- act ---------------------------------------------
    x_pred = arma_predict(a, b, x_hist, hor)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(x_pred, expected_x_pred, decimal=3)
