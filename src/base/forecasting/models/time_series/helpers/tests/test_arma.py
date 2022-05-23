from typing import List

import numpy as np
import pytest

from src.base.forecasting.models.time_series.helpers.arma import (
    ar_fit,
    arma_compute_e_hist,
    arma_compute_initial_e,
    arma_fit,
    arma_predict,
    arma_predict_with_e_hist,
    generate_arma_data,
)


# =================================================================================================
#  AR(MA) - Predict
# =================================================================================================
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


# =================================================================================================
#  AR(MA) - Fit
# =================================================================================================
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_ar_fit(p: int):

    # --- arrange -----------------------------------------
    a_real = np.random.standard_normal(p)
    a_real = a_real * (0.8 / np.linalg.norm(a_real, ord=1))  # make sure model is stable
    n = 10_000
    x = generate_arma_data(a=a_real, n=n)

    # --- act ---------------------------------------------
    a_estimated = ar_fit(x, p=p, wd=0)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(a_real, a_estimated, decimal=1)


def test_arma_fit():

    # --- arrange -----------------------------------------
    a_real = np.array([0.8, 0.1])
    b_real = np.array([1.0])
    n = 10_000
    x = generate_arma_data(a=a_real, b=b_real, n=n)

    # --- act ---------------------------------------------
    a_estimated, b_estimated, cost = arma_fit(x, p=2, q=1)

    # --- assert ------------------------------------------
    np.testing.assert_almost_equal(a_real, a_estimated, decimal=2)
    np.testing.assert_almost_equal(b_real, b_estimated, decimal=2)
    assert cost >= 0


@pytest.mark.parametrize(
    "p, q, a_init, b_init",
    [
        # correct size
        (0, 3, [], [0.1, 0.2, 0.3]),
        (3, 0, [0.1, 0.2, 0.3], []),
        (2, 2, [0.1, 0.2], [0.2, 0.1]),
        # wrong size, but should be tolerated
        (2, 2, [0.1, 0.2, 0.0], [0.2]),
    ],
)
def test_arma_fit_ab_init(p: int, q: int, a_init: List[float], b_init: List[float]):

    # --- arrange -----------------------------------------
    a_real = np.array([0.8, 0.1, 0.01])
    b_real = np.array([0.9, 0.09])
    n = 1_000
    x = generate_arma_data(a=a_real, b=b_real, n=n)

    # --- act ---------------------------------------------
    a_estimated, b_estimated, cost = arma_fit(x, p=p, q=q, a_init=np.array(a_init), b_init=np.array(b_init))

    # --- assert ------------------------------------------
    assert len(a_estimated) == p
    assert len(b_estimated) == q
