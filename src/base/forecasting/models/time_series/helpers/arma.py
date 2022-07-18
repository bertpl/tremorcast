"""
Module with helper functions for prediction & fitting with ARMA time-series models.

Convention:
  - a: 1D np array with AR-coefficients    (a[i] coefficient for x_{k-i-1})
  - b: 1D np array with MA-coefficients    (b[i] coefficient for e_{k-i-1})
"""
import numba
import numpy as np


# =================================================================================================
#  Predict
# =================================================================================================
@numba.jit()
def arma_predict(a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, hor: int) -> np.ndarray:

    if b.size > 0:
        e_hist = arma_compute_e_hist(a, b, x_hist)
    else:
        e_hist = np.zeros_like(x_hist)

    return arma_predict_with_e_hist(a, b, x_hist, e_hist, hor)


@numba.jit()
def arma_predict_with_e_hist(
    a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, e_hist: np.ndarray, hor: int
) -> np.ndarray:
    """Returns e_hist array of same length as x_hist."""

    # --- extend x_hist, e_hist ---------------------------
    x = np.concatenate([x_hist, np.zeros(hor)])
    e = np.concatenate([e_hist, np.zeros(hor)])

    # --- simulate ----------------------------------------
    for i in range(x_hist.size, x.size):
        if a.size > 0:
            x[i] += np.dot(x[i - a.size : i], np.flip(a))
        if b.size > 0:
            x[i] += np.dot(e[i - b.size : i], np.flip(b))

    # --- return ------------------------------------------
    return x[x_hist.size :].copy()


# =================================================================================================
#  Initialization
# =================================================================================================
@numba.jit
def arma_compute_e_hist(a: np.ndarray, b: np.ndarray, x_hist: np.ndarray) -> np.ndarray:
    """Compute e_hist corresponding to x_hist"""

    # --- pad signal --------------------------------------
    n_padding = max(a.size, b.size)
    e_init = arma_compute_initial_e(a, b, x_hist[0])
    e_hist = np.concatenate([np.full(n_padding, e_init), np.zeros_like(x_hist)])
    x_hist = np.concatenate([np.full(n_padding, x_hist[0]), x_hist])

    # --- simulate and compute e --------------------------
    for i in range(n_padding, e_hist.size):
        x_pred = 0
        if a.size > 0:
            x_pred += np.dot(x_hist[i - a.size : i], np.flip(a))
        if b.size > 0:
            x_pred += np.dot(e_hist[i - b.size : i], np.flip(b))
        e_hist[i] = x_hist[i] - x_pred

    # --- return e_hist without padding -------------------
    return e_hist[n_padding:]


@numba.jit
def arma_compute_initial_e(a: np.ndarray, b: np.ndarray, x_init: float) -> float:
    """Returns e_init corresponding to x_init, assuming steady-state"""

    # we want to compute e_init such that...
    #
    #     e_init = x_init - (sum(a) * x_init + sum(b) * e_init)
    # --> e_init = x_init - sum(a) * x_init - sum(b) * e_init
    # --> (1 + sum(b)) * e_init = (1 - sum(a)) * x_init
    # --> e_init = x_init * (1 - sum(a)) / (1 + sum(b))

    return x_init * regularized_division(
        num=1 - np.sum(a),
        den=1 + np.sum(b),
        max_result=10,  # never pick e_init larger than 10*x_init in abs value
    )


@numba.jit
def regularized_division(num: float, den: float, max_result: float) -> float:
    """
    We want to compute num/den without it going to 0 for den ± 0, but instead
    becoming linear as a function of num around den=0 and being limited to [-max_result, max_result]

       (num/den) / (1/c + (num/den)^2)
    =  den*num / (1/c + den^2)
    = c*den*num / (1 + c*den^2)
    """
    c = (2 * max_result) ** 2
    return (c * den * num) / (1 + c * (den**2))


# =================================================================================================
#  Fitting
# =================================================================================================
pass  # TODO
