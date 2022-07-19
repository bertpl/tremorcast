"""
Module with helper functions for prediction & fitting with ARMA time-series models.

Convention:
  - a: 1D np array with AR-coefficients of size p    (a[i] coefficient for x_{k-i-1})
  - b: 1D np array with MA-coefficients of size q    (b[i] coefficient for e_{k-i-1})
"""
from typing import Tuple

import numba
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from src.base.forecasting.models import build_toeplitz
from src.tools.math import remove_nan_rows


# =================================================================================================
#  Predict
# =================================================================================================
@numba.jit(fastmath=True)
def arma_predict(a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, hor: int) -> np.ndarray:

    e_hist = arma_compute_e_hist(a, b, x_hist)
    return arma_predict_with_e_hist(a, b, x_hist, e_hist, hor)


@numba.jit(fastmath=True)
def arma_predict_with_e_hist(
    a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, e_hist: np.ndarray, hor: int
) -> np.ndarray:
    """Returns e_hist array of same length as x_hist."""

    # --- extend x_hist, e_hist ---------------------------
    x = np.concatenate([x_hist, np.zeros(hor)])
    e = np.concatenate([e_hist, np.zeros(hor)])

    # --- simulate ----------------------------------------
    a_flip = np.flip(a).copy()
    b_flip = np.flip(b).copy()
    for i in range(x_hist.size, x.size):
        if a.size > 0:
            x[i] += np.dot(x[i - a.size : i], a_flip)
        if b.size > 0:
            x[i] += np.dot(e[i - b.size : i], b_flip)

    # --- return ------------------------------------------
    return x[x_hist.size :].copy()


# =================================================================================================
#  Initialization
# =================================================================================================
@numba.jit(fastmath=True)
def arma_compute_e_hist(a: np.ndarray, b: np.ndarray, x_hist: np.ndarray) -> np.ndarray:
    """Compute e_hist corresponding to x_hist"""

    # --- pad signal --------------------------------------
    n_padding = max(a.size, b.size)
    e_init = arma_compute_initial_e(a, b, x_hist[0])
    e_hist = np.concatenate((np.full(n_padding, e_init), np.zeros_like(x_hist)))
    x_hist = np.concatenate((np.full(n_padding, x_hist[0]), x_hist))

    # --- simulate and compute e --------------------------
    a_flip = np.flip(a).copy()
    b_flip = np.flip(b).copy()
    for i in range(n_padding, e_hist.size):
        x_pred = 0
        if a.size > 0:
            x_pred += np.dot(x_hist[i - a.size : i], a_flip)
        if b.size > 0:
            x_pred += np.dot(e_hist[i - b.size : i], b_flip)
        e_hist[i] = x_hist[i] - x_pred

    # --- return e_hist without padding -------------------
    return e_hist[n_padding:]


@numba.jit(fastmath=True)
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


@numba.jit(fastmath=True)
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
class ArmaFitMethods:
    POWELL = "Powell"
    BFGS = "BFGS"


def arma_fit(
    x: np.ndarray, p: int, q: int, wd: float = 0.0, method: str = None, tol: float = 1e-6, silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits ARMA model of orders (p, q) to provided data x.  We optimize parameters a, b such as to minimize
    the l2-norm of the error terms e.  Optionally l2 regularization can be applied to the coefficients by means
    of a 'weight decay' parameter wd.
    :param x: (1D np.ndarray) containing data to be fitted
    :param p: (int) AR order of model to be fitted
    :param q: (int) MA order of model to be fitted
    :param wd: (float, default=0) weight decay parameters for l2 regularization on model coefficients
    :param method: (str) method to be provided to scipy.optimize.minimize  (default=ArmaFitMethods.POWELL)
                              (possible values: only members of ArmaFitMethods)
    :param tol: (float, default=1e-6) tolerance parameter; convergence threshold for optimization algorithm.
    :param silent: (bool, default=False) when True, no output is generated.
    :return: (a, b)-tuple with AR- and MA-coefficient respectively as 1D np arrays.
    """

    # --- argument checking -------------------------------
    if x.size < 2 * (p + q):
        raise ValueError(
            f"Need at least {2*(p+q)} samples for fitting an ARMA model of order (p,q)=({p},{q}), here: {x.size}."
        )

    if method is None:
        method = ArmaFitMethods.POWELL

    # --- fitting function --------------------------------
    def arma_fitting_cost(coefs: np.ndarray) -> float:
        """
        Computes squared norm of e-terms (except for first p+q as warmup) + wd-weighted squared norm of coefficients.
        """
        a = coefs[:p]
        b = coefs[p:]
        e_hist = arma_compute_e_hist(a, b, x)
        return (np.linalg.norm(e_hist[p + q :]) ** 2) + wd * (np.linalg.norm(coefs) ** 2)

    # --- initial value -----------------------------------
    if p > 0:
        a_init = ar_fit(x, p, wd)
        b_init = np.zeros(q)
    else:
        a_init = np.zeros(p)
        b_init = np.zeros(q)

    c_init = np.concatenate((a_init, b_init))

    # --- optimize ----------------------------------------
    kwargs = dict()
    if method == ArmaFitMethods.BFGS:
        kwargs["jac"] = "3-point"

    res = minimize(arma_fitting_cost, c_init, method=method, tol=tol, **kwargs)  # type: OptimizeResult

    c_opt = res.x
    if (not res.success) and (not silent):
        print("--- WARNING: fitting ARMA model parameters did not successfully converge ---")

    # --- return ------------------------------------------
    return c_opt[:p], c_opt[p:]


def ar_fit(x: np.ndarray, p: int, wd: float = 0.0) -> np.ndarray:

    # --- argument checking -------------------------------
    if x.size < 2 * p:
        raise ValueError(f"Need at least {2*p} samples for fitting an AR model of order {p}, here: {x.size}.")

    # --- linear regression -------------------------------
    #  find coefficients as solution of A*c=b
    A = build_toeplitz(x, window_size=p + 1, forward=False)[:, 1:]
    b = build_toeplitz(x, window_size=1, forward=True)
    A, b = remove_nan_rows(A, b)

    # add regularization ('weight decay')
    if wd > 0:
        A = np.concatenate((A, np.sqrt(wd) * np.eye(p)), axis=0)
        b = np.concatenate((b, np.zeros((p, 1))), axis=0)

    # solve
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    # return
    return c.flatten()


# =================================================================================================
#  Testing
# =================================================================================================
def generate_arma_data(a: np.ndarray = None, b: np.ndarray = None, n: int = 100) -> np.ndarray:

    # --- prep --------------------------------------------
    if a is None:
        a = np.array([])
    if b is None:
        b = np.array([])
    p = a.size
    q = b.size
    n_ext = n + 2 * max(p, q)

    # --- init --------------------------------------------
    x = np.zeros(n_ext)
    e = np.random.standard_normal(n_ext)

    # --- simulate ----------------------------------------
    for i in range(max(p, q), n_ext):
        if p > 0:
            x[i] += np.dot(x[i - p : i], np.flip(a))
        if q > 0:
            x[i] += np.dot(e[i - q : i], np.flip(b))
        x[i] += e[i]

    # --- return ------------------------------------------
    return x[-n:].copy()  # return copy of last n elements
