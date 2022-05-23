"""
Module with helper functions for prediction & fitting with ARMA time-series models.

Convention:
  - a: 1D np array with AR-coefficients of size p    (a[i] coefficient for x_{k-i-1})
  - b: 1D np array with MA-coefficients of size q    (b[i] coefficient for e_{k-i-1})
"""
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numba
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from src.tools.math import remove_nan_rows

from .toeplitz import build_toeplitz


# =================================================================================================
#  Predict
# =================================================================================================
def arma_predict(a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, hor: int, max_abs_value: float = 1e6) -> np.ndarray:

    e_hist = arma_compute_e_hist(a, b, x_hist, max_abs_value)
    return arma_predict_with_e_hist(a, b, x_hist, e_hist, hor, max_abs_value)


@numba.jit(fastmath=True)
def arma_predict_with_e_hist(
    a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, e_hist: np.ndarray, hor: int, max_abs_value
) -> np.ndarray:
    """Returns e_hist array of same length as x_hist."""

    # --- extend x_hist, e_hist ---------------------------
    x = np.concatenate((x_hist, np.zeros(hor)))
    e = np.concatenate((e_hist, np.zeros(hor)))

    # --- simulate ----------------------------------------
    a_flip = np.flip(a).copy()
    b_flip = np.flip(b).copy()
    for i in range(x_hist.size, x.size):
        if a.size > 0:
            x[i] += np.dot(x[i - a.size : i], a_flip)
        if b.size > 0:
            x[i] += np.dot(e[i - b.size : i], b_flip)
        x[i] = np.maximum(np.minimum(x[i], max_abs_value), -max_abs_value)

    # --- return ------------------------------------------
    return x[x_hist.size :].copy()


# =================================================================================================
#  Initialization
# =================================================================================================
@numba.jit(fastmath=True)
def arma_compute_e_hist(a: np.ndarray, b: np.ndarray, x_hist: np.ndarray, max_abs_value) -> np.ndarray:
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
        e_hist[i] = np.maximum(np.minimum(x_hist[i] - x_pred, max_abs_value), -max_abs_value)

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
def arma_fit(
    x: np.ndarray,
    p: int,
    q: int,
    wd: float = 0.0,
    tol: float = 1e-6,
    silent: bool = False,
    a_init: np.ndarray = None,
    b_init: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits ARMA model of orders (p, q) to provided data x.  We optimize parameters a, b such as to minimize
    the l2-norm of the error terms e.  Optionally l2 regularization can be applied to the coefficients by means
    of a 'weight decay' parameter wd.
    :param x: (1D np.ndarray) containing data to be fitted
    :param p: (int) AR order of model to be fitted
    :param q: (int) MA order of model to be fitted
    :param wd: (float, default=0) weight decay parameters for l2 regularization on model coefficients
    :param tol: (float, default=1e-6) tolerance parameter; convergence threshold for optimization algorithm.
    :param silent: (bool, default=False) when True, no output is generated.
    :param a_init: (np.ndarray) initial value of a (if not provided we start with values obtained from optimal AR-fit)
    :param b_init: (np.ndarray) initial value of b (if not provided, zeros are assumed)
                     NOTE: both a_init & b_init should be provided or none; if one is omitted, the other is ignored.
    :return: (a, b, cost)-tuple with...
               --> a, b:  AR- and MA-coefficients respectively as 1D np arrays.
               --> cost:  float representing the optimal cost value obtained: (sum(e^2) + wd*sum([a, b]^2)
    """

    # --- argument checking -------------------------------
    if x.size < 2 * (p + q):
        raise ValueError(
            f"Need at least {2*(p+q)} samples for fitting an ARMA model of order (p,q)=({p},{q}), here: {x.size}."
        )

    e_max_abs_value = 1e3 * max(np.abs(x))

    # --- fitting function --------------------------------
    def arma_fitting_cost(coefs: np.ndarray) -> float:
        """
        Computes squared norm of e-terms (except for first p+q as warmup) + wd-weighted squared norm of coefficients.
        """
        a = coefs[:p]
        b = coefs[p:]
        e_hist = arma_compute_e_hist(a, b, x, e_max_abs_value)
        return (np.linalg.norm(e_hist[p + q :]) ** 2) + wd * (np.linalg.norm(coefs) ** 2)

    # --- initial value -----------------------------------
    if (a_init is None) or (b_init is None):
        # a_init and/or b_init not set
        if p > 0:
            a_init = ar_fit(x, p, wd)
            b_init = np.zeros(q)
        else:
            a_init = np.zeros(p)
            b_init = np.zeros(q)
    else:
        # a_init & b_init set --> check sizes
        if len(a_init) < p:  # too small
            a_init = np.pad(a_init, (0, p - len(a_init)), mode="constant")
        else:  # possibly too large
            a_init = a_init[:p]
        if len(b_init) < q:  # too small
            b_init = np.pad(b_init, (0, q - len(b_init)), mode="constant")
        else:  # possibly too large
            b_init = b_init[:q]

    c_init = np.concatenate((a_init, b_init))

    # --- optimize ----------------------------------------
    solver_options = dict(disp=(not silent))
    res = minimize(arma_fitting_cost, c_init, method="Powell", tol=tol, options=solver_options)  # type: OptimizeResult

    c_opt = res.x
    if (not res.success) and (not silent):
        print("--- WARNING: fitting ARMA model parameters did not successfully converge ---")

    # --- return ------------------------------------------
    return c_opt[:p], c_opt[p:], res.fun


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
#  Robust Fitting
# =================================================================================================
def arma_fit_robust(
    x: np.ndarray, p: int, q: int, wd: float = 0.0, tol: float = 1e-6, silent: bool = False, robustness: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    This function performs multiple calls to arma_fit and returns the best (lowest-cost) solution.  The approach that
    is taken is to fit lower order models first and use those results to initialize fitting of higher order models.
    In this way we try to make sure that higher order models have fitting cost that is <= the fitting cost of lower
    order models, which is what we would expect if the optimization of arma_fit would always find the global optimum.

    The robustness parameter can be used to control speed vs robustness.

    :param x: (1D np.ndarray) containing data to be fitted
    :param p: (int) AR order of model to be fitted
    :param q: (int) MA order of model to be fitted
    :param wd: (float, default=0) weight decay parameters for l2 regularization on model coefficients
    :param tol: (float, default=1e-6) tolerance parameter; convergence threshold for optimization algorithm.
    :param silent: (bool, default=False) when True, no output is generated.
    :param robustness: (int, default=3, any value >= 0 allowed).
    :return: (a, b, cost)-tuple with...
               --> a, b:  AR- and MA-coefficients respectively as 1D np arrays.
               --> cost:  float representing the optimal cost value obtained: (sum(e^2) + wd*sum([a, b]^2)
    """

    # --- init --------------------------------------------
    fitting_schedule = determine_fitting_schedule(p, q, robustness)

    # fitted_models maps  (p_fit, q_fit) -> (a, b, fitting_cost)
    fitted_models = dict()  # type: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float]]

    # --- fitting procedure -------------------------------
    for (p_fit, q_fit), inits in fitting_schedule.items():

        if not silent:
            print(f"Fitting ARMA model with (p,q)=({p_fit},{q_fit}) ...")

        for pq_init in inits:

            msg = f"   Init with (p,q)={pq_init}".ljust(35)

            if pq_init is None:
                a_init = None
                b_init = None
            else:
                a_init = np.array(fitted_models[pq_init][0])
                b_init = np.array(fitted_models[pq_init][1])

            a, b, fitting_cost = arma_fit(x, p_fit, q_fit, wd, tol, silent=True, a_init=a_init, b_init=b_init)
            msg += f"--> {fitting_cost:.5f}".ljust(15)

            if ((p_fit, q_fit) not in fitted_models.keys()) or (fitting_cost < fitted_models[p_fit, q_fit][2]):
                fitted_models[p_fit, q_fit] = (a, b, fitting_cost)
                msg += " [better]"

            if not silent:
                print(msg)

    # --- return ------------------------------------------
    return fitted_models[p, q]


def determine_fitting_schedule(
    p: int, q: int, robustness: int
) -> Dict[Tuple[int, int], Set[Optional[Tuple[int, int]]]]:
    """
    Determines a robust fitting schedule determining which ARMA models of which orders need to be fitted, in which order
    & how they should be initialized.

    The idea behind this is that we will fit lower order models first and use those results as initializations for
    fitting higher order models, making sure that higher-order models will not end up with less optimal solutions than
    the lower ones, which is essentially what the user expects.

    The returned fitting schedule has the following structure:

        dictionary
            keys:   (p_fit, q_fit)                              --> ARMA model with these parameters should be fitted
            values: set with (p_init,q_init)-tuples or None     --> set of initializations that should be tried

                                                                     (p_init, q_init)
                                                                          -> initialize with this model
                                                                          -> should also be a key in the dict
                                                                          -> should be an 'earlier' key in the dict
                                                                     None
                                                                          -> use default initialization of arma_fit

    :param p: (int) p-parameter for which we want to eventually find a model   (all p_fit, p_init will be <= p)
    :param q: (int) q-parameter for which we want to eventually find a model   (all p_fit, p_init will be <= p)
    :param robustness: (int) allowed values: [0, 1]   (higher will result in more extensive fitting schedules
    :return: (dict) fitting schedule as described above.
    """

    # --- raw fitting schedule ----------------------------
    fitting_schedule = defaultdict(set)
    fitting_schedule[(p, q)].add(None)
    if robustness > 0:

        # determine 'pq-grid'
        pq_max = max(p, q)
        if robustness >= max(p, q):
            c_values = np.linspace(0, pq_max, robustness + 1) / pq_max
        else:
            c_values = np.linspace(0, robustness, robustness + 1) + (pq_max - robustness) * (
                np.linspace(0, 1, robustness + 1) ** 2
            )
            c_values = 1 - (c_values / pq_max)
        pq_values = sorted({(math.ceil(c * p), math.ceil(c * q)) for c in c_values})

        # build fitting schedule
        for (p_init, q_init), (p_fit, q_fit) in zip(pq_values, pq_values[1:]):
            # direction (0,0) -> (p,q)
            fitting_schedule[p_init, q_init].add(None)
            fitting_schedule[p_fit, q_fit].add((p_init, q_init))
            # direction (0,q) -> (p, q)
            fitting_schedule[p_init, q].add(None)
            fitting_schedule[p_fit, q].add((p_init, q))
            # direction (p, 0) -> (p, q)
            fitting_schedule[p, q_init].add(None)
            fitting_schedule[p, q_fit].add((p, q_init))

    # --- make sure keys are sorted & valid ---------------
    fitting_schedule = {
        key: fitting_schedule[key] for key in sorted(fitting_schedule.keys()) if are_valid_arma_pq(key[0], key[1])
    }

    # --- filter out invalid inits ------------------------
    fitting_schedule = {
        pq_fit: {
            pq_init
            for pq_init in inits
            if (pq_init is None)
            or (
                (pq_init in fitting_schedule.keys())
                and are_valid_arma_pq(pq_init[0], pq_init[1])
                and (pq_init < pq_fit)
            )
        }
        for pq_fit, inits in fitting_schedule.items()
    }

    # --- return ------------------------------------------
    return fitting_schedule


def are_valid_arma_pq(p: int, q: int) -> bool:
    return (p >= 0) and (q >= 0) and (p + q > 0)


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
