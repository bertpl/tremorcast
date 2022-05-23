import math
from hashlib import shake_256
from typing import Any, List, Tuple

import fastai
import numpy as np
from scipy import optimize


# =================================================================================================
#  Exponentially spaced indices
# =================================================================================================
def exp_spaced_indices_fixed_base(n: int, exp_base: float) -> List[int]:
    """Returns indices (int) starting at 0 which are exponentially spaced, starting with spacing 1 and increasing
    with exp_base each sample"""

    assert exp_base >= 1.0, "exp_base should be >=1.0"

    if n == 1:
        return [0]
    else:
        spacings = [math.pow(exp_base, i + 1) for i in range(n - 1)]
        indices = [0] + [int(round(x)) for x in np.cumsum(spacings)]
        indices = [0] + list(np.cumsum(sorted(np.diff(indices))))  # make sure deltas are non-decreasing
        return indices


def exp_spaced_indices_fixed_max(n: int, max_index: int) -> List[int]:
    """Returns exponentially spaced indices (int) generated by exp_spaced_indices_fixed_base where exp_base
    is automatically tuned such that the largest index is exactly equal to max_index."""

    assert n >= 2, "need at least 2 indices (n >= 2)"
    assert max_index >= n - 1, "indices cannot be spaced less than 1 apart"

    if n == 2:

        return [0, max_index]

    elif max_index == n - 1:

        return list(range(n))

    else:

        def brentq_zero_function(exp_base: float):
            indices = exp_spaced_indices_fixed_base(n, exp_base)
            return max(indices) - max_index

        exp_base_min = math.pow(max_index / n, 1 / (n - 1))
        exp_base_max = math.pow(1 + max_index, 1 / (n - 1))  # should be conservative

        opt_exp_base, _ = optimize.brentq(brentq_zero_function, exp_base_min, exp_base_max, full_output=True)

        return exp_spaced_indices_fixed_base(n, opt_exp_base)


# =================================================================================================
#  Misc
# =================================================================================================
def remove_nan_rows(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    rows_without_nan = [not (any(x_row) or any(y_row)) for x_row, y_row in zip(np.isnan(x), np.isnan(y))]

    x = x[rows_without_nan]
    y = y[rows_without_nan]

    return x, y


# =================================================================================================
#  Random
# =================================================================================================
def set_all_random_seeds(value: Any):
    """Sets all seeds (random, numpy, torch) based on the provided hashable value, which is hashed into a seed."""

    seed = any_to_int_hash(value)
    fastai.torch_core.set_seed(seed, reproducible=True)  # make initialization as reproducible as possible


def any_to_int_hash(value: Any) -> int:
    """hashes any object into a non-trivial integer, also integer values"""

    # --- value -> seed -----------------------------------
    source = str((type(value).__name__, str(value)))

    h = shake_256()
    h.update(bytes(source, "utf-8"))
    seed = int.from_bytes(h.digest(8), "little", signed=True)

    # --- return ------------------------------------------
    return seed
