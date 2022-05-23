from enum import Enum, auto
from typing import Tuple

import numpy as np


# =================================================================================================
#  Enum & main function
# =================================================================================================
class DataSetType(Enum):
    LINEAR = auto()
    SINE = auto()


def create_dataset(
    dataset_type: DataSetType, n: int, x_noise: float = 0.0, y_noise: float = 0.1, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """returns (x, y) based on requested dataset type & other parameters"""

    # --- get dataset -------------------------------------
    if dataset_type == DataSetType.LINEAR:
        x, y = _dataset_linear(n)
    elif dataset_type == DataSetType.SINE:
        x, y = _dataset_sine(n, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset type: {DataSetType}")

    # --- add noise ---------------------------------------
    x = x + x_noise * np.random.standard_normal(x.shape)
    y = y + y_noise * np.random.standard_normal(y.shape)

    # return
    return x, y


# =================================================================================================
#  Specific data sets
# =================================================================================================
def _dataset_linear(n: int) -> Tuple[np.ndarray, np.ndarray]:

    x = np.linspace(-1.0, 1.0, n).reshape(n, 1)
    y = x

    return x, y


def _dataset_sine(n: int, c: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:

    x = np.linspace(-1.0, 1.0, n).reshape(n, 1)
    y = np.sin(c * x)

    return x, y
