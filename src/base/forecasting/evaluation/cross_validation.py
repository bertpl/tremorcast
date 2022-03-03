import random
from enum import Enum
from itertools import product
from typing import List, Tuple

import numpy as np


# =================================================================================================
#  Data splitting
# =================================================================================================
def split_cv_data(
    x: np.ndarray, n_splits: int, i_split: int, randomize: bool = True, seed: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a matrix into training & validation data for use in k-fold cross-validation.  The matrix is split
    in row-wise fashion and can represent either features (x) or targets (y).
    :param x: (n, k) numpy array
    :param n_splits: (int) number of splits (k) in k-fold cross-validation
    :param i_split: (int) index of split to generate (0 ... n-splits-1)
    :param randomize: (bool, default=True) randomize before splitting.
    :param seed: (int, default=-1) seed used to make randomization reproducible
    :return: tuple (x_train, x_val)
    """

    # --- generate split indices --------------------------
    n_rows = x.shape[0]

    indices = sorted([i % n_splits for i in range(n_rows)])
    if randomize:
        random.seed(seed)
        random.shuffle(indices)

    i_train = list(filter(lambda i: indices[i] != i_split, range(n_rows)))
    i_val = list(filter(lambda i: indices[i] == i_split, range(n_rows)))

    # --- split & return ----------------------------------
    return x[i_train, :], x[i_val, :]


# =================================================================================================
#  Parameter grid processing
# =================================================================================================
def param_grid_dict_to_list(param_grid: dict) -> List[dict]:
    """
    Converts parameter grid dictionary to list of all combinations.

    Example

        input:
            {
                "param_a": [1, 2]
                "param_b": [10, 20]
            }

        output:
            [
                {"param_a": 1, "param_b": 10},
                {"param_a": 1, "param_b": 20},
                {"param_a": 2, "param_b": 10},
                {"param_a": 2, "param_b": 20},
            ]

    """

    return [
        {param_name: param_value for param_name, param_value in zip(param_grid.keys(), param_values)}
        for param_values in product(*param_grid.values())
    ]


# =================================================================================================
#  Optimal parameter selection
# =================================================================================================
class ParamSelectionMethod(Enum):
    OPTIMAL = 1  # look at lowest mean validation score
    BALANCED = 2  # choose the least complex model that is within 1 sigma of optimal validation score


def select_params(cv_results: list, method: ParamSelectionMethod) -> dict:
    """
    Selects best set of parameters based on detailed cross-validation result using the specified selection method.

    cv_results:  list of dicts, with each dict containing following keys:
            "params": dict with parameter values as key-value pairs
            "complexity": indicator of model complexity for this set of parameters
            "training_losses": dict with training set losses information (see below)
            "validation_losses": dict with validation set losses information (see below)

        losses_dict: dict with loss information:
            "all":      list with all losses for all splits
            "mean":     mean value of all losses
            "std":      standard deviation of all losses
            "sigma":    estimate of uncertainty on mean loss (as std / sqrt(n_splits))

    :param cv_results: list with detailed CV results.  See above for details.
    :param method: ParamSelectionMethod.
    :return: element of cv_results list that represents the best set of parameters based on the selected criterion.
    """

    if method == ParamSelectionMethod.OPTIMAL:

        lowest_mean_val_loss = min([result["validation_losses"]["mean"] for result in cv_results])
        return next(filter(lambda result: result["validation_losses"]["mean"] == lowest_mean_val_loss, cv_results))

    elif method == ParamSelectionMethod.BALANCED:

        # first fetch optimal result and determine mean+sigma validation loss threshold
        optimal_result = select_params(cv_results, ParamSelectionMethod.OPTIMAL)
        validation_loss_threshold = (
            optimal_result["validation_losses"]["mean"] + optimal_result["validation_losses"]["std"]
        )

        # select least complex result with mean validation loss below threshold
        selected_result = None
        for result in cv_results:
            if (result["validation_losses"]["mean"] <= validation_loss_threshold) and (
                (selected_result is None)
                or (
                    (result["complexity"], result["validation_losses"]["mean"])
                    < (selected_result["complexity"], selected_result["validation_losses"]["mean"])
                )
            ):

                selected_result = result

        return selected_result
