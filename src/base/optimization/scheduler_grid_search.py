import random
from itertools import product
from typing import Dict, Optional, Tuple

from .scheduler import Scheduler


# =================================================================================================
#  Grid Search
# =================================================================================================
class GridSearch(Scheduler):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, param_grid: Dict, shuffle: bool = True):

        # --- process param grid --------------------------
        n_params = 1
        for key, value in param_grid.items():
            assert isinstance(key, str), f"param_grid keys should be strings.  here: {key} = {type(key)}"
            assert isinstance(value, list), f"param_grid values should be lists.  here: {value} = {type(value)}"
            n_params *= len(value)

        # --- report --------------------------------------
        print(f"GridSearch over {n_params:_} parameter values.")

        # --- superclass constructor ----------------------
        super().__init__(param_grid, max_iter=n_params)

        # --- settings ------------------------------------
        self.param_grid = param_grid
        self.param_tuples = [tuple(param_values) for param_values in product(*param_grid.values())]
        if shuffle:
            random.Random(x=1).shuffle(self.param_tuples)  # reproducible random shuffle

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def _yield_next_param_tuple(self) -> Optional[Tuple]:
        if len(self.param_tuples) > 0:
            return self.param_tuples.pop()
        else:
            return None

    def _register_new_result(self, params: Tuple, result: float):
        pass  # nothing to do here

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def tuple_to_dict(self, param_tuple: Tuple) -> dict:
        return {param_name: param_value for param_name, param_value in zip(self.param_grid.keys(), param_tuple)}
