import random
from typing import Dict, Optional, Set, Tuple

from .scheduler import Scheduler


# =================================================================================================
#  Random Search
# =================================================================================================
class RandomSearch(Scheduler):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        param_grid: Dict,
        max_iter: int = None,
        max_seconds: int = None,
    ):

        # --- process param grid --------------------------
        n_params = 1
        for key, value in param_grid.items():
            assert isinstance(key, str), f"param_grid keys should be strings.  here: {key} = {type(key)}"
            assert isinstance(value, list), f"param_grid values should be lists.  here: {value} = {type(value)}"
            n_params *= len(value)

        # --- report --------------------------------------
        print(f"RandomSearch over {n_params:_} parameter values.")

        # --- superclass constructor ----------------------
        max_iter = min([max_iter or n_params, n_params])
        super().__init__(
            param_grid,
            max_iter=max_iter,
            max_seconds=max_seconds,
        )

        # --- settings ------------------------------------
        self.param_grid = param_grid
        self.random = random.Random(x=1)

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def _yield_next_param_tuple(self) -> Optional[Tuple]:

        # select new tuple
        new_tuple = None
        while (new_tuple is None) or (new_tuple in self._yielded_tuples.keys()):
            new_tuple = tuple([self.random.choice(param_values) for param_values in self.param_grid.values()])

        # register & return new tuple
        return new_tuple

    def _register_new_result(self, params: Tuple, result: float):
        pass  # nothing to do here

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def tuple_to_dict(self, param_tuple: Tuple) -> dict:
        return {param_name: param_value for param_name, param_value in zip(self.param_grid.keys(), param_tuple)}
