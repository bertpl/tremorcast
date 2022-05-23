import math
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance

from .scheduler import Scheduler

PHASES = [(INIT_SEARCH := 0), (LINE_SEARCH := 1), (PLANAR_SEARCH := 2), (RANDOM_SEARCH := 3)]


# =================================================================================================
#  LineSearch
# =================================================================================================
class LineSearch(Scheduler):
    """
    Scheduler that searches in 4 phases:
      1) initial random search to get a reasonable starting point
      2) 1D line searches starting from the current optimum until a local optimum is found
      3) 2D planar search starting from current optimum  (switching back to phase 2 each time we find a new one)
      4) random search until all parameter candidates are evaluated
    All within a preset time or iteration limit.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        param_grid: Dict,
        max_iter: int = None,
        max_seconds: int = None,
        closest_first: bool = True,
        init_iters: int = None,
        hint: dict = None,
    ):

        # --- process param grid --------------------------
        n_params = 1
        for key, value in param_grid.items():
            assert isinstance(key, str), f"param_grid keys should be strings.  here: {key} = {type(key)}"
            assert isinstance(value, list), f"param_grid values should be lists.  here: {value} = {type(value)}"
            n_params *= len(value)

        # --- report --------------------------------------
        print(f"LineSearch over {n_params:_} parameter values.")

        # --- superclass constructor ----------------------
        max_iter = min([max_iter or n_params, n_params])
        super().__init__(
            param_grid,
            max_iter=max_iter,
            max_seconds=max_seconds,
        )

        # --- settings ------------------------------------
        self.closest_first = closest_first
        self.hint = hint

        # --- state ---------------------------------------
        self.random = np.random.default_rng(1)

        # some param_grid dimensions
        self._n_params = len(self.param_grid.keys())
        self._n_param_values = [len(param_values) for param_values in param_grid.values()]

        # set initial phase of the algorithm
        self.init_iters = init_iters  # type: Optional[int]
        self._phase = min(PHASES)  #  type: int

        # tuple if indices of current reference (=optimum)
        self._current_ref = tuple([0] * self._n_params)  # type : Tuple[int]
        self._current_ref_result = None  # type: Optional[float]

        # queue of tuples to be tried out before moving to the next phase
        self._queue = Queue(maxsize=0)  # type: Queue[Tuple[int]]
        self._refresh_queue()

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def _refresh_queue(self):
        """empty queue and refill based on self._current_ref and self._phase"""

        # empty queue
        while not self._queue.empty():
            self._queue.get()

        # --- 1D line search ------------------------------
        if self._phase in [INIT_SEARCH, LINE_SEARCH, PLANAR_SEARCH]:

            # unique tuples to be evaluated
            if self._phase == INIT_SEARCH:

                # determine number of initial iterations
                if self.init_iters is None:
                    n_init_min = max(self._n_param_values)
                    n_init_max = min(self.max_iter, (sum(self._n_param_values) - self._n_params) * self._n_params)
                    n_init = int(math.sqrt(n_init_min * n_init_max))
                    self.report_initial_stat(
                        "init iter", f"{n_init} = geo_mean({n_init_min:_}, {n_init_max:_})  [DEFAULT]"
                    )
                else:
                    n_init = self.init_iters
                    self.report_initial_stat("init iter", f"{n_init}   [USER-SPECIFIED]")

                # determine random tuples with uniform distribution over all parameter values
                indexes_per_param = [
                    [i % n for i in range(n_init)] for n in self._n_param_values
                ]  # type: List[List[int]]

                for i in range(self._n_params):
                    self.random.shuffle(indexes_per_param[i])

                # generate n_init initial trials
                new_index_tuples = {
                    tuple([indexes_per_param[i_param][i_tup] for i_param in range(self._n_params)])
                    for i_tup in range(n_init)
                }

                # also add 'hint'
                if self.hint is not None:
                    self._current_ref = tuple(
                        [self.param_grid[param].index(self.hint[param]) for param in self.param_grid.keys()]
                    )
                    new_index_tuples.add(self._current_ref)

            elif self._phase == LINE_SEARCH:
                new_index_tuples = {
                    self._current_ref[:i_param] + (idx,) + self._current_ref[i_param + 1 :]
                    for i_param in range(self._n_params)
                    for idx in range(self._n_param_values[i_param])
                }
            else:
                new_index_tuples = {
                    self._current_ref[:i_param1]
                    + (idx1,)
                    + self._current_ref[i_param1 + 1 : i_param2]
                    + (idx2,)
                    + self._current_ref[i_param2 + 1 :]
                    for i_param1 in range(self._n_params - 1)
                    for i_param2 in range(i_param1 + 1, self._n_params)
                    for idx1 in range(self._n_param_values[i_param1])
                    for idx2 in range(self._n_param_values[i_param2])
                }

            # sort by distance from self._current_ref
            new_index_tuples_sorted = sorted(new_index_tuples)
            if self.closest_first and self._phase != INIT_SEARCH:
                new_index_tuples_sorted = sorted(
                    new_index_tuples, key=lambda t: distance.euclidean(t, self._current_ref)
                )
            else:
                self.random.shuffle(new_index_tuples_sorted)

            # add to queue
            for new_index_tuple in new_index_tuples_sorted:
                self._queue.put(new_index_tuple)

        # --- random search -------------------------------
        if self._phase == RANDOM_SEARCH:
            pass  # don't do anything, we sample randomly, but don't populate the queue up front

    def _yield_next_param_tuple(self) -> Optional[Tuple]:

        # --- LINE_SEARCH & PLANAR_SEARCH -----------------
        while self._phase in [INIT_SEARCH, LINE_SEARCH, PLANAR_SEARCH]:

            while not self._queue.empty():

                new_index_tuple = self._queue.get()  # type: Tuple[int]
                new_tuple = tuple(
                    [param_values[idx] for idx, param_values in zip(new_index_tuple, self.param_grid.values())]
                )

                if new_tuple not in self._yielded_tuples.keys():
                    return new_tuple

            # if queue is empty, move on to next phase and refresh queue
            self._phase += 1
            self._refresh_queue()

        # --- RANDOM_SEARCH -------------------------------
        if self._phase == RANDOM_SEARCH:
            new_tuple = None
            while (new_tuple is None) or (new_tuple in self._yielded_tuples):
                new_tuple = tuple([self.random.choice(param_values) for param_values in self.param_grid.values()])
            return new_tuple

    def _register_new_result(self, params: Tuple, result: float):
        if result is not None:
            if (self._current_ref_result is None) or (result < self._current_ref_result):

                # update current ref
                self._current_ref_result = result
                self._current_ref = tuple(
                    [
                        param_values.index(param_value)
                        for param_value, param_values in zip(params, self.param_grid.values())
                    ]
                )

                # refresh queue
                if self._phase != INIT_SEARCH:
                    self._phase = LINE_SEARCH
                    self._refresh_queue()

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def tuple_to_dict(self, param_tuple: Tuple) -> dict:
        return {param_name: param_value for param_name, param_value in zip(self.param_grid.keys(), param_tuple)}
