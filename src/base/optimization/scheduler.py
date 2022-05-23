import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.tools.datetime import format_datetime, format_timedelta


# =================================================================================================
#  Parameter Scheduler
# =================================================================================================
class Scheduler(ABC):
    """Class that yields new parameters to try out based on the history of what was already tried."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        param_grid: Dict[str, List],
        max_iter: int = None,
        max_seconds: float = None,
    ):

        # settings
        self.param_grid = param_grid
        self.max_iter = max_iter
        self.max_seconds = max_seconds

        # internal book-keeping
        self._start_time = datetime.datetime.now()
        self._n_params_yielded = 0  # number of parameters yielded
        self._n_results_registered = 0  # number of parameters evaluated
        self._finished = False  # set to True when we decide to start wrapping up

        self._yielded_tuples = dict()  # type: Dict[Tuple, datetime.datetime]
        self._registered_results = dict()  # type: Dict[Tuple, float]

        # report some stats
        print()
        self.report_initial_stat("max iter", f"{self.max_iter:_}" if self.max_iter else "/")
        if self.max_seconds:
            max_sec = (
                f"{format_timedelta(self.max_seconds)} --> "
                f"[{format_datetime(datetime.datetime.now() + datetime.timedelta(seconds=self.max_seconds))}]"
            )
        else:
            max_sec = "/"
        self.report_initial_stat("max time", max_sec)

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    @property
    def seconds_elapsed(self) -> float:
        return (datetime.datetime.now() - self._start_time).total_seconds()

    @property
    def n_param_sets_yielded(self) -> int:
        return self._n_params_yielded

    @property
    def max_iter_reached(self) -> bool:
        return (self.max_iter is not None) and (self.n_param_sets_yielded >= self.max_iter)

    @property
    def max_seconds_reached(self) -> bool:
        return (self.max_seconds is not None) and (
            self.seconds_elapsed + (self.eta_evaluations_in_progress or 0.0) >= self.max_seconds
        )

    @property
    def n_results_registered(self) -> int:
        return self._n_results_registered

    @property
    def eta(self) -> Optional[float]:
        """Return estimated time ahead in seconds, based on max_iter & max_seconds. None if no estimate is available."""

        # --- in case when we're already wrapping up ------
        if self._finished:
            # only time remaining is to wait for the already-ongoing-evaluations to finish
            return self.eta_evaluations_in_progress

        # --- iter-based ETA ------------------------------
        if self.max_iter and self._n_results_registered:
            secs_per_iter = self.seconds_elapsed / self._n_results_registered
            iters_ahead = self.max_iter - self._n_results_registered
            eta_iter = iters_ahead * secs_per_iter
        else:
            eta_iter = np.inf

        # --- time-based ETA ------------------------------
        if self.max_seconds:

            # time-based ETA
            eta_secs = max([0, self.max_seconds - self.seconds_elapsed])

            # evaluations-already-in-progress also form a lower bound
            if (eta_in_progress := self.eta_evaluations_in_progress) is not None:
                eta_secs = max([eta_secs, eta_in_progress])

        else:
            eta_secs = np.inf

        # --- overall ETA ---------------------------------
        return None if np.isinf(eta := min([eta_iter, eta_secs])) else eta

    @property
    def eta_evaluations_in_progress(self) -> Optional[float]:
        """Tries to provide a rough estimate (secs) of remaining time to finish already-ongoing evaluations."""

        if self._n_results_registered:

            secs_per_iter = self.seconds_elapsed / self._n_results_registered
            evaluations_in_progress = self._n_params_yielded - self._n_results_registered

            # Some in-progress-evaluations might already be partially done, but we'll have to wait until
            # the last one has finished until we can actually stop; so not taking partially processed evaluations
            # into account here is the most accurate assumption.
            return secs_per_iter * evaluations_in_progress

        else:

            return None

    @property
    def best_results_per_parameter(self) -> Dict[str, Dict[Any, float]]:

        # --- init ----------------------------------------
        best_results = {
            param_name: {param_value: np.nan for param_value in param_values}
            for param_name, param_values in self.param_grid.items()
        }

        # --- loop over results ---------------------------
        for param_tuple, result in self._registered_results.items():
            if result is not None:
                for param_name, param_value in zip(self.param_grid.keys(), param_tuple):
                    current_result = best_results[param_name][param_value]
                    if np.isnan(current_result) or (result < current_result):
                        best_results[param_name][param_value] = result

        # --- return --------------------------------------
        return best_results

    # -------------------------------------------------------------------------
    #  API - Implemented
    # -------------------------------------------------------------------------
    def yield_next_param_tuple(self) -> Optional[Tuple]:
        if self.max_iter_reached or self.max_seconds_reached:
            self._finished = True
            return None
        else:
            next_param_tuple = self._yield_next_param_tuple()
            if next_param_tuple is not None:
                self._n_params_yielded += 1
                self._yielded_tuples[next_param_tuple] = datetime.datetime.now()
            else:
                self._finished = True
            return next_param_tuple

    def register_new_result(self, params: Tuple, result: Optional[float]):

        # register
        self._n_results_registered += 1
        self._registered_results[params] = result
        self._register_new_result(params, result)

    # -------------------------------------------------------------------------
    #  API - Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def _yield_next_param_tuple(self) -> Optional[Tuple]:
        """Yields next tuple with parameter values to be tried out.  Yields None if optimization should stop."""
        pass

    @abstractmethod
    def _register_new_result(self, params: Tuple, result: Optional[float]):
        """Registers new result when it's available."""
        pass

    # -------------------------------------------------------------------------
    #  Report statistics
    # -------------------------------------------------------------------------
    @staticmethod
    def report_initial_stat(name: str, value: Any):
        print(f"{name.rjust(10)} : {str(value)}")

    def report_stats(self):

        # --- early exit ----------------------------------
        if len(self._yielded_tuples) == 0:
            return

        # --- determine time slices -----------------------
        min_ts = min(ts for tup, ts in self._yielded_tuples.items())
        max_ts = max(ts for tup, ts in self._yielded_tuples.items())
        total_seconds = (max_ts - min_ts).total_seconds()  # type: float

        n_slices = 20
        time_thresholds = [
            min_ts + datetime.timedelta(seconds=c * total_seconds) for c in np.linspace(-0.001, 1, n_slices + 1)
        ]
        time_slices = list(zip(time_thresholds[:-1], time_thresholds[1:]))

        # --- report param by param -----------------------
        print()
        print(f"SAMPLING STATISTICS AFTER ±{format_timedelta(total_seconds)} ...")
        print()

        # best results for all values of all parameters
        best_results_per_parameter = self.best_results_per_parameter

        for i_param, (param_name, param_values) in enumerate(self.param_grid.items()):

            print(f"param '{param_name}':")
            print(
                "VALUE".rjust(13)
                + f"COUNTS PER TIME SLICE OF ±{format_timedelta(total_seconds/n_slices)}".rjust(7 + (5 * n_slices))
                + "TOTAL".rjust(25)
                + "BEST RESULT".rjust(21)
            )

            for i_value, param_value in enumerate(param_values):

                # best result for this value of this parameter
                best_result = best_results_per_parameter[param_name][param_value]

                # prefix
                s = f"   {str(param_value): >10}    -> "

                # total count & counts per slice
                total_cnt = 0
                for ts_start, ts_end in time_slices:
                    cnt_in_slice = len(
                        [
                            None
                            for tup, ts in self._yielded_tuples.items()
                            if ts_start < ts <= ts_end and tup[i_param] == param_value
                        ]
                    )
                    total_cnt += cnt_in_slice
                    if cnt_in_slice > 0:
                        s += f"{cnt_in_slice: >5_}"
                    else:
                        s += f"    ."

                # postfix
                pct = 100 * (total_cnt / len(self._yielded_tuples))
                br = best_result
                if not np.isnan(br) and br == np.nanmin(list(best_results_per_parameter[param_name].values())):
                    indicator = "(*)"
                else:
                    indicator = ""
                s += f"      -> {int(total_cnt): >6_}  [{pct: >5.1f}%]    ->      {br:9.3e}   {indicator}"

                print(s)

            print()
