import datetime
import os
from concurrent import futures
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.tools.datetime import format_datetime, format_timedelta
from src.tools.processes import kill_all_child_processes

from .scheduler import Scheduler


# =================================================================================================
#  Optimizer task
# =================================================================================================
@dataclass(frozen=True)
class OptimizerTask:
    task_idx: int
    future: Future
    params: Tuple
    start_time: datetime.datetime
    i_try: int = 1
    n_max_tries: int = 5


# =================================================================================================
#  Parallel Optimizer
# =================================================================================================
class ParallelOptimizer:
    """
    Class that can optimize object functions by evaluating multiple candidate parameters in parallel.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, n_workers: int = -1, debug: bool = False, report_stats_freq_secs: int = 1_800):
        """
        Initialize a new parallel optimizer.
        :param n_workers: (int, default=-1) number of worker processes.
                            when negative:  n_actual_workers = cpu_count() + n_workers
        :param debug: (bool, default=False) if True, exceptions occurred within objective evaluation are re-raised
        :param report_stats_freq_secs: (int, default=1800) report sampling statistics every so many seconds; set to <=0 to disable
        """

        # --- settings ------------------------------------
        if n_workers <= 0:
            self.n_workers = os.cpu_count() + n_workers
        else:
            self.n_workers = n_workers

        self.debug = debug

        # --- results -------------------------------------
        self.results = dict()  # type: Dict[Tuple, Any]
        self.best_result = None  # type: Optional[Any]
        self.best_params = None  # type: Optional[Tuple]

        # --- task mgmt -----------------------------------
        self._active_tasks = []  # type: List[OptimizerTask]
        self._executor = None  # type: Optional[ProcessPoolExecutor]
        self._objective = None  # type: Optional[Callable]
        self._scheduler = None  # type: Optional[Scheduler]
        self._fixed_param = None

        # --- stats reporting -----------------------------
        if report_stats_freq_secs <= 0:
            report_stats_freq_secs = datetime.timedelta(weeks=1_000).total_seconds()
        self._report_stats_freq_secs = report_stats_freq_secs
        self._report_stats_next_ts = datetime.datetime.now() + datetime.timedelta(seconds=report_stats_freq_secs)

    # -------------------------------------------------------------------------
    #  Optimization
    # -------------------------------------------------------------------------
    def optimize(self, objective: Callable, scheduler: Scheduler, fixed_param: Any = None) -> Tuple:
        """
        Find the optimum (minimum) of the provided object function by means of using the
        provided ParameterScheduler, which decides when which parameter-tuple should be evaluated.

        The objective is evaluated in parallel using multi-processing, for speed.

        The objective should be a Callable that accepts *args and returns a result that is convertible to a float,
        by either being a float, being another numeric type such as int or by being an object that implements the
        __float__ method.  It is acceptable that the objective function returns None if some parameter combinations
        are invalid.

        :param objective: Callable that takes *args and returns a comparable object
        :param scheduler: ParameterScheduler that should be configured in a compatible way with the actual objective
                          function, i.e. it should know how many and which parameters the objective expects.
        :param fixed_param: (default=None) when set, this fixed parameter is provided to the objective function as a
                            first parameter, then the other parameters over which we're optimizing.
        :return: Optimal parameter set.
        """

        # --- init ----------------------------------------
        print()
        kill_all_child_processes(sigkill=False)
        self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        self._objective = objective
        self._scheduler = scheduler
        self._fixed_param = fixed_param

        # --- main loop -----------------------------------
        cont = True
        while cont:

            # fill tasks queue
            cont = self._fill_active_tasks_queue()

            # wait for next task to finish & update bookkeeping
            if cont:
                self._wait_for_new_result_and_register()
                self._report_stats_if_needed()

        # --- terminate -----------------------------------
        print(
            f"\nParallel search algorithm wrapping up. Waiting for {len(self._active_tasks)} evaluations to finish...\n"
        )
        while len(self._active_tasks) > 0:
            self._wait_for_new_result_and_register()

        print(
            f"\n{self._scheduler.n_param_sets_yielded:_} evaluations "
            f"in {format_timedelta(self._scheduler.seconds_elapsed)}.\n"
        )
        self._scheduler.report_stats()

        # --- clean up ------------------------------------
        print(f"Closing down worker processes...".ljust(40), end="")
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        kill_all_child_processes(sigkill=False)
        print("Done.")

        self._active_tasks = []
        self._objective = None
        self._scheduler = None

        # --- return result -------------------------------
        return self.best_params

    # -------------------------------------------------------------------------
    #  Internal - Algorithm
    # -------------------------------------------------------------------------
    def _add_task_to_queue_robust(self, params: Tuple, task_idx: int, i_try: int = 1):
        """Wraps around _ad_task_to_queue and restarts the process pool if it is broken"""

        try:
            self._add_task_to_queue(params, task_idx, i_try)
        except BrokenProcessPool:
            self._restart_process_pool()
            self._add_task_to_queue(params, task_idx, i_try)

    def _add_task_to_queue(self, params: Tuple, task_idx: int, i_try: int = 1):

        # --- add to pool -> create future ----------------
        if self._fixed_param is not None:
            extended_params = tuple([self._fixed_param, *params])
            future = self._executor.submit(self._objective, *extended_params)  # type: Future
        else:
            future = self._executor.submit(self._objective, *params)  # type: Future

        # --- create task & report ------------------------
        task = OptimizerTask(task_idx, future, params, datetime.datetime.now(), i_try=i_try)
        self._active_tasks.append(task)
        self._report_new_task(task)

    def _restart_process_pool(self):

        # kill old process pool
        self._report_restarting_pool()
        self._executor = None
        kill_all_child_processes(sigterm=True, sigkill=True, verbose=False)

        # set up new pool
        self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        self._report_pool_restarted()

        # add all tasks to new one
        tasks_to_restart = self._active_tasks.copy()
        self._active_tasks = []
        for task in tasks_to_restart:
            self._add_task_to_queue(task.params, task.task_idx, task.i_try + 1)

    def _fill_active_tasks_queue(self) -> bool:
        """make sure active_tasks list is filled up, until scheduler wants to stop.  In case we return False"""

        cont = True

        while cont and len(self._active_tasks) < self.n_workers:
            # active_tasks not long enough -> add a new task and keep track of the resulting Future
            params = self._scheduler.yield_next_param_tuple()

            if params is not None:
                self._add_task_to_queue_robust(params, task_idx=self._scheduler.n_param_sets_yielded)
            else:
                # algorithm should terminate
                cont = False

        return cont

    def _wait_for_new_result_and_register(self):
        """wait for at least 1 task to finish, extract result & update internal bookkeeping"""

        # --- wait for new task to finish -----------------
        done_futures, not_done_futures = futures.wait(
            fs=[task.future for task in self._active_tasks], return_when=FIRST_COMPLETED
        )

        # --- go over all finished futures ----------------
        done_tasks = [task for task in self._active_tasks if task.future in done_futures]
        for done_task in done_tasks:

            # already remove this task from _active_tasks
            self._active_tasks = [task for task in self._active_tasks if task != done_task]

            # try to extract & register result
            try:

                # extract result
                result = done_task.future.result()

                # only store result if it is not None
                if result is not None:

                    # store result
                    self.results[done_task.params] = result

                    # update best_result
                    if (self.best_result is None) or (float(result) < float(self.best_result)):
                        self.best_result = result
                        self.best_params = done_task.params

                # report
                self._report_new_result(done_task, result)

                # register with scheduler
                self._scheduler.register_new_result(done_task.params, float(result) if result is not None else None)

            except:

                self._report_task_failed(done_task)
                if done_task.i_try < done_task.n_max_tries:
                    # retry
                    self._add_task_to_queue_robust(
                        done_task.params, task_idx=done_task.task_idx, i_try=done_task.i_try + 1
                    )
                else:
                    # maximum number of tries reached
                    self._scheduler.register_new_result(done_task.params, None)
                    if self.debug:
                        raise

    # -------------------------------------------------------------------------
    #  Internal - reporting
    # -------------------------------------------------------------------------
    def _report_stats_if_needed(self):
        if datetime.datetime.now() > self._report_stats_next_ts:
            self._scheduler.report_stats()
            self._report_stats_next_ts += datetime.timedelta(seconds=self._report_stats_freq_secs)

    def _report_restarting_pool(self):
        prefix = self._reporting_prefix()  # 45 chars
        print(f"{prefix} RESTARTING PROCESS POOL ".ljust(180, "."))

    def _report_pool_restarted(self):
        prefix = self._reporting_prefix()  # 45 chars
        print(f"{prefix} PROCESS POOL RESTARTED ".ljust(180, "."))

    def _report_new_task(self, task: OptimizerTask):
        prefix = self._reporting_prefix(task)  # 45 chars

        if task.i_try == 1:
            print(f"{prefix} START ".ljust(180, "."))
        else:
            print(f"{prefix} START ".ljust(65, ".") + f"[TRY {task.i_try:_}/{task.n_max_tries:_}]".ljust(115, "."))

    def _report_task_failed(self, task: OptimizerTask):

        prefix = self._reporting_prefix(task)  # 45 chars
        postfix = self._reporting_postfix()  # 70 chars

        print(f"{prefix} FAILED ".ljust(110, ".") + postfix)

        # also show parameters
        params_str = ", ".join(
            f"{par_name}={str(par_value)}"
            for par_value, par_name in zip(task.params, self._scheduler.param_grid.keys())
        )
        if len(params_str) > 100:
            params_str = ", ".join(str(par_value) for par_value in task.params)
        print(f"{prefix} FAILED .. [{params_str}]".ljust(180, "."))

    def _report_new_result(self, task: OptimizerTask, result: Any):

        prefix = self._reporting_prefix(task)  # 45 chars
        postfix = self._reporting_postfix()  # 70 chars

        time_elapsed = (datetime.datetime.now() - task.start_time).total_seconds()

        result_str = f"{float(result): >8.2e}" if result is not None else "/"

        print(
            f"{prefix} END ".ljust(65, ".")
            + f"[time: {format_timedelta(time_elapsed): <6}]".ljust(25, ".")
            + f"[result: {result_str}]".rjust(20, ".")
            + postfix
        )

    def _reporting_prefix(self, task: OptimizerTask = None) -> str:
        """returns 45-char prefix for reporting purposes."""

        ts_str = f"[{format_datetime(datetime.datetime.now())}]"

        if task is not None:

            if self._scheduler.max_iter:
                idx_str = f"[{task.task_idx: >5_} / {self._scheduler.max_iter: <5_}]"
            else:
                idx_str = f"[{task.task_idx: >5_}]"

            return f"{ts_str} ... {idx_str} ".ljust(45, ".")

        else:

            return f"{ts_str} ".ljust(45, ".")

    def _reporting_postfix(self) -> str:
        """returns 70-char postfix for reporting purposes"""

        # --- best result ---------------------------------
        if self.best_result is not None:
            best_result_str = f"[best: {float(self.best_result): >8.2e}]"
        else:
            best_result_str = f"[best: ??? ]"

        # --- ETA -----------------------------------------
        eta_secs = self._scheduler.eta
        if eta_secs is not None:
            eta_dt = datetime.datetime.now() + datetime.timedelta(seconds=eta_secs)
            eta_str = f"[eta: {format_timedelta(eta_secs).ljust(7)} -->   {format_datetime(eta_dt)}]"
        else:
            eta_str = f"[eta: ??? ]"

        # --- return --------------------------------------
        return f"{best_result_str} {eta_str}".rjust(70, ".")
