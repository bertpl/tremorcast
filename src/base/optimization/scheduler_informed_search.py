from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .scheduler import Scheduler


# =================================================================================================
#  AutoBalancedSampler
# =================================================================================================
class AutoBalancedSampler:
    """Class with built-in feedback mechanism for actively tracking target sampling probabilities."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, n: int, c: float = 1.0, rnd: np.random.Generator = None):

        # config
        self.n = n
        self.c = c

        # state
        self._rnd = rnd
        self._counts = np.zeros(n)
        self._counts_target = np.zeros(n)

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def sample(self, p: np.ndarray) -> int:

        # --- feedback mechanism --------------------------
        # apply correction to each p that is <1 if it has been over-sampled and >1 when under-sampled
        p_corrected = p * ((1 + self._counts_target) / (1 + self._counts)) ** self.c
        p_corrected = p_corrected / np.sum(p_corrected)  # re-normalized

        # --- sample --------------------------------------
        idx = self._rnd.choice(self.n, p=p_corrected)

        # --- update internal state -----------------------
        self._counts[idx] += 1
        self._counts_target += p

        # --- return --------------------------------------
        return idx


# =================================================================================================
#  AdaptiveSampler
# =================================================================================================
class AdaptiveSampler:
    """Class for adaptively sampling values from a list, using observed optimality of results that are reported back."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, values: List[Any], rnd: np.random.Generator = None):

        self.values = values
        self.n = len(values)
        self.balanced_sampler = AutoBalancedSampler(n=len(values), c=5.0, rnd=rnd)

        # array with the best objective function values observed for each parameter value (np.nan if not yet evaluated)
        self.best_results = np.full(self.n, np.nan)

        # counter how many times each value has been yielded
        self.values_count = np.zeros(self.n)

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def yield_new_value(self, focus: float) -> Any:
        """
        Yields a new value from self.values, based on the provided selectivity.

        selectivity influences probability of each value being chosen:
            p[i] ~= exp(-selectivity * (best_results[i] - min(best_results)))

        selectivity: value =0  --> 100% random selection>= 0, with...
                           >0  --> the higher the selectivity and the higher the best_result, the lower the probability
                                       a certain value will be yielded
        """

        sampling_pmf = self.get_sampling_pmf(focus)
        idx = self.balanced_sampler.sample(p=sampling_pmf)
        return self.values[idx]

    def confirm_value_selected(self, value: Any):
        idx = self.values.index(value)
        self.values_count[idx] += 1

    def register_new_result(self, value: Any, result: float):
        idx = self.values.index(value)
        if np.isnan(self.best_results[idx]):
            self.best_results[idx] = result
        else:
            self.best_results[idx] = np.minimum(self.best_results[idx], result)

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def get_sampling_pmf(self, selectivity: float) -> np.ndarray:

        # if any(self.values_count == 0):
        #     # not all values have already been selected --> randomly select a value we haven't selected yet
        #     pmf = np.array([1.0 if count == 0 else 0.0 for count in self.values_count])
        #
        # elif all(np.isnan(self.best_results)):
        if all(np.isnan(self.best_results)):
            # as long as we didn't get any result back yet
            #  --> randomly select any value (i.e. as if focus = 0.0)
            pmf = np.ones(self.n)

        else:
            # we have selected each value at least once already & we already received at least 1 result back

            # set all missing values to the median
            best_results = self.best_results.copy()
            best_results[np.isnan(best_results)] = np.nanmedian(best_results)

            # convert to probabilities based on 'focus' value
            #  --> p[i] ~= exp(-selectivity * (best_results[i] - min(best_results)))
            pmf = np.exp(-selectivity * (best_results - min(best_results)))

        return pmf / np.sum(pmf)


# =================================================================================================
#  InformedSearch
# =================================================================================================
class InformedSearch(Scheduler):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        param_grid: Dict,
        max_iter: int = None,
        max_seconds: int = None,
        min_focus: float = 0.0,
        max_focus: float = 5.0,
        focus_exponent: float = 1.0,
        backoff_factor: float = 0.95,
    ):
        """
        Rule of thumb: choose max_focus in (n_values/2)*[1, np.sqrt(n_params)]
           with ...      n_values = avg number of values per parameter
                         n_params = number of parameters
           and ... smaller values preferable for shorter run times.
        """

        # --- process param grid --------------------------
        n_params = 1
        for key, value in param_grid.items():
            assert isinstance(key, str), f"param_grid keys should be strings.  here: {key} = {type(key)}"
            assert isinstance(value, list), f"param_grid values should be lists.  here: {value} = {type(value)}"
            n_params *= len(value)

        # --- report --------------------------------------
        print(f"InformedSearch over {n_params:_} parameter values.")

        # --- superclass constructor ----------------------
        max_iter = min([max_iter or n_params, n_params])
        super().__init__(
            param_grid,
            max_iter=max_iter,
            max_seconds=max_seconds,
        )

        # --- settings ------------------------------------
        self.param_grid = param_grid
        self.min_focus = min_focus
        self.max_focus = max_focus
        self.focus_exponent = focus_exponent
        self.backoff_factor = backoff_factor

        # --- state ---------------------------------------
        self.random = np.random.default_rng(1)
        self.samplers = [
            AdaptiveSampler(values=param_values, rnd=self.random) for param_name, param_values in param_grid.items()
        ]  # 1 samples for each parameter

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def _yield_next_param_tuple(self) -> Optional[Tuple]:

        # --- determine focus -----------------------
        focus = self._compute_focus()

        # --- determine new param tuple -------------------
        new_tuple = None
        while (new_tuple is None) or (new_tuple in self._yielded_tuples.keys()):
            selectivity = self._compute_selectivity(focus)
            new_tuple = tuple([sampler.yield_new_value(selectivity) for sampler in self.samplers])
            focus *= self.backoff_factor

        # --- register & return ---------------------------
        for sampler, value in zip(self.samplers, new_tuple):
            sampler.confirm_value_selected(value)

        return new_tuple

    def _register_new_result(self, params: Tuple, result: float):
        if result is not None:
            for sampler, value in zip(self.samplers, params):
                sampler.register_new_result(value, result)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _compute_focus(self) -> float:

        if self.n_results_registered < 2:

            # just random search if we don't have at least 2 results registered
            focus = 0.0

        else:

            # --- compute progress_fraction ---
            seconds_elapsed = self.seconds_elapsed
            seconds_to_go = self.eta
            if (seconds_elapsed is not None) and (seconds_to_go is not None) and (seconds_elapsed + seconds_to_go > 0):
                progress_fraction = seconds_elapsed / (seconds_elapsed + seconds_to_go)
            else:
                progress_fraction = 0.0

            # --- compute focus ---
            focus = self.min_focus + (progress_fraction**self.focus_exponent) * (self.max_focus - self.min_focus)

        # --- return ---
        return focus

    def _compute_selectivity(self, focus: float) -> float:

        if focus == 0.0:
            return 0.0
        else:
            all_best_results = sorted(
                set([x for sampler in self.samplers for x in sampler.best_results if not np.isnan(x)])
            )
            if len(all_best_results) < 2:
                return 0.0
            else:
                if focus <= 1.0:
                    # look at largest of all_best_results
                    return focus / (max(all_best_results) - min(all_best_results))
                else:
                    # start going down the list, proportional to 1/focus
                    ref_best_result = np.quantile(all_best_results, q=1 / focus)
                    return 1.0 / (ref_best_result - min(all_best_results))

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    def tuple_to_dict(self, param_tuple: Tuple) -> dict:
        return {param_name: param_value for param_name, param_value in zip(self.param_grid.keys(), param_tuple)}
