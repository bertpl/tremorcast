from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.base.forecasting.evaluation.metrics.base_metric import BaseMetric
from src.tools.misc import sort_any

from .cv_plot_1d import CrossValidationPlot1D
from .cv_plot_2d import CrossValidationPlot2D


# =================================================================================================
#  CrossValidation result for 1 set of parameters
# =================================================================================================
@dataclass
class CVResult:

    params: dict

    train_metrics: List[float]
    train_metric_mean: float
    train_metric_std: float

    val_metrics: List[float]
    val_metric_mean: float
    val_metric_std: float

    fit_time_mean: float
    fit_time_std: float


# =================================================================================================
#  CrossValidation results for n sets of parameters
# =================================================================================================
@dataclass
class CVResults:

    # --- members -----------------------------------------
    metric: BaseMetric
    best_result: Optional[CVResult]
    all_results: List[CVResult]

    # --- helper functions --------------------------------
    def update_best_result(self):
        """Updates the best_result member based on all_results."""
        for cv_result in self.all_results:
            if (self.best_result is None) or (
                self.metric.metric_to_score(cv_result.val_metric_mean)
                > self.metric.metric_to_score(self.best_result.val_metric_mean)
            ):

                self.best_result = cv_result

    def all_param_values(self) -> Dict:
        """Returns dictionary with all param_names & corresponding values in this cv_result."""
        return {
            param_name: sort_any({cv_result.params[param_name] for cv_result in self.all_results})
            for param_name in sorted(
                {param_name for cv_result in self.all_results for param_name in cv_result.params.keys()}
            )
        }

    def filter(self, param_filter: dict = None) -> CVResults:

        # --- create new subset of all_results ------------
        filtered_results = self.all_results.copy()
        for param_name, param_value in param_filter.items():
            filtered_results = [
                cv_result for cv_result in filtered_results if cv_result.params.get(param_name) == param_value
            ]

        # --- create new CVResults object -----------------
        new_cv_results = CVResults(
            metric=self.metric,
            best_result=None,
            all_results=filtered_results,
        )
        new_cv_results.update_best_result()

        # --- return --------------------------------------
        return new_cv_results

    def sweep_by_filter(self, param_names: List[str], param_filter: dict = None) -> List[Tuple[Any, CVResult]]:
        """
        Returns a 1D sweep across the results, where...
          - param_names determines the x-axis of the sweep  (independent variable) (can be multiple)
          - param_filter, when specified, determines which subset of results to consider
        If multiple results are obtained for the same value of param_name, we take the best one.
        :param param_filter: dictionary with key-value pairs treated as equality constraints when filtering
        :param param_names: 1 or more parameter names over which we want to 1D sweep.  All unique value-tuples will
                             be collected and sorted to form a 1D sweep.
        :return: list of (param_value_tuple, CVResult)-tuples
        """

        # --- argument handling ---------------------------
        param_filter = param_filter or dict()
        if not isinstance(param_names, list):
            param_names = [param_names]

        # --- get all parameter value tuples --------------
        param_value_tuples = sort_any(
            {
                tuple([cv_result.params[param_name] for param_name in param_names])
                for cv_result in self.filter(param_filter).all_results
            }
        )  # type: List[tuple]

        # --- fetch results -------------------------------
        result = []
        for param_value_tuple in param_value_tuples:

            # filter results by provided filter + this specific param value
            for param_name, param_value in zip(param_names, param_value_tuple):
                param_filter[param_name] = param_value
            filtered_results = self.filter(param_filter)

            # extract info from best result
            result.append((param_value_tuple, filtered_results.best_result))

        # --- return result -------------------------------
        return result

    def show_optimal_results(self):
        def summarize_losses(losses: List[float]) -> str:
            mean = np.mean(losses)
            std = np.std(losses)
            return f"{mean:>8.3f} ± {std:<8.3f} <-- [{''.join([f'{x:>9.3f} ' for x in losses])}]"

        # --- all values for each param -------------------
        all_param_values = self.all_param_values()

        # --- process parameter names ---------------------
        param_names = sorted(self.best_result.params.keys())
        max_param_len = max([len(pn) for pn in param_names])
        max_value_len = max([len(str(self.best_result.params[pn])) for pn in param_names])

        # --- show results --------------------------------
        print("-" * 80)
        print("Training losses   : " + summarize_losses(self.best_result.train_metrics))
        print("Validation losses : " + summarize_losses(self.best_result.val_metrics))
        print("Optimal parameter values:")
        for param_name in param_names:
            param_value = self.best_result.params[param_name]
            all_values = all_param_values[param_name]

            print(
                f"  {param_name: <{max_param_len+1}}: {str(param_value): <{max_value_len+2}} "
                + "<-- ["
                + ", ".join([str(v) for v in all_values])
                + "]"
            )

        print("-" * 80)

    def plot_1d(self, param_names: Union[str, List[str]], param_filter: dict = None) -> CrossValidationPlot1D:
        """
        Creates a 1D plot for the provided parameter & filtering.
        """
        return CrossValidationPlot1D(
            param_names=param_names,
            data=self.sweep_by_filter(param_names, param_filter),
            higher_is_better=self.metric.greater_metric_is_better(),
        )

    def plot_2d(self, x_param: str, y_param: str, param_filter: dict = None) -> CrossValidationPlot2D:
        """
        Creates a 2D plot for the provided parameter & filtering.
        """
        return CrossValidationPlot2D(
            x_param,
            y_param,
            data=self.sweep_by_filter([x_param, y_param], param_filter),
            higher_is_better=self.metric.greater_metric_is_better(),
        )
