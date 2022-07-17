from __future__ import annotations

from enum import Enum, auto
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.tools.matplotlib import plot_style_matplotlib_default

from .cv_results import CVMetricResult, CVResult, CVResults


class ErrorBounds(Enum):
    STDEV = auto()
    QUARTILES = auto()

    def get_ub(self, values: CVMetricResult) -> float:
        if self == ErrorBounds.STDEV:
            return values.mean() + values.std()
        else:
            return values.quantile(0.75)

    def get_lb(self, values: CVMetricResult) -> float:
        if self == ErrorBounds.STDEV:
            return values.mean() - values.std()
        else:
            return values.quantile(0.25)


MAX_LINEAR_RANGE = 15


class CrossValidationPlot1D:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, param_names: List[str], data: List[Tuple[Tuple, CVResult]], higher_is_better: bool):

        # --- arguments -----------------------------------
        self.param_names = param_names
        self.data = data
        self.higher_is_better = higher_is_better  # TODO

        # --- defaults ------------------------------------

        # misc
        self.error_bounds = ErrorBounds.STDEV

        # axes settings
        self.y_label = "losses"

        if isinstance(param_names, str) or len(param_names) == 1:

            if not isinstance(param_names, str):
                param_names = param_names[0]

            self.x_label = param_names

            self.param_values = [param_value_tuple[0] for param_value_tuple, _ in data]
            if all([isinstance(v, (int, float)) for v in self.param_values]):
                self.x_values = self.param_values
                self.log_x_scale = (min(self.x_values) > 0) and (
                    min(self.x_values) < max(self.x_values) / MAX_LINEAR_RANGE
                )
            else:
                self.x_values = list(range(len(self.param_values)))
                self.log_x_scale = False

        else:

            self.x_label = "(" + ", ".join([str(pn) for pn in param_names]) + ")"
            self.param_values = [v for v, _ in data]
            self.x_values = list(range(len(self.param_values)))
            self.log_x_scale = False

        # other plot settings
        self.fig_title = "Cross-validation results"

    # -------------------------------------------------------------------------
    #  Modifiers
    # -------------------------------------------------------------------------
    def set_x_label(self, x_label: str) -> CrossValidationPlot1D:
        self.x_label = x_label
        return self

    def set_y_label(self, y_label: str) -> CrossValidationPlot1D:
        self.y_label = y_label
        return self

    def set_fig_title(self, fig_title: str) -> CrossValidationPlot1D:
        self.fig_title = fig_title
        return self

    def set_error_bounds(self, error_bounds: ErrorBounds) -> CrossValidationPlot1D:
        self.error_bounds = error_bounds
        return self

    # -------------------------------------------------------------------------
    #  Actual plotting
    # -------------------------------------------------------------------------
    def create(self, w: float = 8, h: float = 6) -> Tuple[plt.Figure, plt.Axes]:

        # --- init ----------------------------------------
        plot_style_matplotlib_default()
        fig, ax = plt.subplots(nrows=1, ncols=1)  # type: plt.Figure, plt.Axes

        # --- determine values to plot --------------------
        training_metric_mean = np.array([cv_result.train_metrics.overall for _, cv_result in self.data])
        validation_metric_mean = np.array([cv_result.val_metrics.overall for _, cv_result in self.data])

        training_metric_lb = np.array([self.error_bounds.get_lb(cv_result.train_metrics) for _, cv_result in self.data])
        training_metric_ub = np.array([self.error_bounds.get_ub(cv_result.train_metrics) for _, cv_result in self.data])
        validation_metric_lb = np.array([self.error_bounds.get_lb(cv_result.val_metrics) for _, cv_result in self.data])
        validation_metric_ub = np.array([self.error_bounds.get_ub(cv_result.val_metrics) for _, cv_result in self.data])

        # --- plot ----------------------------------------
        ax.fill_between(
            self.x_values,
            training_metric_lb,
            training_metric_ub,
            color="r",
            alpha=0.1,
        )
        ax.fill_between(
            self.x_values,
            validation_metric_lb,
            validation_metric_ub,
            color="g",
            alpha=0.1,
        )

        # --- actual lines ---
        h_train = ax.plot(self.x_values, training_metric_mean, "r-x")
        h_val = ax.plot(self.x_values, validation_metric_mean, "g-x")

        # --- set limits ---

        # ideally, center median of validation losses upper bound
        y_max_ideal = 2 * np.median(validation_metric_ub)

        # make sure minimum validation loss is still discernible
        y_max = min(10 * min(validation_metric_mean), y_max_ideal)

        # set value
        ax.set_ylim(bottom=0.0, top=y_max)

        # --- ticks ---
        if self.log_x_scale:
            ax.set_xscale("log")
        ax.set_xticks(self.x_values)
        ax.set_xticklabels([str(pv) for pv in self.param_values], rotation=45, ha="right")

        # --- decorate ---
        ax.grid(visible=True)

        ax.legend([h_train[0], h_val[0]], ["mean training metric", "mean validation metric"])

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        fig.suptitle(self.fig_title)

        fig.set_size_inches(w=w, h=h)
        fig.tight_layout()

        # --- lines & best performance ---
        selection_criterion = validation_metric_mean
        if self.higher_is_better:
            i_best = list(selection_criterion).index(max(selection_criterion))
        else:
            i_best = list(selection_criterion).index(min(selection_criterion))

        best_metric_crit = selection_criterion[i_best]
        x_best_param = self.x_values[i_best]

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # line + text + dot - SELECTION_CRITERION
        ax.plot(x_best_param, best_metric_crit, "go")
        ax.plot([x_min, x_max], [best_metric_crit, best_metric_crit], "g--", alpha=0.5)
        ax.text(x_min, best_metric_crit - 0.01 * y_max, f" {best_metric_crit:.3f}", ha="left", va="top", color="g")

        # reset limits
        ax.set_xlim(x_min, x_max)

        # --- return ---
        return fig, ax
