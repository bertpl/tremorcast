from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.tools.matplotlib import plot_style_matplotlib_default
from src.tools.misc import sort_any


class CrossValidationPlot2D:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, x_param: str, y_param: str, data: List[Tuple[Tuple, "CVResult"]]):

        # set arguments
        self.x_param = x_param
        self.y_param = y_param
        self.data = data

        # defaults
        self.x_log_scale = False
        self.y_log_scale = False
        self.levels = 20

    # -------------------------------------------------------------------------
    #  Actual plotting
    # -------------------------------------------------------------------------
    def create(self) -> Tuple[plt.Figure, plt.Axes]:

        # --- init ----------------------------------------
        plot_style_matplotlib_default()
        fig, ax = plt.subplots(nrows=1, ncols=1)  # type: plt.Figure, plt.Axes

        # --- values to plot ------------------------------
        x_values, x_ticks, x_tick_labels = self.process_xy_values(values=[v[0] for v, _ in self.data])
        y_values, y_ticks, y_tick_labels = self.process_xy_values(values=[v[1] for v, _ in self.data])
        z_values = [cv_result.val_metric_mean for _, cv_result in self.data]

        # --- actual plot ---------------------------------
        contour = ax.tricontour(x_values, y_values, z_values, levels=self.levels, cmap="coolwarm")
        fig.colorbar(contour, ax=ax)

        # --- decorate ------------------------------------
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)

        ax.set_xlabel(str(self.x_param))
        ax.set_ylabel(str(self.y_param))

        fig.suptitle("Cross-validation results")

        fig.set_size_inches(w=8, h=8)
        fig.tight_layout()

        # --- return --------------------------------------
        return fig, ax

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def process_xy_values(values: list) -> Tuple[List[float], List[str], List[str]]:

        if all([isinstance(v, (int, float)) for v in values]):

            v_values = values
            v_ticks = sorted(set(values))
            v_tick_labels = v_ticks

        else:

            unique_values = sort_any(set(values))

            v_values = [unique_values.index(v) for v in values]
            v_ticks = unique_values
            v_tick_labels = [str(v) for v in unique_values]

        return v_values, v_ticks, v_tick_labels
