from __future__ import annotations

from typing import List, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from src.tools.matplotlib import plot_style_matplotlib_default
from src.tools.misc import sort_any

from .cv_results import CVMetricResult, CVResult, CVResults


# =================================================================================================
#  2D plotting class
# =================================================================================================
class CrossValidationPlot2D:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, x_param: str, y_param: str, data: List[Tuple[Tuple, CVResult]], higher_is_better: bool):

        # set arguments
        self.x_param = x_param
        self.y_param = y_param
        self.data = data
        self.higher_is_better = higher_is_better

        # defaults
        self.n_levels = 10
        self.show_labels = False
        self.show_data_points = True
        self.x_label = None
        self.y_label = None

    # -------------------------------------------------------------------------
    #  Settings
    # -------------------------------------------------------------------------
    def with_contour_settings(
        self, n_levels: int = None, show_labels: bool = None, show_data_points: bool = None
    ) -> CrossValidationPlot2D:
        if n_levels is not None:
            self.n_levels = n_levels
        if show_labels is not None:
            self.show_labels = show_labels
        if show_data_points is not None:
            self.show_data_points = show_data_points

        return self

    def with_x_label(self, x_label: str) -> CrossValidationPlot2D:
        self.x_label = x_label
        return self

    def with_y_label(self, y_label: str) -> CrossValidationPlot2D:
        self.y_label = y_label
        return self

    # -------------------------------------------------------------------------
    #  Actual plotting
    # -------------------------------------------------------------------------
    def create(self, w: float = 8, h: float = 6) -> Tuple[plt.Figure, plt.Axes]:

        # --- init ----------------------------------------
        plot_style_matplotlib_default()
        fig, ax = plt.subplots(nrows=1, ncols=1)  # type: plt.Figure, plt.Axes

        # --- values to plot ------------------------------
        x_values, x_ticks, x_tick_labels = process_xy_values(values=[v[0] for v, _ in self.data])
        y_values, y_ticks, y_tick_labels = process_xy_values(values=[v[1] for v, _ in self.data])
        z_values = [cv_result.val_metrics.overall for _, cv_result in self.data]

        # --- optimal levels ------------------------------
        line_levels = optimal_levels(x_values, y_values, z_values, self.n_levels)
        fill_levels = optimal_levels(x_values, y_values, z_values, 25 * self.n_levels)

        # --- actual plot ---------------------------------
        cntr_lines = ax.tricontour(x_values, y_values, z_values, levels=line_levels, linewidths=0.5, colors="k")
        cntr_fill = ax.tricontourf(
            x_values, y_values, z_values, levels=fill_levels, cmap=get_color_map(self.higher_is_better)
        )

        if self.show_labels:
            ax.clabel(cntr_lines, inline=True, fontsize=8)
        fig.colorbar(cntr_fill, ax=ax)

        # --- show all data points ------------------------
        if self.show_data_points:
            ax.plot(x_values, y_values, "o", mfc="k", mec="k", ms=4)

        # --- indicate best score -------------------------

        # find best score
        if self.higher_is_better:
            z_best = max(z_values)
        else:
            z_best = min(z_values)

        i_best = z_values.index(z_best)
        x_best = x_values[i_best]
        y_best = y_values[i_best]

        # visual indication of best result
        ax.plot([-100, 100], [y_best, y_best], "k--", lw=0.5)
        ax.plot([x_best, x_best], [-100, 100], "k--", lw=0.5)
        ax.plot(x_best, y_best, "x", mfc="k", mec="k", ms=8)

        # show metric value
        x_text, ha = (x_best + 0.05, "left") if (x_best <= np.mean(x_ticks)) else (x_best - 0.05, "right")
        y_text, va = (y_best + 0.05, "bottom") if (y_best <= np.mean(y_ticks)) else (y_best - 0.05, "top")
        ax.text(x_text, y_text, f"{z_best:6.3f}", ha=ha, va=va, weight="semibold")

        # --- decorate ------------------------------------
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)

        ax.set_xlim(left=min(x_ticks) - 0.1, right=max(x_ticks) + 0.1)
        ax.set_ylim(bottom=min(y_ticks) - 0.1, top=max(y_ticks) + 0.1)

        ax.set_xlabel(self.x_label or str(self.x_param))
        ax.set_ylabel(self.y_label or str(self.y_param))

        fig.suptitle("Cross-validation results")

        fig.set_size_inches(w=w, h=h)
        fig.tight_layout()

        # --- return --------------------------------------
        return fig, ax


# =================================================================================================
#  Helpers
# =================================================================================================
def process_xy_values(values: list) -> Tuple[List[int], List[float], List[str]]:

    unique_values = sort_any(set(values))

    v_values = [unique_values.index(v) for v in values]
    v_ticks = list(range(len(unique_values)))
    v_tick_labels = [str(v) for v in unique_values]

    return v_values, v_ticks, v_tick_labels


def get_color_map(higher_is_better: bool):

    red = (0.8, 0.2, 0.2)
    yellow = (1.0, 1.0, 0.2)
    green = (0.2, 0.8, 0.4)

    if higher_is_better:
        c_lo, c_mid, c_hi = red, yellow, green
    else:
        c_lo, c_mid, c_hi = green, yellow, red

    cdict = {
        "red": [(0.0, c_lo[0], c_lo[0]), (0.5, c_mid[0], c_mid[0]), (1.0, c_hi[0], c_hi[0])],
        "green": [(0.0, c_lo[1], c_lo[1]), (0.5, c_mid[1], c_mid[1]), (1.0, c_hi[1], c_hi[1])],
        "blue": [(0.0, c_lo[2], c_lo[2]), (0.5, c_mid[2], c_mid[2]), (1.0, c_hi[2], c_hi[2])],
    }

    return colors.LinearSegmentedColormap("GnYeRd", cdict)


def optimal_levels(x_values: List[float], y_values: List[float], z_values: List[float], n: int) -> List[float]:

    # --- interpolate z values to grid --------------------

    # interpolate onto regular grid
    grid_size = 100
    xi = np.linspace(min(x_values), max(x_values), grid_size)
    yi = np.linspace(min(y_values), max(y_values), grid_size)

    triang = tri.Triangulation(x_values, y_values)
    interpolator = tri.LinearTriInterpolator(triang, z_values)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interpolator(Xi, Yi)

    # --- get sorted values -------------------------------
    zi = sorted(Zi.compressed())

    # --- get proportional & linear levels ----------------
    levels_prop = np.quantile(zi, np.linspace(0, 1, n + 2))
    levels_lin = np.linspace(0.99 * min(zi), 1.01 * max(zi), n + 2)

    # --- return mix --------------------------------------
    return [np.sqrt(a * b) for a, b in zip(levels_prop, levels_lin)]
