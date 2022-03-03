import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_one_vs_n_step_ahead_time_series():
    # --- create fig, ax ----------------------------------
    plot_style_matplotlib_default()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax_left, ax_right = ax[0], ax[1]

    trace_colors = [(c, 0.5 * c, 1 - c) for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    traces = [
        (5, trace_colors[0]),
        (10, trace_colors[1]),
        (15, trace_colors[2]),
        (20, trace_colors[3]),
    ]

    # --- create time series ------------------------------
    np.random.seed(2)
    x = np.arange(start=1, stop=21)
    y = 0.1 + (0.1 * x) - (0.002 * (x ** 2)) + 0.1 * np.random.normal(size=x.shape)

    for ax, p, n in [(ax_left, 7, 1), (ax_right, 7, 5)]:  # type: plt.Axes, int, int

        # plot data
        ax.plot(x, y, "k-o", markerfacecolor="white")

        # plot marked data points
        for i_trace, clr in traces:
            ax.plot(x[i_trace - 1], y[i_trace - 1], "o", markerfacecolor=clr, markeredgecolor=clr)

        # past and future
        i_now = 9
        x_now = x[i_now] + 0.5
        light_grey = (0.7, 0.7, 0.7)
        grey = (0.5, 0.5, 0.5)
        ax.plot([x_now, x_now], [0, 1.6], color=light_grey, linestyle="--", lw=1, zorder=-10)
        ax.text(x_now - 0.5, 1.5, "past", ha="right", color=grey)
        ax.text(x_now + 0.5, 1.5, "future", ha="left", color=grey)

        # features
        x_min, x_max, y_min, y_max = plot_rectangle(x, y, i_now - p + 1, i_now, ax)
        ax.text(x_min - 0.4, y_max + 0.075, "features", va="center", ha="left")

        # targets
        x_min, x_max, y_min, y_max = plot_rectangle(x, y, i_now + 1, i_now + n, ax)
        ax.text(x_min, y_min - 0.075, "targets", va="center", ha="left")

        # organize axes
        ax.set_xlim(left=0, right=21)
        ax.set_ylim(bottom=0, top=1.6)
        ax.set_xticks(list(x))
        ax.tick_params(axis="x", labelsize=9)

        # title
        ax.set_title(f"{n}-step-ahead auto-regressive\nforecast model")

    # --- decorate ----------------------------------------
    fig.set_size_inches(w=10, h=4.5)
    fig.tight_layout()

    # --- save --------------------------------------------
    base_path = _get_output_path("post_5_n_step_ahead")
    png_filename = os.path.join(base_path, f"n_step_ahead_time_series.png")

    fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def plot_rectangle(
    x: np.ndarray, y: np.ndarray, i_first: int, i_last: int, ax: plt.Axes
) -> Tuple[float, float, float, float]:
    x = x[i_first : i_last + 1]
    y = y[i_first : i_last + 1]

    x_min = min(x)
    x_max = max(x)
    y_min = min(y) - 0.02
    y_max = max(y) + 0.02

    rect = FancyBboxPatch(
        (x_min, y_min),
        (x_max - x_min),
        (y_max - y_min),
        boxstyle="round,pad=0.4",
        mutation_aspect=1.6 / 20,
        linewidth=0,
        facecolor=(0.8, 0.8, 0.8),
    )
    ax.add_patch(rect)

    return x_min, x_max, y_min, y_max
