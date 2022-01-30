import dataclasses
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_ols_vs_pls():

    # --- create fig, ax ----------------------------------
    plot_style_matplotlib_default()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax_left, ax_right = ax[0], ax[1]  # type: plt.Axes, plt.Axes

    # --- colors ------------------------------------------
    dark_grey = (0.3, 0.3, 0.3)
    light_grey = (0.7, 0.7, 0.7)

    gradient_from = (0.9, 0.9, 0.9)
    gradient_to = (0.95, 0.95, 0.95)

    # --- OLS ---------------------------------------------
    plot_vector(ax_left, -2, 7, dark_grey, light_grey)
    plot_vector(ax_left, 2, 5, dark_grey, light_grey)

    plot_mapping(ax_left, -2, 2, 7, 5, gradient_from, gradient_to)

    ax_left.set_axis_off()
    ax_left.set_xlim(left=-4.5, right=4.5)
    ax_left.set_ylim(bottom=-4, top=4.5)

    ax_left.set_title("Ordinary Least Squares (OLS)")

    # --- PLS ---------------------------------------------
    plot_vector(ax_right, -3, 7, dark_grey, light_grey)
    plot_vector(ax_right, 0, 2, dark_grey, light_grey)
    plot_vector(ax_right, 3, 5, dark_grey, light_grey)

    plot_mapping(ax_right, -3, 0, 7, 2, gradient_from, gradient_to)
    plot_mapping(ax_right, 0, 3, 2, 5, gradient_from, gradient_to)

    ax_right.set_axis_off()
    ax_right.set_xlim(left=-4.5, right=4.5)
    ax_right.set_ylim(bottom=-4, top=4.5)

    ax_right.set_title("Partial Least Squares (PLS)")

    # --- decorate ----------------------------------------
    fig.set_size_inches(w=10, h=5)
    fig.tight_layout()

    # --- save --------------------------------------------
    base_path = _get_output_path("post_5_n_step_ahead")
    png_filename = os.path.join(base_path, f"ols_vs_pls.png")

    fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def plot_vector(ax: plt.Axes, x_pos: float, n: int, dot_clr: tuple, edge_clr: tuple):

    y_pos = [i - ((n - 1) / 2) for i in range(n)]
    for i in range(n):
        ax.plot(x_pos, y_pos[i], "o", markeredgecolor=dot_clr, markerfacecolor=dot_clr)

    rect = FancyBboxPatch(
        (x_pos, min(y_pos)),
        0,
        n - 1,
        boxstyle="round,pad=0.4,rounding_size=0.4",
        mutation_aspect=1.0,
        linewidth=2,
        facecolor=(0, 0, 0, 0),
        edgecolor=edge_clr,
    )
    ax.add_patch(rect)

    ax.text(x_pos, min(y_pos) - 0.75, f"{n}", va="center", ha="center", fontsize=9)


def plot_mapping(ax: plt.Axes, x_from: float, x_to: float, n_from: float, n_to: float, clr_from, clr_to, margin=0.7):

    # --- prep x & y --------------------------------------
    y_from = (n_from - 1) / 2  # vertically spanning [-y_from, y_from]
    y_to = (n_to - 1) / 2  # vertically spanning [-y_to, y_to]

    # --- connect to point on curvature -------------------
    x_from_orig, x_to_orig = x_from, x_to
    connector_angle = 45
    curve_radius = 0.4

    sn, cs = np.sin(np.deg2rad(connector_angle)), np.cos(np.deg2rad(connector_angle))

    x_from += sn * curve_radius
    x_to -= sn * curve_radius
    y_from += cs * curve_radius
    y_to += cs * curve_radius

    # --- adjust for margin -------------------------------
    # we want to update x_from to x_from+margin and x_to to x_to-margin,
    #  while adjusting the other parameters accordingly
    y_from, y_to = (
        np.interp(x_from_orig + margin, [x_from, x_to], [y_from, y_to]),
        np.interp(x_to_orig - margin, [x_from, x_to], [y_from, y_to]),
    )
    x_from = x_from_orig + margin
    x_to = x_to_orig - margin

    # --- plot with gradient ------------------------------
    n_steps = 100
    for i in range(n_steps):

        sub_x_from = x_from + (i / n_steps) * (x_to - x_from)
        sub_x_to = x_from + ((i + 1.1) / n_steps) * (x_to - x_from)
        sub_y_from = y_from + (i / n_steps) * (y_to - y_from)
        sub_y_to = y_from + ((i + 1.1) / n_steps) * (y_to - y_from)
        c = (i + 0.5) / n_steps
        clr = tuple([c_from + c * (c_to - c_from) for c_from, c_to in zip(clr_from, clr_to)])

        xy = np.array(
            [
                [sub_x_from, sub_y_from],
                [sub_x_to, sub_y_to],
                [sub_x_to, -sub_y_to],
                [sub_x_from, -sub_y_from],
            ]
        )

        poly = Polygon(xy, closed=True, edgecolor=clr, facecolor=clr, linewidth=0)
        ax.add_patch(poly)

    ax.text((x_from + x_to) / 2, 0, f"{n_from}x{n_to}\n\nlinear\nmapping", va="center", ha="center")
