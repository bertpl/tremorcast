import dataclasses
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_lin_vs_nn():

    # -------------------------------------------------------------------------
    #  Init
    # -------------------------------------------------------------------------

    # --- row / col dims ----------------------------------
    col_x_lims = [(-1, 1), (-9, 9), (-10, 10), (-11, 11)]
    row_y_lims = [(-4.5, 8), (-4.5, 9)]

    fig_size = [18, 10]

    col_widths = [x_max - x_min for x_min, x_max in col_x_lims]
    row_heights = [y_max - y_min for y_min, y_max in row_y_lims]

    # --- create fig, ax ----------------------------------
    plot_style_matplotlib_default()
    fig, ax = plt.subplots(nrows=2, ncols=4, gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights})
    ax_top_rowtitle, ax_top_left, ax_top_mid, ax_top_right = (
        ax[0][0],
        ax[0][1],
        ax[0][2],
        ax[0][3],
    )  # type: plt.Axes, plt.Axes, plt.Axes, plt.Axes
    ax_bot_rowtitle, ax_bot_left, ax_bot_mid, ax_bot_right = (
        ax[1][0],
        ax[1][1],
        ax[1][2],
        ax[1][3],
    )  # type: plt.Axes, plt.Axes, plt.Axes, plt.Axes

    # --- colors ------------------------------------------
    dark_grey = (0.3, 0.3, 0.3)
    light_grey = (0.7, 0.7, 0.7)

    gradient_from = (0.9, 0.9, 0.9)
    gradient_to = (0.95, 0.95, 0.95)

    # --- titles ------------------------------------------
    title_style = dict(ha="center", fontsize=14, color=(0.0, 0.0, 0.0), weight="bold")
    subtitle_style = dict(ha="center", fontsize=14, color=(0.2, 0.2, 0.2))
    subsubtitle_style = dict(ha="center", fontsize=12, color=(0.4, 0.4, 0.8), weight="bold")
    rowtitle_style = dict(**title_style, rotation="vertical", va="center")

    y_title_pos = 7.5
    y_subtitle_pos = 5.2
    y_subsubtitle_pos = 4.5
    x_rowtitle_pos = 0
    y_rowtitle_pos = 0

    def title(ax: plt.Axes, txt: str):
        ax.text(0, y_title_pos, txt, **title_style)

    def subtitle(ax: plt.Axes, txt: str):
        ax.text(0, y_subtitle_pos, txt, **subtitle_style)

    def subsubtitle(ax: plt.Axes, txt: str):
        ax.text(0, y_subsubtitle_pos, txt, **subsubtitle_style)

    def rowtitle(ax: plt.Axes, txt: str):
        ax.text(x_rowtitle_pos, y_rowtitle_pos, txt, **rowtitle_style)

    # --- other -------------------------------------------
    debug_lines = False

    # -------------------------------------------------------------------------
    #  Linear
    # -------------------------------------------------------------------------

    # --- row title ---------------------------------------
    rowtitle(ax_top_rowtitle, "Linear Regression")

    ax_top_rowtitle.set_axis_off()
    ax_top_rowtitle.set_xlim(left=-1, right=1)
    ax_top_rowtitle.set_ylim(bottom=-4, top=8)

    # --- 1-step-ahead - OLS ------------------------------
    plot_vector(ax_top_left, -6, 7, None, light_grey, "p")
    plot_vector(ax_top_left, 6, 1, None, light_grey, "1")

    plot_mapping(ax_top_left, -6, 6, 7, 1, gradient_from, gradient_to, mapping_name="p x 1\n\nlinear\nmapping")

    title(ax_top_left, "1-Step-Ahead")
    subtitle(ax_top_left, "Ordinary Least Squares (OLS)")
    subsubtitle(ax_top_left, "(blog post 4 - 'AR' model)")

    # --- n-step-ahead - OLS ------------------------------
    plot_vector(ax_top_mid, -6, 7, None, light_grey, "p")
    plot_vector(ax_top_mid, 6, 5, None, light_grey, "n")

    plot_mapping(ax_top_mid, -6, 6, 7, 5, gradient_from, gradient_to, mapping_name="p x n\n\nlinear\nmapping")

    title(ax_top_mid, "n-Step-Ahead")
    subtitle(ax_top_mid, "Ordinary Least Squares (OLS)")
    subsubtitle(ax_top_mid, "(blog post 5 - 'OLS' model)")

    # --- n-step-ahead - PLS ------------------------------
    plot_vector(ax_top_right, -9, 7, None, light_grey, "p")
    plot_vector(ax_top_right, 0, 2, None, light_grey, "n_PLS")
    plot_vector(ax_top_right, 9, 5, None, light_grey, "n")

    plot_mapping(ax_top_right, -9, 0, 7, 2, gradient_from, gradient_to, mapping_name="linear\nmapping")
    plot_mapping(ax_top_right, 0, 9, 2, 5, gradient_from, gradient_to, mapping_name="linear\nmapping")

    title(ax_top_right, "n-Step-Ahead - Bottlenecked")
    subtitle(ax_top_right, "Partial Least Squares (PLS)")
    subsubtitle(ax_top_right, "(blog post 5 - 'PLS' model)")

    # -------------------------------------------------------------------------
    #  Neural
    # -------------------------------------------------------------------------

    # --- row title ---------------------------------------
    rowtitle(ax_bot_rowtitle, "Neural Networks")

    ax_bot_rowtitle.set_axis_off()
    ax_bot_rowtitle.set_xlim(left=-1, right=1)
    ax_bot_rowtitle.set_ylim(bottom=-4.5, top=8)

    # --- 1-step-ahead - NN-AR ----------------------------
    plot_vector(ax_bot_left, -6, 7, None, light_grey, "p")
    plot_vector(ax_bot_left, -3, 6, None, light_grey, "")
    plot_vector(ax_bot_left, 0, 6, None, light_grey, "hidden layers")
    plot_vector(ax_bot_left, 3, 6, None, light_grey, "")
    plot_vector(ax_bot_left, 6, 1, None, light_grey, "1")

    ax_bot_left.text(-3, 0, "f", ha="center", va="center")
    ax_bot_left.text(0, 0, "f", ha="center", va="center")
    ax_bot_left.text(3, 0, "f", ha="center", va="center")

    plot_mapping(ax_bot_left, -6, -3, 7, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_left, -3, 0, 6, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_left, 0, 3, 6, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_left, 3, 6, 6, 1, gradient_from, gradient_to, mapping_name="")

    subtitle(ax_bot_left, "Multilayer Perceptron")
    subsubtitle(ax_bot_left, '(blog post 6 - "AR-MLP" model)')

    # --- 1-step-ahead - NN-n-step ------------------------
    ax_bot_mid.text(0, 1, "(not part of this post)", ha="center", va="center")

    # --- n-step-ahead - NN-Enc-Dec -----------------------
    plot_vector(ax_bot_right, -9, 7, None, light_grey, "p")
    plot_vector(ax_bot_right, -6, 6, None, light_grey, "")
    plot_vector(ax_bot_right, -3, 6, None, light_grey, "")
    plot_vector(ax_bot_right, 0, 2, None, light_grey, "n_latent")
    plot_vector(ax_bot_right, 3, 4, None, light_grey, "")
    plot_vector(ax_bot_right, 6, 4, None, light_grey, "")
    plot_vector(ax_bot_right, 9, 5, None, light_grey, "n")

    ax_bot_right.text(-6, 0, "f", ha="center", va="center")
    ax_bot_right.text(-3, 0, "f", ha="center", va="center")
    ax_bot_right.text(3, 0, "f", ha="center", va="center")
    ax_bot_right.text(6, 0, "f", ha="center", va="center")

    plot_mapping(ax_bot_right, -9, -6, 7, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_right, -6, -3, 6, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_right, -3, 0, 6, 2, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_right, 0, 3, 2, 4, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_right, 3, 6, 4, 4, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_bot_right, 6, 9, 4, 5, gradient_from, gradient_to, mapping_name="")

    subtitle(ax_bot_right, "Encoder-Decoder Topology")
    subsubtitle(ax_bot_right, '(blog post 6 - "Bottleneck" model)')

    # -------------------------------------------------------------------------
    #  Decorate & Save
    # -------------------------------------------------------------------------

    # --- set axes x/y lims -------------------------------
    for i_row, (y_min, y_max) in enumerate(row_y_lims):
        for i_col, (x_min, x_max) in enumerate(col_x_lims):

            this_ax = ax[i_row][i_col]  # type: plt.Axes

            if debug_lines:
                this_ax.plot(
                    [x_min, x_max, x_max, x_min, x_min], [y_max, y_max, y_min, y_min, y_max], c=(0.8, 0.8, 0.8)
                )

            this_ax.set_axis_off()
            this_ax.set_xlim(x_min, x_max)
            this_ax.set_ylim(y_min, y_max)

    # --- decorate ----------------------------------------
    fig.set_size_inches(w=fig_size[0], h=fig_size[1])
    fig.tight_layout()

    # --- save --------------------------------------------
    base_path = _get_output_path("post_6_nn")
    png_filename = os.path.join(base_path, f"lin_vs_nn.png")

    fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def plot_vector(ax: plt.Axes, x_pos: float, n: int, dot_clr: tuple, edge_clr: tuple, caption_txt: str = None):

    # --- init -----------------------------------
    y_pos = [i - ((n - 1) / 2) for i in range(n)]

    # --- dots -----------------------------------
    if dot_clr is not None:
        for i in range(n):
            ax.plot(x_pos, y_pos[i], "o", markeredgecolor=dot_clr, markerfacecolor=dot_clr)

    # --- rounded box ---------------------------
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

    # --- caption -------------------------------
    if caption_txt is None:
        caption_txt = f"{n}"

    ax.text(x_pos, min(y_pos) - 0.75, caption_txt, va="center", ha="center", fontsize=9)


def plot_mapping(
    ax: plt.Axes,
    x_from: float,
    x_to: float,
    n_from: float,
    n_to: float,
    clr_from,
    clr_to,
    margin=0.7,
    mapping_name: str = None,
):

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

    if mapping_name is None:
        mapping_name = f"{n_from}x{n_to}\n\nlinear\nmapping"

    ax.text((x_from + x_to) / 2, 0, mapping_name, va="center", ha="center")
