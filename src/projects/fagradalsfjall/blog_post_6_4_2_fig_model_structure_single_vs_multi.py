import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_single_vs_multi():

    # -------------------------------------------------------------------------
    #  Init
    # -------------------------------------------------------------------------

    # --- row / col dims ----------------------------------
    col_x_lims = [(-9, 9), (-9, 9)]
    row_y_lims = [(-7, 9)]

    fig_size = [14, 7]

    col_widths = [x_max - x_min for x_min, x_max in col_x_lims]
    row_heights = [y_max - y_min for y_min, y_max in row_y_lims]

    # --- create fig, ax ----------------------------------
    plot_style_matplotlib_default()
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights})
    ax = [ax]

    ax_left, ax_right = ax[0][0], ax[0][1]  # type: plt.Axes, plt.Axes

    # --- colors ------------------------------------------
    dark_grey = (0.3, 0.3, 0.3)
    light_grey = (0.7, 0.7, 0.7)

    gradient_from = (0.9, 0.9, 0.9)
    gradient_to = (0.95, 0.95, 0.95)

    black = (0, 0, 0)
    almost_black = (0.2, 0.2, 0.2)

    light_blue = (0.4, 0.4, 0.8)
    lighter_blue = (0.6, 0.6, 0.9)

    dark_green = (0.2, 0.6, 0.2)
    light_green = (0.4, 0.8, 0.4)

    # --- titles ------------------------------------------
    title_style = dict(ha="center", fontsize=14, color=black, weight="bold")
    subtitle_style = dict(ha="center", fontsize=14, color=almost_black)
    subsubtitle_style = dict(ha="center", fontsize=12, color=light_blue, weight="bold")

    y_title_pos = 9.5
    y_subtitle_pos = 7.2
    y_subsubtitle_pos = 6.5

    def title(ax: plt.Axes, txt: str):
        ax.text(0, y_title_pos, txt, **title_style)

    def subtitle(ax: plt.Axes, txt: str):
        ax.text(0, y_subtitle_pos, txt, **subtitle_style)

    def subsubtitle(ax: plt.Axes, txt: str):
        ax.text(0, y_subsubtitle_pos, txt, **subsubtitle_style)

    # --- other -------------------------------------------
    debug_lines = True

    # -------------------------------------------------------------------------
    #  LEFT - Single-MLP
    # -------------------------------------------------------------------------

    plot_vector(ax_left, -6, 7, None, light_grey, "p")
    plot_vector(ax_left, -3, 6, None, light_grey, "")
    plot_vector(ax_left, 0, 6, None, light_grey, "")
    plot_vector(ax_left, 3, 6, None, light_grey, "")
    plot_vector(ax_left, 6, 5, None, light_grey, "n")

    ax_left.text(-3, 0, "f", ha="center", va="center")
    ax_left.text(0, 0, "f", ha="center", va="center")
    ax_left.text(3, 0, "f", ha="center", va="center")

    plot_mapping(ax_left, -6, -3, 7, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_left, -3, 0, 6, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_left, 0, 3, 6, 6, gradient_from, gradient_to, mapping_name="")
    plot_mapping(ax_left, 3, 6, 6, 5, gradient_from, gradient_to, mapping_name="")

    subtitle(ax_left, "N-step-ahead - Single MLP")

    # -------------------------------------------------------------------------
    #  RIGHT - Multi-MLP
    # ------------------------------------------------------------------------

    plot_vector(ax_right, -6, 7, None, light_grey, "p")

    for y_delta, name in [(4.5, "1"), (1, "2"), (-4.5, "n")]:

        layer_width = 3

        plot_vector(ax_right, -3, layer_width, None, light_grey, "", y_delta=y_delta)
        plot_vector(ax_right, 0, layer_width, None, light_grey, "", y_delta=y_delta)
        plot_vector(ax_right, 3, layer_width, None, light_grey, "", y_delta=y_delta)

        ax_right.text(-3, y_delta, "f", ha="center", va="center")
        ax_right.text(0, y_delta, "f", ha="center", va="center")
        ax_right.text(3, y_delta, "f", ha="center", va="center")

        plot_mapping(
            ax_right, -6, -3, 8, layer_width - 1, gradient_from, gradient_to, mapping_name="", y_delta_right=y_delta
        )
        plot_mapping(
            ax_right,
            -3,
            0,
            layer_width,
            layer_width,
            gradient_from,
            gradient_to,
            mapping_name="",
            y_delta_left=y_delta,
            y_delta_right=y_delta,
        )
        plot_mapping(
            ax_right,
            0,
            3,
            layer_width,
            layer_width,
            gradient_from,
            gradient_to,
            mapping_name="",
            y_delta_left=y_delta,
            y_delta_right=y_delta,
        )
        plot_mapping(
            ax_right,
            3,
            6,
            layer_width,
            1,
            gradient_from,
            gradient_to,
            mapping_name="",
            y_delta_left=y_delta,
            y_delta_right=y_delta,
        )

        plot_vector(ax_right, 6, 1, None, light_grey, "", y_delta=y_delta)
        ax_right.text(6, y_delta, name, ha="center", va="center")

    x_delta_dots = -0.1
    ax_right.text(-3 + x_delta_dots, -1.75, r"$\vdots$", ha="center", va="center", fontsize=18)
    ax_right.text(+x_delta_dots, -1.75, r"$\vdots$", ha="center", va="center", fontsize=18)
    ax_right.text(3 + x_delta_dots, -1.75, r"$\vdots$", ha="center", va="center", fontsize=18)

    subtitle(ax_right, "N-step-ahead - Multiple MLPs")

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
    png_filename = os.path.join(base_path, f"single_vs_multi.png")

    fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def plot_vector(
    ax: plt.Axes, x_pos: float, n: int, dot_clr: tuple, edge_clr: tuple, caption_txt: str = None, y_delta: int = 0
):

    # --- init -----------------------------------
    y_pos = [i - ((n - 1) / 2) + y_delta for i in range(n)]

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

    ax.text(x_pos, min(y_pos) - 0.75, caption_txt, va="center", ha="center", fontsize=12, usetex=True)


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
    y_delta_left: int = 0,
    y_delta_right: int = 0,
):

    # --- prep x & y --------------------------------------
    y_from = (n_from - 1) / 2  # vertically spanning [-y_from, y_from] + y_delta_left
    y_to = (n_to - 1) / 2  # vertically spanning [-y_to, y_to] + y_delta_right

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

    # adjust for y_delta
    y_from_bot = y_delta_left - y_from
    y_from_top = y_delta_left + y_from
    y_to_bot = y_delta_right - y_to
    y_to_top = y_delta_right + y_to

    # --- plot with gradient ------------------------------
    n_steps = 100
    for i in range(n_steps):

        sub_x_from = x_from + (i / n_steps) * (x_to - x_from)
        sub_x_to = x_from + ((i + 1.1) / n_steps) * (x_to - x_from)

        sub_y_from_bot = y_from_bot + (i / n_steps) * (y_to_bot - y_from_bot)
        sub_y_from_top = y_from_top + (i / n_steps) * (y_to_top - y_from_top)

        sub_y_to_bot = y_from_bot + ((i + 1.1) / n_steps) * (y_to_bot - y_from_bot)
        sub_y_to_top = y_from_top + ((i + 1.1) / n_steps) * (y_to_top - y_from_top)

        c = (i + 0.5) / n_steps
        clr = tuple([c_from + c * (c_to - c_from) for c_from, c_to in zip(clr_from, clr_to)])

        xy = np.array(
            [
                [sub_x_from, sub_y_from_top],
                [sub_x_to, sub_y_to_top],
                [sub_x_to, sub_y_to_bot],
                [sub_x_from, sub_y_from_bot],
            ]
        )

        poly = Polygon(xy, closed=True, edgecolor=clr, facecolor=clr, linewidth=0)
        ax.add_patch(poly)

    if mapping_name is None:
        mapping_name = f"{n_from}x{n_to}\n\nlinear\nmapping"

    ax.text((x_from + x_to) / 2, (y_delta_left + y_delta_right) / 2, mapping_name, va="center", ha="center")
