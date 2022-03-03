import os
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.tools.matplotlib import plot_curly_braces, plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_cross_validation():

    # --- create fig, ax ----------------------------------
    plot_style_matplotlib_default()
    fig, ax = plt.subplots(nrows=1, ncols=1)  # type: plt.Figure, plt.Axes

    # --- colors ------------------------------------------
    clr_all_data = (0.6, 0.6, 0.6)  # grey
    clr_train = (0.5, 0.8, 0.5)  # green
    clr_val = (0.9, 0.9, 0.5)  # orange
    clr_test = (0.5, 0.5, 0.8)  # blue

    clr_arrow = (0.8, 0.8, 0.8)  # light grey
    clr_rect = (0.95, 0.95, 0.95)  # lighter grey
    clr_text = (0.3, 0.3, 0.3)
    clr_text_blue = (0.4, 0.4, 1)  # light blue

    # --- create figure -----------------------------------

    # all data
    plot_data_block(ax, row=-2.5, col_from=3, col_to=10, clr=clr_all_data, txt="All data")

    ax.arrow(
        3.5,
        1.75,
        -1,
        -1,
        width=0.2,
        head_length=0.3,
        length_includes_head=True,
        facecolor=clr_arrow,
        edgecolor=clr_arrow,
    )

    ax.arrow(
        9.5,
        1.75,
        1,
        -1,
        width=0.2,
        head_length=0.3,
        length_includes_head=True,
        facecolor=clr_arrow,
        edgecolor=clr_arrow,
    )

    # training & test
    plot_data_block(ax, row=0, col_from=0, col_to=4, clr=clr_train, txt="Training")
    plot_data_block(ax, row=0, col_from=10, col_to=12, clr=clr_test, txt="Test")
    ax.text(
        2,
        -1.2,
        "split in k subsets for model tuning;\nretrain model on entire set after tuning",
        va="center",
        ha="center",
        fontsize=9,
        color=clr_text,
    )
    ax.text(
        11,
        -1.2,
        "keep aside & don't use\nfor training or tuning",
        va="center",
        ha="center",
        fontsize=9,
        color=clr_text,
    )

    # cv splits - arrow
    ax.arrow(
        2, -1.75, 0, -1, width=0.2, head_length=0.3, length_includes_head=True, facecolor=clr_arrow, edgecolor=clr_arrow
    )

    # cv splits - blocks & text
    for r in range(5):
        row = 3.5 + (r * 1.2)

        # plot Vx & Tx blocks
        for c in range(5):
            if r == c:
                plot_data_block(ax, row=row, col_from=c, col_to=c, clr=clr_val, txt=f"$V_{c+1}$")
            else:
                plot_data_block(ax, row=row, col_from=c, col_to=c, clr=clr_train, txt=f"$T_{c+1}$")

        # right arrow
        for x in [4.75, 7.25, 11]:
            ax.arrow(
                x,
                -row,
                0.5,
                0,
                width=0.15,
                head_length=0.2,
                length_includes_head=True,
                facecolor=clr_arrow,
                edgecolor=clr_arrow,
            )

        # text
        v_set_str = ", ".join([f"T_{c+1}" for c in range(5) if c != r])
        ax.text(6.25, -row, f"model {r+1}", ha="center", va="center", fontsize=9)
        ax.text(9.5, -row, f"train on\n${v_set_str}$", ha="center", va="center", fontsize=9)
        ax.text(13, -row, f"evaluate on\n$V_{r + 1}$", ha="center", va="center", fontsize=9)

    # --- train & val rectangles --------------------------
    rect = Rectangle((8.35, -9), 2.3, 6.2, lw=0, facecolor=clr_rect, zorder=-10)
    ax.add_patch(rect)

    rect = Rectangle((11.85, -9), 2.3, 6.2, lw=0, facecolor=clr_rect, zorder=-10)
    ax.add_patch(rect)

    # --- braces ------------------------------------------
    plot_curly_braces(ax, (8.25, -9.25), (10.75, -9.25), height=0.3, lw=1, color="k")
    ax.text(9.5, -9.75, "training set\nperformance", fontsize=10, va="top", ha="center", color=clr_text_blue)
    ax.text(9.5, -10.7, "(while tuning)", fontsize=10, va="top", ha="center")

    plot_curly_braces(ax, (11.75, -9.25), (14.25, -9.25), height=0.3, lw=1, color="k")
    ax.text(13, -9.75, "validation set\nperformance", fontsize=10, va="top", ha="center", color=clr_text_blue)
    ax.text(13, -10.55, "=", fontsize=10, va="top", ha="center")
    ax.text(13, -11, "cross-validation\nperformance", fontsize=10, va="top", ha="center", color=clr_text_blue)

    plot_curly_braces(ax, (-1, 0.75), (-1, -0.75), height=0.3, lw=1, color="k")
    ax.text(-1.5, 0.25, "training set\nperformance", fontsize=10, va="center", ha="right", color=clr_text_blue)
    ax.text(-1.5, -0.5, "(after tuning)", fontsize=10, va="center", ha="right")

    plot_curly_braces(ax, (13, -0.75), (13, 0.75), height=0.3, lw=1, color="k")
    ax.text(13.5, 0, "test set\nperformance", fontsize=10, va="center", ha="left", color=clr_text_blue)

    # --- decoration --------------------------------------
    ax.set_axis_off()
    ax.set_xlim(left=-4.5, right=16)
    ax.set_ylim(bottom=-12, top=3.5)

    fig.suptitle("Data partitioning & performance evaluation using k-fold cross-validation")

    fig.set_size_inches(w=9, h=7.5)
    fig.tight_layout()

    # --- save --------------------------------------------
    base_path = _get_output_path("post_5_n_step_ahead")
    png_filename = os.path.join(base_path, f"cross_validation.png")

    fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def plot_data_block(ax: plt.Axes, row: float, col_from: float, col_to: float, clr: Tuple, txt: str):

    y_top = -row + 0.5
    y_bot = -row - 0.5
    y_mid = -row

    x_left = col_from - 0.5
    x_right = col_to + 0.5
    x_mid = (col_from + col_to) / 2

    rect = Rectangle(
        xy=(x_left, y_bot),
        width=x_right - x_left,
        height=y_top - y_bot,
        facecolor=clr,
        edgecolor=(0.1, 0.1, 0.1),
        linewidth=0.5,
    )

    ax.add_patch(rect)

    ax.text(x_mid, y_mid, txt, va="center", ha="center", fontsize=9)
