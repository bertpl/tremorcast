import dataclasses
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_one_vs_n_step_ahead_tabular():

    # --- create fig, ax ----------------------------------
    plot_style_matplotlib_default()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax_left, ax_right = ax[0], ax[1]  # type: plt.Axes, plt.Axes

    # --- traces ------------------------------------------
    trace_colors = [(c, 0.5 * c, 1 - c) for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    traces = [
        (5, trace_colors[0]),
        (10, trace_colors[1]),
        (15, trace_colors[2]),
        (20, trace_colors[3]),
        (25, trace_colors[4]),
        (30, trace_colors[5]),
    ]

    # --- 1-step-ahead ------------------------------------
    plot_matrices(ax_left, n_samples=30, p=7, n=1, traces=traces)

    ax_left.set_axis_off()
    ax_left.set_xlim(left=-15, right=10)
    ax_left.set_ylim(bottom=-13, top=15)

    ax_left.set_title("tabular data set\nfor 1-step-ahead auto-regressive model")

    # --- n-step-ahead -----------------------------------------
    plot_matrices(ax_right, n_samples=30, p=7, n=5, traces=traces)

    ax_right.set_axis_off()
    ax_right.set_xlim(left=-10, right=15)
    ax_right.set_ylim(bottom=-13, top=15)

    ax_right.set_title("tabular data set\nfor 5-step-ahead auto-regressive model")

    # --- decorate ----------------------------------------
    fig.set_size_inches(w=10, h=5)
    fig.tight_layout()

    # --- save --------------------------------------------
    base_path = _get_output_path("post_5_n_step_ahead")
    png_filename = os.path.join(base_path, f"n_step_ahead_tabular.png")

    fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
@dataclasses.dataclass
class Matrix:
    n_rows: int
    n_cols: int
    x_left: float
    y_top: float

    def row_to_y(self, row: int) -> float:
        return self.y_top - row

    def col_to_x(self, col: int) -> float:
        return self.x_left + col

    def plot_cell_range(
        self,
        ax: plt.Axes,
        row_from: int,
        row_to: int,
        col_from: int,
        col_to: int,
        edge_clr=(0, 0, 0),
        face_clr=(0, 0, 0, 0),
        zorder=0,
    ):
        x_left = self.col_to_x(col_from) - 0.5
        x_right = self.col_to_x(col_to) + 0.5
        y_top = self.row_to_y(row_from) + 0.5
        y_bottom = self.row_to_y(row_to) - 0.5
        rect = Rectangle(
            (x_left, y_bottom),
            x_right - x_left,
            y_top - y_bottom,
            edgecolor=edge_clr,
            facecolor=face_clr,
            zorder=zorder,
        )
        ax.add_patch(rect)

    def plot(self, ax: plt.Axes, name: str):
        self.plot_cell_range(ax, 0, self.n_rows - 1, 0, self.n_cols - 1, zorder=1)
        ax.text(self.x_left - 0.5, self.y_top + 1, name, va="center", ha="left", fontsize=9)

    def show_row_count(self, ax: plt.Axes, clr=(0, 0, 0), inside: bool = False):
        x = self.x_left + (-0.2 if inside else -1.5)
        y = self.row_to_y((self.n_rows - 1) / 2)
        ax.text(x, y, f"{self.n_rows}", fontsize=8, color=clr, va="center", ha="center")

    def show_col_count(self, ax: plt.Axes, clr=(0, 0, 0), inside: bool = False):
        x = self.col_to_x((self.n_cols - 1) / 2)
        y = self.row_to_y(self.n_rows - 1) + (0 if inside else -1.5)
        ax.text(x, y, f"{self.n_cols}", fontsize=8, color=clr, va="center", ha="center")


def plot_matrices(ax: plt.Axes, n_samples: int, p: int, n: int, traces: List[Tuple[int, Any]]):

    k = n_samples - (n + p - 1)
    t_features = np.array([[1 + row + col for col in range(p)] for row in range(k)])
    t_targets = np.array([[1 + p + row + col for col in range(n)] for row in range(k)])

    # initialize matrices
    features = Matrix(n_rows=k, n_cols=p, x_left=-p, y_top=(k - 1) / 2)
    model = Matrix(n_rows=p, n_cols=n, x_left=1, y_top=(p - 1) / 2)
    targets = Matrix(n_rows=k, n_cols=n, x_left=n + 2, y_top=(k - 1) / 2)

    # colors
    transparent = (0, 0, 0, 0)
    light_grey = (0.7, 0.7, 0.7)
    green = (0.1, 0.6, 0.4)

    # features
    features.plot(ax, "features")
    features.plot_cell_range(ax, 3, 3, 0, p - 1, edge_clr=transparent, face_clr=light_grey)
    for i_trace, clr in traces:
        for i_row in range(k):
            for i_col in range(p):
                if t_features[i_row, i_col] == i_trace:
                    features.plot_cell_range(
                        ax,
                        col_from=i_col,
                        col_to=i_col,
                        row_from=i_row,
                        row_to=i_row,
                        edge_clr=transparent,
                        face_clr=clr,
                    )

    features.show_row_count(ax)
    features.show_col_count(ax)

    # model
    ax.text(0, 0, "x", va="center", ha="center")
    model.plot_cell_range(ax, 0, p - 1, 0, n - 1, edge_clr=transparent, face_clr=green)
    model.plot(ax, "")

    model.show_row_count(ax, inside=True, clr=(1, 1, 1))
    model.show_col_count(ax, inside=True, clr=(1, 1, 1))

    # targets
    ax.text(n + 1, 0, "=", va="center", ha="center")
    targets.plot(ax, "targets")
    targets.plot_cell_range(ax, 3, 3, 0, n - 1, edge_clr=transparent, face_clr=light_grey)
    for i_trace, clr in traces:
        for i_row in range(k):
            for i_col in range(n):
                if t_targets[i_row, i_col] == i_trace:
                    targets.plot_cell_range(
                        ax,
                        col_from=i_col,
                        col_to=i_col,
                        row_from=i_row,
                        row_to=i_row,
                        edge_clr=transparent,
                        face_clr=clr,
                    )

    targets.show_col_count(ax)
