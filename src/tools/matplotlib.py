import datetime
from typing import Iterable, List, Tuple

import darts
import matplotlib.pyplot as plt
import numpy as np

from .datetime import float_to_ts, ts_to_float


# -------------------------------------------------------------------------
#  Axes scales
# -------------------------------------------------------------------------
def set_x_scale_daily(
    ax: plt.Axes, ts_from: datetime.datetime, ts_to: datetime.datetime, margin: float = 0.01
) -> Tuple[float, float]:

    # x_ticks
    ts_min = ts_from.replace(hour=0, minute=0, second=0, microsecond=0)
    ts_max = ts_to.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)

    n_days = (ts_max - ts_min).days

    dates = [(ts_min + datetime.timedelta(days=i)).date() for i in range(n_days + 1)]
    x_ticks = [ts_to_float(date) for date in dates]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([date.strftime("%d.%m") for date in dates], ha="right", rotation=45)

    # scale
    x_min, x_max = __compute_scale_with_margin(min(x_ticks), max(x_ticks), margin)
    ax.set_xlim(x_min, x_max)

    # return
    return x_min, x_max


def set_y_scale(
    ax: plt.Axes,
    *,
    y_min: float,
    y_max: float,
    y_data: Iterable[float] = None,
    y_ticks: List[float] = None,
    n_ticks: int = 10,
    margin: float = 0.01,
) -> Tuple[float, float]:

    if y_ticks is None:
        y_ticks = optimal_ticks(n_ticks, min_value=y_min, max_value=y_max, values=y_data)

    # y_ticks
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], ha="right")

    # scale
    y_min, y_max = __compute_scale_with_margin(min(y_ticks), max(y_ticks), margin)
    ax.set_ylim(y_min, y_max)

    # return
    return y_min, y_max


def enable_grid_lines(ax: plt.Axes):
    ax.grid(True, which="both")


def optimal_ticks(n_ticks: int, *, min_value: float = None, max_value: float = None, values: Iterable[float] = None):

    # --- determine data range ----------------------------
    if values is not None:
        min_value = min(values)
        max_value = max(values)

    if (min_value is not None) and (max_value is not None):
        data_range = max_value - min_value
    else:
        raise ValueError(
            "Could not determine data range; please specify either 'values' or both 'min_value' and 'max_value' arguments."
        )

    # --- optimal ticks -----------------------------------
    tick_delta_min = np.power(10, np.floor(np.log10(data_range / n_ticks)))  # power of 10 as lower bound for tick_delta

    opt_tick_delta = None
    opt_ticks = None
    opt_score = None
    for cand_tick_delta in [tick_delta_min * i for i in [1, 2, 5, 10, 20, 50]]:

        cand_ticks = __compute_ticks(min_value, max_value, cand_tick_delta)
        score = abs(np.log(len(cand_ticks) / n_ticks))

        if (opt_score is None) or (score < opt_score):
            opt_tick_delta = cand_tick_delta
            opt_ticks = cand_ticks
            opt_score = score

    if opt_tick_delta >= 1:
        opt_ticks = [int(tick) for tick in opt_ticks]

    return opt_ticks


# -------------------------------------------------------------------------
#  Plotting special shapes
# -------------------------------------------------------------------------
def plot_curly_braces(
    ax: plt.Axes, xy_from: Tuple[float, float], xy_to: Tuple[float, float], height: float = None, **line_kwargs
) -> np.ndarray:
    """
    Plot curly braces on matplotlib axes by providing coordinates from and to + height.
    :param ax: matplotlib Axes
    :param xy_from: (x, y) from which bracket starts
    :param xy_to: (x, y) where bracket ends
    :param height: (float, default = 0.1*length) abs height of braces
    :param line_kwargs: optional kwargs for determining plot style
    :return: (x, y) top of the pointy end of curly brace
    """

    # --- process xy_from, xy_to, height ------------------
    xy_from = np.array(xy_from)
    xy_to = np.array(xy_to)
    xy_mid = (xy_from + xy_to) / 2

    xy_len = xy_to - xy_from
    length = np.linalg.norm(xy_len)
    if height is None:
        height = length * 0.1
    height = min(height, length / 2)  # height should be <= length/2

    xy_height = np.array([xy_len[1], -xy_len[0]]) * (height / length)

    c = (height / 2) / length  # fraction of circle segment radius wrt length

    # --- process line_kwargs -----------------------------
    line_kwargs = line_kwargs or dict()
    line_kwargs["solid_capstyle"] = "round"
    line_kwargs["solid_joinstyle"] = "round"
    line_kwargs["dash_capstyle"] = "round"
    line_kwargs["dash_joinstyle"] = "round"

    # --- helper - circle segments ------------------------
    def circle_segment(center, v1, v2, angle_from, angle_to) -> np.array:

        n_coords = 30
        coords = np.zeros((n_coords, 2))

        for i, angle_deg in enumerate(np.linspace(angle_from, angle_to, n_coords)):
            angle_rad = np.deg2rad(angle_deg)
            coords[i, :] = (center + np.sin(angle_rad) * v1 + np.cos(angle_rad) * v2).reshape(1, 2)

        return coords

    # --- plot actual brace -------------------------------
    circ_v1 = c * xy_len
    circ_v2 = 0.5 * xy_height

    all_coords = np.concatenate(
        [
            circle_segment(xy_from + circ_v1, circ_v1, circ_v2, -90, 0),
            circle_segment(xy_mid - circ_v1 + xy_height, circ_v1, circ_v2, 180, 90),
            circle_segment(xy_mid + circ_v1 + xy_height, circ_v1, circ_v2, 270, 180),
            circle_segment(xy_to - circ_v1, circ_v1, circ_v2, 0, 90),
        ],
        axis=0,
    )

    ax.plot(all_coords[:, 0], all_coords[:, 1], **line_kwargs)

    # --- return tip --------------------------------------
    return xy_mid + xy_height


# -------------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------------
def __compute_ticks(min_value: float, max_value: float, tick_delta: float) -> List[float]:
    return [
        tick_delta * i for i in range(int(np.floor(min_value / tick_delta)), int(np.ceil(max_value / tick_delta)) + 1)
    ]


def __compute_scale_with_margin(scale_min: float, scale_max: float, margin: float) -> Tuple[float, float]:

    delta = abs(scale_max - scale_min)
    if delta == 0:
        delta = 1

    return scale_min - (margin * delta), scale_max + (margin * delta)


# -------------------------------------------------------------------------
#  Plotting styles
# -------------------------------------------------------------------------
def plot_style_darts():
    # this is what darts does by default (triggered by 'import darts')
    plt.rcParams.update(darts.u8plots_mplstyle)


def plot_style_matplotlib_default():
    # this overrides the overrides from darts back to the default values
    plt.rcdefaults()
    plt.rcParams["savefig.facecolor"] = "white"
