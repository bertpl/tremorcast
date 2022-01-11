from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.tools.matplotlib import plot_style_matplotlib_default


def plot_mae_curves(mae_curves: Dict[str, np.ndarray], threshold: float, title: str) -> Tuple[plt.Figure, plt.Axes]:

    plot_style_matplotlib_default()

    # --- init --------------------------------------------
    max_value = max(max(mae_curve) for mae_curve in mae_curves.values())
    n_curves = len(mae_curves)
    n_samples = min(len(mae_curve) for mae_curve in mae_curves.values())

    # --- prepare data ------------------------------------
    all_mae_curves = np.concatenate(
        [mae_curve.reshape((1, len(mae_curve))) for mae_curve in mae_curves.values()], axis=0
    )
    x_values = np.log(np.arange(1, n_samples + 1).reshape((1, n_samples)))

    # --- create plot -------------------------------------
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    fig.suptitle(title)

    ax.plot(x_values.transpose(), all_mae_curves.transpose(), lw=1)
    ax.plot([-1, 1000], [threshold, threshold], ls="--", c="grey", lw=1)

    # --- finalize layout ---------------------------------
    ax.set_ylim(bottom=0, top=max_value * 1.1)
    ax.set_xlim(left=0, right=np.log(10 * 96))

    ax.set_xticks(np.log([1, 2, 4, 8, 16, 24, 48, 96, 1.5 * 96, 2 * 96, 3 * 96, 4 * 96, 6 * 96, 8 * 96, 10 * 96]))
    ax.set_xticklabels(["15m", "30m", "1h", "2h", "4h", "6h", "12h", "24h", "36h", "2d", "3d", "4d", "6d", "8d", "10d"])
    ax.grid(visible=True, axis="x")

    ax.legend(list(mae_curves.keys()), loc="upper left")

    ax.set_xlabel("Forecast Lead Time")
    ax.set_ylabel("Mean-Absolute-Deviation \n (log-magnitude)")

    fig.set_size_inches(w=9, h=6)
    fig.tight_layout()

    # --- return ------------------------------------------
    return fig, ax
