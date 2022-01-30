import os
import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
from matplotlib.patheffects import Normal, Stroke
from scipy.linalg import lstsq
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_pls_vs_pcr():
    def rotate(deg: float) -> np.ndarray:
        theta = np.radians(deg)
        cs, sn = np.cos(theta), np.sin(theta)
        return np.array([[cs, -sn], [sn, cs]])

    def scale(a: float, b: float) -> np.ndarray:
        return np.array([[a, 0], [0, b]])

    # --- generate data -----------------------------------

    # random
    n = 1000
    r = 0.7 * np.random.normal(size=(n, 2))

    # input & output
    x = r @ rotate(-10) @ scale(0.5, 1) @ rotate(45)
    y = r @ rotate(-90) @ scale(0.2, 1) @ rotate(45)

    # color
    c = r[:, 0].flatten()
    c = c - np.mean(c)
    c = c / np.max(np.abs(c))
    c = 0.5 + 0.5 * c

    # --- perform PLS & PCA -------------------------------
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(x, y)

    # left and right component vectors are situated in the rows
    L_pls = pls.x_weights_.transpose()
    R_pls = pls.y_weights_.transpose()

    L_pca = np.array([[1.0, 1.0], [0.5, -0.5]])

    # --- plot --------------------------------------------
    plot_style_matplotlib_default()

    for plot_type in ["pcr", "pls"]:

        # --- create fig, ax --------------------
        fig, ax = plt.subplots(nrows=2, ncols=2)  # type: plt.Figure, List[plt.Axes]

        ax_top_left, ax_top_right, ax_bot_left, ax_bot_right = ax[0][0], ax[0][1], ax[1][0], ax[1][1]

        # --- plot data -------------------------
        for i in range(n):
            clr = (c[i], 0.5 * c[i], 1 - c[i])  # blue -> orange ==>  (0, 0, 1) -> (1, 0.5, 0)
            ax_top_left.plot(x[i, 0], x[i, 1], "o", markersize=2, markeredgecolor=clr, markerfacecolor=clr)
            ax_top_right.plot(y[i, 0], y[i, 1], "o", markersize=2, markeredgecolor=clr, markerfacecolor=clr)

        # --- components ------------------------
        if plot_type == "pcr":
            L_comp = L_pca
            R_comp = None
            clr = (0.0, 0.0, 0.0)
            name = "PC"
        else:
            L_comp = L_pls
            R_comp = R_pls
            # clr = (0.0, 0.7, 0.3)
            clr = (0.0, 0.0, 0.0)
            name = "PLS"

        # features
        if L_comp is not None:
            plot_components(ax_top_left, L_comp, clr, name)

        # targets
        if R_comp is not None:
            plot_components(ax_top_right, R_comp, clr, name)

        # --- latent space ----------------------
        comp = L_comp[0, :]  # type: np.ndarray
        comp /= np.linalg.norm(comp)

        l = x @ comp.transpose()
        l = l.reshape((n, 1))

        for i in range(n):
            clr = (c[i], 0.5 * c[i], 1 - c[i])  # blue -> orange ==>  (0, 0, 1) -> (1, 0.5, 0)
            ax_bot_left.plot(l[i, 0], 0, "o", markersize=2, markeredgecolor=clr, markerfacecolor=clr)

        grey = (0.5, 0.5, 0.5)
        light_grey = (0.8, 0.8, 0.8)
        ax_bot_left.arrow(
            0,
            0.7,
            0,
            -0.4,
            width=0.1,
            head_length=0.1,
            length_includes_head=True,
            facecolor=light_grey,
            edgecolor=light_grey,
        )
        ax_bot_left.text(0.8, 0.5, "projection\nof data\nonto first\ncomponent", ha="center", va="center", color=grey)

        # --- predictions -----------------------
        ols = LinearRegression(fit_intercept=False)
        ols.fit(l, y)
        y_pred = ols.predict(l)

        # plot targets in lightgrey
        for i in range(n):
            clr = (0.9, 0.9, 0.9)
            ax_bot_right.plot(y[i, 0], y[i, 1], "o", markersize=2, markeredgecolor=clr, markerfacecolor=clr)

        # plot deltas
        for i in range(n):
            clr = (0.8, 0.8, 0.8)
            ax_bot_right.plot([y[i, 0], y_pred[i, 0]], [y[i, 1], y_pred[i, 1]], "-", color=clr, lw=0.2)

        # plot predictions
        for i in range(n):
            clr = (c[i], 0.5 * c[i], 1 - c[i])  # blue -> orange ==>  (0, 0, 1) -> (1, 0.5, 0)
            ax_bot_right.plot(y_pred[i, 0], y_pred[i, 1], "o", markersize=2, markeredgecolor=clr, markerfacecolor=clr)

        ax_bot_right.arrow(
            -1.8,
            0,
            0.8,
            0,
            width=0.12,
            head_length=0.2,
            length_includes_head=True,
            facecolor=light_grey,
            edgecolor=light_grey,
        )
        ax_bot_right.text(-1.4, 0.7, "prediction\nbased on\nlatent\nvariables", ha="center", va="center", color=grey)

        # --- decorate --------------------------
        ax_top_left.set_xlim(left=-2, right=2)
        ax_top_left.set_ylim(bottom=-2, top=2)
        ax_top_left.set_xlabel("$x_1$")
        ax_top_left.set_ylabel("$x_2$")
        ax_top_left.set_title("features")

        ax_top_right.set_xlim(left=-2, right=2)
        ax_top_right.set_ylim(bottom=-2, top=2)
        ax_top_right.set_xlabel("$y_1$")
        ax_top_right.set_ylabel("$y_2$")
        ax_top_right.set_title("targets")

        ax_bot_left.set_xlim(left=-2, right=2)
        ax_bot_left.set_ylim(bottom=-1, top=1)
        ax_bot_left.set_xlabel("$l_1$")
        ax_bot_left.set_yticks([])
        ax_bot_left.set_title("latent space")

        ax_bot_right.set_xlim(left=-2, right=2)
        ax_bot_right.set_ylim(bottom=-2, top=2)
        ax_bot_right.set_xlabel("$\hat{y}_1$")
        ax_bot_right.set_ylabel("$\hat{y}_2$")
        ax_bot_right.set_title("predictions")

        fig.set_size_inches(w=9, h=9)
        if plot_type == "pcr":
            fig.suptitle("Principal Component Regression (PCR)")
        else:
            fig.suptitle("Partial Least Squares (PLS)")

        fig.tight_layout()

        # --- save --------------------------------------------
        base_path = _get_output_path("post_5_n_step_ahead")
        png_filename = os.path.join(base_path, f"ex_{plot_type}.png")

        fig.savefig(png_filename, dpi=600)

    plt.show()


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def plot_components(ax: plt.Axes, comps: np.ndarray, clr, label: str, label_dist=0.25):
    """Plots vectors in rows of comps on ax in given color with provided label"""

    # --- arrows ------------------------------------------
    draw_arrows(
        ax,
        [
            (comps[0, 0], comps[0, 1], clr),
            (comps[1, 0], comps[1, 1], clr),
        ],
    )

    # --- labels -----------------------------------------
    for i in range(comps.shape[0]):
        x, y = comps[i, 0], comps[i, 1]
        this_label = f"${label}_{i+1}$"

        nrm = np.sqrt(x * x + y * y)
        x = x * (nrm + label_dist) / nrm
        y = y * (nrm + label_dist) / nrm

        ax.text(x, y, this_label, **text_style(clr))


def draw_arrows(ax, arrow_info):
    for shadow in [True, False]:
        for x, y, clr in arrow_info:
            ax.arrow(
                0,
                0,
                x,
                y,
                length_includes_head=True,
                width=0.02,
                head_width=0.1,
                zorder=10,
                facecolor=clr,
                edgecolor=clr,
                path_effects=[Stroke(linewidth=3, foreground="white")] if shadow else [Normal()],
            )


def text_style(clr) -> dict:
    return dict(
        color=clr,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=20,
        path_effects=[Stroke(linewidth=4, foreground="white"), Normal()],
    )
