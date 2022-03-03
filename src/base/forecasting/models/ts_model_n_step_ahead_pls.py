from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from src.tools.matplotlib import plot_style_matplotlib_default

from .ts_model_n_step_ahead_linear import TimeSeriesModelMultiStepLinear


class TimeSeriesModelMultiStepPLS(TimeSeriesModelMultiStepLinear):
    """Linear auto-regressive n-step-ahead predictor using PLS."""

    # colors for creating PLS component plots
    CLR_GREY = (0.8, 0.8, 0.8)
    CLR_GREEN = (0.4, 0.8, 0.4)
    CLR_BLUE = (0.4, 0.4, 1.0)

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, signal_name: str, p: int, n: int, rank: int, cv: dict = None):
        super().__init__(
            model_type="n-step-pls",
            signal_name=signal_name,
            p=p,
            n=n,
            avoid_training_nans=True,  # remove rows with at least 1 NaN in either features or targets
            cv=cv,
        )

        self.rank = 0
        self._pls = None  # type: Optional[PLSRegression]
        self.set_param("rank", rank)

    # -------------------------------------------------------------------------
    #  Set parameters
    # -------------------------------------------------------------------------
    def set_param(self, param_name: str, param_value: Any):

        super().set_param(param_name, param_value)

        self._pls = PLSRegression(
            n_components=self.rank,
            scale=False,  # x & y are already scaled
            max_iter=10_000,  # increased accuracy & robustness
            tol=1e-9,  # increased accuracy
            copy=True,
        )

    def _model_complexity(self) -> float:
        # roughly speaking the number of free parameters in the model
        return self.p * self.rank

    # -------------------------------------------------------------------------
    #  Fit
    # -------------------------------------------------------------------------
    def _fit_tabulated(self, x: np.ndarray, y: np.ndarray):

        print(f"Training n-step-pls with (p, n, rank)=({self.p}, {self.n}, {self.rank}).")

        self._pls.fit(x, y)

        self.C = self._pls.coef_.copy()

    # -------------------------------------------------------------------------
    #  Interpretability
    # -------------------------------------------------------------------------
    @property
    def L(self) -> np.ndarray:
        # matrix L = (rank * n_features) such that each row contains a principal vector in feature space
        return self._pls.x_weights_.transpose()

    @property
    def R(self) -> np.ndarray:
        # matrix R = (rank * n_targets) such that each row contains a principal vector in target space
        return self._pls.y_weights_.transpose()

    def plot_components(self, n: int) -> Tuple[plt.Figure, List[plt.Axes]]:

        # --- argument checking ---------------------------
        assert n <= self.rank

        # --- prepare graphics ----------------------------
        plot_style_matplotlib_default()
        fig, axes = plt.subplots(nrows=n, ncols=1, sharex="col")  # type: plt.Figure, List[plt.Axes]

        # --- plot components -----------------------------
        for i in range(n):

            # prepare axis
            ax = axes[i]  # type: plt.Axes
            ax.set_xlim(left=-self.p, right=self.n)
            ax.set_xticks([x for x in range(-self.p, self.n + 1) if (x % 96) == 0])

            # extract data
            l_vec = np.flip(self.L[i, :].flatten())  # flip due to how we implemented build_toeplitz
            r_vec = self.R[i, :].flatten()

            # equalize norms of l & r
            l_norm = np.linalg.norm(l_vec)
            r_norm = np.linalg.norm(r_vec)
            target_norm = np.sqrt(l_norm * r_norm)

            l_vec = target_norm * (l_vec / l_norm)
            r_vec = target_norm * (r_vec / r_norm)

            # plot hor & vec ref lines
            ax.plot([-self.p - 10, self.n + 10], [0, 0], c=self.CLR_GREY, lw=0.5)  # hor
            ax.plot([0, 0], [-10, 10], c=self.CLR_GREY, lw=0.5)  # ver

            # plot l & r
            ax.plot(np.arange(-self.p, 0), l_vec, c=self.CLR_BLUE)
            ax.plot(np.arange(0, self.n), r_vec, c=self.CLR_GREEN)

            # vertical scale
            lr_max = max([max(np.abs(l_vec)), max(np.abs(r_vec))])
            ax.set_ylim(bottom=-1.1 * lr_max, top=1.1 * lr_max)
            ax.set_ylabel(f"$PLS_{i+1}$")

        # style figure
        fig.suptitle(
            f"First {n} components of PLS regression \n \n "
            "LEFT: features = past samples (blue) \n "
            "RIGHT: targets = future samples (green)"
        )

        fig.set_size_inches(w=8, h=8)
        fig.tight_layout()

        return fig, axes
