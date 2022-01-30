import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.tools.matplotlib import plot_style_matplotlib_default
from src.base.forecasting.models import TimeSeriesModelMultiStepPLS

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_pls_components():

    # --- load CV model -----------------------------------
    base_path = _get_output_path("post_5_n_step_ahead")
    pkl_filename = os.path.join(base_path, "n-step-pls-288-288-7_retraining_off_simdata.pkl")

    with open(pkl_filename, "rb") as f:
        model, *_ = pickle.load(f)      # Type: TimeSeriesModelMultiStepPLS, Any

    # --- plot --------------------------------------------
    fig, ax = model.plot_components(n=7)   # type: plt.Figure, plt.Axes

    # final figure decoration
    # fig.suptitle("PLS components in feature & target space")
    fig.set_size_inches(w=8, h=6)
    fig.tight_layout()

    # save
    png_filename = os.path.join(base_path, "pls_components.png")
    fig.savefig(png_filename, dpi=900)

    plt.show()

    print()
