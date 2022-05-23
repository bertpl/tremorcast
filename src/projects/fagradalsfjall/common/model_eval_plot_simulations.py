import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.projects.fagradalsfjall.common.dataset import load_test_data_numpy, load_test_data_vedur, load_train_data_numpy
from src.projects.fagradalsfjall.common.model_eval import ModelEvalResult
from src.projects.fagradalsfjall.common.project_settings import FORECAST_SIGNAL_COLOR
from src.tools.datetime import ts_to_float
from src.tools.matplotlib import plot_style_matplotlib_default


def plot_simulations(simulations: Dict[str, ModelEvalResult], folder: Path) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:

    return {
        model_name: plot_simulation(model_name, model_eval_result, folder)
        for model_name, model_eval_result in simulations.items()
    }


def plot_simulation(model_name: str, model_eval_result: ModelEvalResult, folder: Path = None):

    # --- init --------------------------------------------
    plot_style_matplotlib_default()

    # --- load data ---------------------------------------
    test_data = load_test_data_vedur()  # type: VedurHarmonicMagnitudes

    x_train = load_train_data_numpy()
    x_test = load_test_data_numpy()

    # --- base plot ---------------------------------------
    title = f"Test set simulations - '{model_name}'"
    fig, ax = test_data.create_plot(title=title, aspect_ratio=1.5)

    # --- plot main signal --------------------------------
    x_values = [ts_to_float(t) for t in test_data.time]
    plot_clr = [c / 255 for c in FORECAST_SIGNAL_COLOR]

    ax.plot(x_values, x_test, scalex=False, scaley=False, c=plot_clr, lw=2)
    ax.plot(x_values, x_test, scalex=False, scaley=False, c="k", ls="--", lw=0.5)

    # --- plot forecasts ----------------------------------
    for i, prediction in model_eval_result.test_simulations:

        if prediction.size > 0:
            i_within_test_set = i - len(x_train)  # i refresh to sample # within [x_train, x_test]

            x = x_values[i_within_test_set : i_within_test_set + len(prediction)]

            ax.plot(x[0], prediction[0], "ko", scalex=False, scaley=False)
            ax.plot(x, prediction, "k", lw=1, scalex=False, scaley=False)

    # --- formatting --------------------------------------
    fig.set_size_inches(w=16, h=7)

    # --- save to disk ------------------------------------
    if folder:
        os.makedirs(folder, exist_ok=True)
        fig.savefig(folder / f"simulation_{model_name.replace('-', '_')}.png", dpi=600)

    # --- return ------------------------------------------
    return fig, ax
