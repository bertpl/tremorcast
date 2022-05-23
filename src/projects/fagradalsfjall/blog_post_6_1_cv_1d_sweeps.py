import os
import pickle
from enum import Enum, auto
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base.forecasting.models import TimeSeriesModelMultiStepNeuralMLP, loss_mae
from src.projects.fagradalsfjall.evaluate_models import get_dataset_train
from src.tools.matplotlib import plot_style_matplotlib_default

from ._project_settings import FORECAST_SIGNAL_NAME
from .evaluate_models.evaluate_forecast_models import _get_output_path


# =================================================================================================
#  Sweep ENUM
# =================================================================================================
class Sweep(Enum):
    N_EPOCHS_VALLEY = auto()
    N_EPOCHS_MEDIUM = auto()
    N_EPOCHS_MINIMUM = auto()
    N_EPOCHS_WD_LO = auto()
    N_EPOCHS_WD_HI = auto()
    N_EPOCHS_SHALLOW = auto()
    N_EPOCHS_DEEP = auto()
    WD = auto()
    P = auto()
    N_LAYERS = auto()

    def lower_name(self):
        return self.name.lower()


# =================================================================================================
#  Main function
# =================================================================================================
def blog_6_cv_1d_sweeps(n: int, do_train: bool = True, do_plot: bool = True):

    # --- define sweeps -----------------------------------

    sweeps = [
        Sweep.N_EPOCHS_VALLEY,
        Sweep.N_EPOCHS_MEDIUM,
        Sweep.N_EPOCHS_MINIMUM,
        Sweep.N_EPOCHS_WD_LO,
        Sweep.N_EPOCHS_WD_HI,
        Sweep.N_EPOCHS_SHALLOW,
        Sweep.N_EPOCHS_DEEP,
        Sweep.WD,
        Sweep.P,
        Sweep.N_LAYERS,
    ]

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- perform 1D sweeps -------------------------------
    for sweep in sweeps:

        # --- train model -------------
        if do_train:

            print("-" * 120)
            print(f"SWEEP: {sweep}")
            print("-" * 120)
            print()

            # cv settings
            cv_settings, *_ = _get_cv_settings(sweep, n)
            cv_model = TimeSeriesModelMultiStepNeuralMLP(signal_name=FORECAST_SIGNAL_NAME, n=0, p=0, cv=cv_settings)

            # perform sweep
            cv_model.fit(df_train)

            # save model
            model_filename = get_filename_model(sweep, n)
            with open(model_filename, "wb") as f:
                pickle.dump(cv_model, f)

        # --- plot result -------------
        if do_plot:
            plot_sweep_result(sweep, n)


# =================================================================================================
#  Helpers - CV Settings
# =================================================================================================
def _get_cv_settings(sweep: Sweep, n: int) -> Tuple[dict, str, bool, str]:
    """
    Returns cv_settings dict & related info based on parameter over which we want to do 1D sweep
    :param sweep: (Sweep) which 1D sweep for which to return info
    :param n: (int) n steps ahead
    :return: (cv_settings, param_name, log_x_scale, sub_title)
    """

    # --- nominal settings --------------------------------
    cv_settings = dict(
        n_splits=10,
        randomize=True,
        randomize_runs=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [3],
            "n": [n],
            "p": [96],
            "wd": [0.1],
            "n_epochs": [100],
            "lr_max_method": ["minimum"],
        },
    )

    # --- 1D sweep ----------------------------------------
    log_x_scale = True
    sub_title = None

    if sweep == Sweep.N_EPOCHS_WD_LO:
        cv_settings["param_grid"]["wd"] = [0.0]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "wd=0"
    elif sweep == Sweep.N_EPOCHS_WD_HI:
        cv_settings["param_grid"]["wd"] = [1.0]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "wd=1.0"
    elif sweep == Sweep.N_EPOCHS_VALLEY:
        cv_settings["param_grid"]["lr_max_method"] = ["valley"]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "lr_max='valley'"
    elif sweep == Sweep.N_EPOCHS_MEDIUM:
        cv_settings["param_grid"]["lr_max_method"] = ["intermediate"]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "lr_max='intermediate'"
    elif sweep == Sweep.N_EPOCHS_MINIMUM:
        cv_settings["param_grid"]["lr_max_method"] = ["minimum"]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "lr_max='minimum'"
    elif sweep == Sweep.N_EPOCHS_SHALLOW:
        cv_settings["param_grid"]["n_hidden_layers"] = [1]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "n_hidden_layers=1"
    elif sweep == Sweep.N_EPOCHS_DEEP:
        cv_settings["param_grid"]["n_hidden_layers"] = [10]
        cv_settings["param_grid"]["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        param_name = "n_epochs"
        sub_title = "n_hidden_layers=10"
    elif sweep == Sweep.WD:
        cv_settings["param_grid"]["wd"] = [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            20,
            50,
            100,
        ]
        param_name = "wd"
    elif sweep == Sweep.P:
        cv_settings["param_grid"]["p"] = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288, 384]
        param_name = "p"
    elif sweep == Sweep.N_LAYERS:
        cv_settings["param_grid"]["n_hidden_layers"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        param_name = "n_hidden_layers"
        log_x_scale = False
    else:
        raise NotImplementedError(f"sweep name '{sweep}' not implemented.")

    # --- return ------------------------------------------
    return cv_settings, param_name, log_x_scale, sub_title


# =================================================================================================
#  Helpers - File Names
# =================================================================================================
def get_filename_base(sweep: Sweep, n: int) -> str:
    return os.path.join(_get_output_path("post_6_nn"), f"1d_sweep_{n}_step_{sweep.lower_name()}")


def get_filename_fig(sweep: Sweep, n: int) -> str:
    return get_filename_base(sweep, n) + ".png"


def get_filename_model(sweep: Sweep, n: int) -> str:
    return get_filename_base(sweep, n) + ".pkl"


# =================================================================================================
#  Helpers - Plotting
# =================================================================================================
def plot_sweep_result(sweep: Sweep, n: int):

    # --- override style overrides ------------------------
    plot_style_matplotlib_default()

    # --- load model --------------------------------------
    model_filename = get_filename_model(sweep, n)
    with open(model_filename, "rb") as f:
        cv_model = pickle.load(f)  # type: TimeSeriesModelMultiStepNeuralMLP

    # --- get info ----------------------------------------
    _, param_name, log_x_scale, sub_title = _get_cv_settings(sweep, n)

    # --- extract results ---------------------------------
    param_values = np.array([cv_result["params"][param_name] for cv_result in cv_model.cv_results["all"]])
    training_losses_mean = np.array([cv_result["training_losses"]["mean"] for cv_result in cv_model.cv_results["all"]])
    training_losses_std = np.array([cv_result["training_losses"]["std"] for cv_result in cv_model.cv_results["all"]])
    validation_losses_mean = np.array(
        [cv_result["validation_losses"]["mean"] for cv_result in cv_model.cv_results["all"]]
    )
    validation_losses_std = np.array(
        [cv_result["validation_losses"]["std"] for cv_result in cv_model.cv_results["all"]]
    )

    # --- create figure -----------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1)  # type: plt.Figure, plt.Axes

    # --- uncertainty bands ---
    ax.fill_between(
        param_values,
        training_losses_mean - training_losses_std,
        training_losses_mean + training_losses_std,
        color="r",
        alpha=0.1,
    )
    ax.fill_between(
        param_values,
        validation_losses_mean - validation_losses_std,
        validation_losses_mean + validation_losses_std,
        color="g",
        alpha=0.1,
    )

    # --- actual lines ---
    h_train = ax.plot(param_values, training_losses_mean, "r-x")
    h_val = ax.plot(param_values, validation_losses_mean, "g-x")

    # --- decorate ---
    ax.grid(visible=True)

    ax.legend([h_train[0], h_val[0]], ["mean training loss", "mean validation loss"])

    ax.set_xlabel(param_name)
    if log_x_scale:
        ax.set_xscale("log")
    ax.set_xticks(param_values)
    ax.set_xticklabels([str(pv) for pv in param_values])

    ax.set_ylabel("MAE")
    y_max = max(max(training_losses_mean), max(validation_losses_mean))
    y_max = min(10 * min(validation_losses_mean), 1.2 * y_max)  # make sure minimum validation loss is still discernible
    ax.set_ylim(bottom=0.0, top=y_max)

    fig.suptitle("1D sweep cross-validation results" + (f"\n({sub_title})" if sub_title else ""))

    fig.set_size_inches(w=8, h=6)
    fig.tight_layout()

    # --- lines & best performance ---
    min_val_loss = min(validation_losses_mean)
    param_best = param_values[list(validation_losses_mean).index(min_val_loss)]

    x_min, x_max = ax.get_xlim()

    # line + text
    ax.plot([x_min, x_max], [min_val_loss, min_val_loss], "g--", alpha=0.5)
    ax.text(x_min, 0.95 * min_val_loss, f" {min_val_loss:.3f}", ha="left", va="top", color="g")

    # circle around best result
    ax.plot(param_best, min_val_loss, "go")

    ax.set_xlim(x_min, x_max)

    # --- save fig ----------------------------------------
    fig_filename = get_filename_fig(sweep, n)
    fig.savefig(fig_filename, dpi=600)
