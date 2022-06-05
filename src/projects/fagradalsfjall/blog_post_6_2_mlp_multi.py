import itertools
import os
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base.forecasting.models import (
    ScoreMetric,
    TimeSeriesModelRegressionMultiMLP,
)
from src.projects.fagradalsfjall.evaluate_models import get_dataset_train
from src.tools.matplotlib import plot_style_matplotlib_default

from ._project_settings import FORECAST_SIGNAL_NAME
from .evaluate_models.evaluate_forecast_models import _get_output_path, evaluate_forecast_models


# =================================================================================================
#  Main functions
# =================================================================================================
def blog_6_2_mlp_multi_sub_model_cv(n: int):

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- model with default hyper-params -----------------
    p = 2*4*24      # 2 days = 192 samples
    model = TimeSeriesModelRegressionMultiMLP(signal_name=FORECAST_SIGNAL_NAME, p=p, n=n)

    model.regressor.set_sub_params(
        input_selection_indices=,
        n_hidden_layers=3,
        layer_width=100,
        # wd=0.1,
        # n_epochs=100,
        lr_max="minimum",
    )








def blog_6_1_mlp_single_1d_sweeps(n: int, do_train: bool = True, do_plot: bool = True, sweeps: List[Sweep] = None):

    # --- define sweeps -----------------------------------
    sweeps = sweeps or list(Sweep)

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- perform 1D sweeps -------------------------------
    for sweep in sweeps:

        # --- train model -------------
        if do_train:

            print("-" * 120)
            print(f"SWEEP: {sweep} --- N: {n}")
            print("-" * 120)
            print()

            # cv settings
            sweep_settings = get_cv_settings_1d(sweep, n)
            ts_mlp_model = sweep_settings.model

            # perform sweep
            ts_mlp_model.cv.grid_search(
                training_data=df_train, param_grid=sweep_settings.param_grid, **sweep_settings.grid_search_kwargs
            )

            # save model
            model_filename = get_filename_1d_sweep_model(sweep, n)
            with open(model_filename, "wb") as f:
                pickle.dump(ts_mlp_model, f)

        # --- plot result -------------
        if do_plot:
            plot_sweep_result(sweep, n)


def blog_6_1_mlp_single_grid_search(n: int):

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- CV model ----------------------------------------
    ts_mlp_model, cv_param_grid, cv_kwargs, *_ = get_cv_settings_full(n)

    # --- cross-validation --------------------------------
    print("-" * 120)
    print(f"CV GRID SEARCH --- N: {n}")
    print("-" * 120)
    print()

    # perform grid search
    ts_mlp_model.cv.grid_search(training_data=df_train, param_grid=cv_param_grid, **cv_kwargs)

    # --- save --------------------------------------------
    model_filename = get_filename_full_cv_model(n)
    with open(model_filename, "wb") as f:
        pickle.dump(ts_mlp_model, f)


# def blog_6_cv_final_comparisons():
#
#     # --- simulate ----------------------------------------
#     naive_models = {
#         "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
#         "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
#     }
#
#     earlier_models = {
#         "ar-192": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=1),
#         "n-step-ols-288-288": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=288, n=288),
#         "n-step-pls-288-288-7": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=7),
#     }
#
#     # --- load models -----------------------------------------
#     selection_method = ParamSelectionMethod.OPTIMAL
#     models = dict()
#     for n in [1, 16]:
#
#         print(f"--- N={n} -------------")
#
#         # load
#         with open(get_filename_full_cv_model(n=n), "rb") as f:
#             model = pickle.load(f)  # type: TimeSeriesModelMultiStepNeuralMLP
#
#         # select params & set
#         params = model.cv_results["best_params_by_method"][selection_method]["params"]
#         for param_name, value in params.items():
#             model.set_param(param_name, value)
#             print(f"{param_name}: {value} ")
#         print()
#
#         # remove cv settings --> avoid redoing CV all over again
#         model.cv_settings = None
#
#         # keep track in dict
#         models[n] = model
#
#     nn_models = {
#         "1-step-mlp": models[1],
#         "16-step-mlp": models[16],
#     }
#
#     # --- simulate ----------------------------------------
#     evaluate_forecast_models(
#         models=naive_models | earlier_models | nn_models,
#         retrain=False,
#         stride=1,
#         results_sub_folder="post_6_nn",
#         simulate=True,
#         evaluate=True,
#         set_name=f"final_result",
#     )


# =================================================================================================
#  Helpers - CV Settings
# =================================================================================================
@dataclass
class SweepSettings:
    sweep: Sweep
    n: int
    model: TimeSeriesModelRegressionMLP
    param_grid: dict
    grid_search_kwargs: dict
    param_name: str
    x_label: str
    log_x_scale: bool
    sub_title: str


def get_cv_settings_1d(sweep: Sweep, n: int) -> SweepSettings:
    """
    Returns cv_settings dict & related info based on parameter over which we want to do 1D sweep
    :param sweep: (Sweep) which 1D sweep for which to return info
    :param n: (int) n steps ahead
    :return: (model, cv_param_grid, cv_kwargs, param_name, x_label, log_x_scale, sub_title)
    """

    # --- cv_kwargs ---------------------------------------
    cv_kwargs = dict(n_splits=5, n_jobs=-1, score_metric=ScoreMetric.MAE)

    # --- misc settings -----------------------------------
    log_x_scale = True
    sub_title = None
    x_label = None

    # --- param grid --------------------------------------

    # larger n seems to benefit from larger n_epochs without risk of overfitting or instability
    if n == 1:
        nominal_lr_max_method = "aggressive"
        nominal_n_epochs = 100
    else:
        nominal_lr_max_method = "minimum"
        nominal_n_epochs = 50

    # default parameters
    cv_param_grid = {
        "input_selection_first_n": [16],
        "n_hidden_layers": [3],
        "layer_width": [100],
        "wd": [0.1],
        "n_epochs": [nominal_n_epochs],
        "lr_max": [nominal_lr_max_method],
    }

    if sweep == Sweep.N_EPOCHS_WD_LO:
        cv_param_grid["wd"] = [0.0]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "wd=0"
    elif sweep == Sweep.N_EPOCHS_WD_HI:
        cv_param_grid["wd"] = [1.0]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "wd=1.0"
    elif sweep == Sweep.N_EPOCHS_LR_MAX_VALLEY:
        cv_param_grid["lr_max"] = ["valley"]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "lr_max='valley'"
    elif sweep == Sweep.N_EPOCHS_LR_MAX_INTERMEDIATE:
        cv_param_grid["lr_max"] = ["intermediate"]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "lr_max='intermediate'"
    elif sweep == Sweep.N_EPOCHS_LR_MAX_MINIMUM:
        cv_param_grid["lr_max"] = ["minimum"]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "lr_max='minimum'"
    elif sweep == Sweep.N_EPOCHS_LR_MAX_AGGRESSIVE:
        cv_param_grid["lr_max"] = ["aggressive"]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "lr_max='aggressive'"
    elif sweep == Sweep.N_EPOCHS_SHALLOW:
        cv_param_grid["n_hidden_layers"] = [1]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "n_hidden_layers=1"
    elif sweep == Sweep.N_EPOCHS_DEEP:
        cv_param_grid["n_hidden_layers"] = [10]
        cv_param_grid["n_epochs"] = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        param_name = "n_epochs"
        sub_title = "n_hidden_layers=10"
    elif sweep == Sweep.WD:
        cv_param_grid["wd"] = sorted([a * b for a, b in itertools.product([1, 2, 5], [0.001, 0.01, 0.1, 1, 10])])
        param_name = "wd"
    elif sweep == Sweep.P:
        cv_param_grid["input_selection_first_n"] = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288]
        param_name = "input_selection_first_n"
        x_label = "p"
    elif sweep == Sweep.LAYER_WIDTH:
        cv_param_grid["layer_width"] = [10, 20, 50, 75, 100, 150, 200, 500, 1000]
        param_name = "layer_width"
    elif sweep == Sweep.N_LAYERS:
        cv_param_grid["n_hidden_layers"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        param_name = "n_hidden_layers"
        log_x_scale = False
    else:
        raise NotImplementedError(f"sweep name '{sweep}' not implemented.")

    if x_label is None:
        x_label = param_name

    # --- model -------------------------------------------
    model = TimeSeriesModelRegressionMLP(
        signal_name=FORECAST_SIGNAL_NAME,
        n=n,
        p=max(cv_param_grid["input_selection_first_n"]),
    )

    # --- return ------------------------------------------
    return SweepSettings(
        sweep=sweep,
        n=n,
        model=model,
        param_grid=cv_param_grid,
        grid_search_kwargs=cv_kwargs,
        param_name=param_name,
        x_label=x_label,
        log_x_scale=log_x_scale,
        sub_title=sub_title,
    )


def get_cv_settings_full(n: int) -> Tuple[TimeSeriesModelRegressionMLP, dict, dict]:

    # --- model -------------------------------------------
    model = TimeSeriesModelRegressionMLP(
        signal_name=FORECAST_SIGNAL_NAME,
        n=n,
        p=32,
    )

    # --- param grid --------------------------------------
    param_grid = {
        "n_hidden_layers": [1, 2, 3],
        "layer_width": [100],
        "input_selection_first_n": [4, 8, 16, 32],
        "wd": [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "n_epochs": [10, 20, 50, 100, 200],
        "lr_max": ["minimum", "aggressive"],
    }

    # --- grid search arguments ---------------------------
    grid_search_kwargs = dict(n_splits=5, n_jobs=6, score_metric=ScoreMetric.MAE)

    # --- return ------------------------------------------
    return model, param_grid, grid_search_kwargs


# =================================================================================================
#  Helpers - File Names
# =================================================================================================
def get_filename_1d_sweep_base(sweep: Sweep, n: int) -> str:
    return os.path.join(_get_output_path("post_6_nn"), f"1d_sweep_{n}_step_{sweep.lower_name()}")


def get_filename_1d_sweep_fig(sweep: Sweep, n: int) -> str:
    return get_filename_1d_sweep_base(sweep, n) + ".png"


def get_filename_1d_sweep_model(sweep: Sweep, n: int) -> str:
    return get_filename_1d_sweep_base(sweep, n) + ".pkl"


def get_filename_full_cv_model(n: int) -> str:
    return os.path.join(_get_output_path("post_6_nn"), f"cv_grid_search_{n}_step.pkl")


# =================================================================================================
#  Helpers - Plotting
# =================================================================================================
def plot_sweep_result(sweep: Sweep, n: int):

    # --- override style overrides ------------------------
    plot_style_matplotlib_default()

    # --- load model --------------------------------------
    model_filename = get_filename_1d_sweep_model(sweep, n)
    with open(model_filename, "rb") as f:
        model = pickle.load(f)  # type: TimeSeriesModelRegressionMLP

    # --- get sweep settings ------------------------------
    sweep_settings = get_cv_settings_1d(sweep, n)

    # --- extract results ---------------------------------
    (
        param_values,
        training_loss_mean,
        training_loss_std,
        validation_loss_mean,
        validation_loss_std,
    ) = model.cv.results.sweep_by_filter(
        param_name=sweep_settings.param_name,
    )
    training_loss_mean = np.array(training_loss_mean)
    training_loss_std = np.array(training_loss_std)
    validation_loss_mean = np.array(validation_loss_mean)
    validation_loss_std = np.array(validation_loss_std)

    # --- create figure -----------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1)  # type: plt.Figure, plt.Axes

    # --- uncertainty bands ---
    ax.fill_between(
        param_values,
        training_loss_mean - training_loss_std,
        training_loss_mean + training_loss_std,
        color="r",
        alpha=0.1,
    )
    ax.fill_between(
        param_values,
        validation_loss_mean - validation_loss_std,
        validation_loss_mean + validation_loss_std,
        color="g",
        alpha=0.1,
    )

    # --- actual lines ---
    h_train = ax.plot(param_values, training_loss_mean, "r-x")
    h_val = ax.plot(param_values, validation_loss_mean, "g-x")

    # --- decorate ---
    ax.grid(visible=True)

    ax.legend([h_train[0], h_val[0]], ["mean training loss", "mean validation loss"])

    ax.set_xlabel(sweep_settings.x_label)
    if sweep_settings.log_x_scale:
        ax.set_xscale("log")
    ax.set_xticks(param_values)
    ax.set_xticklabels([str(pv) for pv in param_values])

    ax.set_ylabel("MAE")

    y_max_ideal = 2 * np.median(
        validation_loss_mean + validation_loss_std
    )  # center median of mean+std validation losses
    y_max = min(10 * min(validation_loss_mean), y_max_ideal)  # make sure minimum validation loss is still discernible
    ax.set_ylim(bottom=0.0, top=y_max)

    fig.suptitle(
        "1D sweep cross-validation results" + (f"\n({sweep_settings.sub_title})" if sweep_settings.sub_title else "")
    )

    fig.set_size_inches(w=8, h=6)
    fig.tight_layout()

    # --- lines & best performance ---
    selection_criterion = validation_loss_mean
    i_best = list(selection_criterion).index(min(selection_criterion))

    best_loss_crit = selection_criterion[i_best]
    best_param = param_values[i_best]

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # line + text + dot - SELECTION_CRITERION
    ax.plot(best_param, best_loss_crit, "go")
    ax.plot([x_min, x_max], [best_loss_crit, best_loss_crit], "g--", alpha=0.5)
    ax.text(x_min, best_loss_crit - 0.01 * y_max, f" {best_loss_crit:.3f}", ha="left", va="top", color="g")

    # reset limits
    ax.set_xlim(x_min, x_max)

    # --- save fig ----------------------------------------
    fig_filename = get_filename_1d_sweep_fig(sweep, n)
    fig.savefig(fig_filename, dpi=600)
