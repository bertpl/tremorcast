import os
import pickle
from enum import Enum, auto
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base.forecasting.models import ScoreMetric, TimeSeriesModelRegressionMultiMLP
from src.projects.fagradalsfjall.evaluate_models import get_dataset_train
from src.tools.math import exp_spaced_indices_fixed_max
from src.tools.matplotlib import plot_style_matplotlib_default

from ._project_settings import FORECAST_SIGNAL_NAME
from .evaluate_models.evaluate_forecast_models import _get_output_path, evaluate_forecast_models


# =================================================================================================
#  Enums
# =================================================================================================
class MultiModelSweep(Enum):
    FULL_GRID = auto()
    REFERENCE = auto()


# =================================================================================================
#  Main functions - Simulate
# =================================================================================================
def blog_6_2_mlp_multi_sub_model_cv_simulate(n: int, n_models: int, sweep: MultiModelSweep):

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- model with default hyper-params -----------------
    model = _get_nominal_model(n, p=192)

    # --- param_grid --------------------------------------
    if sweep == MultiModelSweep.REFERENCE:

        param_grid = dict(
            n_hidden_layers=[1],
            layer_width=[100],
            n_epochs=[50],
            wd=[0.1],
            lr_max=["minimum"],
            input_selection_indices=exp_spaced_indices_fixed_max(n=16, max_index=15)
        )

    elif sweep == MultiModelSweep.FULL_GRID:

        param_grid = dict(
            n_hidden_layers=[1],
            layer_width=[100],
            n_epochs=[1, 10, 100],
            wd=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            lr_max=['valley', 'minimum', 'aggressive'],
            input_selection_indices=[
                tuple(exp_spaced_indices_fixed_max(n=16, max_index=15)),    # 16 = 4h
                tuple(exp_spaced_indices_fixed_max(n=16, max_index=31)),    # 32 = 8h
                tuple(exp_spaced_indices_fixed_max(n=16, max_index=63)),    # 64 = 16h
                tuple(exp_spaced_indices_fixed_max(n=16, max_index=127)),   # 128 = 32h
                tuple(exp_spaced_indices_fixed_max(n=16, max_index=191)),   # 192 = 48h
            ]
        )

    else:

        raise NotImplementedError(f"sweep not implemented: {sweep}.")

    # --- perform grid search -----------------------------
    model.sub_cv.grid_search(
        training_data=df_train,
        param_grid=param_grid,
        score_metric=ScoreMetric.MAE,
        n_splits=5,
        i_sub_models=exp_spaced_indices_fixed_max(n=n_models, max_index=n - 1),
    )

    # --- save results ------------------------------------
    model_file_name = get_filename_multi_mlp_sub_cv_model(n, n_models, sweep)
    with open(model_file_name, "wb") as f:
        pickle.dump(model, f)


def blog_6_2_mlp_multi_sub_model_optimal_cv_simulate(n: int, n_models: int):

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- model with default hyper-params -----------------
    model = _get_optimal_model(n)

    # --- param_grid --------------------------------------
    param_grid = dict(layer_width=[50])

    # --- perform grid search -----------------------------
    model.sub_cv.grid_search(
        training_data=df_train,
        param_grid=param_grid,
        score_metric=ScoreMetric.MAE,
        n_splits=5,
        i_sub_models=exp_spaced_indices_fixed_max(n=n_models, max_index=n - 1),
    )

    model.regressor.sub_cv.show_results()

    # --- save results ------------------------------------
    model_file_name = get_filename_multi_mlp_sub_optimal_model(n, n_models)
    with open(model_file_name, "wb") as f:
        pickle.dump(model, f)


# =================================================================================================
#  Plotting
# =================================================================================================
def blog_6_2_mlp_multi_sub_model_cv_plots(n: int, n_models: int, sweep: MultiModelSweep):

    # --- load results ------------------------------------
    model_file_name = get_filename_multi_mlp_sub_cv_model(n, n_models, sweep)
    with open(model_file_name, "rb") as f:
        model = pickle.load(f)  # type: TimeSeriesModelRegressionMultiMLP

    # --- show results in console -------------------------
    model.regressor.sub_cv.show_results()

    # --- create plots ------------------------------------
    plot_style_matplotlib_default()

    results = model.regressor.sub_cv.get_best_hyper_params()
    param_names = ['loss', 'lr_max_value', 'n_epochs', 'wd']

    for param_name in param_names:
        x_values = np.array(list(results.keys()))
        y_values = [result[param_name] for result in results.values()]

        fig, ax = plt.subplots()        # type: plt.Figure, plt.Axes

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel("Lead time (samples)")
        ax.set_ylabel(param_name)

        # cv values
        ax.plot(x_values+1, y_values)

        # straight line
        # try:
        if param_name != "loss":
            x_line = list(range(1, 289))
            y_line = [_optimal_param_value(param_name, x) for x in x_line]
            ax.plot(x_line, y_line, 'r')
        # except:
        #     pass

        fig.suptitle("Optimal parameter values - " + param_name)
        fig.set_size_inches(w=10, h=6)
        fig.tight_layout()


# =================================================================================================
#  Helpers - File Names
# =================================================================================================
def get_filename_multi_mlp_sub_cv_model(n: int, n_models: int, sweep: MultiModelSweep) -> str:
    return os.path.join(
        _get_output_path("post_6_nn"), f"mlp_multi_sub_cv_{n}_step_{sweep.name.lower()}_sweep_{n_models}_sub_models.pkl"
    )


def get_filename_multi_mlp_sub_optimal_model(n: int, n_models: int) -> str:
    return os.path.join(
        _get_output_path("post_6_nn"), f"mlp_multi_sub_optimal_{n}_step_{n_models}_sub_models.pkl"
    )


def _optimal_param_value(param_name: str, lead_time: int) -> float:

    # --- settings ----------------------------------------
    params = {
        "wd": [(1, 1e-3), (3, 1), (288, 1000)],
        "lr_max_value": [(1, 0.1), (10, 0.02), (288, 3e-4)],
        "n_epochs": [(1, 30), (288, 3)]
    }
    xfyf = params[param_name]
    xf = [a[0] for a in xfyf]
    yf = [a[1] for a in xfyf]
    # xf, yf = zip(params[param_name])  # type: List[float], List[float]
    log_xf = np.log(xf)
    log_yf = np.log(yf)

    # --- log-log interpolation ---------------------------
    log_x = np.log(lead_time)
    log_y = np.interp(log_x, log_xf, log_yf)
    y = np.exp(log_y)

    return float(y)


def _get_nominal_model(n: int, p: int = 192) -> TimeSeriesModelRegressionMultiMLP:

    model = TimeSeriesModelRegressionMultiMLP(signal_name=FORECAST_SIGNAL_NAME, p=p, n=n)
    model.regressor.set_sub_params(
        input_selection_indices=exp_spaced_indices_fixed_max(n=16, max_index=p - 1),
        n_hidden_layers=1,
        layer_width=100,
    )

    return model


def _get_optimal_model(n: int, p: int = 192) -> TimeSeriesModelRegressionMultiMLP:

    model = _get_nominal_model(n, p)

    for i in range(model.n):
        lead_time = i+1
        model.regressor.set_sub_params(
            i_sub_models=[i],
            wd=_optimal_param_value("wd", lead_time),
            n_epochs=round(_optimal_param_value("n_epochs", lead_time)),
            lr_max=_optimal_param_value("lr_max_value", lead_time)
        )

    return model
