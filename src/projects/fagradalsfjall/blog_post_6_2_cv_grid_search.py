import datetime
import os
import pickle
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from src.base.forecasting.models import (
    TimeSeriesModelAutoRegressiveNeural,
    TimeSeriesModelMultiStepNeuralBottleneck,
    TimeSeriesModelMultiStepOLS,
    TimeSeriesModelMultiStepPLS,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
    loss_mae,
    loss_rmse,
)
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models, get_dataset_train
from src.tools.matplotlib import plot_style_matplotlib_default

from ._project_settings import FORECAST_SIGNAL_NAME
from .evaluate_models.evaluate_forecast_models import _get_output_path


# =================================================================================================
#  Epochs & Learning Rates
# =================================================================================================
def n_step_ahead_neural_cv_sweep():

    # -------------------------------------------------------------------------
    #  Bottleneck
    # -------------------------------------------------------------------------
    cv_settings_bottleneck_full = dict(
        n_splits=10,
        randomize=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [2],
            "n_latent_dims": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50],
            "wd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "n_epochs": [1000],
            "lr_max_method": ["minimum"],
        },
    )

    cv_settings_bottleneck_mini = dict(
        n_splits=5,
        randomize=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [2],
            "n_latent_dims": [1, 2, 4, 6, 8],
            "wd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "n_epochs": [500],
            "lr_max_method": ["minimum"],
        },
    )

    cv_settings_bottleneck_micro = dict(
        n_splits=4,
        randomize=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [2],
            "n_latent_dims": [1, 4, 16],
            "wd": [1e-3, 1e-2, 1e-1],
            "n_epochs": [1000],
            "lr_max_method": ["minimum"],
        },
    )

    model_params_bottleneck = dict(signal_name=FORECAST_SIGNAL_NAME, p=288, n=288)

    # -------------------------------------------------------------------------
    #  Auto-regressive
    # -------------------------------------------------------------------------
    cv_settings_ar_full = dict(
        n_splits=10,
        randomize=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [1, 2, 3],
            "p": [12, 24, 48, 96, 144, 192, 288, 384],
            "wd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "n_epochs": [1000],
            "lr_max_method": ["minimum"],
        },
    )

    cv_settings_ar_mini = dict(
        n_splits=5,
        randomize=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [2],
            "p": [48, 96, 192, 288],
            "wd": [1e-3, 1e-2, 1e-1],
            "n_epochs": [500],
            "lr_max_method": ["minimum"],
        },
    )

    cv_settings_ar_micro = dict(
        n_splits=4,
        randomize=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [2],
            "p": [48, 96, 192],
            "wd": [1e-3, 1e-2, 1e-1],
            "n_epochs": [500],
            "lr_max_method": ["minimum"],
        },
    )

    model_params_ar = dict(signal_name=FORECAST_SIGNAL_NAME, p=288)

    # -------------------------------------------------------------------------
    #  Perform sweep
    # -------------------------------------------------------------------------
    cv_models = {
        # --- BOTTLENECK ----------------------------------
        # "n-step-nn-288-288-cv-micro": TimeSeriesModelMultiStepNeuralBottleneck(
        #     **model_params_bottleneck, cv=cv_settings_bottleneck_micro
        # ),
        # "n-step-nn-288-288-cv-mini": TimeSeriesModelMultiStepNeuralBottleneck(
        #     **model_params_bottleneck, cv=cv_settings_bottleneck_mini
        # ),
        "n-step-nn-288-288-cv-full": TimeSeriesModelMultiStepNeuralBottleneck(
            **model_params_bottleneck, cv=cv_settings_bottleneck_full
        ),
        # --- AUTO-REGRESSIVE -----------------------------
        # "ar-nn-cv-micro": TimeSeriesModelAutoRegressiveNeural(
        #     **model_params_ar, cv=cv_settings_ar_micro
        # ),
        # "ar-nn-cv-mini": TimeSeriesModelAutoRegressiveNeural(
        #     **model_params_ar, cv=cv_settings_ar_mini
        # ),
        # "ar-nn-cv-full": TimeSeriesModelAutoRegressiveNeural(
        #     **model_params_ar, cv=cv_settings_ar_full
        # ),
    }

    evaluate_forecast_models(
        models=cv_models,
        retrain=False,
        stride=1,
        results_sub_folder="post_6_nn",
        simulate=True,
        evaluate=True,
        set_name="n_step_ahead_neural_cv_results",
    )


def n_step_ahead_neural_cv_figure():

    # --- load CV model -----------------------------------
    base_path = _get_output_path("post_6_nn")
    pkl_filename = os.path.join(base_path, "n-step-nn-288-288-cv-micro_retraining_off_simdata.pkl")

    with open(pkl_filename, "rb") as f:
        model, *_ = pickle.load(f)

    # --- extract results ---------------------------------
    all_cv_results = model.cv_results["all"]

    print()

    # rank_values = [result["params"]["rank"] for result in all_cv_results]
    #
    # val_mean = np.array([result["validation_losses"]["mean"] for result in all_cv_results])
    # val_std = np.array([result["validation_losses"]["std"] for result in all_cv_results])
    #
    # train_mean = np.array([result["training_losses"]["mean"] for result in all_cv_results])
    # train_std = np.array([result["training_losses"]["std"] for result in all_cv_results])
