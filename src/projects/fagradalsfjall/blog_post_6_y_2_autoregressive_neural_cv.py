import datetime
import os
import pickle
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from src.base.forecasting.models import (
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


# =================================================================================================
#  Full Cross-Validation Grid Search
# =================================================================================================
def autoregressive_neural_cv_sweep():

    # -------------------------------------------------------------------------
    #  Auto-regressive
    # -------------------------------------------------------------------------
    cv_settings_full = dict(
        n_splits=10,
        randomize=True,
        randomize_runs=True,
        loss=loss_mae,
        param_grid={
            "n_hidden_layers": [3],
            "p": [4, 12, 48, 96, 192, 384],
            "wd": [0.05, 0.1, 0.2, 0.5, 1, 2],
            "n_epochs": [50, 100, 200],
            "lr_max_method": ["minimum"],
        },
    )

    model_params = dict(signal_name=FORECAST_SIGNAL_NAME, p=288)

    # -------------------------------------------------------------------------
    #  Perform sweep
    # -------------------------------------------------------------------------
    cv_models = {
        "ar-nn-cv-full": TimeSeriesModelAutoRegressiveNeural(**model_params, cv=cv_settings_full),
    }

    evaluate_forecast_models(
        models=cv_models,
        retrain=False,
        stride=1,
        results_sub_folder="post_6_nn",
        simulate=True,
        evaluate=True,
        set_name="autoregressive_cv",
    )
