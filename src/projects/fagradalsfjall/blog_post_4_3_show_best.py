import os
import pickle
from typing import Dict, List

import numpy as np

from src.base.forecasting.models import (
    TimeSeriesModelDartsArima,
    TimeSeriesModelDartsLinearRegression,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
)
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models

from ._project_settings import FORECAST_SIGNAL_NAME


def show_best_linear_models():

    # --- models to evaluate ------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    best_lin_models = {
        "arma-12-12": TimeSeriesModelDartsArima(FORECAST_SIGNAL_NAME, p=12, q=12, d=0),
        "ar-192": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=192),
    }

    # --- plot results ------------------------------------
    # EVALUATE ALL
    evaluate_forecast_models(
        models=naive_models | best_lin_models,
        retrain=True,
        stride=0,
        results_sub_folder="post_4_linear_models",
        simulate=False,
        evaluate=True,
        set_name="best_linear",
    )
