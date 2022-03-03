import os
import pickle
from typing import Dict, List

import numpy as np

from src.base.forecasting.models import (
    TimeSeriesModelDartsLinearRegression,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
)
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models

from ._project_settings import FORECAST_SIGNAL_NAME


def evaluate_high_order_ar_models():

    # --- models to evaluate ------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    ar_models = {
        "ar-12": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=12),
        "ar-24": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=24),
        "ar-48": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=48),
        "ar-96": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=96),
        "ar-192": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=192),
        "ar-384": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=384),
    }

    # --- evaluate ----------------------------------------
    # AR models with stride 48
    # evaluate_forecast_models(
    #     models=ar_models,
    #     retrain=True,
    #     stride=48,
    #     results_sub_folder="post_4_linear_models",
    #     simulate=True,
    #     evaluate=False
    # )

    # EVALUATE ALL
    evaluate_forecast_models(
        models=naive_models | ar_models,
        retrain=True,
        stride=0,
        results_sub_folder="post_4_linear_models",
        simulate=False,
        evaluate=True,
        set_name="high_order_ar",
    )
