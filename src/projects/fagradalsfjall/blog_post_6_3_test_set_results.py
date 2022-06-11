import itertools
import os
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base.forecasting.evaluation.cross_validation import ParamSelectionMethod
from src.base.forecasting.models import (
    ScoreMetric,
    TimeSeriesModelMultiStepOLS,
    TimeSeriesModelMultiStepPLS,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
    TimeSeriesModelRegressionMLP,
)
from src.projects.fagradalsfjall.evaluate_models import get_dataset_train
from src.tools.matplotlib import plot_style_matplotlib_default

from ._project_settings import FORECAST_SIGNAL_NAME
from .evaluate_models.evaluate_forecast_models import _get_output_path, evaluate_forecast_models

from .blog_post_6_1_mlp_single import get_filename_full_cv_model
from .blog_post_6_2_mlp_multi import _get_optimal_model


# =================================================================================================
#  Part I - MLP-Single
# =================================================================================================
def blog_6_3_mlp_single_final_comparisons():

    # --- simulate ----------------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    earlier_models = {
        "ar-192": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=1),
        "n-step-ols-288-288": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=288, n=288),
        "n-step-pls-288-288-7": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=7),
    }

    # --- load models -----------------------------------------
    models = dict()
    for n in [1, 16]:

        print(f"--- N={n} -------------")

        # load
        with open(get_filename_full_cv_model(n=n), "rb") as f:
            model = pickle.load(f)  # type: TimeSeriesModelRegressionMLP

        # select params & set
        best_params = model.cv.results.best_result.params  # type: dict
        for param_name, value in best_params.items():
            setattr(model, param_name, value)
            print(f"{param_name}: {value} ")

        # keep track in dict
        models[n] = model

    nn_models = {
        "1-step-mlp": models[1],
        "16-step-mlp": models[16],
    }

    # --- simulate ----------------------------------------
    evaluate_forecast_models(
        models=naive_models | earlier_models | nn_models,
        retrain=False,
        stride=1,
        results_sub_folder="post_6_nn",
        simulate=True,
        evaluate=True,
        set_name=f"final_result_single",
    )


# =================================================================================================
#  Part II - MLP-Multi
# =================================================================================================
def blog_6_3_mlp_multi_final_comparisons():

    # --- simulate ----------------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    earlier_models = {
        "ar-192": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=1),
        "n-step-ols-288-288": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=288, n=288),
        "n-step-pls-288-288-7": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=7),
    }

    # --- load models -----------------------------------------
    nn_models = {
        "288-step-mlp-multi": _get_optimal_model(n=288)
    }

    # --- simulate ----------------------------------------
    evaluate_forecast_models(
        models=naive_models | earlier_models | nn_models,
        retrain=False,
        stride=1,
        results_sub_folder="post_6_nn",
        simulate=True,
        evaluate=True,
        set_name=f"final_result_multi",
    )
