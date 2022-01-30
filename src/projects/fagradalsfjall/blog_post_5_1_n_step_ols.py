from src.base.forecasting.models import (
    TimeSeriesModelDartsLinearRegression,
    TimeSeriesModelMultiStepOLS,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
)
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models

from ._project_settings import FORECAST_SIGNAL_NAME


def evaluate_n_step_ahead_ols():

    # --- models to evaluate ------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    ar_models = {
        "ar-192": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, p=192),
        "n-step-ols-192-1": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=1),
        "n-step-ols-192-24": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=24),
        "n-step-ols-192-96": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=96),
        "n-step-ols-192-192": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=192),
        "n-step-ols-288-288": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=288, n=288),
    }

    # --- evaluate ----------------------------------------
    # AR models with stride 48
    evaluate_forecast_models(
        models=naive_models | ar_models,
        retrain=False,
        stride=1,
        results_sub_folder="post_5_n_step_ahead",
        simulate=True,
        evaluate=True,
        set_name="n_step_ahead_ols",
    )

    # # EVALUATE ALL
    # evaluate_forecast_models(
    #     models=ar_models,
    #     retrain=False,
    #     stride=0,
    #     results_sub_folder="post_5_n_step_ahead",
    #     simulate=False,
    #     evaluate=True,
    #     set_name="n_step_ahead_ols",
    # )
