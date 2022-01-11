from src.base.forecasting.models import (
    TimeSeriesModelDartsArima,
    TimeSeriesModelDartsLinearRegression,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
)
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models

from ._project_settings import FORECAST_SIGNAL_NAME


def evaluate_arima_models():

    # --- models to evaluate ------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    arima_models = {
        # AR models
        "ar-12": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, n=12),
        "ar-18": TimeSeriesModelDartsLinearRegression(FORECAST_SIGNAL_NAME, n=18),
        # ARMA models
        "arma-1-1": TimeSeriesModelDartsArima(FORECAST_SIGNAL_NAME, p=1, q=1, d=0),
        "arma-6-6": TimeSeriesModelDartsArima(FORECAST_SIGNAL_NAME, p=6, q=6, d=0),
        "arma-12-12": TimeSeriesModelDartsArima(FORECAST_SIGNAL_NAME, p=12, q=12, d=0),
        "arma-24-24": TimeSeriesModelDartsArima(FORECAST_SIGNAL_NAME, p=24, q=24, d=0),
    }

    # --- evaluate ----------------------------------------

    # NAIVE models with stride 1
    # evaluate_forecast_models(
    #     models=naive_models,
    #     retrain=True,
    #     stride=1,
    #     results_sub_folder="post_4_linear_models",
    #     simulate=True,
    #     evaluate=False
    # )

    # # ARIMA models with stride 48
    # evaluate_forecast_models(
    #     models=arima_models,
    #     retrain=True,
    #     stride=48,
    #     results_sub_folder="post_4_linear_models",
    #     simulate=True,
    #     evaluate=False
    # )

    # EVALUATE ALL
    evaluate_forecast_models(
        models=naive_models | arima_models,
        retrain=True,
        stride=0,
        results_sub_folder="post_4_linear_models",
        simulate=False,
        evaluate=True,
        set_name="arima",
    )
