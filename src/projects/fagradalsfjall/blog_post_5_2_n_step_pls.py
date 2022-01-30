from src.base.forecasting.models import (
    TimeSeriesModelMultiStepOLS,
    TimeSeriesModelMultiStepPLS,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
    loss_mae,
    loss_rmse,
)
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models

from ._project_settings import FORECAST_SIGNAL_NAME


def evaluate_n_step_ahead_pls():

    # --- models to evaluate ------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    earlier_models = {
        "ar-192": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=192, n=1),
        "n-step-ols-288-288": TimeSeriesModelMultiStepOLS(FORECAST_SIGNAL_NAME, p=288, n=288),
    }

    cv_settings = dict(
        n_splits=10,
        randomize=True,
        param_grid=dict(rank=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 50]),
        loss=loss_mae,
    )

    pls_models_initial_sweep = {
        "n-step-pls-288-288-1": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=1),
        "n-step-pls-288-288-2": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=2),
        "n-step-pls-288-288-3": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=3),
        "n-step-pls-288-288-5": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=5),
        "n-step-pls-288-288-10": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=10),
        "n-step-pls-288-288-20": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=20),
        "n-step-pls-288-288-40": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=50),
        "n-step-pls-288-288-80": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=100),
    }

    pls_model_cv = {
        "n-step-pls-288-288-cv": TimeSeriesModelMultiStepPLS(
            FORECAST_SIGNAL_NAME, p=288, n=288, rank=1, cv=cv_settings
        ),
    }

    pls_models_cv_results = {
        "n-step-pls-288-288-7": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=7),
        # "n-step-pls-288-288-12": TimeSeriesModelMultiStepPLS(FORECAST_SIGNAL_NAME, p=288, n=288, rank=12),
    }

    # --- evaluate ----------------------------------------
    # simulate & evaluate all
    # evaluate_forecast_models(
    #     models=naive_models | earlier_models | pls_models_initial_sweep,
    #     retrain=False,
    #     stride=1,
    #     results_sub_folder="post_5_n_step_ahead",
    #     simulate=True,
    #     evaluate=True,
    #     set_name="n_step_ahead_pls_sweep",
    # )

    # evaluate_forecast_models(
    #     models=pls_model_cv,
    #     retrain=False,
    #     stride=1,
    #     results_sub_folder="post_5_n_step_ahead",
    #     simulate=True,
    #     evaluate=True,
    #     set_name="n_step_ahead_pls_cv",
    # )

    evaluate_forecast_models(
        models=naive_models | earlier_models | pls_models_cv_results,
        retrain=False,
        stride=1,
        results_sub_folder="post_5_n_step_ahead",
        simulate=True,
        evaluate=True,
        set_name="n_step_ahead_pls_cv_results",
    )
