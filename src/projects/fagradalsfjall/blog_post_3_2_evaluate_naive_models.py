from src.base.forecasting.models import TimeSeriesModelNaiveConstant, TimeSeriesModelNaiveMean
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models

from ._project_settings import FORECAST_SIGNAL_NAME


def evaluate_naive_models():

    # --- models to evaluate ------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    # --- evaluate ----------------------------------------
    evaluate_forecast_models(
        models=naive_models,
        retrain=[False, True],
        results_sub_folder="post_3_naive_models",
        simulate=False,
        evaluate=True,
    )
