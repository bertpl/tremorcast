import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.base.forecasting.models import TimeSeriesForecastModel, TimeSeriesModelNaiveConstant, TimeSeriesModelNaiveMean
from src.base.forecasting.simulation import simulate_time_series_model

from ._project_settings import (
    FILE_DATASET_TEST,
    FILE_DATASET_TRAIN,
    FORECAST_MAD_THRESHOLD,
    FORECAST_SIGNAL_NAME,
    PATH_RESULTS,
)
from .plot_forecasts import plot_forecasts
from .plot_mad_curves import plot_mad_curves


def evaluate_forecast_models():

    # --- load test & training data sets ------------------
    with open(FILE_DATASET_TRAIN + ".pkl", "rb") as f:
        data_train = pickle.load(f)  # type: VedurHarmonicMagnitudes
        df_train = data_train.to_dataframe()

    with open(FILE_DATASET_TEST + ".pkl", "rb") as f:
        data_test = pickle.load(f)  # type: VedurHarmonicMagnitudes
        df_test = data_test.to_dataframe()

    # --- construct untrained forecast models -------------
    models = [
        TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    ]  # type: List[TimeSeriesForecastModel]

    models_dict = {model.model_type: model for model in models}

    # --- evaluate ----------------------------------------
    model_types_to_evaluate = list(models_dict.keys())  # override with list of strings literal to evaluate a subset

    for retrain_model in [False, True]:

        mad_curves = dict()  # type: Dict[str, np.ndarray]

        for model_type in model_types_to_evaluate:

            # --- select model --------
            model = models_dict[model_type]

            # --- simulate ------------
            forecasts, mad_curve, horizon = simulate_time_series_model(
                model=model,
                training_data=df_train,
                test_data=df_test,
                accuracy_threshold=FORECAST_MAD_THRESHOLD,
                horizon=10 * 96,  # 10 days
                retrain_model=retrain_model,
            )
            mad_curves[model_type] = mad_curve

            # --- plot forecasts ------
            indices = [96 + i * 48 for i in range(20)]
            title = f"Sample forecasts (model: {model_type}, retraining {'ON' if retrain_model else 'OFF'})"
            fig, ax = plot_forecasts(
                data_test=data_test, forecasts=forecasts, horizon=3 * 96, indices=indices, title=title
            )

            base_fig_filename = os.path.join(
                PATH_RESULTS, model_type + f"_retraining_{'on' if retrain_model else 'off'}"
            )
            fig.savefig(base_fig_filename + "_forecasts.png", dpi=600)

            # --- plot MAD ------------
            title = f"MAD curve (model: {model_type}, retraining {'ON' if retrain_model else 'OFF'})"
            fig, ax = plot_mad_curves({model_type: mad_curve}, FORECAST_MAD_THRESHOLD, title)

            fig.savefig(base_fig_filename + "_mad_curve.png", dpi=600)

        # --- compare MAD curves ----------------
        title = f"MAD curves (retraining {'ON' if retrain_model else 'OFF'})"
        fig, ax = plot_mad_curves(mad_curves, FORECAST_MAD_THRESHOLD, title)

        filename = os.path.join(PATH_RESULTS, f"mad_curves_retraining_{'on' if retrain_model else 'off'}.png")
        fig.savefig(filename, dpi=600)
