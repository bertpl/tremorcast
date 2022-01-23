import os
import pickle
from typing import Dict, List, Union

import numpy as np

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.base.forecasting.evaluation import simulate_time_series_model
from src.base.forecasting.models import TimeSeriesForecastModel
from src.projects.fagradalsfjall.evaluate_models.plot_forecasts import plot_forecasts
from src.projects.fagradalsfjall.evaluate_models.plot_mae_curves import plot_mae_curves


# =================================================================================================
#  SIMULATE & EVALUATE
# =================================================================================================
def evaluate_forecast_models(
    models: Dict[str, TimeSeriesForecastModel],
    retrain: Union[bool, List[bool]] = False,
    data_train: VedurHarmonicMagnitudes = None,
    data_test: VedurHarmonicMagnitudes = None,
    results_sub_folder: str = None,
    stride: int = 1,
    simulate: bool = True,
    evaluate: bool = True,
    set_name: str = "",
):

    # --- argument handling -------------------------------
    if not isinstance(retrain, list):
        retrain = [retrain]

    # --- import general settings -------------------------
    from src.projects.fagradalsfjall._project_settings import (
        FILE_DATASET_TEST,
        FILE_DATASET_TRAIN,
        FORECAST_MAE_THRESHOLD,
    )

    # --- load test & training data sets ------------------
    if data_train is None:
        with open(FILE_DATASET_TRAIN + ".pkl", "rb") as f:
            data_train = pickle.load(f)  # type: VedurHarmonicMagnitudes
    df_train = data_train.to_dataframe()

    if data_test is None:
        with open(FILE_DATASET_TEST + ".pkl", "rb") as f:
            data_test = pickle.load(f)  # type: VedurHarmonicMagnitudes
            df_test = data_test.to_dataframe()

    # --- simulate ----------------------------------------
    output_path = _get_output_path(results_sub_folder)

    for retrain_model in retrain:

        mae_curves = dict()  # type: Dict[str, np.ndarray]

        for model_name, model in models.items():

            print(f"=== {model_name} === retrain_model: {retrain_model} ====================================")

            # --- filename ----------------------
            base_filename = _get_base_filename(output_path, model_name, retrain_model)

            # --- simulate & save OR load -------
            if simulate:

                # simulate
                forecasts, mae_curve, max_reliable_lead_time = simulate_time_series_model(
                    model=model,
                    training_data=df_train,
                    test_data=df_test,
                    accuracy_threshold=FORECAST_MAE_THRESHOLD,
                    horizon=10 * 96,  # 10 days
                    retrain_model=retrain_model,
                    stride=stride,
                )

                # save
                with open(base_filename + "_simdata.pkl", "wb") as f:
                    pickle.dump((model, forecasts, mae_curve, max_reliable_lead_time), f)

            else:

                # load
                with open(base_filename + "_simdata.pkl", "rb") as f:
                    _, forecasts, mae_curve, max_reliable_lead_time = pickle.load(f)

            # --- prep MAE curve plotting -----------------
            if evaluate:

                # --- MAE metrics -----
                print()
                print(f"min(MAE)               = {min(mae_curve):.1f}")
                print(f"max(MAE)               = {max(mae_curve):.1f}")
                print(f"max_reliable_lead_time = {max_reliable_lead_time:.3f} samples")
                print()

                # --- prep MAE curves plotting
                mae_curves[model_name] = mae_curve

                # --- plot forecasts --
                indices = [96 + i * 48 for i in range(20)]
                title = f"Sample forecasts (model: {model_name}, retraining {'ON' if retrain_model else 'OFF'})"
                fig, ax = plot_forecasts(
                    data_test=data_test, forecasts=forecasts, horizon=3 * 96, indices=indices, title=title
                )

                fig.savefig(base_filename + "_forecasts.png", dpi=600)

                # --- plot MAE --------
                title = f"MAE curve (model: {model_name}, retraining {'ON' if retrain_model else 'OFF'})"
                fig, ax = plot_mae_curves({model_name: mae_curve}, FORECAST_MAE_THRESHOLD, title)

                fig.savefig(base_filename + "_mae_curve.png", dpi=600)

        # --- compare MAE curves ----------------
        if evaluate:
            title = f"MAE curves (retraining {'ON' if retrain_model else 'OFF'})"
            fig, ax = plot_mae_curves(mae_curves, FORECAST_MAE_THRESHOLD, title)

            if set_name:
                filename = os.path.join(
                    output_path, f"mae_curves_{set_name}_retraining_{'on' if retrain_model else 'off'}.png"
                )
            else:
                filename = os.path.join(output_path, f"mae_curves_retraining_{'on' if retrain_model else 'off'}.png")

            fig.savefig(filename, dpi=600)


# =================================================================================================
#  HELPERS
# =================================================================================================
def _get_output_path(results_sub_folder: str) -> str:

    # --- import general settings -------------------------
    from src.projects.fagradalsfjall._project_settings import PATH_RESULTS

    # --- output folder -----------------------------------
    if results_sub_folder:
        output_path = os.path.join(PATH_RESULTS, results_sub_folder)
    else:
        output_path = PATH_RESULTS
    os.makedirs(output_path, exist_ok=True)

    # --- return ------------------------------------------
    return output_path


def _get_base_filename(output_path: str, model_name: str, retrain_model: bool) -> str:

    return os.path.join(output_path, model_name + f"_retraining_{'on' if retrain_model else 'off'}")
