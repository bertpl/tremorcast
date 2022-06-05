import os
import pickle

import pandas as pd

from src.base.forecasting.models import ScoreMetric, TimeSeriesModelRegressionMultiMLP
from src.projects.fagradalsfjall.evaluate_models import get_dataset_train
from src.tools.math import exp_spaced_indices_fixed_max
from src.tools.matplotlib import plot_style_matplotlib_default

from ._project_settings import FORECAST_SIGNAL_NAME
from .evaluate_models.evaluate_forecast_models import _get_output_path, evaluate_forecast_models


# =================================================================================================
#  Main functions
# =================================================================================================
def blog_6_2_mlp_multi_sub_model_cv(n: int, n_models: int):

    # --- load training data set --------------------------
    df_train = get_dataset_train().to_dataframe()  # type: pd.DataFrame

    # --- model with default hyper-params -----------------
    p = 2 * 4 * 24  # 2 days = 192 samples
    n_features = 16

    model = TimeSeriesModelRegressionMultiMLP(signal_name=FORECAST_SIGNAL_NAME, p=p, n=n)
    model.regressor.set_sub_params(
        input_selection_indices=exp_spaced_indices_fixed_max(n=n_features, max_index=p - 1),
        n_hidden_layers=3,
        layer_width=50,
    )

    # --- param_grid --------------------------------------
    param_grid = dict(n_epochs=[20, 50, 100], wd=[0.001, 0.01, 0.1, 1, 10], lr_max=["minimum", "aggressive"])

    # --- perform grid search -----------------------------
    model.sub_cv.grid_search(
        training_data=df_train,
        param_grid=param_grid,
        score_metric=ScoreMetric.MAE,
        n_splits=5,
        i_sub_models=exp_spaced_indices_fixed_max(n=n_models, max_index=n - 1),
    )

    # --- save_results ------------------------------------
    model_file_name = get_filename_multi_mlp_sub_cv_model(n, n_models)
    with open(model_file_name, "wb") as f:
        pickle.dump(model, f)


# =================================================================================================
#  Helpers - File Names
# =================================================================================================
def get_filename_multi_mlp_sub_cv_model(n: int, n_models: int) -> str:
    return os.path.join(_get_output_path("post_6_nn"), f"mlp_multi_sub_cv_{n}_step_{n_models}_sub_models.pkl")
