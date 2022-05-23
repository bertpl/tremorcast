from typing import Tuple

import matplotlib.pyplot as plt

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.base.forecasting.models import (
    LrMaxCriterion,
    TimeSeriesModelAutoRegressiveMLP,
    TimeSeriesModelAutoRegressiveOLS,
    TimeSeriesModelAutoRegressivePLS,
    TimeSeriesModelNaiveConstant,
    TimeSeriesModelNaiveMean,
)
from src.projects.fagradalsfjall.common.project_settings import FORECAST_SIGNAL_NAME
from src.projects.fagradalsfjall.evaluate_models import evaluate_forecast_models
from src.tools.matplotlib import plot_style_matplotlib_default

from .create_dataset import create_datasets


# =================================================================================================
#  Main function
# =================================================================================================
def test_mlp_models():

    # --- get datasets ------------------------------------
    data_train, data_test = get_data()

    # --- plot --------------------------------------------
    plot_style_matplotlib_default()
    data_train.create_plot()
    plt.show()

    # --- simulate ----------------------------------------
    evaluate_models(data_train, data_test)  # test on training set


def test_mlp_models_grid_search_cv(n: int):

    # --- get datasets ------------------------------------
    data_train, data_test = get_data()

    # --- define model & grid -----------------------------
    model = TimeSeriesModelAutoRegressiveMLP(FORECAST_SIGNAL_NAME, p=288, n=n, n_seeds=2)

    param_grid = dict(
        n_epochs=[1, 5, 10, 25, 50, 100],
        wd=[0.0, 0.001, 0.01, 0.1, 1.0],
        n_hidden_layers=[1, 3, 5],
        lr_max=[LrMaxCriterion.VALLEY, LrMaxCriterion.MINIMUM],
    )

    model.cv.grid_search(
        training_data=data_train.to_dataframe(), param_grid=param_grid, score_metric=ScoreMetric.MAE, n_splits=10
    )

    model.cv.results.show_optimal_results()


# =================================================================================================
#  Helpers
# =================================================================================================
def get_data() -> Tuple[VedurHarmonicMagnitudes, VedurHarmonicMagnitudes]:

    t_sine_var = 50  # default = 50
    non_linearity = 0.75  # default = 0.05
    random_vertical_drift = 500  # default = 500

    return create_datasets(
        t_sine_var=t_sine_var, non_linearity=non_linearity, random_vertical_drift=random_vertical_drift
    )  # type: VedurHarmonicMagnitudes, VedurHarmonicMagnitudes


def evaluate_models(data_train: VedurHarmonicMagnitudes, data_test: VedurHarmonicMagnitudes):

    # --- define models -----------------------------------
    naive_models = {
        "naive-constant": TimeSeriesModelNaiveConstant(FORECAST_SIGNAL_NAME),
        "naive-mean": TimeSeriesModelNaiveMean(FORECAST_SIGNAL_NAME),
    }

    earlier_models = {
        "ar-192": TimeSeriesModelAutoRegressiveOLS(FORECAST_SIGNAL_NAME, p=192, n=1),
        "n-step-ols-288-288": TimeSeriesModelAutoRegressiveOLS(FORECAST_SIGNAL_NAME, p=288, n=288),
        "n-step-pls-288-288-7": TimeSeriesModelAutoRegressivePLS(FORECAST_SIGNAL_NAME, p=288, n=288, n_components=7),
    }

    # --- cross-validation ------------------------------------
    cv_models = dict()
    # for n in [16, 32, 288]:
    for n in [288]:

        cv_model = TimeSeriesModelAutoRegressiveMLP(FORECAST_SIGNAL_NAME, p=288, n=288, n_seeds=2)

        param_grid = dict(
            n_epochs=[5, 10, 25, 50, 100],
            wd=[0.01, 0.1, 1.0, 10.0],
            n_hidden_layers=[1],
            lr_max=[LrMaxCriterion.AGGRESSIVE],
        )

        cv_model.cv.grid_search(
            training_data=data_train.to_dataframe(),
            param_grid=param_grid,
            score_metric=ScoreMetric.MAE,
            n_splits=10,
        )

        cv_models[n] = cv_model

    # --- load models -----------------------------------------
    nn_models = {
        # "16-step-mlp-single-manual": TimeSeriesModelRegressionMLP(FORECAST_SIGNAL_NAME, p=288, n=16, n_epochs=100),
        # "16-step-mlp-single-cv": cv_models[16],
        # "32-step-mlp-single-manual": TimeSeriesModelRegressionMLP(FORECAST_SIGNAL_NAME, p=288, n=32, n_epochs=100),
        # "32-step-mlp-single-cv": cv_models[32],
        "288-step-mlp-single-manual": TimeSeriesModelAutoRegressiveMLP(
            FORECAST_SIGNAL_NAME, p=288, n=288, n_epochs=100, n_seeds=10
        ),
        "288-step-mlp-single-cv": cv_models[288],
    }

    # --- simulate ----------------------------------------
    evaluate_forecast_models(
        models=naive_models | earlier_models | nn_models,
        retrain=False,
        stride=1,
        results_sub_folder="post_6_nn",
        simulate=True,
        evaluate=True,
        set_name=f"debug_mlp",
        data_train=data_train,
        data_test=data_test,
    )
