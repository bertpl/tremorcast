import sys
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from src.base.forecasting.evaluation.cross_validation import TimeSeriesCVSplitter
from src.base.forecasting.evaluation.metrics import TabularMetric, TimeSeriesMetric
from src.base.forecasting.evaluation.metrics.timeseries_metrics import MaxAccurateLeadTime
from src.base.forecasting.models import TimeSeriesModel
from src.base.forecasting.models.time_series.helpers.timeseries_cv import TimeSeriesCrossValidation, evaluate_ts_model
from src.projects.fagradalsfjall.common.project_settings import (
    CV_HORIZON_SAMPLES,
    CV_METRIC_RMSE_THRESHOLD,
    CV_MIN_SAMPLES_TRAIN,
    CV_MIN_SAMPLES_VALIDATE,
    CV_N_SPLITS,
    SIMULATION_STRIDE,
)

from .dataset import load_test_data_numpy, load_train_data_numpy


# =================================================================================================
#  Enums
# =================================================================================================
class ValidationType(Enum):
    CV_TRAIN = 0  # results on training sets of cross-validation
    CV_VAL = 1  # results on validation sets of cross-validation
    TRAIN = 10  # results on training set
    TEST = 20  # results on test set

    def get_display_name(self) -> str:
        return {
            ValidationType.CV_TRAIN: "Cross-Validation - Training Performance",
            ValidationType.CV_VAL: "Cross-Validation - Validation Performance",
            ValidationType.TRAIN: "Training Set Performance",
            ValidationType.TEST: "Test Set Performance",
        }[self]


# =================================================================================================
#  Validation functions
# =================================================================================================
def validate_models(
    models: Dict[str, TimeSeriesModel], metric: TimeSeriesMetric = None, hor: int = None
) -> Dict[str, Dict[ValidationType, np.ndarray]]:

    return {
        model_name: validate_model(model, metric, hor)
        for model_name, model in tqdm(models.items(), desc="Validating models ", file=sys.stdout)
    }


def validate_model(
    ts_model: TimeSeriesModel,
    metric: TimeSeriesMetric = None,
    hor: int = None,
    validation_types: List[ValidationType] = None,
) -> Dict[ValidationType, np.ndarray]:

    # --- argument handling -------------------------------
    metric = metric or MaxAccurateLeadTime(TabularMetric.rmse(), metric_threshold=CV_METRIC_RMSE_THRESHOLD)
    hor = hor or CV_HORIZON_SAMPLES
    validation_types = validation_types or list(ValidationType)

    # --- init --------------------------------------------
    x_train = load_train_data_numpy()  # type: np.ndarray
    x_test = load_test_data_numpy()  # type: np.ndarray
    results = dict()  # type: Dict[ValidationType, np.ndarray]

    # --- cross-validation --------------------------------
    if (ValidationType.CV_TRAIN in validation_types) or (ValidationType.CV_VAL in validation_types):
        cv_result = TimeSeriesCrossValidation.cross_validate_ts_model(
            model=ts_model,
            x=x_train,
            metric=metric,
            ts_cv_splitter=TimeSeriesCVSplitter(CV_MIN_SAMPLES_TRAIN, CV_MIN_SAMPLES_VALIDATE, CV_N_SPLITS),
            hor=hor,
            retrain=False,
        )

        if ValidationType.CV_TRAIN in validation_types:
            results[ValidationType.CV_TRAIN] = cv_result.train_metrics.overall_metric_curve()

        if ValidationType.CV_VAL in validation_types:
            results[ValidationType.CV_VAL] = cv_result.val_metrics.overall_metric_curve()

    # --- training set ------------------------------------
    if ValidationType.TRAIN in validation_types:
        results[ValidationType.TRAIN] = evaluate_ts_model(
            model=ts_model,
            x_hist=x_train[: ts_model.min_hist()],
            x_val=x_train[ts_model.min_hist() :],
            metric=metric,
            hor=hor,
            silent=False,
        )

    # --- test set ----------------------------------------
    if ValidationType.TEST in validation_types:
        results[ValidationType.TEST] = evaluate_ts_model(
            model=ts_model, x_hist=x_train, x_val=x_test, metric=metric, hor=hor, silent=False
        )

    # --- return ------------------------------------------
    return results


# =================================================================================================
#  Simulations
# =================================================================================================
def run_test_set_simulations(
    models: Dict[str, TimeSeriesModel],
) -> Dict[str, List[Tuple[int, np.ndarray]]]:

    # --- load data ---------------------------------------
    x_train = load_train_data_numpy()  # type: np.ndarray
    x_test = load_test_data_numpy()  # type: np.ndarray

    # --- simulate ----------------------------------------
    return run_simulations(models, x_train, x_test)


def run_simulations(
    models: Dict[str, TimeSeriesModel],
    x_hist: np.ndarray,
    x_val: np.ndarray,
    hor: int = CV_HORIZON_SAMPLES,
    stride: int = SIMULATION_STRIDE,
) -> Dict[str, List[Tuple[int, np.ndarray]]]:

    return {
        model_name: ts_model.batch_predict(
            x=np.concatenate([x_hist, x_val]),
            first_sample=len(x_hist),
            hor=hor,
            overlap_end=False,
            stride=stride,
            silent=False,
        )
        for model_name, ts_model in models.items()
    }
