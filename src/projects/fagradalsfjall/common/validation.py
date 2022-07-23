import sys
from enum import Enum
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm

from src.base.forecasting.evaluation.cross_validation import TimeSeriesCVSplitter
from src.base.forecasting.evaluation.metrics import TabularMetric, TimeSeriesMetric
from src.base.forecasting.evaluation.metrics.timeseries_metrics import MaxAccurateLeadTime
from src.base.forecasting.models import TimeSeriesModel
from src.base.forecasting.models.time_series.helpers.timeseries_cv import (
    TimeSeriesCrossValidation,
    ValidationPredictions,
    evaluate_ts_model,
)
from src.projects.fagradalsfjall._project_settings import (
    CV_HORIZON,
    CV_METRIC_RMSE_THRESHOLD,
    CV_MIN_SAMPLES_TRAIN,
    CV_MIN_SAMPLES_VALIDATE,
    CV_N_SPLITS,
)

from .dataset import load_test_data, load_train_data


# =================================================================================================
#  Enums
# =================================================================================================
class ValidationType(Enum):
    CV_TRAIN = 0  # results on training sets of cross-validation
    CV_VAL = 1  # results on validation sets of cross-validation
    TRAIN = 10  # results on training set
    TEST = 20  # results on test set


# =================================================================================================
#  Validation functions
# =================================================================================================
def validate_model(
    ts_model: TimeSeriesModel,
    metric: TimeSeriesMetric = None,
    hor: int = None,
    validation_types: List[ValidationType] = None,
) -> Dict[ValidationType, ValidationPredictions]:

    # --- argument handling -------------------------------
    metric = metric or MaxAccurateLeadTime(TabularMetric.rmse(), threshold=CV_METRIC_RMSE_THRESHOLD)
    hor = hor or CV_HORIZON
    validation_types = validation_types or list(ValidationType)

    # --- init --------------------------------------------
    x_train = load_train_data()  # type: np.ndarray
    x_test = load_test_data()  # type: np.ndarray
    results = dict()  # type: Dict[ValidationType, ValidationPredictions]

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
            results[ValidationType.CV_TRAIN] = cv_result.train_metrics.overall_val_preds()

        if ValidationType.CV_VAL in validation_types:
            results[ValidationType.CV_VAL] = cv_result.val_metrics.overall_val_preds()

    # --- training set ------------------------------------
    if ValidationType.TRAIN in validation_types:
        results[ValidationType.TRAIN] = evaluate_ts_model(
            model=ts_model, x_hist=x_train[: ts_model.min_hist()], x_val=x_train[ts_model.min_hist() :], hor=hor, silent=False
        )

    # --- test set ----------------------------------------
    if ValidationType.TEST in validation_types:
        results[ValidationType.TEST] = evaluate_ts_model(model=ts_model, x_hist=x_train, x_val=x_test, hor=hor, silent=False)

    # --- return ------------------------------------------
    return results


def validate_models(
    models: Dict[str, TimeSeriesModel], metric: TimeSeriesMetric = None, hor: int = None
) -> Dict[str, Dict[ValidationType, ValidationPredictions]]:

    return {
        model_name: validate_model(model, metric, hor)
        for model_name, model in tqdm(models.items(), desc="Validating models ", file=sys.stdout)
    }
