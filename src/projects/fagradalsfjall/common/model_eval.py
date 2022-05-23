from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.base.forecasting.evaluation.metrics import TimeSeriesMetric
from src.base.forecasting.models import TimeSeriesCrossValidation, TimeSeriesModel
from src.base.forecasting.models.time_series.helpers.timeseries_cv import TimeSeriesCVResult, evaluate_ts_model
from src.projects.fagradalsfjall.common.dataset import load_test_data_numpy, load_train_data_numpy
from src.projects.fagradalsfjall.common.project_settings import (
    CV_HORIZON_SAMPLES,
    DATASET_TRAIN_N_SAMPLES,
    SIMULATION_STRIDE,
    TS_ALL_METRICS,
    TS_CV_SPLITTER,
)


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
#  Model Evaluation Result
# =================================================================================================
class ModelEvalResult:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        ts_model: TimeSeriesModel,
        metrics: List[TimeSeriesMetric] = None,
        hor: int = None,
        compute_metrics: bool = True,
        compute_simulations: bool = True,
    ):

        # --- process arguments ---------------------------
        if metrics is None:
            metrics = TS_ALL_METRICS

        if hor is None:
            hor = CV_HORIZON_SAMPLES

        assert len({metric.tabular_metric for metric in metrics}) == 1, "metrics should have identical tabular_metric"

        # --- remember arguments --------------------------
        self.ts_model = ts_model
        self.metrics = metrics
        self.hor = hor

        self.tabular_metric = self.metrics[0].tabular_metric

        # --- init properties -----------------------------
        # metrics curves for all validation types
        # note: metrics curves only depend on the tabular metric, which should be the same
        #       for all provided time series metrics
        self.metric_curves = dict()  # type: Dict[ValidationType, np.ndarray]

        # resulting time series metric value for each (validation type, ts metric)
        self.metric_values = dict()  # type: Dict[Tuple[ValidationType, TimeSeriesMetric], float]

        # simulations (train & test)
        self.simulations = []  # type: List[Tuple[int, np.ndarray]]

        # --- compute results -----------------------------
        if compute_metrics:
            self._compute_metrics()
        if compute_simulations:
            self._compute_simulations()

    @property
    def test_simulations(self) -> List[Tuple[int, np.ndarray]]:
        return [(i, pred) for i, pred in self.simulations if i >= DATASET_TRAIN_N_SAMPLES]

    @property
    def train_simulations(self) -> List[Tuple[int, np.ndarray]]:
        return [(i, pred) for i, pred in self.simulations if i < DATASET_TRAIN_N_SAMPLES]

    # -------------------------------------------------------------------------
    #  Evaluation functionality
    # -------------------------------------------------------------------------
    def _compute_metrics(self):
        # computes values of self.metric_curves & self.metric_values

        # --- load data -----------------------------------
        x_train = load_train_data_numpy()  # type: np.ndarray
        x_test = load_test_data_numpy()  # type: np.ndarray

        # --- metrics curves ------------------------------

        # --- CV_TRAIN, CV_VAL ---
        self.report_progress("Metrics: CV_TRAIN, CV_VAL")
        cv_result = TimeSeriesCrossValidation.cross_validate_ts_model(
            model=self.ts_model,
            x=x_train,
            metric=self.metrics[0],  # pick one, metrics curves only depend on tabular metric
            ts_cv_splitter=TS_CV_SPLITTER,
            hor=self.hor,
            retrain=False,
        )  # type: TimeSeriesCVResult
        self.metric_curves[ValidationType.CV_TRAIN] = cv_result.train_metrics.overall_metric_curve()
        self.metric_curves[ValidationType.CV_VAL] = cv_result.val_metrics.overall_metric_curve()

        # --- TRAIN ---
        self.report_progress("Metrics: TRAIN")
        self.metric_curves[ValidationType.TRAIN] = evaluate_ts_model(
            model=self.ts_model,
            x_hist=x_train[: self.ts_model.min_hist()],
            x_val=x_train[self.ts_model.min_hist() :],
            metric=self.metrics[0],  # pick one, metrics curves only depend on tabular metric
            hor=self.hor,
            silent=False,
        )

        # --- TEST ---
        self.report_progress("Metrics: TEST")
        self.metric_curves[ValidationType.TEST] = evaluate_ts_model(
            model=self.ts_model,
            x_hist=x_train,
            x_val=x_test,
            metric=self.metrics[0],  # pick one, metrics curves only depend on tabular metric
            hor=self.hor,
            silent=False,
        )

        # --- ts metric values ----------------------------
        for validation_type in ValidationType:
            for metric in self.metrics:
                self.metric_values[validation_type, metric] = metric.compute(self.metric_curves[validation_type])

    def _compute_simulations(self):
        # computes self.simulations

        # --- load data -----------------------------------
        x_train = load_train_data_numpy()  # type: np.ndarray
        x_test = load_test_data_numpy()  # type: np.ndarray

        # --- simulate ------------------------------------
        self.report_progress("Simulations")
        first_sample = int(math.ceil(self.ts_model.min_hist() / SIMULATION_STRIDE) * SIMULATION_STRIDE)
        self.simulations = self.ts_model.batch_predict(
            x=np.concatenate([x_train, x_test]),
            first_sample=first_sample,
            hor=self.hor,
            overlap_end=False,
            stride=SIMULATION_STRIDE,
            silent=False,
        )

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def report_progress(self, task: str):
        print(f"Evaluating model '{self.ts_model.name}'... ".ljust(60) + f"[{task}]")

    # -------------------------------------------------------------------------
    #  Evaluate many
    # -------------------------------------------------------------------------
    @classmethod
    def eval_many(
        cls, ts_models: Dict[str, TimeSeriesModel], compute_metrics: bool = True, compute_simulations: bool = True
    ) -> Dict[str, ModelEvalResult]:
        return {
            model_id: ModelEvalResult(
                ts_model, compute_metrics=compute_metrics, compute_simulations=compute_simulations
            )
            for model_id, ts_model in ts_models.items()
        }

    @classmethod
    def all_metric_values_as_df(
        cls, model_eval_results: Dict[str, ModelEvalResult], validation_type: ValidationType
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        for model_id, model_eval_result in model_eval_results.items():
            for ts_metric in model_eval_result.metrics:
                df.at[model_id, ts_metric.name] = model_eval_result.metric_values[validation_type, ts_metric]

        return df
