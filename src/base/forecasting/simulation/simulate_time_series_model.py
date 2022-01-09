import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.base.forecasting.metrics import compute_mae_curve, compute_maximum_reliable_lead_time
from src.base.forecasting.models import TimeSeriesForecastModel


def simulate_time_series_model(
    model: TimeSeriesForecastModel,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    accuracy_threshold: float,
    horizon: int,
    retrain_model: bool = False,
) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray, float]:
    """
    Evaluates time series forecast model on the test set after training on the training set, assuming the training and
    test sets chronologically complement each other without a gap in between.

    Procedure:
      - if retrain_model==False --> train model on training data
      - for each sample with index i in the test set:
         - construct history as training data + first i-1 test data samples
         - if retrain_model==True --> train model on history
         - use model to predict next min(horizon, remaining samples in test set) samples
      -  compute MAE curve & maximum reliable forecast horizon

    :param model: TimeSeriesForecastModel to be evaluated; need not be trained.
    :param training_data: (pd.DataFrame) training data set
    :param test_data: (pd.DataFrame) test data set
    :param accuracy_threshold: (float) threshold to be used to compute max_reliable_lead_time
    :param horizon: (int) (max) number of samples to predict ahead
    :param retrain_model: (bool, default=False) if True the model is retrained as more data becomes available.
    :return: (forecasts, mae_curve, max_reliable_lead_time)
    """

    # --- initial training --------------------------------
    if not retrain_model:
        model.fit(training_data)

    # --- perform simulation ------------------------------
    forecasts = []  # type: List[Tuple[int, np.ndarray]]
    for i in tqdm(
        range(len(test_data)),
        desc=f"Evaluating model '{model.model_type}' (retrain={retrain_model})".ljust(60),
        file=sys.stdout,
    ):

        # construct history
        if i > 0:
            history = training_data.append(test_data.iloc[0:i])
        else:
            history = training_data.copy()

        # retrain if needed
        if retrain_model:
            model.fit(history)

        # create forecast
        n_samples = min(horizon, len(test_data) - i)
        forecast = model.predict(history, n_samples)

        forecasts.append((i, forecast))

    # --- compute metrics & return ------------------------
    observations = test_data[model.signal_name].to_numpy()
    mae_curve = compute_mae_curve(observations, forecasts)
    max_reliable_lead_time = compute_maximum_reliable_lead_time(mae_curve, accuracy_threshold)

    print()
    print(f"min(MAE)               = {min(mae_curve):.1f}")
    print(f"max(MAE)               = {max(mae_curve):.1f}")
    print(f"max_reliable_lead_time = {max_reliable_lead_time:.3f} samples")
    print()

    return forecasts, mae_curve, max_reliable_lead_time
