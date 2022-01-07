"""In this file we compute the Mean Absolute Deviation as a function of n_samples_ahead."""

from collections import defaultdict
from typing import List, Tuple

import numpy as np


def compute_mad_curve(observations: np.ndarray, forecasts: List[Tuple[int, np.ndarray]]) -> np.ndarray:
    """
    Computes the Mean-Absolute-Deviation (MAD) as a function of n_samples_ahead based on a series of forecasts.

    The resulting MAD-curve is an indicator of how good the forecasts are as a function of how far ahead we're forecasting.

    :param observations: (1D numpy array) observations we want to approximate with the forecasts.
    :param forecasts: a series of forecasts as a list of (i_first, forecast)-tuples, where...
                         i_first: index of first element of the forecast into the observations array.
                                   In other words:  observations[i_first+n] can be compared with forecast[n] for n>=0
                         forecast: 1D numpy array with forecast
    :return: 1D numpy array with MAD curve (as long as the longest forecast),
                        with first element corresponding to forecasting 1 samples ahead.
    """

    # group absolute deviations by n_samples_ahead
    devs = defaultdict(list)  # dict mapping (n_samples_ahead-1) -> list of abs(forecast-observation) values
    for i_first, forecast in forecasts:
        for n, forecasted_value in enumerate(forecast):
            devs[n].append(abs(forecasted_value - observations[i_first + n]))

    # compute MAD curve
    mad_curve = np.zeros(len(devs))
    for n, devs_group in devs.items():
        mad_curve[n] = np.mean(devs_group)

    # return
    return mad_curve
