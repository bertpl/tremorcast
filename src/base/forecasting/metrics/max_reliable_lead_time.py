"""In this file we compute the Maximum Reliable Lead Time based on the MAE curve"""

import numpy as np


def compute_maximum_reliable_lead_time(mae_curve: np.ndarray, threshold: float) -> float:
    """
    Computes 'Maximum Reliable Lead Time', expressed in # of samples, by evaluating for how many samples
    the MAE curve does not exceed the threshold.

    The result is returned as a float, by interpolation between the first sample exceeding the threshold and the
    sample before.

    :param mae_curve: (1D numpy array) containing the MAE curve, where the first value represents the MAE of forecasting
                             1 sample ahead.
    :param threshold: (float >= 0)
    :return: Computed metric expressed in number of samples, value between 1 and len(mae_curve)+1, except for the
                following corner cases:
                  1) if mae_curve[0] > threshold: a value between 0 and 1 is returned
                  2) if all(mae_curve < threshold): np.inf is returned
    """

    # --- corner case 2 -----------------------------------
    # detect corner case 2
    if all(mae_curve < threshold):
        return np.inf

    # --- corner case 1 & regular case --------------------
    # prepend 0 to gracefully include corner case 2 + to adjust for element 0 representing 1 sample ahead, not 0.
    mae_curve = np.concatenate([[0], mae_curve])

    # we are guaranteed to find one (as we're not in corner case 2) + i_first will not be 0
    i_first = next(i for i, mad in enumerate(mae_curve) if mad >= threshold)

    # interpolate between mad curve values i_first-1 and i_first to find intersection point with threshold
    return np.interp(x=threshold, xp=[mae_curve[i_first - 1], mae_curve[i_first]], fp=[i_first - 1, i_first])
