"""In this file we compute the Maximum Reliable Lead Time based on the MAD curve"""

import numpy as np


def compute_maximum_reliable_lead_time(mad_curve: np.ndarray, threshold: float) -> float:
    """
    Computes 'Maximum Reliable Lead Time', expressed in # of samples, by evaluating for how many samples
    the MAD curve does not exceed the threshold.

    The result is returned as a float, by interpolation between the first sample exceeding the threshold and the
    sample before.

    :param mad_curve: (1D numpy array) containing the MAD curve, where the first value represents the MAD of forecasting
                             1 sample ahead.
    :param threshold: (float >= 0)
    :return: Computed metric expressed in number of samples, value between 1 and len(mad_curve)+1, except for the
                following corner cases:
                  1) if mad_curve[0] > threshold: a value between 0 and 1 is returned
                  2) if all(mad_curve < threshold): np.inf is returned
    """

    # --- corner case 2 -----------------------------------
    # detect corner case 2
    if all(mad_curve < threshold):
        return np.inf

    # --- corner case 1 & regular case --------------------
    # prepend 0 to gracefully include corner case 2 + to adjust for element 0 representing 1 sample ahead, not 0.
    mad_curve = np.concatenate([[0], mad_curve])

    # we are guaranteed to find one (as we're not in corner case 2) + i_first will not be 0
    i_first = next(i for i, mad in enumerate(mad_curve) if mad >= threshold)

    # interpolate between mad curve values i_first-1 and i_first to find intersection point with threshold
    return np.interp(x=threshold, xp=[mad_curve[i_first - 1], mad_curve[i_first]], fp=[i_first - 1, i_first])
