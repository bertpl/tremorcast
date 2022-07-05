"""In this file we compute the Maximum Reliable Lead Time based on the MAE curve"""

import numpy as np


def compute_maximum_reliable_lead_time(score_curve: np.ndarray, threshold: float) -> float:
    """
    Computes 'Maximum Reliable Lead Time', expressed in # of samples, by evaluating for how many samples
    the error curve does not exceed the threshold.

    The result is returned as a float, by interpolation between the first sample exceeding the threshold and the
    sample before.

    :param score_curve: (1D numpy array) containing the score curve, where the first value represents the score of
                             forecasting 1 sample ahead.  Scores are within range [0,1] and higher is always better.
    :param threshold: (float >= 0)
    :return: Computed metric expressed in number of samples, value between 1 and len(score_curve)+1, except for the
                following corner cases:
                  1) if mae_curve[0] < threshold: a value between 0 and 1 is returned
                  2) if all(score_curve > threshold): np.inf is returned
    """

    # --- corner case 2 -----------------------------------
    if all(score_curve > threshold):
        return np.inf

    # --- corner case 1 -----------------------------------
    if score_curve[0] <= threshold:
        # return value in [0, 1]
        return abs(threshold) / (abs(threshold) + abs(score_curve[0] - threshold))

    # --- regular case ------------------------------------

    # we are guaranteed to find one (as we're not in corner case 2) + i_first will not be 0 (which is corner case 1)
    i_first = next(i for i, score in enumerate(score_curve) if score <= threshold)

    # interpolate between score curve values i_first-1 and i_first to find intersection point with threshold
    return 1 + np.interp(x=threshold, xp=[score_curve[i_first], score_curve[i_first - 1]], fp=[i_first, i_first - 1])
