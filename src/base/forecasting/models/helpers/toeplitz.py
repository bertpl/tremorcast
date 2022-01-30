import numpy as np
from scipy import linalg


def build_toeplitz(ts: np.ndarray, window_size: int, forward: bool) -> np.ndarray:
    """
    Returns (n x window_size) Toeplitz matrix, with each row containing the full
      backward or forward-looking window for each sample in the TimeSeries.

    This helper is handy for creating auto-regressive tabulated datasets from a time series.

    Example

        ts = [1, 3, 10, 20]

        window_size = 3, forward = False

            toeplitz = [ [ 1, NaN, NaN],
                         [ 3,   1, NaN],
                         [10,   3,   1],
                         [20,  10,   3] ]

        window_size = 4, forward = True

            toeplitz = [ [ 1,   3,  10,  20],
                         [ 3,  10,  20, NaN],
                         [10,  20, NaN, NaN],
                         [20, NaN, NaN, NaN] ]


    :param ts: (1D n-element array) time series to convert to Toeplitz matrix
    :param window_size: (int>0) number of past or future samples to include in Toeplitz matrix
    :param forward: (bool) if False, past time series values are used, otherwise future time series values are used.
    """

    # prep
    first_row = np.full(shape=(1, window_size), fill_value=np.NaN)

    # compute
    if forward:
        return np.flipud(linalg.toeplitz(c=np.flip(ts), r=first_row))
    else:
        return linalg.toeplitz(c=ts, r=first_row)
