from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.applications.vedur_is.vedur import VedurColors
from src.tools.datetime import ts_to_float

from ._project_settings import FORECAST_SIGNAL_NAME


def plot_forecasts(
    data_test: VedurHarmonicMagnitudes,
    forecasts: List[Tuple[int, np.ndarray]],
    horizon: int,
    indices: List[int],
    title: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot requested forecasts.
    """

    # --- base plot ---------------------------------------
    i_first = max([0, min(indices) - 1 * 96])  # 1 day before start of first forecast
    i_last = min([data_test.n_samples, max(indices) + horizon + 96])  # 1 day after start of first forecast
    data_test_subset = data_test.slice(i_first, i_last)

    fig, ax = data_test_subset.create_plot(title=title, aspect_ratio=1.5)

    # --- plot main signal --------------------------------
    x_values = [ts_to_float(t) for t in data_test.time]
    signal = data_test[FORECAST_SIGNAL_NAME].data

    plot_clr = [c / 255 for c in VedurColors.PURPLE.value]

    ax.plot(x_values[i_first:i_last], signal[i_first:i_last], scalex=False, scaley=False, c=plot_clr, lw=2)

    # --- plot forecasts ----------------------------------
    for i in indices:
        forecast = forecasts[i][1]  # type: np.ndarray

        forecast = forecast[0:horizon]
        x = x_values[i : i + len(forecast)]

        ax.plot(x[0], forecast[0], "ko", scalex=False, scaley=False)
        ax.plot(x, forecast, "k", lw=1, scalex=False, scaley=False)

    # --- return ------------------------------------------
    return fig, ax
