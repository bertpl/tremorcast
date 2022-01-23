from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.applications.vedur_is.vedur import VedurColors
from src.projects.fagradalsfjall._project_settings import FORECAST_SIGNAL_NAME
from src.tools.datetime import ts_to_float
from src.tools.matplotlib import plot_style_matplotlib_default


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

    plot_style_matplotlib_default()

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

        # find forecast that most closely starts at request i
        best_i = None  # type: Optional[int]
        best_forecast = None  # type: Optional[np.ndarray]
        for i_start, forecast in forecasts:
            if (best_i is None) or (abs(i_start - i) < abs(best_i - i)):
                best_i = i_start
                best_forecast = forecast

        # forecast = forecasts[i][1]  # type: np.ndarray

        forecast = best_forecast[0:horizon]
        x = x_values[best_i : i + len(forecast)]

        ax.plot(x[0], forecast[0], "ko", scalex=False, scaley=False)
        ax.plot(x, forecast, "k", lw=1, scalex=False, scaley=False)

    # --- return ------------------------------------------
    return fig, ax
