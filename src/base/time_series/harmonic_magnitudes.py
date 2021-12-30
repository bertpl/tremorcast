from __future__ import annotations

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.tools.datetime import ts_to_float
from src.tools.matplotlib import enable_grid_lines, set_x_scale_daily, set_y_scale

from .time_series import MultiTimeSeries, TimeSeries


class HarmonicMagnitudes(MultiTimeSeries):
    """
    Class to represent a group of time series representing the magnitudes (in undefined units)
      of harmonic tremors at multiple frequency bands.

    No assumptions are made regarding these frequency bands.  For each frequency band multiple
      time series can be added representing different processing methods (min, max, mean, median, ...).

    Each individual time series is represented by a TimeSeries object and is characterized by a
      tag = (freq, signal_type), both of which are strings.
    """

    # -------------------------------------------------------------------------
    #  Constructor & co
    # -------------------------------------------------------------------------
    def __init__(self, freq_bands: List[Any], colors: List[Tuple[int, int, int]]):
        self.freq_bands = freq_bands
        self.colors = colors
        super().__init__()

    def copy(self) -> HarmonicMagnitudes:
        cpy = HarmonicMagnitudes(self.freq_bands, self.colors)
        for tag, series in self.tags():
            self[tag] = series
        return cpy

    def __str__(self):
        return f"HarmonicMagnitudes(t0={self.t0}, ts={self.ts}), {len(self)} signals of length {len(self.time)})"

    # -------------------------------------------------------------------------
    #  Get / set
    # -------------------------------------------------------------------------
    def set_signal(self, freq_band: Any, signal_type: Any, signal: TimeSeries):
        """Convenience equivalent of __setitem__"""
        self[self.__construct_tag(freq_band, signal_type)] = signal

    def get_signal(self, freq_band: Any, signal_type: Any) -> TimeSeries:
        """Convenience equivalent of __getitem__"""
        return self[self.__construct_tag(freq_band, signal_type)]

    @staticmethod
    def __construct_tag(freq_band: Any, signal_type: Any) -> str:
        return f"{str(freq_band)}_{str(signal_type)}"

    # -------------------------------------------------------------------------
    #  Graphs
    # -------------------------------------------------------------------------
    def create_plot(self, signal_types: List[Any] = None, title: str = None) -> Tuple[plt.Figure, plt.Axes]:

        # --- determine signal tags to be plotted ---------
        if signal_types is not None:
            tags = [
                candidate_tag
                for candidate_tag in [
                    self.__construct_tag(freq_band, signal_type)
                    for freq_band in self.freq_bands
                    for signal_type in signal_types
                ]
                if candidate_tag in self.tags()
            ]
        else:
            tags = self.tags()

        # --- actual work ---------------------------------
        fig, ax = self._prepare_fig_and_axes(tags)
        self._draw_signals(fig, ax, tags)
        self._finalize_fig_and_axes(fig, ax, title)

        # --- return --------------------------------------
        return fig, ax

    def _prepare_fig_and_axes(self, tags: List[str]) -> Tuple[plt.Figure, plt.Axes]:

        # create plot
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes

        # x scale
        set_x_scale_daily(ax, ts_from=min(self.time), ts_to=max(self.time), margin=0.01)

        # y scale
        y_min = min([np.min(self[tag].data) for tag in tags])
        y_max = max([np.max(self[tag].data) for tag in tags])
        set_y_scale(ax, y_min=y_min, y_max=y_max, margin=0.05)

        # grid lines
        enable_grid_lines(ax)

        # return
        return fig, ax

    def _draw_signals(self, fig: plt.Figure, ax: plt.Axes, tags: List[str]) -> List[plt.Line2D]:

        x = [ts_to_float(ts) for ts in self.time]

        all_handles = []
        for tag in tags:
            color = self.colors[self.freq_bands.index(tag[0])]
            h = ax.plot(x, self[tag].data, scalex=False, scaley=False, c=[x / 255 for x in color], lw=0.5)
            all_handles.append(h[0])

        return all_handles

    def _finalize_fig_and_axes(self, fig: plt.Figure, ax: plt.Axes, title: str):

        # size
        x_min, x_max = ax.get_xlim()
        w = (x_max - x_min) / 150_000  # divide by ~2 days
        h = 6
        fig.set_size_inches(w=w, h=h)

        # title
        fig.suptitle(title)
        fig.tight_layout()
