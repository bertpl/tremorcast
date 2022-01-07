from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt

from src.base.time_series import HarmonicMagnitudes
from src.tools.datetime import ts_to_float

from .custom_time_series import MinMidMaxTimeSeries
from .vedur import VEDUR_DATA_COLORS, VedurColors, VedurFreqBands


class VedurHarmonicMagnitudes(HarmonicMagnitudes):
    """
    Class representing harmonic tremor magnitudes as obtained from (graphs produced by) vedur.is,
      such as found on http://hraun.vedur.is/ja/oroi/allarsort.html

    Specifics are:
      - 3 frequency bands    (0.5-1.0Hz, 1-2Hz, 2-4Hz)
      - each frequency band has a min, mid & max value
      - colors are set to be consistent with the above linked graphs

    """

    # --- constructor -------------------------------------
    def __init__(self, low: MinMidMaxTimeSeries, mid: MinMidMaxTimeSeries, hi: MinMidMaxTimeSeries):

        # --- superclass constructor ------------
        super().__init__(
            freq_bands=list(VEDUR_DATA_COLORS.keys()), colors=[clr.value for clr in VEDUR_DATA_COLORS.values()]
        )

        # --- add 3x3 signals -------------------
        self.set_signal(VedurFreqBands.LOW, "min", low.min)
        self.set_signal(VedurFreqBands.LOW, "mid", low.mid)
        self.set_signal(VedurFreqBands.LOW, "max", low.max)

        self.set_signal(VedurFreqBands.MID, "min", mid.min)
        self.set_signal(VedurFreqBands.MID, "mid", mid.mid)
        self.set_signal(VedurFreqBands.MID, "max", mid.max)

        self.set_signal(VedurFreqBands.HI, "min", hi.min)
        self.set_signal(VedurFreqBands.HI, "mid", hi.mid)
        self.set_signal(VedurFreqBands.HI, "max", hi.max)

    # --- convenience -------------------------------------
    def low(self) -> MinMidMaxTimeSeries:
        return MinMidMaxTimeSeries(
            min=self.get_signal(VedurFreqBands.LOW, "min"),
            mid=self.get_signal(VedurFreqBands.LOW, "mid"),
            max=self.get_signal(VedurFreqBands.LOW, "max"),
        )

    def mid(self) -> MinMidMaxTimeSeries:
        return MinMidMaxTimeSeries(
            min=self.get_signal(VedurFreqBands.MID, "min"),
            mid=self.get_signal(VedurFreqBands.MID, "mid"),
            max=self.get_signal(VedurFreqBands.MID, "max"),
        )

    def hi(self) -> MinMidMaxTimeSeries:
        return MinMidMaxTimeSeries(
            min=self.get_signal(VedurFreqBands.HI, "min"),
            mid=self.get_signal(VedurFreqBands.HI, "mid"),
            max=self.get_signal(VedurFreqBands.HI, "max"),
        )

    # --- plotting ----------------------------------------
    def _draw_signals(self, fig: plt.Figure, ax: plt.Axes, tags: List[str]) -> List[plt.Line2D]:

        # --- prep ----------------------------------------
        x = [ts_to_float(ts) for ts in self.time]
        min_thickness = (max(ax.get_ylim()) - min(ax.get_ylim())) / 1000
        all_handles = []

        # --- purple --------------------------------------
        y_min = self.get_signal(VedurFreqBands.LOW, "min").data - min_thickness
        y_max = self.get_signal(VedurFreqBands.LOW, "max").data + min_thickness
        h = ax.fill_between(x, y_min, y_max, color=[x / 255 for x in VedurColors.PURPLE.value], alpha=0.5, lw=0)
        all_handles.append(h)

        # --- green ---------------------------------------
        y_min = self.get_signal(VedurFreqBands.MID, "min").data - min_thickness
        y_max = self.get_signal(VedurFreqBands.MID, "max").data + min_thickness
        h = ax.fill_between(x, y_min, y_max, color=[x / 255 for x in VedurColors.GREEN.value], alpha=0.5, lw=0)
        all_handles.append(h)

        # --- blue ----------------------------------------
        y_min = self.get_signal(VedurFreqBands.HI, "min").data - min_thickness
        y_max = self.get_signal(VedurFreqBands.HI, "max").data + min_thickness
        h = ax.fill_between(x, y_min, y_max, color=[x / 255 for x in VedurColors.BLUE.value], alpha=0.5, lw=0)
        all_handles.append(h)

        return all_handles

    # --- other -------------------------------------------
    def copy(self) -> VedurHarmonicMagnitudes:
        """By overriding this method, also the __or__ & resample() methods automatically return an object of this class."""
        return VedurHarmonicMagnitudes(self.low(), self.mid(), self.hi())
