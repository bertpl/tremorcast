import numpy as np

from src.base.time_series import MultiTimeSeries, TimeSeries


class MinMidMaxTimeSeries(MultiTimeSeries):

    # --- constructor -------------------------------------
    def __init__(self, min: TimeSeries, mid: TimeSeries, max: TimeSeries):
        super().__init__([("min", min), ("mid", mid), ("max", max)])

    # --- convenience -------------------------------------
    @property
    def min(self) -> TimeSeries:
        return self["min"]

    @property
    def mid(self) -> TimeSeries:
        return self["mid"]

    @property
    def max(self) -> TimeSeries:
        return self["max"]
