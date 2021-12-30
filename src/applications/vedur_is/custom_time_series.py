import numpy as np

from src.base.time_series import MultiTimeSeries, TimeSeries


class MinMeanMaxTimeSeries(MultiTimeSeries):

    # --- constructor -------------------------------------
    def __init__(self, min: TimeSeries, mean: TimeSeries, max: TimeSeries):
        super().__init__([("min", min), ("mean", mean), ("max", max)])

    # --- convenience -------------------------------------
    @property
    def min(self) -> TimeSeries:
        return self["min"]

    @property
    def mean(self) -> TimeSeries:
        return self["mean"]

    @property
    def max(self) -> TimeSeries:
        return self["max"]
