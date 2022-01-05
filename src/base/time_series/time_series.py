"""
This module implements classes for representing and easily manipulating
  equidistantly sampled time series.  Classes for single & multiple time series are provided.
"""

from __future__ import annotations

import datetime
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

from src.tools.datetime import float_to_ts, ts_to_float


# =================================================================================================
#  TimeSeries
# =================================================================================================
class TimeSeries:
    """Class to represent a time series of float-values at equidistantly spaced points in time."""

    # --- constructor -----------------------------------------
    def __init__(
        self,
        t0: datetime.datetime,
        ts: datetime.timedelta,
        data: Union[np.ndarray, List[float]],
    ):
        self.ts = ts
        self.t0 = t0
        self.data = data.flatten() if isinstance(data, np.ndarray) else np.array(data)

    # --- convenience -------------------------------------
    @property
    def time(self) -> List[datetime.datetime]:
        return [self.t0 + i * self.ts for i in range(len(self))]

    @property
    def n_samples(self) -> int:
        return len(self.data)

    def get_closest_index(self, t: datetime.datetime) -> int:
        i = round((t - self.t0).total_seconds() / self.ts.total_seconds())
        return max(0, min(i, self.n_samples - 1))

    # --- list-like ---------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            # slice -> return new TimeSeries object
            return TimeSeries(t0=self.t0 + (self.ts * subscript.start), ts=self.ts, data=self.data[subscript])
        else:
            # single element -> return float
            return self.data[subscript]

    # --- other magic methods -----------------------------
    def __str__(self) -> str:
        return f"TimeSeries(t0={self.t0}, ts={self.ts}), data=<{len(self)}-element array>)"

    def copy(self) -> TimeSeries:
        return TimeSeries(self.t0, self.ts, self.data.copy())

    def __or__(self, other: TimeSeries):
        cpy = self.copy()
        cpy |= other
        return cpy

    def __ior__(self, other: TimeSeries) -> TimeSeries:

        # error checking
        assert self.ts == other.ts, "TimeSeries.__ior__: ts mismatch."

        # determine t0, ts
        new_t0 = min(self.t0, other.t0)
        new_ts = self.ts

        # construct merge data
        i0_self = round((self.t0 - new_t0).total_seconds() / new_ts.total_seconds())
        i0_other = round((other.t0 - new_t0).total_seconds() / new_ts.total_seconds())

        new_len = max(i0_self + len(self), i0_other + len(other))
        new_data = np.full(new_len, np.nan)

        new_data[i0_self : i0_self + len(self)] = self.data
        for i, value in enumerate(other.data):
            if not np.isnan(value):
                # old values should not be overwritten with NaN values from new signal
                new_data[i0_other + i] = other.data[i]

        # assign to self
        self.t0 = new_t0
        self.data = new_data

        # return self
        return self

    # --- resampling --------------------------------------
    def resample(self, new_ts: datetime.timedelta) -> TimeSeries:
        """Resamples time series to new sampling time.  This will always
        use the same 'reference t0' for defining the alignment of the new
        sampling times, i.e. tools.datetime.__TS_REF"""

        # --- helpers -------------------------------------
        def ts_to_sample_nr(ts: datetime.datetime) -> float:
            """returns how many sampling times (new_ts) have passed between ts_ref and ts (as a float)"""
            return ts_to_float(ts) / new_ts.total_seconds()

        def sample_nr_to_ts(f: float) -> datetime.datetime:
            return float_to_ts(f * new_ts.total_seconds())

        # --- resampling ----------------------------------
        time = self.time  # current ts-values for each sample in current TimeSeries
        i_first = int(np.ceil(ts_to_sample_nr(min(time))))
        i_last = int(np.floor(ts_to_sample_nr(max(time))))

        new_t0 = sample_nr_to_ts(i_first)

        xp = [ts_to_sample_nr(ts) for ts in time]
        fp = self.data
        x = list(range(i_first, i_last + 1))

        new_data = np.interp(x, xp, fp)

        # --- new TimeSeries ------------------------------
        return TimeSeries(new_t0, new_ts, new_data)

    # -------------------------------------------------------------------------
    #  Export
    # -------------------------------------------------------------------------
    def to_dataframe(self) -> pd.Dataframe:
        return pd.DataFrame(index=self.time, columns=["data"], data=self.data)


# =================================================================================================
#  MultiTimeSeries
# =================================================================================================
class MultiTimeSeries:
    """
    Acts as an unsorted collection of time series that have a common time axis
     and can be easily merged (| and |=) in a single operation.

    Individual time series can be obtained using a dict-like interface.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, items: Iterable[Tuple[str, TimeSeries]] = None):
        """
        Constructor of MultiTimeSeries class.
        :param items: iterable of (tag, series)-tuples
        """
        self._series = dict()  # type: Dict[str, TimeSeries]
        if items:
            for tag, series in items:
                self.__setitem__(tag, series)

    # -------------------------------------------------------------------------
    #  Dict-like - get / set single TimeSeries
    # -------------------------------------------------------------------------
    def __setitem__(self, tag: str, series: TimeSeries):
        if len(self) > 0:
            if series.t0 != self.t0:
                raise ValueError(f"t0 mismatch upon insertion of TimeSeries: {series.t0} vs {self.t0}.")
            if series.ts != self.ts:
                raise ValueError(f"ts mismatch upon insertion of TimeSeries: {series.ts} vs {self.ts}.")

        self._series[tag] = series

    def __getitem__(self, tag: str) -> TimeSeries:
        return self._series[tag]

    def __len__(self):
        return len(self._series)

    def tags(self) -> List[str]:
        return sorted(self._series.keys())  # deterministic order

    def series(self) -> List[TimeSeries]:
        return [self._series[tag] for tag in self.tags()]  # same order as tags()

    def items(self) -> Iterable[Tuple[str, TimeSeries]]:
        for tag in self.tags():
            yield tag, self._series[tag]  # same order as tags()

    # -------------------------------------------------------------------------
    #  TimeSeries-like - Properties
    # -------------------------------------------------------------------------
    @property
    def t0(self) -> datetime.datetime:
        return self.series()[0].t0

    @property
    def ts(self) -> datetime.timedelta:
        return self.series()[0].ts

    @property
    def time(self) -> List[datetime.datetime]:
        return self.series()[0].time

    @property
    def n_samples(self) -> int:
        return self.series()[0].n_samples

    def get_closest_index(self, t: datetime.datetime) -> int:
        i = round((t - self.t0).total_seconds() / self.ts.total_seconds())
        return max(0, min(i, self.n_samples - 1))

    def to_array(self) -> np.array:
        """Returns a (n_signals, n_samples)-shaped numpy array (copy of original data)."""
        arr = np.zeros(shape=(len(self), len(self.time)))
        for i, series in enumerate(self.series()):
            arr[i, :] = series.data.copy()
        return arr

    def slice(self, i_from: int, i_to: int) -> MultiTimeSeries:
        """Returns a subset [i_from:i_to] of samples."""
        cpy = self.copy()  # this make sure we return an appropriate object type, also for child classes
        for tag, series in self.items():
            # use cpy._series[tag] instead of cpy[tag] to bypass sanity checks,
            # since temporarily the time series will have incompatible time axes.
            cpy._series[tag] = series[i_from:i_to]
        return cpy

    # -------------------------------------------------------------------------
    #  TimeSeries-like - Merge
    # -------------------------------------------------------------------------
    def copy(self) -> MultiTimeSeries:
        """Deep copy of MultiTimeSeries object"""
        copied_items = [(tag, time_series.copy()) for tag, time_series in self.items()]
        return MultiTimeSeries(copied_items)

    def clear(self):
        self._series.clear()

    def __or__(self, other: MultiTimeSeries):
        cpy = self.copy()
        cpy |= other
        return cpy

    def __ior__(self, other: MultiTimeSeries):
        if set(self.tags()) != set(other.tags()):
            raise ValueError(
                f"Mismatch of tags when merging two MultiTimeSeries objects: {set(self.tags())} vs {set(self.tags())}"
            )
        for tag, series in self.items():
            series |= other[tag]
        return self

    # -------------------------------------------------------------------------
    #  TimeSeries-like - Resample
    # -------------------------------------------------------------------------
    def resample(self, new_ts: datetime.timedelta) -> MultiTimeSeries:
        new = self.copy()
        new.clear()
        for tag, series in self.items():
            new[tag] = series.resample(new_ts)
        return new

    # -------------------------------------------------------------------------
    #  Export
    # -------------------------------------------------------------------------
    def to_dataframe(self) -> pd.Dataframe:

        df = pd.DataFrame()

        for tag, series in self.items():
            df[tag] = series.to_dataframe()

        return df
