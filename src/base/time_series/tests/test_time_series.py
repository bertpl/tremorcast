import datetime

import numpy as np
import pytest

from src.base.time_series import MultiTimeSeries, TimeSeries

# -------------------------------------------------------------------------
#  Sample data
# -------------------------------------------------------------------------
some_t0 = datetime.datetime(2021, 9, 1, 0, 0, 0)

ts_15min = datetime.timedelta(minutes=15)
ts_20min = datetime.timedelta(minutes=20)
ts_30min = datetime.timedelta(minutes=30)

one_minute = datetime.timedelta(minutes=1)


# =================================================================================================
#  TimeSeries
# =================================================================================================
@pytest.mark.parametrize(
    "data, expected_data",
    [([0, 0, 0], np.zeros(3)), (np.array([0, 0, 0]), np.zeros(3)), (np.array([[0, 0, 0]]), np.zeros(3))],
)
def test_time_series_constructor(data, expected_data):
    """Simple test to see if the constructor does not fail under various use cases"""

    # --- act ---------------------------------------------
    time_series = TimeSeries(some_t0, ts_30min, data)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(time_series.data, expected_data)


def test_time_series_misc():
    """Test TimeSeries.time, TimeSeries.__len__, TimeSeries.__getitem__"""

    # --- arrange -----------------------------------------
    t0 = some_t0
    ts = ts_15min
    data = [0, 0.1, 0.3]

    expected_time = [t0, t0 + ts, t0 + 2 * ts]
    expected_len = 3

    item_idx = 2
    expected_item = 0.3

    # --- act ---------------------------------------------
    time_series = TimeSeries(t0, ts, data)

    # --- assert ------------------------------------------
    assert time_series.time == expected_time
    assert len(time_series) == expected_len
    assert time_series[item_idx] == 0.3


def test_time_series_copy():

    # --- arrange -----------------------------------------
    t0 = some_t0
    ts = ts_15min
    data = [0, 0.1, 0.3]

    time_series = TimeSeries(t0, ts, data)

    # --- act ---------------------------------------------
    time_series_copy = time_series.copy()

    # --- assert ------------------------------------------
    assert id(time_series) != id(time_series_copy), "copy() should return a different object"

    assert time_series_copy.t0 == time_series.t0
    assert time_series_copy.ts == time_series.ts
    np.testing.assert_array_equal(time_series_copy.data, time_series.data)


@pytest.mark.parametrize(
    "time_series_a, time_series_b, expected_time_series",
    [
        # regular case
        (
            TimeSeries(some_t0, ts_30min, [1, 2, 3]),
            TimeSeries(some_t0 + ts_30min, ts_30min, [10, 20, 30]),
            TimeSeries(some_t0, ts_30min, [1, 10, 20, 30]),
        ),
        # gap between both time series
        (
            TimeSeries(some_t0, ts_30min, [1]),
            TimeSeries(some_t0 + (2 * ts_30min), ts_30min, [10]),
            TimeSeries(some_t0, ts_30min, [1, np.nan, 10]),
        ),
        # NaN in time series B, which should not overwrite corresponding value of time series A
        (
            TimeSeries(some_t0, ts_30min, [1, 2, 3]),
            TimeSeries(some_t0 + ts_30min, ts_30min, [10, np.nan, 30]),
            TimeSeries(some_t0, ts_30min, [1, 10, 3, 30]),
        ),
    ],
)
def test_time_series_or(time_series_a: TimeSeries, time_series_b: TimeSeries, expected_time_series: TimeSeries):
    """Test TimeSeries.__or__ and .__ior__."""

    # --- act ---------------------------------------------
    time_series_or = time_series_a | time_series_b

    # --- assert ------------------------------------------
    assert time_series_or.t0 == expected_time_series.t0
    assert time_series_or.ts == expected_time_series.ts
    np.testing.assert_array_equal(time_series_or.data, expected_time_series.data)


@pytest.mark.parametrize(
    "time_series, new_ts, expected_time_series",
    [
        (TimeSeries(some_t0, ts_30min, [0, 1, 2]), ts_15min, TimeSeries(some_t0, ts_15min, [0, 0.5, 1, 1.5, 2])),
        (TimeSeries(some_t0, ts_30min, [0, 3, 6, 9]), ts_20min, TimeSeries(some_t0, ts_20min, [0, 2, 4, 6, 8])),
        (TimeSeries(some_t0 - one_minute, ts_30min, [0, 30, 60]), ts_30min, TimeSeries(some_t0, ts_30min, [1, 31])),
    ],
)
def test_time_series_resample(time_series: TimeSeries, new_ts: datetime.timedelta, expected_time_series: TimeSeries):

    # --- act ---------------------------------------------
    resampled_time_series = time_series.resample(new_ts)

    # --- assert ------------------------------------------
    assert resampled_time_series.t0 == expected_time_series.t0
    assert resampled_time_series.ts == expected_time_series.ts
    np.testing.assert_array_almost_equal(resampled_time_series.data, expected_time_series.data)

    assert id(resampled_time_series) != time_series, "resample() should return a new object"


@pytest.mark.parametrize(
    "t, expected_index",
    [
        # before start of time series
        (some_t0 - ts_30min, 0),
        # around sample 0
        (some_t0 - one_minute, 0),
        (some_t0, 0),
        (some_t0 + one_minute, 0),
        # around sample 1
        (some_t0 + ts_30min - one_minute, 1),
        (some_t0 + ts_30min, 1),
        (some_t0 + ts_30min + one_minute, 1),
        # around sample 2
        (some_t0 + 2 * ts_30min - one_minute, 2),
        (some_t0 + 2 * ts_30min, 2),
        (some_t0 + 2 * ts_30min + one_minute, 2),
        # beyond end of time series
        (some_t0 + 3 * ts_30min, 2),
        (some_t0 + 100 * ts_30min, 2),
    ],
)
def test_time_series_get_closest_sample(t: datetime.datetime, expected_index: int):

    # --- arrange -----------------------------------------
    time_series = TimeSeries(some_t0, ts_30min, [0, 1, 2])

    # --- act ---------------------------------------------
    index = time_series.get_closest_index(t)

    # --- assert ------------------------------------------
    assert index == expected_index


# =================================================================================================
#  MultiTimeSeries
# =================================================================================================
@pytest.mark.parametrize(
    "items, expected_len, expected_tags",
    [
        (None, 0, set()),
        ([], 0, set()),
        (
            [("tag_a", TimeSeries(some_t0, ts_30min, [0])), ("tag_b", TimeSeries(some_t0, ts_30min, [1]))],
            2,
            {"tag_a", "tag_b"},
        ),
    ],
)
def test_multi_time_series_constructor_valid(items, expected_len, expected_tags):
    """Tests constructor with valid arguments + __len__ + tags"""

    # --- act ---------------------------------------------
    mts = MultiTimeSeries(items)

    # --- assert ------------------------------------------
    assert len(mts) == expected_len
    assert set(mts.tags()) == expected_tags


@pytest.mark.parametrize(
    "items",
    [
        # t0 mismatch
        [("tag_a", TimeSeries(some_t0, ts_30min, [0])), ("tag_b", TimeSeries(some_t0 + ts_15min, ts_30min, [1]))],
        # ts mismatch
        [("tag_a", TimeSeries(some_t0, ts_30min, [0])), ("tag_b", TimeSeries(some_t0, ts_15min, [1]))],
    ],
)
def test_multi_time_series_constructor_invalid(items):
    """Tests constructor with invalid arguments"""

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        mts = MultiTimeSeries(items)


def test_multi_time_series_copy():

    # --- arrange -----------------------------------------
    time_series_a = TimeSeries(some_t0, ts_30min, [1])
    time_series_b = TimeSeries(some_t0, ts_30min, [2])
    mts = MultiTimeSeries([("tag_1", time_series_a), ("tag_b", time_series_b)])

    # --- act ---------------------------------------------
    mts_copy = mts.copy()

    # --- assert ------------------------------------------
    assert id(mts_copy) != mts, "copy() should return a different object"
    assert len(mts) == len(mts_copy)
    assert set(mts.tags()) == set(mts_copy.tags())
    assert all(
        [time_series not in mts.series() for time_series in mts_copy.series()]
    ), "MultiTimeSeries.copy() should perform a deep copy"


def test_multi_time_series_to_array():

    # --- arrange -----------------------------------------
    time_series_a = TimeSeries(some_t0, ts_30min, [1, 2, 3])
    time_series_b = TimeSeries(some_t0, ts_30min, [10, 20, 30])
    mts = MultiTimeSeries([("tag_1", time_series_a), ("tag_b", time_series_b)])

    expected_array = np.array([[1, 2, 3], [10, 20, 30]])

    # --- act ---------------------------------------------
    array = mts.to_array()

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(array, expected_array)


def test_multi_time_series_resample():

    # --- arrange -----------------------------------------
    time_series_a = TimeSeries(some_t0, ts_30min, [1, 2, 3])
    time_series_b = TimeSeries(some_t0, ts_30min, [10, 20, 30])
    mts = MultiTimeSeries([("tag_1", time_series_a), ("tag_b", time_series_b)])

    new_ts = ts_15min
    expected_resampled_array = np.array([[1, 1.5, 2, 2.5, 3], [10, 15, 20, 25, 30]])

    # --- act ---------------------------------------------
    mts_resampled = mts.resample(new_ts)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(mts_resampled.to_array(), expected_resampled_array)


def test_multi_time_series_slice():

    # --- arrange -----------------------------------------
    time_series_a = TimeSeries(some_t0, ts_30min, [0, 1, 2, 3, 4, 5])
    time_series_b = TimeSeries(some_t0, ts_30min, [0, 10, 20, 30, 40, 50])
    mts = MultiTimeSeries([("tag_1", time_series_a), ("tag_b", time_series_b)])

    expected_sliced_array = np.array([[1, 2, 3], [10, 20, 30]])

    # --- act ---------------------------------------------
    mts_sliced = mts.slice(1, 4)

    # --- assert ------------------------------------------
    assert mts_sliced.tags() == mts.tags()
    np.testing.assert_array_equal(mts_sliced.to_array(), expected_sliced_array)


@pytest.mark.parametrize(
    # same parameters as equivalent test for TimeSeries class
    *test_time_series_get_closest_sample.pytestmark[0].args
)
def test_multi_time_series_get_closest_sample(t: datetime.datetime, expected_index: int):

    # --- arrange -----------------------------------------
    time_series_a = TimeSeries(some_t0, ts_30min, [0, 1, 2])
    time_series_b = TimeSeries(some_t0, ts_30min, [10, 11, 12])
    multi_time_series = MultiTimeSeries([("series_a", time_series_a), ("series_b", time_series_b)])

    # --- act ---------------------------------------------
    index = multi_time_series.get_closest_index(t)

    # --- assert ------------------------------------------
    assert index == expected_index
