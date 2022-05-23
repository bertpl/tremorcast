import datetime
from typing import Tuple, Union

__TS_REF = datetime.datetime(2020, 1, 1, 0, 0, 0)


# -------------------------------------------------------------------------
#  Conversion
# -------------------------------------------------------------------------
def ts_to_float(ts: Union[datetime.datetime, datetime.date]) -> float:
    if not isinstance(ts, datetime.datetime):
        ts = datetime.datetime.combine(ts, datetime.time(0, 0, 0))
    return (ts - __TS_REF).total_seconds()


def float_to_ts(f: float) -> datetime.datetime:
    return __TS_REF + datetime.timedelta(seconds=f)


# -------------------------------------------------------------------------
#  ETA
# -------------------------------------------------------------------------
def estimate_eta(start_time: datetime.datetime, work_fraction_done: float) -> Tuple[datetime.datetime, float]:
    """Estimate (eta_time, secs_to_go) based on start_time and fraction of work done"""

    if (work_fraction_done is None) or (work_fraction_done < 0) or (work_fraction_done > 1):
        work_fraction_done = 0.5

    secs_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    secs_total = secs_elapsed / work_fraction_done
    secs_to_go = secs_total * (1 - work_fraction_done)
    eta_time = start_time + datetime.timedelta(seconds=secs_total)

    return eta_time, secs_to_go


# -------------------------------------------------------------------------
#  Formatting
# -------------------------------------------------------------------------
def format_datetime(dt: datetime.datetime) -> str:
    if dt is not None:
        return dt.strftime("%a - %Y-%m-%d - %H:%M:%S")
    else:
        return "???"


def format_timedelta(total_seconds: float) -> str:
    """Format time delta in string with adaptive precision.  Result is max 6 chars long."""
    if total_seconds is not None:
        d, h, m, s = _split_sec(total_seconds)
        if d + h + m == 0:
            if s < 10:
                return f"{s:.2f}s"
            else:
                return f"{s:.1f}s"
        elif d + h == 0:
            return f"{m}m{s:.0f}s"
        elif d == 0:
            return f"{h}h{m}m"
        else:
            return f"{d}d{h}h"
    else:
        return "???"


def _split_sec(sec: float) -> Tuple[int, int, int, float]:
    """split total seconds in (days_int, hours_int, minutes_int, secs_float)"""
    d = int(sec // (24 * 60 * 60))
    sec -= d * 24 * 60 * 60

    h = int(sec // (60 * 60))
    sec -= h * 60 * 60

    m = int(sec // 60)
    sec -= m * 60

    return d, h, m, sec
