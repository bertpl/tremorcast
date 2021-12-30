import datetime
from typing import Union

__TS_REF = datetime.datetime(2020, 1, 1, 0, 0, 0)


def ts_to_float(ts: Union[datetime.datetime, datetime.date]) -> float:
    if not isinstance(ts, datetime.datetime):
        ts = datetime.datetime.combine(ts, datetime.time(0, 0, 0))
    return (ts - __TS_REF).total_seconds()


def float_to_ts(f: float) -> datetime.datetime:
    return __TS_REF + datetime.timedelta(seconds=f)
