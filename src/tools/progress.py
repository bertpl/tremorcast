import datetime
import sys
from math import nan
from typing import Tuple

from fastai.callback.core import Callback
from fastai.imports import noop
from tqdm import tqdm


# =================================================================================================
#  Fast.AI callback
# =================================================================================================
class ProgressCallbackTqdm(Callback):
    def before_fit(self):
        desc = f"  [{datetime.datetime.now().strftime('%Y-%m-%d - %H:%M:%S')}] - Training"
        self.tqdm = tqdm(total=self.n_epoch, desc=desc, file=sys.stdout, ncols=120)

    def before_epoch(self):
        self.tqdm.update(1)

    def after_fit(self):
        self.tqdm.close()


def add_tqdm_callback(learner, enabled=True):

    try:
        if hasattr(learner, "progress"):
            learner.remove_cb(learner.progress)  # remove default progress indicator
        if hasattr(learner, "logger"):
            learner.logger = noop  # disable logging (print)
    except Exception as e:
        print(e)
        pass

    if enabled:
        learner.add_cb(ProgressCallbackTqdm)


def remove_tqdm_callback(learner):

    for cb in list(learner.cbs):
        if isinstance(cb, ProgressCallbackTqdm):
            learner.remove_cb(cb)


# =================================================================================================
#  Generic timer / progress estimator
# =================================================================================================
class ProgressTimer:
    def __init__(self, *, total: int, auto_start: bool = True):
        self.total = total
        self.progress = None
        self.t_start = None
        if auto_start:
            self.start()

    def start(self):
        self.t_start = datetime.datetime.now()

    def update_progress(self, *, progress: int):
        """progress: 0 if 1st iteration is done; total-1 if it's completely done"""
        self.progress = progress

    def iter_done(self, n: int):
        if self.progress is None:
            self.progress = n - 1
        else:
            self.progress += n

    def sec_elapsed(self) -> float:
        if self.t_start is not None:
            return (datetime.datetime.now() - self.t_start).total_seconds()
        else:
            return 0.0

    def eta_available(self) -> bool:
        return (
            (self.t_start is not None)
            and (self.progress is not None)
            and (self.progress >= 0)
            and (self.progress < self.total)
        )

    def eta_sec(self) -> float:
        if self.eta_available():

            iter_done = self.progress + 1
            iter_todo = self.total - iter_done

            return self.sec_elapsed() * (iter_todo / iter_done)

        else:
            return nan

    def eta_str(self) -> str:
        if self.eta_available():
            d, h, m, s = self._split_sec(self.eta_sec())
            if d + h + m == 0:
                if s < 10:
                    return f"{s:.2f}s"
                else:
                    return f"{s:.1f}s"
            elif d + h == 0:
                return f"{m}m{s:.0f}s"
            elif d == 0:
                return f"{h}h{m}m{s:.0f}s"
            else:
                return f"{d}d{h}h{m}m{s:.0f}s"
        else:
            return "???"

    def estimated_end_time_str(self) -> str:
        if self.eta_available():
            est_end_time = datetime.datetime.now() + datetime.timedelta(seconds=int(self.eta_sec()))
            return est_end_time.strftime("%a - %Y-%m-%d - %H:%M:%S")
        else:
            return "???"

    @staticmethod
    def _split_sec(sec: float) -> Tuple[int, int, int, float]:
        """split total seconds in (days_int, hours_int, minutes_int, secs_float)"""
        d = int(sec // (24 * 60 * 60))
        sec -= d * 24 * 60 * 60

        h = int(sec // (60 * 60))
        sec -= h * 60 * 60

        m = int(sec // 60)
        sec -= m * 60

        return d, h, m, sec
