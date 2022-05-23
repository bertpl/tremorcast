import datetime
import sys
from math import nan

from fastai.callback.core import Callback
from fastai.imports import noop
from tqdm import tqdm

from .datetime import format_datetime, format_timedelta


# =================================================================================================
#  Fast.AI callback
# =================================================================================================
class ProgressCallbackTqdm(Callback):
    def __init__(self, extra_msg: str = "", **kwargs):
        self.extra_msg = extra_msg
        super().__init__(**kwargs)

    def before_fit(self):
        if self.extra_msg:
            desc = f"  [{datetime.datetime.now().strftime('%Y-%m-%d - %H:%M:%S')}] [{self.extra_msg}] - Training"
        else:
            desc = f"  [{datetime.datetime.now().strftime('%Y-%m-%d - %H:%M:%S')}] - Training"
        self.tqdm = tqdm(total=self.n_epoch, desc=desc, file=sys.stdout, ncols=120)

    def before_epoch(self):
        self.tqdm.update(1)

    def after_fit(self):
        self.tqdm.close()


def add_tqdm_callback(learner, enabled=True, extra_msg=""):

    try:
        if hasattr(learner, "progress"):
            learner.remove_cb(learner.progress)  # remove default progress indicator
        if hasattr(learner, "logger"):
            learner.logger = noop  # disable logging (print)
    except Exception as e:
        print(e)
        pass

    if enabled:
        learner.add_cb(ProgressCallbackTqdm(extra_msg))


def remove_tqdm_callback(learner):

    for cb in list(learner.cbs):
        if isinstance(cb, ProgressCallbackTqdm):
            learner.remove_cb(cb)


# =================================================================================================
#  Generic timer / progress estimator
# =================================================================================================
class ProgressTimer:
    def __init__(self, *, total: int = 100, auto_start: bool = True):
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
            return format_timedelta(self.eta_sec())
        else:
            return "???"

    def estimated_end_time_str(self) -> str:
        if self.eta_available():
            est_end_time = datetime.datetime.now() + datetime.timedelta(seconds=int(self.eta_sec()))
            return format_datetime(est_end_time)
        else:
            return "???"
