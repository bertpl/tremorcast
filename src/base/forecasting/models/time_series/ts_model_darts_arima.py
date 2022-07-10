from __future__ import annotations

from darts.models import ARIMA

from .ts_model_darts import TimeSeriesModelDarts


class TimeSeriesModelDartsArima(TimeSeriesModelDarts):
    """ARIMA forecast model based on the darts package."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, p: int, d: int = 0, q: int = 0):
        # p = order of AR-part, d = order of I-part, q = order of MA-part.

        super().__init__("", None)

        self.p = p
        self.d = d
        self.q = q
        self._update_darts_model()
        self._update_model_name()

    # -------------------------------------------------------------------------
    #  Parameter handling
    # -------------------------------------------------------------------------
    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self._update_darts_model()
        self._update_model_name()

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _update_darts_model(self):
        self.darts_model = ARIMA(p=self.p, d=self.d, q=self.q)

    def _update_model_name(self):

        if (self.d == 0) and (self.q == 0):
            name = f"ar-{self.p}"
        elif (self.d == 0) and (self.p == 0):
            name = f"ma-{self.q}"
        elif self.d == 0:
            name = f"arma-{self.p}-{self.q}"
        else:
            name = f"arima-{self.p}-{self.d}-{self.q}"

        self.name = name
