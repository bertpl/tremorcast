from darts.models import ARIMA

from .ts_model_darts import TimeSeriesModelDarts


class TimeSeriesModelDartsArima(TimeSeriesModelDarts):
    """ARIMA forecast model based on the darts package."""

    def __init__(self, signal_name: str, p: int, d: int = 0, q: int = 0):
        darts_model = ARIMA(p=p, d=d, q=q)

        if (d == 0) and (q == 0):
            model_type = f"ar-{p}"
        elif (d == 0) and (p == 0):
            model_type = f"ma-{q}"
        elif d == 0:
            model_type = f"arma-{p}-{q}"
        else:
            model_type = f"arima-{p}-{d}-{q}"

        super().__init__(model_type, signal_name, darts_model)
        self.p = p
        self.d = d
        self.q = q
