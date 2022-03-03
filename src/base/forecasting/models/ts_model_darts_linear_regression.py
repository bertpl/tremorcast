import numpy as np
import pandas as pd
from darts.models import RegressionModel
from darts.timeseries import TimeSeries
from sklearn.linear_model import Ridge

from .ts_model import TimeSeriesForecastModelAutoScaled

# --- AR regularization ---
#
# Exponentially increasing L2 regularization weights:
#
# [1] https://uwspace.uwaterloo.ca/bitstream/handle/10012/3766/Bei'finalthesis.pdf?sequence=1
#
# [2] Gel, Y. R. and Barabanov, A. (2007). “Strong consistency of the regularized least-squares
#     estimates of infinite autoregressive models”.Journal of Statistical Planning and Inference,
#     137, 1260-1277


class TimeSeriesModelDartsLinearRegression(TimeSeriesForecastModelAutoScaled):
    """Linear regression model based on the darts package."""

    def __init__(self, signal_name: str, p: int, alpha: float = 0.0):
        if alpha == 0.0:
            model_type = f"ar-{p}"
        else:
            model_type = f"ar-{p}-{alpha:.1e}"
        super().__init__(model_type, signal_name)
        self.p = p
        self.model = RegressionModel(lags=p, model=Ridge(alpha=alpha, fit_intercept=False))

    def _fit(self, training_data: pd.DataFrame):
        ts_train = TimeSeries.from_series(training_data[self.signal_name])
        self.model.fit(ts_train)
        print()

    def _predict(self, history: pd.DataFrame, n_samples: int) -> np.ndarray:
        ts_history = TimeSeries.from_series(history[self.signal_name])
        ts_forecast = self.model.predict(n=n_samples, series=ts_history)
        return ts_forecast.data_array().to_numpy().flatten()
