import matplotlib.pyplot as plt
import numpy as np

from src.base.forecasting.models.tabular.legacy.tabular_regressor_mlp import TabularRegressorMLP
from src.base.forecasting.models.tabular.tabular_regressor import ScoreMetric

from .create_dataset import DataSetType, create_dataset


def test_grid_search_cv():

    # --- training set ------------------------------------
    x_train, y_train = create_dataset(DataSetType.SINE, n=1000, c=7.0)

    # --- settings ----------------------------------------
    param_grid = {"wd": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "n_epochs": [25, 50, 100]}
    score_metric = ScoreMetric.MAE

    # --- actual cv ---------------------------------------
    mlp = TabularRegressorMLP(n_inputs=1, n_outputs=1, n_hidden_layers=3)
    mlp.cv.grid_search(x_train, y_train, param_grid, score_metric, n_jobs=-1, shuffle_data=True, n_splits=10)
    # mlp.fit(x_train, y_train)

    # --- test set & evaluate -----------------------------
    x_min = min(x_train)
    x_max = max(x_train)
    x_min, x_max = x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)

    y_min = min(y_train)
    y_max = max(y_train)
    y_min, y_max = y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)

    x_test = np.linspace(x_min, x_max, 1000)
    y_test = mlp.predict(x_test)

    # --- plot --------------------------------------------
    fig, ax = plt.subplots(1, 1)  # type: plt.Figure, plt.Axes

    # plot training data
    ax.plot(x_train, y_train, ls="", marker="x", c=(0.6, 0.6, 0.6))

    # plot predictions
    ax.plot(x_test, y_test, ls="-")

    ax.set_ylim(bottom=y_min, top=y_max)

    fig.set_size_inches(w=12, h=8)
    fig.tight_layout()

    plt.show()
