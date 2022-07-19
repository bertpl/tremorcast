from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .create_dataset import DataSetType, create_dataset
from .regression import LinearRegressor, NeuralRegressor, Regressor


def test_regression():

    # --- training set ------------------------------------
    x_train, y_train = create_dataset(DataSetType.SINE, n=1000, c=5.0)

    dataset_kwargs = dict(n_features=x_train.shape[1], n_targets=y_train.shape[1])
    other_params = dict(n_hidden_layers=5, layer_width=100, n_epochs=100, lr_max="minimum")

    # --- regressors --------------------------------------
    regressors = [
        # LinearRegressor(**dataset_kwargs, name="lin"),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_0", wd=0.0),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_1", wd=1),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_10", wd=10),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_100", wd=100),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_100", wd=200),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_100", wd=500),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_1_000", wd=1_000),
        NeuralRegressor(**dataset_kwargs, **other_params, name="nn-wd_10_000", wd=10_000),
    ]  # type: List[Regressor]

    # --- train -------------------------------------------
    for regressor in regressors:
        print(f"Training regressor '{regressor.name}'...")
        regressor.train(x_train, y_train)

    # --- test set & evaluate -----------------------------
    x_min = min(x_train)
    x_max = max(x_train)
    x_min, x_max = x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)

    y_min = min(y_train)
    y_max = max(y_train)
    y_min, y_max = y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)

    x_test = np.linspace(x_min, x_max, 1000)
    y_tests = [regressor.predict(x_test) for regressor in regressors]

    # --- plot --------------------------------------------
    fig, ax = plt.subplots(1, 1)  # type: plt.Figure, plt.Axes

    # plot training data
    ax.plot(x_train, y_train, ls="", marker="x", c=(0.6, 0.6, 0.6))

    # plot predictions
    for y_test, regressor in zip(y_tests, regressors):
        ax.plot(x_test, y_test, ls="-")

    legend_names = ["training set"] + [f"model - {regressor.name}" for regressor in regressors]

    ax.legend(legend_names)

    ax.set_ylim(bottom=y_min, top=y_max)

    fig.set_size_inches(w=12, h=8)
    fig.tight_layout()

    plt.show()
