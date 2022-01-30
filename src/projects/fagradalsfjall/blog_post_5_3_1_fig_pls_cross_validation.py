import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.tools.matplotlib import plot_style_matplotlib_default

from .evaluate_models.evaluate_forecast_models import _get_output_path


def fig_pls_cross_validation():

    # --- load CV model -----------------------------------
    base_path = _get_output_path("post_5_n_step_ahead")
    pkl_filename = os.path.join(base_path, "n-step-pls-288-288-cv_retraining_off_simdata.pkl")

    with open(pkl_filename, "rb") as f:
        model, *_ = pickle.load(f)

    # --- extract results ---------------------------------
    all_cv_results = model.cv_results["all"]

    rank_values = [result["params"]["rank"] for result in all_cv_results]

    val_mean = np.array([result["validation_losses"]["mean"] for result in all_cv_results])
    val_std = np.array([result["validation_losses"]["std"] for result in all_cv_results])

    train_mean = np.array([result["training_losses"]["mean"] for result in all_cv_results])
    train_std = np.array([result["training_losses"]["std"] for result in all_cv_results])

    # --- plot --------------------------------------------

    # prep plot
    plot_style_matplotlib_default()
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    x_values = [1 + i for i in range(len(rank_values))]

    # plotting curves
    ax.fill_between(x_values, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(x_values, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")

    h_train = ax.plot(x_values, train_mean, color="r")
    h_val = ax.plot(x_values, val_mean, color="g")

    # annotations
    i_rank_7 = rank_values.index(7)
    xy_7 = (x_values[i_rank_7], val_mean[i_rank_7])

    i_rank_12 = rank_values.index(12)
    xy_12 = (x_values[i_rank_12], val_mean[i_rank_12])

    ax.plot(xy_7[0], xy_7[1], "ko", fillstyle="none")
    ax.plot(xy_12[0], xy_12[1], "ko", fillstyle="none")

    ax.plot([xy_12[0] + 0.2, 13.5], [xy_12[1] - 0.001, 0.77], color=(0.7, 0.7, 0.7), linewidth=0.75)
    ax.plot([xy_7[0] + 0.2, 8.4], [xy_7[1] + 0.005, 0.83], color=(0.7, 0.7, 0.7), linewidth=0.75)

    ax.text(13.7, 0.77, "optimal model", ha="left", va="center", fontsize=10, color=(0.2, 0.2, 0.2))
    ax.text(
        8.5,
        0.83,
        "simplest model \nwithin 1-$\sigma$ of \noptimal model",
        ha="left",
        va="center",
        fontsize=10,
        color=(0.2, 0.2, 0.2),
    )

    # plot threshold
    threshold = val_mean[i_rank_12] + val_std[i_rank_12]
    ax.plot([-10, 100], [threshold, threshold], linestyle="--", linewidth=1, color="g", alpha=0.5)
    ax.text(len(rank_values) + 0.8, threshold + 0.002, "1-$\sigma$ threshold", ha="right", fontsize=10, color="g")

    # decoration
    ax.set_xlim(left=0, right=len(rank_values) + 1)
    ax.set_ylim(bottom=0.74, top=0.88)

    ax.set_xticks(x_values)
    ax.set_xticklabels(rank_values)

    ax.grid(visible=True)

    ax.legend([h_train[0], h_val[0]], ["mean training loss", "mean validation loss"])

    ax.set_xlabel("# of PLS components")
    ax.set_ylabel("MAE")

    fig.suptitle("Cross-validation results")
    fig.set_size_inches(w=8, h=6)
    fig.tight_layout()

    # save
    png_filename = os.path.join(base_path, "pls_cross_validation.png")
    fig.savefig(png_filename, dpi=600)

    plt.show()

    print()
