import datetime
import os
import pickle
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from src.applications.vedur_is import VedurHarmonicMagnitudes, VedurHarmonicMagnitudesGraph

from ._project_settings import FILE_DATASET_FULL, FILE_DATASET_SELECTION, FILE_DATASET_TEST, FILE_DATASET_TRAIN


# -------------------------------------------------------------------------
#  PROCESSING
# -------------------------------------------------------------------------
from ...tools.datetime import ts_to_float


def prepare_train_and_test_data():

    # --- load full dataset -------------------------------
    print("Loading dataset...    ", end="")
    with open(FILE_DATASET_FULL + ".pkl", "rb") as f:
        all_data = pickle.load(f)
    print("Done.")

    # --- time range --------------------------------------
    ts_from = datetime.datetime(2021, 7, 20, 0, 0)
    ts_to = datetime.datetime(2021, 9, 4, 0, 0)

    i_from = all_data.get_closest_index(ts_from)
    i_to = all_data.get_closest_index(ts_to)

    data_selection = all_data.slice(i_from, i_to)   # type: VedurHarmonicMagnitudes
    data_selection_extra = all_data.slice(i_from-96, i_to+96)   # type: VedurHarmonicMagnitudes

    i_train_from = 0
    i_train_to = 20*96
    i_test_from = 20*96
    i_test_to = data_selection.n_samples

    data_train = data_selection.slice(i_train_from, i_train_to)
    data_test = data_selection.slice(i_test_from, i_test_to)

    print()
    print(f"Training set : {data_train.n_samples} samples.")
    print(f"Test set     : {data_test.n_samples} samples.")

    # --- save selection ----------------------------------
    print(f"Saving DATA SELECTION to disk ...".ljust(45), end="")

    # .PKL
    with open(FILE_DATASET_SELECTION + ".pkl", "wb") as f:
        pickle.dump(data_selection, f)

    # .CSV
    data_selection.to_dataframe().to_csv(FILE_DATASET_SELECTION + ".csv")

    # .PNG
    fig, ax = data_selection_extra.create_plot(title="Fagradalsfjall (faf) - TRAINING & TEST SETS")

    x_train_from = ts_to_float(data_selection.time[i_train_from])
    x_train_to = ts_to_float(data_selection.time[i_train_to-1])
    x_test_from = ts_to_float(data_selection.time[i_test_from])
    x_test_to = ts_to_float(data_selection.time[i_test_to-1])

    train_rect = patches.Rectangle((x_train_from, 250), x_train_to-x_train_from, 6500, alpha=0.1, edgecolor=None, facecolor='green')
    ax.add_patch(train_rect)
    ax.text(x_train_from + (6*60*60), 6250, "TRAINING DATA", fontsize=16, fontweight=600)

    test_rect = patches.Rectangle((x_test_from, 250), x_test_to-x_test_from, 6500, alpha=0.1, edgecolor=None, facecolor='blue')
    ax.add_patch(test_rect)
    ax.text(x_test_from + (6*60*60), 6250, "TEST DATA", fontsize=16, fontweight=600)

    fig.savefig(FILE_DATASET_SELECTION + ".png", dpi=450)

    print("Done.")

    # --- save TRAINING data ------------------------------
    print(f"Saving TRAINING DATA to disk ...".ljust(45), end="")

    # .PKL
    with open(FILE_DATASET_TRAIN + ".pkl", "wb") as f:
        pickle.dump(data_train, f)

    # .CSV
    data_train.to_dataframe().to_csv(FILE_DATASET_TRAIN + ".csv")

    # .PNG
    fig, ax = data_train.create_plot(title="Fagradalsfjall (faf) - TRAINING SET")
    fig.savefig(FILE_DATASET_TRAIN + ".png", dpi=450)

    print("Done.")

    # --- save TEST data ------------------------------
    print(f"Saving TEST DATA to disk ...".ljust(45), end="")

    # .PKL
    with open(FILE_DATASET_TEST + ".pkl", "wb") as f:
        pickle.dump(data_test, f)

    # .CSV
    data_test.to_dataframe().to_csv(FILE_DATASET_TEST + ".csv")

    # .PNG
    fig, ax = data_test.create_plot(title="Fagradalsfjall (faf) - TEST SET")
    fig.savefig(FILE_DATASET_TEST + ".png", dpi=450)

    print("Done.")
