"""
Module with dataset loading and saving
"""
import pickle

import numpy as np

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.projects.fagradalsfjall.common.project_settings import (
    FILE_DATASET_FULL,
    FILE_DATASET_TEST,
    FILE_DATASET_TRAIN,
    FORECAST_SIGNAL_NAME,
)


# -------------------------------------------------------------------------
#  Save / Load .pkl
# -------------------------------------------------------------------------
def save_dataset_pickle(data: VedurHarmonicMagnitudes):

    with open(FILE_DATASET_FULL + ".pkl", "wb") as f:
        pickle.dump(data, f)


def load_dataset_pickle() -> VedurHarmonicMagnitudes:

    with open(FILE_DATASET_FULL + ".pkl", "rb") as f:
        data = pickle.load(f)

    return data


# -------------------------------------------------------------------------
#  Save / Load .csv
# -------------------------------------------------------------------------
def save_dataset_csv(data: VedurHarmonicMagnitudes):
    data.to_dataframe().to_csv(FILE_DATASET_FULL + ".csv")


# -------------------------------------------------------------------------
#  Save / Load TRAIN & TEST data
# -------------------------------------------------------------------------
def load_train_data_vedur() -> VedurHarmonicMagnitudes:
    with open(FILE_DATASET_TRAIN + ".pkl", "rb") as f:
        train_data = pickle.load(f)  # type: VedurHarmonicMagnitudes

    return train_data


def load_test_data_vedur() -> VedurHarmonicMagnitudes:
    with open(FILE_DATASET_TEST + ".pkl", "rb") as f:
        test_data = pickle.load(f)  # type: VedurHarmonicMagnitudes

    return test_data


def load_train_data_numpy() -> np.ndarray:
    return load_train_data_vedur().to_dataframe()[FORECAST_SIGNAL_NAME].to_numpy().flatten()


def load_test_data_numpy() -> np.ndarray:
    return load_test_data_vedur().to_dataframe()[FORECAST_SIGNAL_NAME].to_numpy().flatten()
