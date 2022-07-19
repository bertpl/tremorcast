"""
Module with dataset loading and saving
"""
import pickle

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.projects.fagradalsfjall._project_settings import FILE_DATASET_FULL


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
