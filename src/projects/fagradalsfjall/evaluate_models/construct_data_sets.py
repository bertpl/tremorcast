import pickle

import pandas as pd

from src.applications.vedur_is import VedurHarmonicMagnitudes


def get_dataset_train() -> VedurHarmonicMagnitudes:

    from src.projects.fagradalsfjall._project_settings import FILE_DATASET_TRAIN

    with open(FILE_DATASET_TRAIN + ".pkl", "rb") as f:
        data_train = pickle.load(f)  # type: VedurHarmonicMagnitudes

    return data_train


def get_dataset_test() -> VedurHarmonicMagnitudes:

    from src.projects.fagradalsfjall._project_settings import FILE_DATASET_TEST

    with open(FILE_DATASET_TEST + ".pkl", "rb") as f:
        data_test = pickle.load(f)  # type: VedurHarmonicMagnitudes

    return data_test
