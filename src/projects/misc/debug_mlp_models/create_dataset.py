import pickle
from typing import Tuple

import numpy as np

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.projects.fagradalsfjall.common.project_settings import FILE_DATASET_SELECTION
from src.tools.math import set_all_random_seeds


# =================================================================================================
#  Enum & main function
# =================================================================================================
def create_datasets(
    t_sine_mean: float = 200,
    t_sine_var: float = 50,
    t_drift: float = 96 * 4.7891,
    noise_std: float = 200,
    non_linearity: float = 0.05,
    random_vertical_drift: float = 500,
) -> Tuple[VedurHarmonicMagnitudes, VedurHarmonicMagnitudes]:

    # make dataset reproducible
    set_all_random_seeds(1)

    # --- load original dataset ---------------------------
    with open(FILE_DATASET_SELECTION + ".pkl", "rb") as f:
        vedur_dataset = pickle.load(f)  # type: VedurHarmonicMagnitudes

    # --- generate new fake signal ------------------------
    n_samples = vedur_dataset.n_samples

    # --- base_signal ---
    time = np.arange(0, n_samples)
    t_sine = t_sine_mean + (t_sine_var * np.sin((time / t_drift) * (2 * np.pi)))
    base_signal = np.sin(np.cumsum((2 * np.pi) / t_sine))

    def non_linear_transform(x: float):
        c = non_linearity / (1 - non_linearity)
        return np.tanh(c * x) / np.tanh(c)

    if non_linearity > 0.0:
        base_signal = non_linear_transform(base_signal)

    # --- noise ---
    base_noise = noise_std * np.random.random(size=(n_samples,))

    # --- drift ---
    c_drift = np.random.uniform(0, 1, 5)
    c_drift = c_drift / np.sum(c_drift)
    base_drift = np.zeros(n_samples)
    for c, t in zip(c_drift, [0.5, 0.75, 1.0, 1.5, 2.0]):
        t *= t_drift
        base_drift += random_vertical_drift * c * np.sin(1 + (2 * np.pi / t) * np.arange(0, n_samples))

    signal_low = 3500 + (750 * base_signal) + base_drift + base_noise
    signal_mid = 3500 + (1500 * base_signal) + base_drift + base_noise * 1.25
    signal_hi = 3500 + (2500 * base_signal) + base_drift + base_noise * 1.5

    # --- replace original with new -----------------------
    vedur_dataset.low().min.data = signal_low - noise_std
    vedur_dataset.low().mid.data = signal_low
    vedur_dataset.low().max.data = signal_low + noise_std

    vedur_dataset.mid().min.data = signal_mid - noise_std
    vedur_dataset.mid().mid.data = signal_mid
    vedur_dataset.mid().max.data = signal_mid + noise_std

    vedur_dataset.hi().min.data = signal_hi - noise_std
    vedur_dataset.hi().mid.data = signal_hi
    vedur_dataset.hi().max.data = signal_hi + noise_std

    # --- split -------------------------------------------
    return split_train_test(vedur_dataset)


# =================================================================================================
#  Generic helpers
# =================================================================================================
def split_train_test(data: VedurHarmonicMagnitudes) -> Tuple[VedurHarmonicMagnitudes, VedurHarmonicMagnitudes]:

    n_samples = data.n_samples
    n_samples_train = round(n_samples * 0.4)

    data_train = data.slice(0, n_samples_train)
    data_test = data.slice(n_samples_train, n_samples)

    return data_train, data_test
