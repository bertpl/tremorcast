"""Plot ACF & PACF and interpret results."""

import os
import pickle

import matplotlib.pyplot as plt
from darts.timeseries import TimeSeries
from darts.utils.statistics import plot_acf, plot_pacf, stationarity_test_adf, stationarity_test_kpss

from src.applications.vedur_is import VedurHarmonicMagnitudes
from src.projects.fagradalsfjall._project_settings import (
    FILE_DATASET_TEST,
    FILE_DATASET_TRAIN,
    FORECAST_MAE_THRESHOLD,
    FORECAST_SIGNAL_NAME,
    PATH_RESULTS,
)
from src.tools.matplotlib import plot_style_darts


def linear_data_analysis():

    # --- load data ---------------------------------------
    with open(FILE_DATASET_TRAIN + ".pkl", "rb") as f:
        data_train = pickle.load(f)  # type: VedurHarmonicMagnitudes
    df_train = data_train.to_dataframe()

    ts_train = series = TimeSeries.from_series(df_train[FORECAST_SIGNAL_NAME])

    # --- prep output folder ------------------------------
    results_sub_folder = "post_4_linear_models"
    output_path = os.path.join(PATH_RESULTS, results_sub_folder)
    os.makedirs(output_path, exist_ok=True)

    # --- stationarity tests ------------------------------

    # p_value < 0.05 indicates the time series IS stationary
    adf_stat, adf_p_value, *na = stationarity_test_adf(
        ts=ts_train,
        maxlag=4 * 4 * 24,  # 4 days
    )
    print(f"ADF stationarity p-value: {adf_p_value}")

    # p_value < 0.05 indicates the time series IS NOT stationary
    kpss_stat, kpss_p_value, *na = stationarity_test_kpss(
        ts=ts_train,
        nlags=4 * 4 * 24,  # 4 days
    )
    print(f"KPSS stationarity p-value: {kpss_p_value}")

    # --- ACF & PACF --------------------------------------

    plot_style_darts()

    # ACF
    plot_acf(ts_train, max_lag=4 * 24)

    png_file_name = os.path.join(output_path, "acf.png")
    fig = plt.gcf()  # type: plt.Figure
    fig.tight_layout()
    fig.savefig(png_file_name, dpi=600)

    # PACF
    plot_pacf(ts_train, max_lag=4 * 24)

    png_file_name = os.path.join(output_path, "pacf.png")
    fig = plt.gcf()  # type: plt.Figure
    fig.tight_layout()
    fig.savefig(png_file_name, dpi=600)
