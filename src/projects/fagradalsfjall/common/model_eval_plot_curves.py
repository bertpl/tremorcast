import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.projects.fagradalsfjall.common.project_settings import CV_HORIZON_SAMPLES, CV_METRIC_RMSE_THRESHOLD
from src.tools.matplotlib import plot_style_matplotlib_default

from .model_eval import ModelEvalResult, ValidationType


# =================================================================================================
#  Main plotting functions
# =================================================================================================
def generate_all_rmse_plots(
    model_eval_results: Dict[str, ModelEvalResult],
    folder: Union[Path, str],
    restrict_single_model_plots_to_validation_types: List[ValidationType] = None,
    highlight_models: List[str] = None,
    plot_rmse_threshold: bool = True,
):

    # --- argument processing -----------------------------
    if isinstance(folder, str):
        folder = Path(folder)

    os.makedirs(folder, exist_ok=True)

    # --- single-model-plots ------------------------------
    for model_name, model_eval_result in model_eval_results.items():
        if (highlight_models is None) or (model_name in highlight_models):
            fig, ax = plot_rmse_curves_single_model(
                model_name,
                model_eval_result,
                restrict_single_model_plots_to_validation_types,
                plot_rmse_threshold=plot_rmse_threshold,
            )
            fig.savefig(folder / f"curves_single_model_{model_name.replace('-', '_')}.png", dpi=600)

    # --- multi-model plots -------------------------------
    for validation_type in ValidationType:
        fig, ax = plot_rmse_curves_multi_model(
            model_eval_results,
            validation_type,
            highlight_models=highlight_models,
            plot_rmse_threshold=plot_rmse_threshold,
        )
        fig.savefig(folder / f"curves_multi_model_{validation_type.name.lower()}.png", dpi=600)


def plot_rmse_curves_single_model(
    model_name: str,
    model_eval_result: ModelEvalResult,
    restrict_to_validation_types: List[ValidationType] = None,
    plot_rmse_threshold: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots RMSE curves for a single model, but potentially multiple validation types.
    :param model_name: (str) name of the model for which we're creating a plot
    :param model_eval_result: ModelEvalResult for this model
    :param restrict_to_validation_types: if provided, plotting is restricted to these validation types
    :param plot_rmse_threshold: (default=True) plot horizontal line representing RMSE threshold
    :return: (fig, ax)-tuple.
    """

    return _plot_rmse_curves(
        curves={
            validation_type.get_display_name(): metric_curve
            for validation_type, metric_curve in model_eval_result.metric_curves.items()
            if restrict_to_validation_types is None or validation_type in restrict_to_validation_types
        },
        title=f"RMSE curves - model '{model_name}'",
        plot_rmse_threshold=plot_rmse_threshold,
    )


def plot_rmse_curves_multi_model(
    model_eval_results: Dict[str, ModelEvalResult],
    validation_type: ValidationType,
    highlight_models: List[str] = None,
    plot_rmse_threshold: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots RMSE curves for multiple models, for 1 validation type.
    :param model_eval_results: dict mapping model_name --> ModelEvalResult object
    :param validation_type: validation type to be plotted
    :param highlight_models: (list of str; default None) models to highlight
    :param plot_rmse_threshold: (default=True) plot horizontal line representing RMSE threshold
    :return: (fig, ax)-tuple.
    """

    return _plot_rmse_curves(
        curves={
            model_name: model_eval_result.metric_curves[validation_type]
            for model_name, model_eval_result in model_eval_results.items()
        },
        title=f"RMSE curves - {validation_type.get_display_name()}",
        highlight_curves=highlight_models,
        plot_rmse_threshold=plot_rmse_threshold,
    )


# =================================================================================================
#  Internal
# =================================================================================================
def _plot_rmse_curves(
    curves: Dict[str, np.ndarray], title: str, highlight_curves: List[str] = None, plot_rmse_threshold: bool = True
) -> Tuple[plt.Figure, plt.Axes]:

    plot_style_matplotlib_default()

    # --- argument handling -------------------------------
    if highlight_curves is None:
        highlight_curves = []

    # --- init --------------------------------------------
    y_max = max(max([max(curve), 1.2 * curve[0]]) for curve in curves.values())

    # --- prep data ---------------------------------------
    # all_curves = np.concatenate(
    #     [curve[:CV_HORIZON_SAMPLES].reshape(1, CV_HORIZON_SAMPLES) for curve in curves.values()], axis=0
    # )
    x_values = np.log(np.arange(1, CV_HORIZON_SAMPLES + 1).reshape(1, CV_HORIZON_SAMPLES))

    # --- create plot -------------------------------------
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    fig.suptitle(title)

    for curve_name, curve in curves.items():
        if curve_name in highlight_curves:
            kwargs = dict(lw=2.5)
        else:
            kwargs = dict()

        ax.plot(x_values.flatten(), curve.flatten(), **kwargs)

    if plot_rmse_threshold:
        ax.plot([-1, 1000], [CV_METRIC_RMSE_THRESHOLD, CV_METRIC_RMSE_THRESHOLD], ls="--", c="grey", lw=1)

    # --- finalize layout ---------------------------------
    ax.set_ylim(bottom=0, top=y_max * 1.1)
    ax.set_xlim(left=0, right=np.log(CV_HORIZON_SAMPLES))

    if y_max > 2000:
        y_step = 500
    elif y_max > 1000:
        y_step = 200
    else:
        y_step = 100
    ax.set_yticks(np.arange(0, y_max + y_step, y_step))

    x_ticks, x_labels = _get_x_ticks_and_labels()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.grid(visible=True, axis="x")

    ax.legend(list(curves.keys()), loc="upper left")

    ax.set_xlabel("Forecast Lead Time")
    ax.set_ylabel("RMSE")

    fig.set_size_inches(w=9, h=6)
    fig.tight_layout()

    # --- return ------------------------------------------
    return fig, ax


def _get_x_ticks_and_labels() -> Tuple[np.ndarray, List[str]]:

    # --- all ticks & labels ------------------------------
    all_ticks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 72, 96, 1.5 * 96, 2 * 96, 3 * 96, 4 * 96]
    all_labels = ["15m", "30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "18h", "24h", "36h", "2d", "3d", "4d"]

    # --- filter ------------------------------------------
    all_ticks = [tick for tick in all_ticks if tick <= CV_HORIZON_SAMPLES]
    all_labels = all_labels[: len(all_ticks)]

    # --- return ------------------------------------------
    return np.log(all_ticks), all_labels
