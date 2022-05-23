import datetime
import os
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.projects.fagradalsfjall.common.model_eval import ModelEvalResult
from src.projects.fagradalsfjall.common.model_eval_plot_curves import plot_rmse_curves_single_model
from src.projects.fagradalsfjall.common.model_eval_plot_simulations import plot_simulation
from src.projects.fagradalsfjall.common.project_settings import PATH_MODEL_REPO


# =================================================================================================
#  Model Repository
# =================================================================================================
class ModelRepo:

    # -------------------------------------------------------------------------
    #  Single Models
    # -------------------------------------------------------------------------
    @classmethod
    def save_model(cls, model_id: str, model_eval_result: ModelEvalResult):

        model_path = cls._get_model_path(model_id, for_save=True)

        # --- save data -----------------------------------
        with open(model_path / "data.pkl", "wb") as f:
            pickle.dump(model_eval_result, f)

        # --- create & save figures -----------------------
        fig, ax = plot_rmse_curves_single_model(model_id, model_eval_result)
        fig.savefig(model_path / "rmse_curves.png", dpi=600)
        plt.close(fig)

        fig, ax = plot_simulation(model_id, model_eval_result)
        fig.savefig(model_path / "simulation.png", dpi=600)
        plt.close(fig)

        # --- report --------------------------------------
        print(f"Saved model result for '{model_id}'".ljust(60) + f"[{datetime.datetime.now()}]")

    @classmethod
    def load_model(cls, model_id: str) -> ModelEvalResult:
        """Returns ModelEvalResult object based on model_id."""

        model_path = cls._get_model_path(model_id, for_load=True)

        with open(model_path / "data.pkl", "rb") as f:
            model_eval_result = pickle.load(f)

        return model_eval_result

    # -------------------------------------------------------------------------
    #  Multiple Models
    # -------------------------------------------------------------------------
    @classmethod
    def save_models(cls, model_eval_results: Dict[str, ModelEvalResult]):
        """Saves all models"""
        for model_id, model_eval_result in model_eval_results.items():
            cls.save_model(model_id, model_eval_result)

    @classmethod
    def load_models(cls, model_ids: List[str]) -> Dict[str, ModelEvalResult]:
        """
        Returns a dict mapping model_id to ModelEvalResult object
        """
        return {model_id: cls.load_model(model_id) for model_id in model_ids}

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_model_path(model_id: str, *, for_load: bool = False, for_save: bool = False) -> Path:

        path = Path(PATH_MODEL_REPO) / model_id

        if for_save:
            os.makedirs(path, exist_ok=True)
        if for_load:
            if not path.is_dir():
                raise FileNotFoundError(f"Model with id '{model_id}' not loadable from model repo.")

        return path
