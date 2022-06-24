import itertools
from enum import Enum, auto
from typing import List, Optional

from src.base.forecasting.models import FeatureSelector, LrMaxCriterion


# =================================================================================================
#  Sweep definitions & properties
# =================================================================================================
class SweepType(Enum):

    SWEEP_1D = auto()
    SWEEP_2D = auto()
    GRID = auto()


class Sweep(Enum):

    # -------------------------------------------------------------------------
    #  Members
    # -------------------------------------------------------------------------
    N_EPOCHS_LR_MAX_VALLEY = auto()
    N_EPOCHS_LR_MAX_INTERMEDIATE = auto()
    N_EPOCHS_LR_MAX_MINIMUM = auto()
    N_EPOCHS_LR_MAX_AGGRESSIVE = auto()
    N_EPOCHS_WD_LO = auto()
    N_EPOCHS_WD_HI = auto()
    N_EPOCHS_SHALLOW = auto()
    N_EPOCHS_DEEP = auto()
    WD = auto()
    DROPOUT = auto()
    # REGULARIZATION_1D = auto()
    REGULARIZATION_2D = auto()
    LAGS = auto()
    LAYER_WIDTH = auto()
    N_LAYERS = auto()

    # -------------------------------------------------------------------------
    #  Properties
    # -------------------------------------------------------------------------
    def lower_name(self):
        return self.name.lower()

    def sweep_type(self) -> SweepType:
        if self in [Sweep.REGULARIZATION_2D]:
            return SweepType.SWEEP_2D
        else:
            return SweepType.SWEEP_1D

    def sub_title(self) -> Optional[str]:
        if self in [Sweep.N_EPOCHS_WD_LO, Sweep.N_EPOCHS_WD_HI]:
            wd = self.sweep_param_grid()["wd"]
            return f"wd={wd:.1f}"
        elif self in [
            Sweep.N_EPOCHS_LR_MAX_VALLEY,
            Sweep.N_EPOCHS_LR_MAX_INTERMEDIATE,
            Sweep.N_EPOCHS_LR_MAX_MINIMUM,
            Sweep.N_EPOCHS_LR_MAX_AGGRESSIVE,
        ]:
            lr_max = self.sweep_param_grid()["lr_max"]  # type: LrMaxCriterion
            return f"lr_max='{lr_max.name.lower()}'"
        elif self in [Sweep.N_EPOCHS_SHALLOW, Sweep.N_EPOCHS_DEEP]:
            n_hidden_layers = self.sweep_param_grid()["n_hidden_layers"]
            return f"n_hidden_layers='{n_hidden_layers}'"
        else:
            return None

    def sweep_param_grid(self) -> dict:
        """Only the param values that deviate from the nominal"""

        if self.lower_name().startswith("n_epochs"):

            param_grid = {"n_epochs": [1, 2, 5, 10, 20, 50, 100, 200, 500]}

            if self == Sweep.N_EPOCHS_WD_LO:
                param_grid["wd"] = [0.0]
            elif self == Sweep.N_EPOCHS_WD_HI:
                param_grid["wd"] = [1.0]
            elif self == Sweep.N_EPOCHS_LR_MAX_VALLEY:
                param_grid["lr_max"] = [LrMaxCriterion.VALLEY]
            elif self == Sweep.N_EPOCHS_LR_MAX_INTERMEDIATE:
                param_grid["lr_max"] = [LrMaxCriterion.INTERMEDIATE]
            elif self == Sweep.N_EPOCHS_LR_MAX_MINIMUM:
                param_grid["lr_max"] = [LrMaxCriterion.MINIMUM]
            elif self == Sweep.N_EPOCHS_LR_MAX_AGGRESSIVE:
                param_grid["lr_max"] = [LrMaxCriterion.AGGRESSIVE]
            elif self == Sweep.N_EPOCHS_SHALLOW:
                param_grid["n_hidden_layers"] = [1]
            elif self == Sweep.N_EPOCHS_DEEP:
                param_grid["n_hidden_layers"] = [10]
            else:
                raise NotImplementedError(f"unknown n_epochs sweep: {self}")

            return param_grid

        elif self == Sweep.WD:
            return {"wd": sorted([a * b for a, b in itertools.product([1, 2, 5], [0.001, 0.01, 0.1, 1, 10])])}
        elif self == Sweep.DROPOUT:
            return {"dropout": [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]}
        elif self == Sweep.REGULARIZATION_2D:
            return {"wd": [0.0, 0.001, 0.01, 0.1, 1, 10], "dropout": [0.0, 0.2, 0.4, 0.6, 0.8]}
        elif self == Sweep.LAGS:
            return {
                "feature_selector": [
                    [FeatureSelector.first(n_inputs) for n_inputs in [4, 8, 16, 32, 64, 128, 192, 288]]
                    + [
                        FeatureSelector.exp_spaced(first_index=0, last_index=horizon - 1, n_features=16)
                        for horizon in [32, 64, 128, 192, 288]
                    ]
                ]
            }
        elif self == Sweep.LAYER_WIDTH:
            return {"layer_width": [10, 20, 50, 75, 100, 150, 200, 500, 1000]}
        elif self == Sweep.N_LAYERS:
            return {"n_hidden_layers": [1, 2, 3, 4, 6, 8, 10]}
        else:
            raise NotImplementedError(f"sweep name '{self}' not implemented.")

    def get_param_names(self) -> List[str]:
        param_grid = self.sweep_param_grid()
        param_names = []

        for param_name, param_values in param_grid:
            if len(param_values) > 1:
                if isinstance(param_name, str):
                    param_names.append(param_name)
                else:
                    param_names.extend(list(param_name))

        return param_names


def get_nominal_parameters(n: int) -> dict:

    # --- return final defaults ---------------------------
    return {
        "feature_selector": [FeatureSelector.first(16) if (n <= 16) else FeatureSelector.exp_spaced(0, n - 1, 16)],
        "n_hidden_layers": [3],
        "layer_width": [100],
        "wd": [0.1],
        "dropout": [0.2],
        "n_epochs": [50],
        "lr_max": [LrMaxCriterion.AGGRESSIVE],
    }
