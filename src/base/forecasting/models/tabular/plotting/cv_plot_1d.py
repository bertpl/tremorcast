from typing import List, Tuple
from enum import Enum, auto

from src.tools.misc import sort_any


class ErrorBounds(Enum):
    STDEV = auto()
    QUARTILES = auto()

MAX_LINEAR_RANGE = 15


class CrossValidationPlot1D:

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, param_names: List[str], data: List[Tuple[Tuple, "CVResult"]]):

        # --- arguments -----------------------------------
        self.param_names = param_names
        self.data = data
