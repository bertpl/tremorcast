from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Union

# =================================================================================================
#  Metadata
# =================================================================================================
CV_METADATA_PARAM = "cv_metadata"


@dataclass(frozen=True)
class CVMetaData:
    start_time: datetime.datetime
    i_param_set: int
    n_param_sets: int


# =================================================================================================
#  Parameter grid handling
# =================================================================================================
def materialize_param_grid(
    param_grid: Union[Dict, List[Dict]],
    shuffle: bool = True,
    add_meta_info: bool = True,
    encapsulate_param_values_in_list: bool = True,
) -> List[Dict]:
    """Materializes a GridSearchCV param_grid into a list of single-param-set dicts with cv-meta-info attached"""

    # --- argument handling ---------------------------
    if isinstance(param_grid, dict):
        param_grid = [param_grid]

    # --- make sure that each grid is tuple-valued ----
    # this will convert grids of following structure:
    #    {
    #       "param_a": [1, 3, 7]
    #       ("param_b", "param_c"): [(0.1, 0.5), (0.2, 0.7), (0.3, 0.8)]
    #    }
    # to grids of following structure:
    #    {
    #       ("param_a",): [(1,), (3,), (7,)]
    #       ("param_b", "param_c"): [(0.1, 0.5), (0.2, 0.7), (0.3, 0.8)]
    #    }
    param_grid = [
        {
            param
            if isinstance(param, tuple)
            else (param,): [value if isinstance(value, tuple) else (value,) for value in param_values]
            for param, param_values in one_grid.items()
        }
        for one_grid in param_grid
    ]

    # --- materialize ---------------------------------
    param_set_list = [
        {
            param: value
            for param_tuple, param_value_tuple in param_set.items()
            for param, value in zip(param_tuple, param_value_tuple)
        }
        for param_set in [
            {param_name: param_value for param_name, param_value in zip(one_grid.keys(), param_values)}
            for one_grid in param_grid
            for param_values in product(*one_grid.values())
        ]
    ]

    # --- shuffle -------------------------------------
    if shuffle:
        # mainly intended to make sure slow-fitting vs fast-fitting parameter sets are evenly spread out,
        #  so our time estimates are more reliable and less skewed.
        random.shuffle(param_set_list)

    # --- add meta-info -------------------------------
    if add_meta_info:
        now = datetime.datetime.now()
        n_param_sets = len(param_set_list)
        for i_param_set, param_set in enumerate(param_set_list):
            param_set[CV_METADATA_PARAM] = CVMetaData(now, i_param_set, n_param_sets)

    # --- optionally encapsulate in lists -------------
    if encapsulate_param_values_in_list:
        # sklearn's GridSearchCV expects it like this.  Our own internal grid search cv implementations do not.
        param_set_list = [
            {param_name: [param_value] for param_name, param_value in param_set.items()} for param_set in param_set_list
        ]

    # --- return --------------------------------------
    return param_set_list
