from typing import List, Union

import pytest

from src.base.forecasting.evaluation.cross_validation.param_grids import materialize_param_grid
from src.tools.misc import sort_any


@pytest.mark.parametrize(
    "param_grid, expected_result",
    [
        # regular case
        (
            # input:
            {"a": [1, 10], "b": [2, 3]},
            # expected output:
            [
                {"a": [1], "b": [2]},
                {"a": [1], "b": [3]},
                {"a": [10], "b": [2]},
                {"a": [10], "b": [3]},
            ],
        ),
        # list of dicts
        (
            # input:
            [
                {"a": [1], "b": [2, 3]},
                {"a": [10], "b": [3, 5]},
            ],
            # expected output:
            [
                {"a": [1], "b": [2]},
                {"a": [1], "b": [3]},
                {"a": [10], "b": [3]},
                {"a": [10], "b": [5]},
            ],
        ),
        # tuple-valued dicts
        (
            # input:
            [
                {("a", "b"): [(1, 2), (1, 3)]},
                {("a", "b"): [(10, 3), (10, 5)]},
            ],
            # expected output:
            [
                {"a": [1], "b": [2]},
                {"a": [1], "b": [3]},
                {"a": [10], "b": [3]},
                {"a": [10], "b": [5]},
            ],
        ),
    ],
)
def test_materialize_param_grid(param_grid: Union[dict, List[dict]], expected_result: List[dict]):

    # --- act ---------------------------------------------
    materialized_grid = materialize_param_grid(param_grid, add_meta_info=False)

    # --- assert ------------------------------------------
    assert sort_any(materialized_grid) == sort_any(expected_result)
