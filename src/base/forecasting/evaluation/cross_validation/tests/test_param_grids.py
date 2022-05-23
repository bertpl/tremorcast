from typing import List, Union

import pytest

from src.base.forecasting.evaluation.cross_validation.param_grids import materialize_param_grid
from src.tools.misc import sort_any


@pytest.mark.parametrize(
    "param_grid, encapsulate_in_list, expected_result",
    [
        # regular case - with encapsulation
        (
            # input:
            {"a": [1, 10], "b": [2, 3]},
            True,
            # expected output:
            [
                {"a": [1], "b": [2]},
                {"a": [1], "b": [3]},
                {"a": [10], "b": [2]},
                {"a": [10], "b": [3]},
            ],
        ),
        # regular case - without encapsulation
        (
            # input:
            {"a": [1, 10], "b": [2, 3]},
            False,
            # expected output:
            [
                {"a": 1, "b": 2},
                {"a": 1, "b": 3},
                {"a": 10, "b": 2},
                {"a": 10, "b": 3},
            ],
        ),
        # list of dicts
        (
            # input:
            [
                {"a": [1], "b": [2, 3]},
                {"a": [10], "b": [3, 5]},
            ],
            True,
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
            True,
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
def test_materialize_param_grid(
    param_grid: Union[dict, List[dict]], encapsulate_in_list: bool, expected_result: List[dict]
):

    # --- act ---------------------------------------------
    materialized_grid = materialize_param_grid(
        param_grid, add_meta_info=False, encapsulate_param_values_in_list=encapsulate_in_list
    )

    # --- assert ------------------------------------------
    assert sort_any(materialized_grid) == sort_any(expected_result)
