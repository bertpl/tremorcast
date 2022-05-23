import itertools

import pytest

from src.base.forecasting.evaluation.cross_validation.cv_splits import TimeSeriesCVSplitter
from src.tools.testing import all_values_unique, is_sorted


@pytest.mark.parametrize("n_splits, n_samples_tot", itertools.product([1, 5, 10, 100], [1_000, 2_000, 5_000, 10_000]))
def test_time_series_cv_splitter(n_splits: int, n_samples_tot: int):

    # --- arrange -----------------------------------------
    min_samples_train = 100
    min_samples_validate = 50
    cv_splitter = TimeSeriesCVSplitter(
        min_samples_train=min_samples_train, min_samples_validate=min_samples_validate, n_splits=n_splits
    )

    # --- act ---------------------------------------------
    splits = cv_splitter.get_splits(n_samples_tot=n_samples_tot)

    # --- assert ------------------------------------------
    assert len(splits) == n_splits

    all_n_train = [n_train for n_train, n_val in splits]
    all_n_val = [n_val for n_train, n_val in splits]

    # all n_train values should be unique and increasing
    assert is_sorted(all_n_train)
    assert all_values_unique(all_n_train)

    # n_train + n_val should fit within n_samples_tot and should hit it for exactly 1 split
    assert all([n_train + n_val <= n_samples_tot for n_train, n_val in splits])
    assert max([n_train + n_val for n_train, n_val in splits]) == n_samples_tot

    # all n_val & n_train values need to satisfy their minimum values
    assert all([n_val >= min_samples_validate for n_val in all_n_val])
    assert all([n_train >= min_samples_train for n_train in all_n_train])


@pytest.mark.parametrize("n_samples_tot, exception_expected", [(150, True), (158, True), (159, False), (1000, False)])
def test_time_series_cv_splitter_exceptions(n_samples_tot: int, exception_expected: bool):

    # --- arrange -----------------------------------------
    min_samples_train = 100
    min_samples_validate = 50
    n_splits = 10
    cv_splitter = TimeSeriesCVSplitter(
        min_samples_train=min_samples_train, min_samples_validate=min_samples_validate, n_splits=n_splits
    )

    # --- act ---------------------------------------------
    if exception_expected:
        with pytest.raises(ValueError):
            _ = cv_splitter.get_splits(n_samples_tot=n_samples_tot)
    else:
        _ = cv_splitter.get_splits(n_samples_tot=n_samples_tot)
