"""Unit-level splitting. See docs/adr/0003."""

from __future__ import annotations

import numpy as np
import pytest

from prognos.split import assert_disjoint, assign_test_units, make_splits

UNITS = [2, 5, 10, 11, 14, 15, 16, 18, 20]


def test_test_units_are_reproducible_for_a_seed() -> None:
    """The same seed picks the same test group, on every machine."""
    assert assign_test_units(UNITS, 0.34, seed=1) == assign_test_units(UNITS, 0.34, seed=1)


def test_test_units_change_with_the_seed() -> None:
    """Different seeds give different groups, so seed sensitivity is measurable."""
    picks = {assign_test_units(UNITS, 0.34, seed=s) for s in range(12)}
    assert len(picks) > 1


def test_test_group_never_leaks_into_training() -> None:
    """No unit is in both the final test group and any fold's training set."""
    splits = make_splits(UNITS, n_splits=5, test_fraction=0.34, seed=7)
    for split in splits:
        assert_disjoint(split.train_units, split.test_units, split.name)
        assert_disjoint(split.val_units, split.test_units, split.name)


def test_final_test_group_is_identical_across_folds() -> None:
    """The test group is fixed, so it can be reserved and opened once."""
    splits = make_splits(UNITS, n_splits=5, test_fraction=0.34, seed=7)
    groups = {split.test_units for split in splits}
    assert len(groups) == 1


def test_every_unit_appears_exactly_once_per_fold() -> None:
    """Within a fold, train and held-out units partition the non-test units."""
    splits = make_splits(UNITS, n_splits=5, test_fraction=0.34, seed=7)
    cv_units = set(UNITS) - set(splits[0].test_units)
    for split in splits:
        held = set(split.val_units)
        assert held <= cv_units
        assert set(split.train_units) | held == cv_units
        assert_disjoint(split.train_units, held, split.name)


def test_val_mask_and_test_mask_are_different_selections() -> None:
    """The two masks select different rows; conflating them would be a leak."""
    split = make_splits(UNITS, n_splits=3, test_fraction=0.34, seed=7)[0]
    units = np.asarray(UNITS, dtype=np.int64)
    assert not np.array_equal(split.val_mask(units), split.test_mask(units))
    assert set(units[split.val_mask(units)].tolist()) == set(split.val_units)
    assert set(units[split.test_mask(units)].tolist()) == set(split.test_units)


def test_assert_disjoint_raises_on_overlap() -> None:
    """The guard is a test, not a convention."""
    with pytest.raises(ValueError, match="both groups"):
        assert_disjoint([1, 2, 3], [3, 4], "unit check")


def test_insufficient_units_raises_rather_than_silently_degrading() -> None:
    """Two engines cannot produce a train/test split, and we say so."""
    with pytest.raises(ValueError, match="at least 3 engines"):
        assign_test_units([1, 2], test_fraction=0.5, seed=1)


def test_splits_are_deterministic() -> None:
    """Rebuilding the splits with the same arguments gives the same folds."""
    first = make_splits(UNITS, n_splits=5, test_fraction=0.34, seed=7)
    second = make_splits(UNITS, n_splits=5, test_fraction=0.34, seed=7)
    assert [(s.train_units, s.test_units) for s in first] == [
        (s.train_units, s.test_units) for s in second
    ]
