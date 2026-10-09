"""Grouped cross-validation by engine.

Windows overlap heavily, so a split over windows puts near-duplicates on both sides
of the boundary and reports an error that is not a generalisation error. The engine
is what gets split on. See docs/adr/0003.

The unit of grouping here is the engine's `GroupKey`, a (sub-dataset, unit) pair.
N-CMAPSS restarts unit numbers at 1 in every sub-dataset, so DS01 unit 2 and DS02
unit 2 are different engines; grouping on the bare number would put two different
engines in the same fold and stop the reported error from measuring anything real.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True, slots=True)
class UnitSplit:
    """One cross-validation fold.

    `val_units` are held out for this fold's evaluation. `test_units` is the final
    test group, identical across every fold, and is not touched until the model is
    frozen. Keeping the two separate is the point: they serve different purposes,
    and conflating them silently turns the final test set into a validation set.
    """

    name: str
    train_units: tuple[int, ...]
    val_units: tuple[int, ...]
    test_units: tuple[int, ...]

    def val_mask(self, units: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        """Boolean mask over rows, True where the row's engine is held out for scoring."""
        return np.isin(units, np.asarray(self.val_units, dtype=units.dtype))

    def test_mask(self, units: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        """Boolean mask over rows, True where the row's engine is in the final test group."""
        return np.isin(units, np.asarray(self.test_units, dtype=units.dtype))


def assign_test_units(
    groups: Sequence[int] | npt.NDArray[np.integer[Any]],
    test_fraction: float,
    seed: int,
) -> tuple[int, ...]:
    """Choose the final test group, seeded so the choice is reproducible."""
    unique = np.asarray(sorted({int(g) for g in groups}), dtype=np.int64)
    if len(unique) < 3:
        raise ValueError(f"need at least 3 engines to hold out a test group, got {len(unique)}")
    n_test = max(1, round(len(unique) * test_fraction))
    n_test = min(n_test, len(unique) - 2)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique, size=n_test, replace=False)
    return tuple(sorted(map(int, chosen)))


def make_splits(
    groups: Sequence[int],
    n_splits: int,
    test_fraction: float,
    seed: int,
) -> list[UnitSplit]:
    """Build one split per fold: train on the CV engines, evaluate on the held-out ones.

    The final test group is constant across folds on purpose. It is the final
    evaluation set and must stay untouched until the model is frozen; varying it
    per fold would turn it into a cross-validation set.
    """
    unique = np.asarray(sorted({int(g) for g in groups}), dtype=np.int64)
    test_units = assign_test_units(unique, test_fraction, seed)
    cv_units = np.asarray([u for u in unique if u not in test_units], dtype=np.int64)
    if len(cv_units) < n_splits:
        n_splits = len(cv_units)
    if n_splits < 2:
        raise ValueError(f"not enough non-test engines for {n_splits}-fold CV: {len(cv_units)}")

    dummy = np.zeros(len(cv_units), dtype=np.float64)
    splits: list[UnitSplit] = []
    for fold, (_, held) in enumerate(GroupKFold(n_splits=n_splits).split(dummy, dummy, cv_units)):
        held_units = tuple(sorted(int(u) for u in cv_units[held]))
        train = tuple(sorted(int(u) for u in cv_units if u not in set(held_units)))
        splits.append(
            UnitSplit(
                name=f"fold{fold}",
                train_units=train,
                val_units=held_units,
                test_units=test_units,
            )
        )
    return splits


def assert_disjoint(left: Sequence[int], right: Sequence[int], label: str) -> None:
    """Raise if two engine groups overlap."""
    overlap = {int(u) for u in left} & {int(u) for u in right}
    if overlap:
        raise ValueError(f"{label}: engines appear in both groups: {sorted(overlap)}")
