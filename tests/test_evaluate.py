"""Grouped cross-validation harness.

Three things here are load-bearing and easy to get wrong:

1. **Grouping.** Windows overlap, so the split must be on engine unit. This test
   file exists mainly to prove the harness cannot leak a unit across a boundary.
2. **The final test group.** It must stay closed until the model is frozen. There is
   a test that the harness refuses to evaluate on it during cross-validation.
3. **Aggregation.** Per-unit metrics are averaged with their interval attached, so
   a reader sees how little evidence there is behind a comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from prognos.dataset import GroupKey
from prognos.evaluate import (
    FoldResult,
    aggregate_folds,
    build_folds,
    evaluate_model,
    report_run,
)
from prognos.split import UnitSplit

UNITS = [2, 5, 10, 11, 14, 15, 16, 18, 20]


class FakeWindowSet:
    """A stand-in for the real WindowSet with controllable contents."""

    def __init__(self, windows, targets, groups, cycles, fractions, max_cycle=200):
        self.windows = windows
        self.targets = targets
        self.groups = groups
        self.cycles = cycles
        self.fractions = fractions
        self.max_cycle = max_cycle
        self.splits = tuple("dev" for _ in groups)
        self.subdatasets = tuple("SYNTH" for _ in groups)
        self.group_labels = tuple(GroupKey("SYNTH", int(g)) for g in range(int(groups.max()) + 1))

    def subset(self, mask):
        """Subset by boolean mask, matching the real interface."""
        return FakeWindowSet(
            self.windows[mask],
            self.targets[mask],
            self.groups[mask],
            self.cycles[mask],
            self.fractions[mask],
            self.max_cycle,
        )

    def summary(self):
        """Counts for the report."""
        return {
            "n_windows": int(self.groups.size),
            "n_engines": int(np.unique(self.groups).size),
            "engines": [str(g) for g in sorted(set(self.groups.tolist()))],
            "subdatasets": sorted(set(self.subdatasets)),
            "source_splits": sorted(set(self.splits)),
            "max_cycle": int(self.max_cycle),
            "rul_cap": int(self.targets.max()),
        }


def window_set_for(units: list[int], rows_per_unit: int = 40, seed: int = 0) -> FakeWindowSet:
    """A window set with a learnable relationship: RUL falls with cycle number."""
    rng = np.random.default_rng(seed)
    all_windows, targets, groups, cycles, fractions = [], [], [], [], []
    for index, unit in enumerate(units):
        unit_cycles = np.arange(1, rows_per_unit + 1, dtype=np.int64)
        block = rng.normal(size=(rows_per_unit, 10, 3)).astype(np.float32)
        block[:, :, 0] += unit_cycles[:, None] * 0.1
        all_windows.append(block)
        targets.append((200 - unit_cycles).astype(np.float32))
        groups.append(np.full(rows_per_unit, index, dtype=np.int64))
        cycles.append(unit_cycles)
        fractions.append(1.0 - unit_cycles / rows_per_unit)
    return FakeWindowSet(
        np.concatenate(all_windows),
        np.concatenate(targets),
        np.concatenate(groups),
        np.concatenate(cycles),
        np.concatenate(fractions).astype(np.float32),
    )


class TestFoldConstruction:
    def test_no_unit_appears_on_both_sides_of_a_fold(self) -> None:
        """The core guarantee: no unit is in train and validation at once."""
        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)
        for fold in folds:
            assert not set(fold.split.train_units) & set(fold.split.val_units)
            assert not set(fold.split.train_units) & set(fold.split.test_units)
            assert not set(fold.split.val_units) & set(fold.split.test_units)

    def test_every_window_lands_in_exactly_one_side(self) -> None:
        """A dropped window would silently shrink the evaluation set."""
        data = window_set_for(UNITS)
        for fold in build_folds(data, n_splits=5, test_fraction=0.34, seed=7):
            train_mask = fold.train_mask
            val_mask = fold.val_mask
            assert not np.any(train_mask & val_mask)
            accounted = int(train_mask.sum() + val_mask.sum())
            usable = int(
                np.isin(data.groups, list(fold.split.train_units + fold.split.val_units)).sum()
            )
            assert accounted == usable

    def test_folds_are_deterministic(self) -> None:
        """Same seed, same folds, so a committed result can be reproduced."""
        data = window_set_for(UNITS)
        first = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)
        second = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)
        assert [(f.train_mask.tobytes()) for f in first] == [
            (f.train_mask.tobytes()) for f in second
        ]

    def test_final_test_group_is_the_same_across_folds(self) -> None:
        """It has to be fixed, or it stops being a test set and becomes validation."""
        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)
        assert len({f.split.test_units for f in folds}) == 1

    def test_different_seeds_give_different_test_groups(self) -> None:
        """Seed sensitivity is real and should be measurable."""
        data = window_set_for(UNITS)
        groups = {
            build_folds(data, n_splits=5, test_fraction=0.34, seed=s)[0].split.test_units
            for s in range(10)
        }
        assert len(groups) > 1


class TestHarnessDoesNotPeek:
    def test_cross_validation_never_scores_the_test_group(self) -> None:
        """Training and scoring in a fold must exclude the final test units.

        If this fails, the reported cross-validation number is optimistic and the
        final test set is no longer untouched.
        """
        data = window_set_for(UNITS)
        captured: list[float] = []
        fold = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)[0]
        result = evaluate_model(
            data,
            fold.train_mask,
            fold.val_mask,
            model_name="fake",
            fit=lambda w, c, t: captured.append(float(np.mean(t))),
            predict=lambda w, c: np.full(w.shape[0], captured[-1]),
            test_units=fold.split.test_units,
        )
        assert result.n_train == int(fold.train_mask.sum())
        assert result.n_eval == int(fold.val_mask.sum())
        assert result.eval_units == tuple(sorted(fold.split.val_units))

    def test_evaluate_rejects_a_mask_that_touches_test_units(self) -> None:
        """Passing test units in as training data is a hard error, not a warning."""
        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)
        split: UnitSplit = folds[0].split
        bad_train = np.isin(data.groups, split.test_units)
        with pytest.raises(ValueError, match="test group"):
            evaluate_model(
                data,
                bad_train,
                folds[0].val_mask,
                model_name="fake",
                fit=lambda w, c, t: None,
                predict=lambda w, c: np.full(w.shape[0], 1.0),
                test_units=split.test_units,
            )

    def test_rejects_overlapping_masks(self) -> None:
        """A row in both masks would be scored on data it was trained on."""
        data = window_set_for(UNITS)
        fold = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)[0]
        with pytest.raises(ValueError, match="overlap"):
            evaluate_model(
                data,
                fold.train_mask,
                fold.train_mask,
                model_name="fake",
                fit=lambda w, c, t: None,
                predict=lambda w, c: np.full(w.shape[0], 1.0),
                test_units=fold.split.test_units,
            )

    def test_rejects_an_empty_mask(self) -> None:
        """Scoring on nothing would report a metric of zero by accident."""
        data = window_set_for(UNITS)
        fold = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)[0]
        with pytest.raises(ValueError, match="no (training|evaluation)"):
            evaluate_model(
                data,
                np.zeros_like(fold.train_mask),
                fold.val_mask,
                model_name="fake",
                fit=lambda w, c, t: None,
                predict=lambda w, c: np.full(w.shape[0], 1.0),
                test_units=fold.split.test_units,
            )


class TestResultContents:
    def test_records_units_and_counts(self) -> None:
        """A result without its units is not auditable."""
        data = window_set_for(UNITS)
        fold = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)[0]
        result = evaluate_model(
            data,
            fold.train_mask,
            fold.val_mask,
            model_name="fake",
            fit=lambda w, c, t: None,
            predict=lambda w, c: np.full(w.shape[0], 50.0),
        )
        assert isinstance(result, FoldResult)
        assert result.n_eval == int(fold.val_mask.sum())
        assert result.n_train == int(fold.train_mask.sum())
        assert len(result.eval_units) > 0

    def test_bootstrap_brackets_the_point_estimate(self) -> None:
        """The observed metric should sit inside its own confidence interval."""
        data = window_set_for(UNITS)
        fold = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)[0]
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 5.0, size=data.targets.size).astype(np.float32)

        result = evaluate_model(
            data,
            fold.train_mask,
            fold.val_mask,
            model_name="fake",
            fit=lambda w, c, t: None,
            predict=lambda w, c: c.astype(np.float32),
        )
        assert result.rmse_ci[0] <= result.rmse <= result.rmse_ci[2]
        assert result.nasa_ci[0] <= result.nasa_score <= result.nasa_ci[2]
        assert np.isfinite(result.nasa_score)
        del noise

    def test_life_stage_breakdown_covers_all_eval_windows(self) -> None:
        """Stage counts must sum to the evaluation set, or a stage is being dropped."""
        data = window_set_for(UNITS)
        fold = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)[0]
        result = evaluate_model(
            data,
            fold.train_mask,
            fold.val_mask,
            model_name="fake",
            fit=lambda w, c, t: None,
            predict=lambda w, c: np.full(w.shape[0], 50.0),
        )
        assert sum(int(stage["n"]) for stage in result.by_stage.values()) == result.n_eval
        assert set(result.by_stage) == {"early", "mid", "late"}


class TestAggregation:
    def test_aggregates_across_folds_with_an_interval(self) -> None:
        """The headline number needs its spread, or it invites over-reading."""
        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=5, test_fraction=0.34, seed=7)
        results = [
            evaluate_model(
                data,
                f.train_mask,
                f.val_mask,
                model_name="fake",
                fit=lambda w, c, t: None,
                predict=lambda w, c: np.full(w.shape[0], 60.0),
            )
            for f in folds
        ]
        summary = aggregate_folds("fake", results, target_coverage=0.90)
        assert summary["model"] == "fake"
        assert summary["n_folds"] == len(results)
        assert summary["rmse"]["ci_low"] <= summary["rmse"]["mean"] <= summary["rmse"]["ci_high"]

    def test_aggregation_records_n_folds_beside_the_mean(self) -> None:
        """With 5 folds a mean is easy to over-read; the count must travel with it."""
        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=3, test_fraction=0.34, seed=7)
        results = [
            evaluate_model(
                data,
                f.train_mask,
                f.val_mask,
                model_name="fake",
                fit=lambda w, c, t: None,
                predict=lambda w, c: np.full(w.shape[0], 60.0),
            )
            for f in folds
        ]
        summary = aggregate_folds("fake", results, target_coverage=0.90)
        assert summary["n_folds"] == 3


class TestRunReport:
    def test_report_is_json_serialisable(self) -> None:
        """The report is committed under results/, so it must serialise cleanly."""
        import json

        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=3, test_fraction=0.34, seed=7)
        results = [
            evaluate_model(
                data,
                f.train_mask,
                f.val_mask,
                model_name="fake",
                fit=lambda w, c, t: None,
                predict=lambda w, c: np.full(w.shape[0], 60.0),
            )
            for f in folds
        ]
        report = report_run(
            config={
                "window": {"length": 10, "stride": 5, "downsample": 10},
                "split": {"n_splits": 3, "seed": 7},
            },
            data=data,
            folds=folds,
            models={"fake": aggregate_folds("fake", results, 0.90)},
            hardware="test",
        )
        text = json.dumps(report, indent=2, sort_keys=True)
        assert "fake" in text
        assert json.loads(text)["hardware"] == "test"

    def test_report_states_the_final_test_group_is_closed(self) -> None:
        """A reader of results/ must be able to tell the test set was not opened."""
        data = window_set_for(UNITS)
        folds = build_folds(data, n_splits=3, test_fraction=0.34, seed=7)
        report = report_run(
            config={"window": {}, "split": {}},
            data=data,
            folds=folds,
            models={},
            hardware="test",
        )
        assert report["final_test_group"]["status"] == "closed"
        assert report["final_test_group"]["engines"]


class TestEngineIdentity:
    """Grouping must be by (sub-dataset, unit), not by unit number.

    N-CMAPSS restarts unit numbers in every sub-dataset, so DS01 unit 2 and DS02
    unit 2 are different engines. Grouping on the bare number puts two different
    engines in one fold, and the reported error stops measuring generalisation.
    """

    def test_same_unit_number_in_two_subdatasets_is_two_engines(self) -> None:
        """The distinguishing case: identical unit numbers, different engines."""
        assert GroupKey("DS01-005", 2) != GroupKey("DS02-006", 2)

    def test_same_subdataset_and_unit_is_the_same_engine(self) -> None:
        """And the identity is stable, so grouping is well-defined."""
        assert GroupKey("DS02-006", 2) == GroupKey("DS02-006", 2)

    def test_keys_sort_deterministically(self) -> None:
        """Ordering must not depend on insertion order, or splits are not reproducible."""
        keys = [GroupKey("DS02-006", 5), GroupKey("DS01-005", 5), GroupKey("DS02-006", 2)]
        assert sorted(keys) == [
            GroupKey("DS01-005", 5),
            GroupKey("DS02-006", 2),
            GroupKey("DS02-006", 5),
        ]

    def test_label_round_trips_through_text(self) -> None:
        """Cached labels are stored as text, so parsing must be exact."""
        key = GroupKey("DS02-006", 11)
        parsed = GroupKey(
            subdataset=key.label().split("/")[0],
            unit=int(key.label().split("unit")[1]),
        )
        assert parsed == key

    def test_a_fold_never_merges_two_engines_that_share_a_unit_number(self) -> None:
        """Built directly: same unit number, different sub-dataset, both in one set."""
        keys = [GroupKey("DS01-005", u) for u in range(1, 7)]
        keys += [GroupKey("DS02-006", u) for u in range(1, 7)]
        labels = tuple(sorted(set(keys)))
        groups = np.asarray([labels.index(k) for k in keys] * 20, dtype=np.int64)

        folds = build_folds(FakeGroupSet(groups), n_splits=3, test_fraction=0.34, seed=7)
        for fold in folds:
            held_labels = {labels[g] for g in fold.split.val_units}
            train_labels = {labels[g] for g in fold.split.train_units}
            assert not held_labels & train_labels


class FakeGroupSet:
    """Minimal WindowSource carrying only what build_folds reads."""

    def __init__(self, groups: np.ndarray) -> None:
        self.groups = groups

    @property
    def units(self) -> np.ndarray:
        """Not used by build_folds; present to satisfy the protocol shape."""
        return self.groups

    def subset(self, mask: np.ndarray) -> FakeGroupSet:
        """Subset by mask."""
        return FakeGroupSet(self.groups[mask])

    def summary(self) -> dict[str, object]:
        """Counts for the report."""
        return {"n_windows": int(self.groups.size)}
