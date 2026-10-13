"""Tests for the Phase 4 harness.

This module exists to turn a fitted model into a claim about coverage and cost, so
it is where a plausible-looking but meaningless result is easiest to produce. Both
of the bugs it shipped with looked like success:

- episodes built from window indices instead of cycle numbers, so no engine could
  fail and every policy cost about nothing
- a coverage confidence interval computed over the interval bounds, which produced
  [18.0, 20.7] around a coverage of 0.926

So these tests are aimed at the arithmetic and the splits, not at the happy path.
They drive small synthetic datasets with known answers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prognos.dataset import GroupKey, WindowSet
from prognos.decision import CostModel, EngineEpisode
from prognos.evaluate import build_folds
from prognos.policy_eval import (
    CalibratedFold,
    build_episodes,
    evaluate_policies,
    split_fit_and_calibration,
    summarise,
)

N_CHANNELS = 18
WINDOW = 50


def synthetic_dataset(
    n_engines: int = 20,
    windows_per_engine: int = 30,
    seed: int = 0,
    stride: int = 10,
) -> WindowSet:
    """A dataset whose targets are a known function of the cycle number.

    Deterministic targets matter: a test can then assert that episodes are built in
    cycle units, because the true life is known exactly rather than inferred.
    """
    rng = np.random.default_rng(seed)
    total = n_engines * windows_per_engine
    groups = np.repeat(np.arange(n_engines), windows_per_engine).astype(np.int64)
    cycles = (np.arange(windows_per_engine) * stride + WINDOW)[np.newaxis, :]
    cycles = np.repeat(cycles, n_engines, axis=0).reshape(-1).astype(np.int64)
    # RUL falls by `stride` per window, so a prediction of the remaining life is
    # recoverable from the cycle number alone.
    targets = (windows_per_engine * stride - cycles + WINDOW).astype(np.float32)
    return WindowSet(
        windows=rng.normal(size=(total, WINDOW, N_CHANNELS)).astype(np.float32),
        targets=targets,
        groups=groups,
        group_labels=tuple(GroupKey("SYNTH", int(u)) for u in range(n_engines)),
        cycles=cycles,
        fractions=(1.0 - cycles / float(cycles.max() + 1)).astype(np.float32),
        splits=tuple("dev" for _ in range(total)),
        subdatasets=tuple("SYNTH" for _ in range(total)),
        max_cycle=int(cycles.max()),
    )


class TestSplitFitAndCalibration:
    def test_the_three_sets_are_disjoint(self) -> None:
        """The whole coverage claim rests on this being true."""
        data = synthetic_dataset()
        folds = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)
        fold = folds[0]
        fit_mask, calibration_mask, calibration_units = split_fit_and_calibration(
            np.asarray(data.groups), fold.train_mask, seed=1
        )

        assert not (fit_mask & calibration_mask).any(), "fitting and calibration overlap"
        assert not (fit_mask & fold.val_mask).any(), "fitting and evaluation overlap"
        assert not (calibration_mask & fold.val_mask).any(), "calibration and evaluation overlap"
        assert set(calibration_units) <= set(fold.split.train_units), (
            "a calibration engine came from outside this fold's training set"
        )

    def test_both_sides_are_nonempty(self) -> None:
        """An empty calibration set gives an infinite interval, which is not a result."""
        data = synthetic_dataset()
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        fit_mask, calibration_mask, _ = split_fit_and_calibration(
            np.asarray(data.groups), fold.train_mask, seed=1
        )
        assert fit_mask.any() and calibration_mask.any()

    def test_excludes_the_final_test_group(self) -> None:
        """The held-out ten engines must not become calibration data."""
        data = synthetic_dataset()
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        _, calibration_mask, _ = split_fit_and_calibration(
            np.asarray(data.groups), fold.train_mask, seed=1
        )
        test_units = set(fold.split.test_units)
        used = {int(u) for u in np.asarray(data.groups)[calibration_mask]}
        assert not used & test_units

    def test_is_deterministic_for_a_seed(self) -> None:
        """Same seed, same split. Otherwise the coverage number cannot be reproduced."""
        data = synthetic_dataset()
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        first = split_fit_and_calibration(np.asarray(data.groups), fold.train_mask, seed=7)
        second = split_fit_and_calibration(np.asarray(data.groups), fold.train_mask, seed=7)
        assert first[2] == second[2]
        np.testing.assert_array_equal(first[0], second[0])

    def test_rejects_too_few_engines(self) -> None:
        """With two training engines there is nothing to hold out."""
        data = synthetic_dataset(n_engines=20, windows_per_engine=10)
        groups = np.asarray(data.groups)
        mask = np.isin(groups, [0, 1])
        with pytest.raises(ValueError, match="at least 3 training engines"):
            split_fit_and_calibration(groups, mask, seed=1)


class TestBuildEpisodes:
    def test_episode_length_is_measured_in_cycles(self) -> None:
        """An episode's life is its last cycle, not its window count.

        This is the bug that produced an entire run in which nothing could fail. With
        `stride` between windows and a window covering `WINDOW` rows, a window count
        understates the life by roughly two orders of magnitude, and a simulator given
        such an episode never reaches the end of an engine.
        """
        data = synthetic_dataset(n_engines=4, windows_per_engine=10, stride=10)
        cycles = np.asarray(data.cycles)
        groups = np.asarray(data.groups)
        expected_life = int(cycles[groups == 0].max()) + 1

        fold = CalibratedFold(
            name="f",
            y_true=np.asarray(data.targets, dtype=np.float64),
            point=np.asarray(data.targets, dtype=np.float64),
            cycles=cycles,
            split_lower=np.asarray(data.targets, dtype=np.float64),
            split_upper=np.asarray(data.targets, dtype=np.float64) + 5.0,
            cqr_lower=np.asarray(data.targets, dtype=np.float64),
            cqr_upper=np.asarray(data.targets, dtype=np.float64) + 6.0,
            groups=groups,
            stages=np.full(groups.size, "mid"),
            eval_engines=(0, 1, 2, 3),
            calibration_engines=(10,),
            fit_engines=(20,),
            split_width=5.0,
            cqr_width=6.0,
        )
        episodes = build_episodes([fold])
        assert len(episodes) == 4
        assert episodes[0].true_life == expected_life
        assert expected_life > 10 * (10 - 1), "life in cycles should exceed the window count"

    def test_predictions_land_on_their_own_cycle(self) -> None:
        """A prediction must be indexed by cycle, not packed at the front.

        Packing them contiguously is what a window-rank implementation does, and it
        makes the policy fire at the wrong moment in the engine's life while looking
        entirely plausible.
        """
        data = synthetic_dataset(n_engines=2, windows_per_engine=6, stride=10)
        groups = np.asarray(data.groups)
        cycles = np.asarray(data.cycles)
        point = np.asarray(data.targets, dtype=np.float64)
        fold = CalibratedFold(
            name="f",
            y_true=point.copy(),
            point=point.copy(),
            cycles=cycles,
            split_lower=point.copy(),
            split_upper=point + 5.0,
            cqr_lower=point.copy(),
            cqr_upper=point + 6.0,
            groups=groups,
            stages=np.full(groups.size, "mid"),
            eval_engines=(0, 1),
            calibration_engines=(10,),
            fit_engines=(20,),
            split_width=5.0,
            cqr_width=6.0,
        )
        episode = build_episodes([fold])[0]
        for row, cycle in enumerate(cycles[groups == 0]):
            assert episode.prediction_at(int(cycle)) is not None
            assert episode.prediction_at(int(cycle)) is not None
            assert episode.point[int(cycle)] == pytest.approx(point[row])

    def test_cycles_with_no_window_have_no_prediction(self) -> None:
        """A gap must read as no decision available, not as a fabricated zero."""
        data = synthetic_dataset(n_engines=2, windows_per_engine=6, stride=10)
        groups = np.asarray(data.groups)
        point = np.asarray(data.targets, dtype=np.float64)
        fold = CalibratedFold(
            name="f",
            y_true=point.copy(),
            point=point.copy(),
            cycles=np.asarray(data.cycles),
            split_lower=point.copy(),
            split_upper=point + 5.0,
            cqr_lower=point.copy(),
            cqr_upper=point + 6.0,
            groups=groups,
            stages=np.full(groups.size, "mid"),
            eval_engines=(0, 1),
            calibration_engines=(10,),
            fit_engines=(20,),
            split_width=5.0,
            cqr_width=6.0,
        )
        episode = build_episodes([fold])[0]
        # Windows start at WINDOW cycles and step by the stride, so there are gaps.
        assert episode.prediction_at(0) is None
        assert episode.prediction_at(episode.true_life + 5) is None

    def test_one_episode_per_engine(self) -> None:
        data = synthetic_dataset(n_engines=7, windows_per_engine=4)
        groups = np.asarray(data.groups)
        point = np.asarray(data.targets, dtype=np.float64)
        fold = CalibratedFold(
            name="f",
            y_true=point.copy(),
            point=point.copy(),
            cycles=np.asarray(data.cycles),
            split_lower=point.copy(),
            split_upper=point + 5.0,
            cqr_lower=point.copy(),
            cqr_upper=point + 6.0,
            groups=groups,
            stages=np.full(groups.size, "mid"),
            eval_engines=tuple(range(7)),
            calibration_engines=(10,),
            fit_engines=(20,),
            split_width=5.0,
            cqr_width=6.0,
        )
        assert len(build_episodes([fold])) == 7


def _calibrated_folds(seed: int = 0) -> list[CalibratedFold]:
    """Two folds of perfectly-covering intervals, for the summarise tests."""
    rng = np.random.default_rng(seed)
    folds = []
    for index in range(2):
        groups = np.repeat(np.arange(3), 20).astype(np.int64)
        truth = rng.uniform(5.0, 50.0, groups.size)
        folds.append(
            CalibratedFold(
                name=f"fold{index}",
                y_true=truth,
                point=truth,
                cycles=np.arange(groups.size),
                split_lower=truth - 10.0,
                split_upper=truth + 10.0,
                cqr_lower=truth - 12.0,
                cqr_upper=truth + 12.0,
                groups=groups,
                stages=np.full(groups.size, "mid"),
                eval_engines=(0, 1, 2),
                calibration_engines=(10, 11),
                fit_engines=(20, 21),
                split_width=10.0,
                cqr_width=12.0,
            )
        )
    return folds


class TestSummarise:
    def test_coverage_of_containing_intervals_is_one(self) -> None:
        summary = summarise(_calibrated_folds(), "split", 0.9, seed=0)
        assert summary["empirical_coverage"] == pytest.approx(1.0)

    def test_the_confidence_interval_brackets_the_estimate(self) -> None:
        """The bug this exists for averaged the bounds instead of the indicator."""
        summary = summarise(_calibrated_folds(), "split", 0.9, seed=0)
        low, high = summary["coverage_ci"]
        assert 0.0 <= low <= summary["empirical_coverage"] <= high <= 1.0

    def test_widths_differ_between_methods(self) -> None:
        """CQR is built wider in the fixture, so the summary must say so."""
        folds = _calibrated_folds()
        assert (
            summarise(folds, "cqr", 0.9, seed=0)["mean_width"]
            > summarise(folds, "split", 0.9, seed=0)["mean_width"]
        )

    def test_records_the_engine_split_per_fold(self) -> None:
        """Per-fold engine sets, because the disjointness claim is per fold."""
        summary = summarise(_calibrated_folds(), "split", 0.9, seed=0)
        assert [f["name"] for f in summary["folds"]] == ["fold0", "fold1"]
        for fold in summary["folds"]:
            assert fold["fit_engines"] == [20, 21]
            assert fold["calibration_engines"] == [10, 11]
            assert fold["eval_engines"] == [0, 1, 2]

    def test_stage_counts_sum_to_the_scored_rows(self) -> None:
        summary = summarise(_calibrated_folds(), "split", 0.9, seed=0)
        stages = summary["by_stage"]
        counted = sum(int(stage["n"]) for stage in stages.values())
        assert counted == summary["n_eval_rows"]


def cost_model(ratio: float = 25.0) -> CostModel:
    """A cost model with the ratio as the only thing that varies."""
    return CostModel(
        planned_maintenance_cost=1.0,
        failure_cost_ratio=ratio,
        downtime_cost_per_cycle=0.05,
        maintenance_window_cycles=2,
    )


def engine_episodes(offset: float = 19.0) -> list[EngineEpisode]:
    """Engines whose interval is genuinely offset below the point estimate.

    `lower` is the true remaining life; `point` sits `offset` cycles above it, which
    is the optimistic bias a real model shows and what separates the lower-bound policy
    from the point-estimate one. With `offset = 0` the two policies are the same
    policy and any test comparing them is vacuous.
    """
    episodes = []
    for life in (60, 80, 100):
        lower = np.array([float(life - t) for t in range(life)])
        episodes.append(
            EngineEpisode(engine=f"e{life}", true_life=life, lower=lower, point=lower + offset)
        )
    return episodes


class TestEvaluatePolicies:
    def test_planning_beats_waiting_for_failures(self) -> None:
        """The headline: preventing failures is cheaper than paying for them."""
        result = evaluate_policies(
            engine_episodes(),
            cost_model(),
            horizon=600,
            thresholds=[10.0, 20.0],
            fixed_intervals=[50],
            ratios=[1.0, 25.0],
            seed=0,
        )
        rows = {row["policy"]: row for row in result["policies"]}
        assert rows["run_to_failure"]["n_failures"] > 0, "the baseline must actually fail"
        assert rows["threshold_lower_10"]["n_failures"] == 0, "the policy must prevent them"
        assert rows["threshold_lower_10"]["total_cost"] < rows["run_to_failure"]["total_cost"]

    def test_every_policy_gets_a_row(self) -> None:
        result = evaluate_policies(
            engine_episodes(),
            cost_model(),
            horizon=600,
            thresholds=[10.0, 20.0],
            fixed_intervals=[50, 100],
            ratios=[5.0],
            seed=0,
        )
        names = {row["policy"] for row in result["policies"]}
        assert names == {
            "run_to_failure",
            "fixed_interval_50",
            "fixed_interval_100",
            "threshold_lower_10",
            "threshold_point_10",
            "threshold_lower_20",
            "threshold_point_20",
        }

    def test_records_episode_lives_and_the_cost_model(self) -> None:
        """A cost curve without its assumptions is not a result."""
        result = evaluate_policies(
            engine_episodes(),
            cost_model(ratio=7.0),
            horizon=600,
            thresholds=[10.0],
            fixed_intervals=[50],
            ratios=[5.0],
            seed=0,
        )
        assert result["cost_model"]["failure_cost_ratio"] == 7.0
        assert result["episode_lives"]["min"] == 60
        assert result["episode_lives"]["max"] == 100
        assert result["n_episodes"] == 3

    def test_one_sweep_per_threshold(self) -> None:
        result = evaluate_policies(
            engine_episodes(),
            cost_model(),
            horizon=600,
            thresholds=[10.0, 20.0],
            fixed_intervals=[50],
            ratios=[1.0, 5.0],
            seed=0,
        )
        assert [s["threshold"] for s in result["ratio_sweeps"]] == [10.0, 20.0]
        for sweep in result["ratio_sweeps"]:
            assert len(sweep["rows"]) == 2

    def test_the_lower_bound_policy_is_priced_on_the_lower_bound(self) -> None:
        """The two threshold variants must be distinct policies, not one reused."""
        result = evaluate_policies(
            engine_episodes(),
            cost_model(),
            horizon=600,
            thresholds=[10.0],
            fixed_intervals=[50],
            ratios=[25.0],
            seed=0,
        )
        rows = {row["policy"]: row for row in result["policies"]}
        assert rows["threshold_lower_10"]["n_planned"] != rows["threshold_point_10"]["n_planned"], (
            "acting on the lower bound and on the point estimate gave the same policy"
        )


class TestEvaluateFold:
    """The fold pass itself: fit, calibrate on held-out engines, score on the rest."""

    def test_produces_both_interval_methods_from_one_pass(self) -> None:
        from prognos.policy_eval import evaluate_fold

        data = synthetic_dataset(n_engines=20, windows_per_engine=25, seed=3)
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        result = evaluate_fold(data, fold, seed=1, target_coverage=0.9)

        assert result.n > 0
        assert np.all(result.split_lower <= result.split_upper)
        assert np.all(result.cqr_lower <= result.cqr_upper)
        assert np.all(result.split_lower >= 0.0), "RUL bounds must not go negative"

    def test_calibration_engines_are_withheld_from_the_fit(self) -> None:
        """The claim the coverage number rests on, checked on the returned object."""
        from prognos.policy_eval import evaluate_fold

        data = synthetic_dataset(n_engines=20, windows_per_engine=25, seed=3)
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        result = evaluate_fold(data, fold, seed=1, target_coverage=0.9)

        calibration = set(result.calibration_engines)
        fitting = set(result.fit_engines)
        evaluation = set(result.eval_engines)
        assert calibration and fitting and evaluation
        assert not calibration & fitting
        assert not calibration & evaluation
        assert not fitting & evaluation

    def test_scored_rows_are_the_folds_validation_engines(self) -> None:
        from prognos.policy_eval import evaluate_fold

        data = synthetic_dataset(n_engines=20, windows_per_engine=25, seed=3)
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        result = evaluate_fold(data, fold, seed=1, target_coverage=0.9)
        assert {int(g) for g in result.groups} == set(fold.split.val_units)
        assert result.n == int(fold.val_mask.sum())

    def test_widths_are_finite_and_non_negative(self) -> None:
        """Widths must be usable numbers, and split conformal must be non-degenerate.

        A CQR correction of exactly zero is a legitimate outcome on data whose
        quantile band already covers, so it is not asserted against. What must hold
        is that neither width is NaN or infinite and that the split conformal
        half-width reflects real residual spread rather than collapsing to nothing.
        """
        from prognos.policy_eval import evaluate_fold

        data = synthetic_dataset(n_engines=20, windows_per_engine=25, seed=3)
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        result = evaluate_fold(data, fold, seed=1, target_coverage=0.9)
        assert math.isfinite(result.split_width)
        assert math.isfinite(result.cqr_width)
        assert result.split_width >= 0.0
        assert result.cqr_width >= 0.0
        assert result.split_width > 0.0, (
            "the synthetic targets have real residual spread, so a zero half-width "
            "would mean the calibration was not applied"
        )

    def test_no_calibration_engine_comes_from_the_final_test_group(self) -> None:
        """A held-out engine used to calibrate is a held-out engine that is not held out."""
        from prognos.policy_eval import evaluate_fold

        data = synthetic_dataset(n_engines=20, windows_per_engine=25, seed=3)
        fold = build_folds(data, n_splits=3, test_fraction=0.3, seed=1)[0]
        result = evaluate_fold(data, fold, seed=1, target_coverage=0.9)
        assert not set(result.calibration_engines) & set(fold.split.test_units)
