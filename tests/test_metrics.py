"""Metrics, written against the published definitions before the implementation.

The NASA scoring function is the reason this file exists: it is easy to write a
plausible-looking asymmetric loss that has the branches the wrong way round, and
the result still "works" — it just rewards exactly the error that gets an engine
flown past its life. The reference values below are computed from the formula in
Saxena et al. (2008) by hand, not from `prognos.metrics`.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prognos.metrics import (
    EXPONENT_EARLY,
    EXPONENT_LATE,
    by_life_stage,
    grouped_bootstrap_ci,
    life_stage,
    mae,
    mean_ci,
    nasa_score,
    rmse,
)


def reference_nasa_score(d: np.ndarray) -> float:
    """Independent implementation straight from the paper, for cross-checking."""
    total = 0.0
    for value in d:
        if value < 0:
            total += np.exp(-value / 13.0) - 1.0
        else:
            total += np.exp(value / 10.0) - 1.0
    return float(total)


class TestNasaScore:
    def test_perfect_prediction_scores_zero(self) -> None:
        """Only an exact prediction can score zero, in either branch."""
        y = np.array([0.0, 50.0, 125.0])
        assert nasa_score(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_matches_hand_computed_values(self) -> None:
        """Values computed from exp(-d/13) and exp(d/10) by hand.

        A 20-cycle early prediction costs exp(20/13) - 1 = 3.657.
        A 20-cycle late prediction costs exp(20/10) - 1 = 6.389.
        """
        early = nasa_score(np.array([100.0]), np.array([80.0]))
        late = nasa_score(np.array([100.0]), np.array([120.0]))
        assert early == pytest.approx(3.6574, abs=1e-3)
        assert late == pytest.approx(6.3891, abs=1e-3)

    def test_late_prediction_costs_more_than_early(self) -> None:
        """The whole point of the metric: the dangerous direction is dearer.

        The ratio grows with the magnitude, because the late branch's smaller
        exponent compounds faster. At 20 cycles it is about 1.75x; at 25, about
        1.91x. This is the property that makes the score meaningful for
        maintenance — a symmetric implementation fails here.
        """
        for magnitude in (10.0, 20.0, 25.0, 40.0):
            early = nasa_score(np.array([120.0]), np.array([120.0 - magnitude]))
            late = nasa_score(np.array([120.0]), np.array([120.0 + magnitude]))
            assert late > early, f"late cost not higher at magnitude {magnitude}"
        ratio_20 = nasa_score(np.array([120.0]), np.array([140.0])) / nasa_score(
            np.array([120.0]), np.array([100.0])
        )
        assert ratio_20 == pytest.approx(1.747, abs=0.01)

    def test_error_at_the_branch_boundary_is_continuous(self) -> None:
        """An error of exactly zero must not jump between the two score branches."""
        just_below = nasa_score(np.array([100.0]), np.array([100.0 - 1e-9]))
        just_above = nasa_score(np.array([100.0]), np.array([100.0 + 1e-9]))
        assert just_below == pytest.approx(just_above, abs=1e-8)

    def test_is_a_sum_not_an_average(self) -> None:
        """The score grows with the number of predictions, so report it per-window too."""
        y = np.full(10, 100.0)
        pred = np.full(10, 110.0)
        single = nasa_score(np.array([100.0]), np.array([110.0]))
        assert nasa_score(y, pred) == pytest.approx(10 * single)

    def test_never_negative(self) -> None:
        """Both branches are non-negative, so the score cannot reward error."""
        rng = np.random.default_rng(0)
        true = rng.uniform(0, 125, size=200)
        pred = rng.uniform(0, 125, size=200)
        assert nasa_score(true, pred) >= 0.0

    def test_exponents_match_the_paper(self) -> None:
        """13 for early, 10 for late. Swapping these inverts the safety meaning."""
        assert EXPONENT_EARLY == 13.0
        assert EXPONENT_LATE == 10.0

    @settings(max_examples=50, deadline=None)
    @given(
        d=st.floats(min_value=-60, max_value=60, allow_nan=False, allow_infinity=False),
    )
    def test_matches_the_reference_implementation(self, d: float) -> None:
        """Property test against the hand-written loop.

        Vectorised `np.where` has a habit of evaluating both branches and of
        mishandling the `d == 0` edge. A property test catches that; a few example
        cases would not.
        """
        true = np.array([100.0])
        pred = np.array([100.0 + d])
        assert nasa_score(true, pred) == pytest.approx(reference_nasa_score(np.array([d])))

    def test_rejects_empty_input(self) -> None:
        """An empty prediction set is an error, not a score of zero."""
        with pytest.raises(ValueError, match="empty"):
            nasa_score(np.array([]), np.array([]))

    def test_rejects_shape_mismatch(self) -> None:
        """Mismatched lengths are an error, not a silent truncation."""
        with pytest.raises(ValueError, match="shape mismatch"):
            nasa_score(np.array([1.0, 2.0]), np.array([1.0]))


class TestRmseAndMae:
    def test_rmse_of_a_known_offset(self) -> None:
        """Constant error e gives RMSE = |e|."""
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y + 5.0) == pytest.approx(5.0)

    def test_rmse_is_root_mean_of_squared_errors(self) -> None:
        """Checked against the definition rather than a second call."""
        y = np.array([0.0, 0.0])
        pred = np.array([3.0, 4.0])
        assert rmse(y, pred) == pytest.approx(np.sqrt((9 + 16) / 2))

    def test_mae_of_a_known_offset(self) -> None:
        """Constant error e gives MAE = |e|."""
        y = np.array([10.0, 20.0, 30.0])
        assert mae(y, y - 7.0) == pytest.approx(7.0)

    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            rmse(np.array([]), np.array([]))
        with pytest.raises(ValueError, match="empty"):
            mae(np.array([]), np.array([]))


class TestLifeStage:
    def test_thirds_split(self) -> None:
        """The cut points are thirds of remaining life."""
        assert life_stage(0.9) == "early"
        assert life_stage(0.5) == "mid"
        assert life_stage(0.1) == "late"

    def test_boundaries(self) -> None:
        """Each third is closed at the bottom, so the stages tile [0, 1].

        Exactly one third of remaining life counts as late: at that point the
        engine is a third of the way from failure back to new, and the decision it
        drives is imminent.
        """
        assert life_stage(1 / 3) == "late"
        assert life_stage(2 / 3) == "mid"
        assert life_stage(0.0) == "late"
        assert life_stage(1.0) == "early"

    def test_breakdown_partitions_all_rows(self) -> None:
        """Every row lands in exactly one stage, and the counts sum."""
        rng = np.random.default_rng(0)
        fractions = rng.uniform(0, 1, size=300)
        truth = rng.uniform(0, 125, size=300)
        pred = rng.uniform(0, 125, size=300)
        result = by_life_stage(truth, pred, fractions)
        assert sum(int(stage["n"]) for stage in result.values()) == 300
        assert set(result) == {"early", "mid", "late"}

    def test_empty_stage_is_reported_not_dropped(self) -> None:
        """A stage with no windows shows n = 0 so a reader sees it was empty."""
        fractions = np.array([0.9, 0.8])
        result = by_life_stage(np.array([100.0, 100.0]), np.array([90.0, 95.0]), fractions)
        assert result["late"]["n"] == 0
        assert result["mid"]["n"] == 0
        assert result["early"]["n"] == 2

    def test_late_stage_metrics_come_from_late_rows_only(self) -> None:
        """Late-stage numbers are computed on late rows, not on everything."""
        fractions = np.array([0.9, 0.05])
        truth = np.array([100.0, 5.0])
        pred = np.array([101.0, 15.0])
        result = by_life_stage(truth, pred, fractions)
        assert result["late"]["n"] == 1
        assert result["late"]["rmse"] == pytest.approx(10.0)


class TestConfidenceIntervals:
    def test_mean_ci_brackets_the_mean(self) -> None:
        """A confidence interval that excludes its own estimate is wrong."""
        rng = np.random.default_rng(0)
        values = rng.normal(10.0, 2.0, size=8)
        mean, low, high = mean_ci(values)
        assert low < mean < high

    def test_mean_ci_needs_more_than_one_sample(self) -> None:
        """One sample has no spread, so the interval is undefined rather than zero."""
        mean, low, high = mean_ci(np.array([5.0]))
        assert mean == 5.0
        assert np.isnan(low) and np.isnan(high)

    def test_bootstrap_is_wider_when_units_correlate(self) -> None:
        """Resampling windows would understate the interval; resampling units does not.

        Two windows from one engine share most of their rows, so an engine's errors
        are correlated. Resampling rows treats them as independent and reports an
        interval that is too narrow. Here the same data is scored both ways and the
        unit-level interval must be materially wider.
        """
        rng = np.random.default_rng(1)
        n_units, per_unit = 8, 200
        groups = np.repeat(np.arange(n_units), per_unit)
        offset = rng.normal(0, 20.0, size=n_units)[groups]
        truth = np.full(groups.size, 100.0)
        pred = truth + offset

        _, unit_low, unit_high = grouped_bootstrap_ci(
            truth, pred, groups, "rmse", n_resamples=500, seed=0
        )
        unit_width = unit_high - unit_low

        # Same statistic, resampling rows instead of engines.
        rng_rows = np.random.default_rng(0)
        row_samples = np.array(
            [
                rmse(
                    truth[rng_rows.integers(0, truth.size, truth.size)],
                    pred[rng_rows.integers(0, pred.size, pred.size)],
                )
                for _ in range(500)
            ]
        )
        row_width = float(np.quantile(row_samples, 0.975) - np.quantile(row_samples, 0.025))

        assert unit_width > row_width * 1.5, (
            f"unit-level CI width {unit_width:.2f} should exceed the "
            f"row-level {row_width:.2f} by a clear margin"
        )

    def test_bootstrap_interval_contains_the_observed_statistic(self) -> None:
        """The point estimate is the most likely value; the interval should hold it."""
        rng = np.random.default_rng(2)
        groups = np.repeat(np.arange(6), 20)
        truth = rng.uniform(0, 125, size=groups.size)
        pred = truth + rng.normal(0, 5.0, size=groups.size)
        observed, low, high = grouped_bootstrap_ci(
            truth, pred, groups, "rmse", n_resamples=300, seed=0
        )
        assert low <= observed <= high

    def test_bootstrap_is_seed_deterministic(self) -> None:
        """Same seed, same interval, so a committed result can be reproduced."""
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(5), 10)
        truth = rng.uniform(0, 100, size=groups.size)
        pred = truth + rng.normal(0, 3.0, size=groups.size)
        first = grouped_bootstrap_ci(truth, pred, groups, "rmse", n_resamples=200, seed=7)
        second = grouped_bootstrap_ci(truth, pred, groups, "rmse", n_resamples=200, seed=7)
        assert first == second

    def test_nasa_score_statistic_is_supported(self) -> None:
        """The asymmetric metric can be bootstrapped like the symmetric one."""
        rng = np.random.default_rng(4)
        groups = np.repeat(np.arange(5), 10)
        truth = rng.uniform(0, 100, size=groups.size)
        pred = truth + rng.normal(0, 4.0, size=groups.size)
        value, low, high = grouped_bootstrap_ci(
            truth, pred, groups, "nasa_score", n_resamples=100, seed=0
        )
        assert value > 0.0
        assert low <= value <= high

    def test_unknown_statistic_raises(self) -> None:
        """A typo in a statistic name fails rather than returning a wrong number."""
        rng = np.random.default_rng(5)
        with pytest.raises(KeyError):
            grouped_bootstrap_ci(
                rng.uniform(0, 10, 10), rng.uniform(0, 10, 10), np.zeros(10, dtype=int), "nope"
            )


class TestCoverageRate:
    """Coverage is a bootstrap statistic, and its signature is a trap.

    `grouped_bootstrap_ci` calls the statistic as `func(y_true, y_pred)`, and
    coverage is a mean over the first argument alone. Passing anything else as the
    second argument silently averages the wrong thing: it produced a "95% confidence
    interval" of [18.0, 20.7] around an empirical coverage of 0.926, which is the
    kind of number that survives a review because it looks like a number.
    """

    def test_averages_the_indicator(self) -> None:
        from prognos.metrics import coverage_rate

        indicator = np.array([1.0, 0.0, 1.0, 1.0])
        assert coverage_rate(np.zeros(4), indicator) == pytest.approx(0.75)

    def test_first_argument_does_not_affect_the_result(self) -> None:
        """The first argument is the ignored half of the shared bootstrap signature."""
        from prognos.metrics import coverage_rate

        indicator = np.array([1.0, 0.0, 1.0])
        assert coverage_rate(np.zeros(3), indicator) == coverage_rate(
            np.array([99.0, -99.0, 99.0]), indicator
        )

    def test_bootstrap_interval_brackets_the_observed_coverage(self) -> None:
        """The interval must contain the point estimate it is an interval for."""
        from prognos.metrics import grouped_bootstrap_ci

        rng = np.random.default_rng(0)
        groups = rng.integers(0, 12, 4000)
        truth = rng.normal(100.0, 15.0, 4000)
        lower = truth - 10.0
        upper = truth + 10.0
        covered = ((truth >= lower) & (truth <= upper)).astype(np.float64)
        observed, low, high = grouped_bootstrap_ci(
            covered, covered, groups, statistic="coverage", n_resamples=200, seed=1
        )
        assert observed == pytest.approx(1.0)
        assert low <= observed <= high
        assert low == pytest.approx(1.0)
