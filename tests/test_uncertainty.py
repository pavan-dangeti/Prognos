"""Tests for conformal prediction intervals.

The guarantee these tests defend is narrow and worth stating plainly: split
conformal gives *marginal* coverage over exchangeable points. It does not promise
coverage per engine, per life stage, or on a shifted distribution. A test that let
this code claim more than the theorem gives would be worse than no test, so most of
what is checked here is the boundary: the finite-sample quantile index, the honest
failure when there is too little calibration data, and the fact that the intervals
widen where the residuals are wider.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prognos.uncertainty import (
    CqrCalibrator,
    Interval,
    SplitConformalCalibrator,
    conformal_quantile,
    cqr_scores,
    interval_metrics,
)


class TestConformalQuantile:
    """The finite-sample quantile index is where conformal implementations go wrong."""

    def test_uses_the_finite_sample_correction(self) -> None:
        """The index is ceil((n+1)*coverage)/n, not the plain quantile.

        With n calibration residuals at coverage 0.9, the plain 0.9-quantile is the
        ceil(0.9n)-th order statistic. The finite-sample correction asks for the
        ceil(0.9(n+1))-th, which is one rank higher and is what makes the guarantee
        hold for every n rather than asymptotically.
        """
        scores = np.arange(1, 11, dtype=np.float64)  # 1..10, n = 10
        # ceil(0.9 * 11) = ceil(9.9) = 10 -> the 10th score.
        assert conformal_quantile(scores, 0.9) == 10.0
        # ceil(0.9 * 21) = ceil(18.9) = 19 -> the 19th score, not the 18th that a
        # plain 0.9-quantile of 20 points would give.
        twenty = np.arange(1, 21, dtype=np.float64)
        assert conformal_quantile(twenty, 0.9) == 19.0

    def test_reports_infinity_when_there_is_not_enough_data(self) -> None:
        """Too few calibration points cannot support the target, and must say so.

        At n = 5 and coverage 0.9 the required rank is ceil(0.9 * 6) = 6, which is
        past the end of the sample. The honest interval is unbounded, not a
        confidently-too-narrow one.
        """
        scores = np.arange(1, 6, dtype=np.float64)
        assert math.isinf(conformal_quantile(scores, 0.9))

    def test_rejects_invalid_inputs(self) -> None:
        """A nonsense target must raise, not silently return something usable."""
        with pytest.raises(ValueError, match="coverage"):
            conformal_quantile(np.arange(5.0), 1.5)
        with pytest.raises(ValueError, match="coverage"):
            conformal_quantile(np.arange(5.0), 0.0)
        with pytest.raises(ValueError, match="empty"):
            conformal_quantile(np.array([], dtype=np.float64), 0.9)


class TestSplitConformalCalibrator:
    def test_interval_is_symmetric_about_the_point_estimate(self) -> None:
        """Split conformal adds a constant to both sides of the point estimate."""
        rng = np.random.default_rng(0)
        residuals = rng.normal(loc=0.0, scale=5.0, size=400)
        calibrator = SplitConformalCalibrator(target_coverage=0.9).calibrate(residuals)
        interval = calibrator.predict(np.array([50.0]))
        assert interval.point == pytest.approx(50.0)
        assert interval.lower == pytest.approx(50.0 - calibrator.width)
        assert interval.upper == pytest.approx(50.0 + calibrator.width)

    def test_lower_bound_is_clipped_at_zero(self) -> None:
        """Remaining useful life cannot be negative.

        Without the clip, a point estimate near failure produces an interval
        reaching into impossible territory, and a threshold policy acting on that
        bound would trigger on arithmetic rather than on evidence.
        """
        rng = np.random.default_rng(1)
        residuals = rng.normal(loc=0.0, scale=20.0, size=500)
        calibrator = SplitConformalCalibrator(target_coverage=0.9).calibrate(residuals)
        interval = calibrator.predict(np.array([1.0]))
        assert interval.lower == 0.0
        assert interval.upper > interval.lower

    def test_width_grows_with_residual_spread(self) -> None:
        """A noisier model must produce wider intervals.

        If width did not track residual spread, the coverage guarantee would be
        vacuous: the same interval would be claimed for a model that is accurate
        and one that is not.
        """
        rng = np.random.default_rng(2)
        tight = SplitConformalCalibrator(0.9).calibrate(rng.normal(0.0, 1.0, 500))
        loose = SplitConformalCalibrator(0.9).calibrate(rng.normal(0.0, 20.0, 500))
        assert loose.width > tight.width * 5

    def test_higher_target_gives_wider_intervals(self) -> None:
        """Coverage 95% must dominate coverage 80% pointwise."""
        rng = np.random.default_rng(3)
        residuals = rng.normal(0.0, 5.0, 500)
        narrow = SplitConformalCalibrator(0.8).calibrate(residuals)
        wide = SplitConformalCalibrator(0.95).calibrate(residuals)
        assert wide.width > narrow.width

    def test_refuses_to_calibrate_on_nothing(self) -> None:
        """An uncalibrated predictor must not silently emit a zero-width interval."""
        with pytest.raises(ValueError, match="calibrat"):
            SplitConformalCalibrator(0.9).predict(np.array([10.0]))

    def test_infinite_width_is_reported_as_infinite(self) -> None:
        """When the calibration set cannot support the target, the interval says so."""
        calibrator = SplitConformalCalibrator(0.9).calibrate(np.arange(3.0))
        interval = calibrator.predict(np.array([40.0]))
        assert math.isinf(interval.upper)
        assert interval.lower == 0.0


class TestCqr:
    def test_score_is_the_worse_of_the_two_violations(self) -> None:
        """The CQR conformity score is max(lower - y, y - upper).

        A truth below the lower edge and one above the upper edge score the same
        positive amount. A truth *inside* scores negatively, because the maximum of
        two negative numbers is the one closer to zero: the quantile regressors got
        lucky there. A negative score is legitimate and is what lets CQR trim an
        over-wide quantile interval rather than only ever widening it.
        """
        lower = np.array([10.0, 10.0, 10.0])
        upper = np.array([20.0, 20.0, 20.0])
        truth = np.array([5.0, 15.0, 25.0])
        # below by 5, inside by 5 (score -5), above by 5
        np.testing.assert_allclose(cqr_scores(lower, upper, truth), [5.0, -5.0, 5.0])

    def test_score_is_symmetric_about_the_interval(self) -> None:
        """Over- and under-prediction by the same amount score identically."""
        lower = np.array([10.0, 90.0])
        upper = np.array([20.0, 100.0])
        truth = np.array([5.0, 105.0])  # 5 below the first interval, 5 above the second
        np.testing.assert_allclose(cqr_scores(lower, upper, truth), [5.0, 5.0])

    def test_width_follows_the_quantile_regressors(self) -> None:
        """The point of CQR is that width adapts locally.

        A region where the regressors already predict a wide interval should come
        out wider than one where they predict a narrow interval, and the conformal
        correction is what keeps coverage honest around it.
        """
        rng = np.random.default_rng(4)
        calib_lower = rng.normal(50.0, 3.0, 400)
        calib_truth = rng.normal(55.0, 3.0, 400)
        calibrator = CqrCalibrator(target_coverage=0.9).calibrate(
            calib_lower, calib_lower + 10.0, calib_truth
        )
        narrow = calibrator.predict(np.array([50.0]), np.array([55.0]))
        wide = calibrator.predict(np.array([50.0]), np.array([85.0]))
        assert wide.upper - wide.lower > narrow.upper - narrow.lower

    def test_correction_is_applied_verbatim_to_both_edges(self) -> None:
        """The conformal term shifts the raw bounds by exactly its own width.

        This is the invariant that makes the method checkable: given the
        calibrator's width, the bounds are fully determined, so a test can assert
        the arithmetic rather than inferring the correction from outputs.
        """
        rng = np.random.default_rng(5)
        lower = rng.normal(50, 5, 300)
        upper = lower + 10
        truth = rng.normal(55, 5, 300)
        calibrator = CqrCalibrator(target_coverage=0.9).calibrate(lower, upper, truth)
        got_lower, got_upper = calibrator.bounds(np.array([55.0]), np.array([65.0]))
        assert got_lower[0] == pytest.approx(max(55.0 - calibrator.width, 0.0))
        assert got_upper[0] == pytest.approx(65.0 + calibrator.width)

    def test_rejects_mismatched_shapes(self) -> None:
        """A shape mismatch must raise rather than broadcast something wrong."""
        with pytest.raises(ValueError, match="shape"):
            CqrCalibrator(0.9).calibrate(np.zeros(5), np.ones(4), np.zeros(5))


class TestIntervalMetrics:
    def test_coverage_counts_inclusive_bounds(self) -> None:
        """A true value exactly on the bound is covered.

        Getting this wrong is the classic off-by-one that quietly under-reports
        coverage by a little on every run.
        """
        lower = np.array([0.0, 10.0])
        upper = np.array([10.0, 20.0])
        truth = np.array([10.0, 15.0])
        assert interval_metrics(truth, lower, upper)["coverage"] == pytest.approx(1.0)

    def test_excludes_values_outside(self) -> None:
        lower = np.array([0.0, 10.0])
        upper = np.array([10.0, 20.0])
        truth = np.array([-1.0, 25.0])
        assert interval_metrics(truth, lower, upper)["coverage"] == pytest.approx(0.0)

    def test_mean_width_is_the_mean_of_upper_minus_lower(self) -> None:
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 30.0])
        truth = np.array([5.0, 15.0])
        assert interval_metrics(truth, lower, upper)["mean_width"] == pytest.approx(20.0)

    def test_rejects_empty_input(self) -> None:
        """Metrics over no rows are undefined and must not read as perfect coverage."""
        with pytest.raises(ValueError):
            interval_metrics(np.array([]), np.array([]), np.array([]))


class TestSplitConformalProperties:
    """The statistical claim, on data where it actually holds."""

    @settings(max_examples=15, deadline=None)
    @given(
        scale=st.floats(min_value=0.5, max_value=30.0),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    def test_marginal_coverage_lands_near_the_target(self, scale: float, seed: int) -> None:
        """Calibration and evaluation drawn from one distribution must cover ~90%.

        The finite-sample index is checked exactly, on a known array, in
        `test_uses_the_finite_sample_correction`. This checks the thing that index
        exists to buy: that the resulting interval really does cover.

        Point estimates sit well above zero because the lower bound is clipped at
        zero. That clip is correct for remaining useful life and would otherwise
        make this a test of the clip rather than of coverage.
        """
        rng = np.random.default_rng(seed)
        point = np.full(2000, 100.0)
        calibration = rng.normal(0.0, scale, 2000)
        evaluation = 100.0 + rng.normal(0.0, scale, 2000)

        calibrator = SplitConformalCalibrator(target_coverage=0.9).calibrate(calibration)
        lower, upper = calibrator.bounds(point)
        covered = interval_metrics(evaluation, lower, upper)["coverage"]

        # Measured, not guessed. Across 100 seeds at n = 2000 the observed coverage runs
        # 0.879 to 0.925 with a standard deviation of 0.009, against a pure binomial
        # standard deviation of 0.0067; the calibration quantile contributes the rest.
        # The bound is set from that measurement rather than from whatever happened to
        # pass on one run, and it still sits far inside the 0.1 gap to the next
        # plausible wrong target level, so a level error fails this.
        assert 0.86 <= covered <= 0.94, f"coverage {covered:.3f} off target at scale {scale}"

    @settings(max_examples=10, deadline=None)
    @given(
        target=st.floats(min_value=0.5, max_value=0.95),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    def test_bounds_never_invert(self, target: float, seed: int) -> None:
        """Lower <= upper for every prediction, whatever the input.

        An inverted interval silently reports zero width and full coverage of
        nothing, so the invariant is worth stating directly.
        """
        rng = np.random.default_rng(seed)
        residuals = rng.normal(0.0, 8.0, 300)
        calibrator = SplitConformalCalibrator(target_coverage=target).calibrate(residuals)
        point = rng.uniform(0.0, 200.0, 100)
        lower, upper = calibrator.bounds(point)
        assert np.all(lower <= upper)
        assert np.all(np.isfinite(upper) | np.isinf(upper))


class TestIntervalIsImmutable:
    def test_cannot_be_mutated(self) -> None:
        """An interval that gets edited in place is not a value any more."""
        interval = Interval(lower=0.0, point=5.0, upper=10.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            interval.lower = 1.0  # type: ignore[misc]

    def test_ordering_is_checked(self) -> None:
        """An interval with lower above upper is a bug, not a valid value."""
        with pytest.raises(ValueError):
            Interval(lower=10.0, point=5.0, upper=0.0)


class TestDisjointnessAndEdgeCases:
    """The guards that stop a plausible-looking invalid claim."""

    def test_disjoint_engines_pass(self) -> None:
        from prognos.uncertainty import ensure_disjoint

        ensure_disjoint(np.array([0, 1]), np.array([2, 3]))

    def test_overlapping_engines_raise(self) -> None:
        """Calibrating on an engine that was fitted on invalidates the coverage.

        The residuals would be optimistically small, the interval too narrow, and the
        failure invisible in the reported number. Checked in code because it cannot
        be checked from the result.
        """
        from prognos.uncertainty import ensure_disjoint

        with pytest.raises(ValueError, match="also used for fitting"):
            ensure_disjoint(np.array([0, 1, 2]), np.array([2, 3]))

    def test_width_and_covers_on_the_value_object(self) -> None:
        interval = Interval(lower=10.0, point=20.0, upper=30.0)
        assert interval.width == pytest.approx(20.0)
        assert interval.covers(20.0)
        assert interval.covers(10.0)
        assert not interval.covers(9.9)

    def test_width_before_calibration_raises(self) -> None:
        """Reading the width off an uncalibrated predictor must not return zero."""
        from prognos.uncertainty import CqrCalibrator, SplitConformalCalibrator

        with pytest.raises(ValueError, match="not been calibrated"):
            _ = SplitConformalCalibrator(0.9).width
        with pytest.raises(ValueError, match="not been calibrated"):
            _ = CqrCalibrator(0.9).width

    def test_cqr_bounds_before_calibration_raise(self) -> None:
        from prognos.uncertainty import CqrCalibrator

        with pytest.raises(ValueError, match="not been calibrated"):
            CqrCalibrator(0.9).bounds(np.array([1.0]), np.array([2.0]))

    def test_empty_calibration_sets_raise(self) -> None:
        from prognos.uncertainty import CqrCalibrator, SplitConformalCalibrator

        with pytest.raises(ValueError, match="empty"):
            SplitConformalCalibrator(0.9).calibrate(np.array([]))
        with pytest.raises(ValueError, match="empty"):
            CqrCalibrator(0.9).calibrate(np.array([]), np.array([]), np.array([]))

    def test_interval_metrics_rejects_mismatched_shapes(self) -> None:
        """Broadcasting a wrong shape would produce a confident, meaningless coverage."""
        with pytest.raises(ValueError, match="shape mismatch"):
            interval_metrics(np.zeros(4), np.zeros(3), np.zeros(4))

    def test_target_coverage_is_validated_on_both_calibrators(self) -> None:
        from prognos.uncertainty import CqrCalibrator, SplitConformalCalibrator

        for bad in (0.0, 1.0, 1.5, -0.2):
            with pytest.raises(ValueError, match="target coverage"):
                SplitConformalCalibrator(bad)
            with pytest.raises(ValueError, match="target coverage"):
                CqrCalibrator(bad)

    def test_an_empty_stage_is_reported_not_dropped(self) -> None:
        """A stage with no windows must show n = 0, not vanish from the table."""
        from prognos.metrics import LIFE_STAGES
        from prognos.uncertainty import coverage_by_stage

        truth = np.array([5.0, 5.0])
        stages = np.array(["early", "early"])
        table = coverage_by_stage(truth, truth - 1.0, truth + 1.0, stages, tuple(LIFE_STAGES))
        assert set(table) == set(LIFE_STAGES)
        assert table["late"]["n"] == 0
        assert math.isnan(table["late"]["coverage"])

    def test_split_calibrator_target_is_readable(self) -> None:
        calibrator = SplitConformalCalibrator(0.95).calibrate(np.arange(100.0))
        assert calibrator.target_coverage == pytest.approx(0.95)
