"""Conformal prediction intervals.

A point estimate of remaining useful life is not actionable, because the decision is
"replace or do not replace" and the two mistakes cost different amounts. What the
maintenance policy needs is a range it can act on conservatively.

Two methods, reported side by side so the width difference is visible:

- **Split conformal** (`SplitConformalCalibrator`) turns held-out residuals into a
  constant half-width. Cheap and model-agnostic, but every prediction gets the same
  width, which is the wrong shape when early-life windows are far more uncertain
  than late-life ones.
- **Conformalised quantile regression** (`CqrCalibrator`) calibrates around a pair of
  quantile regressors, so width adapts locally. A wider region is a region the model
  itself is less sure about, and conformalisation keeps the coverage honest around
  that.

The guarantee is **marginal coverage under exchangeability**, and nothing stronger.
It does not promise coverage per engine, per life stage, or under distribution
shift. `src/prognos/decision.py` and the live SLO in the streaming phase exist because
marginal coverage is what you get, not what you want.

Calibration data must be disjoint from the data the model was fitted on, and
disjoint from the data coverage is reported on. `ensure_disjoint` states that
requirement in one place so a caller cannot satisfy it by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "CqrCalibrator",
    "Interval",
    "SplitConformalCalibrator",
    "conformal_quantile",
    "cqr_scores",
    "ensure_disjoint",
    "interval_metrics",
]


@dataclass(frozen=True, slots=True)
class Interval:
    """A prediction with a range around it.

    Frozen because an interval that gets edited in place is no longer a value; it is
    a mutable object that used to hold one, and the difference only shows up in a
    policy decision nobody re-reads.
    """

    lower: float
    point: float
    upper: float

    def __post_init__(self) -> None:
        """Reject an inverted interval at construction rather than downstream."""
        if self.lower > self.upper:
            raise ValueError(f"interval is inverted: {self.lower} > {self.upper}")

    @property
    def width(self) -> float:
        """Width of the range."""
        return self.upper - self.lower

    def covers(self, value: float) -> bool:
        """Whether the value falls inside the interval, bounds included."""
        return self.lower <= value <= self.upper


def conformal_quantile(scores: npt.NDArray[np.float64], target_coverage: float) -> float:
    """The finite-sample conformal quantile of `scores`.

    With `n` calibration scores the conformal quantile is the order statistic at rank
    `ceil((n + 1) * target_coverage)`, which is one rank higher than the plain
    `target_coverage` quantile and is what makes the coverage guarantee hold at every
    sample size rather than only asymptotically.

    When that rank exceeds `n` the target cannot be supported by the data available,
    and the answer is infinity. Returning the largest observed score instead would
    produce a confidently-too-narrow interval, which is the specific failure this
    module exists to prevent.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError(f"target coverage must be in (0, 1), got {target_coverage}")
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot take a conformal quantile of an empty calibration set")
    rank = math.ceil((values.size + 1) * target_coverage)
    if rank > values.size:
        return math.inf
    # `rank` is 1-based and at most values.size, so the index is in range.
    return float(np.sort(values)[rank - 1])


def ensure_disjoint(
    fitted_groups: npt.NDArray[np.int64],
    calibration_groups: npt.NDArray[np.int64],
) -> None:
    """Raise if any engine was both fitted on and used to calibrate.

    Split conformal is invalid when the residuals it calibrates on come from engines
    the model memorised: the residuals are then optimistically small, the interval is
    too narrow, and the coverage claim is false on any engine the model has not seen.
    The check is cheap and the failure is invisible in the numbers, so it lives here
    rather than in a comment.
    """
    overlap = sorted({int(g) for g in fitted_groups} & {int(g) for g in calibration_groups})
    if overlap:
        raise ValueError(
            f"calibration engines were also used for fitting: {overlap}. "
            "Conformal intervals require a disjoint calibration set."
        )


def interval_metrics(
    y_true: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Empirical coverage and mean width.

    Coverage is inclusive of both bounds. A truth exactly on a bound is covered, and
    getting that wrong under-reports coverage on every run by a small amount that is
    easy to mistake for a real effect.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if truth.size == 0:
        raise ValueError("cannot compute interval metrics over an empty set")
    if truth.shape != low.shape or truth.shape != high.shape:
        raise ValueError(
            f"shape mismatch: truth {truth.shape}, lower {low.shape}, upper {high.shape}"
        )
    covered = (truth >= low) & (truth <= high)
    finite = np.isfinite(high)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(high[finite] - low[finite])),
        "n": int(truth.size),
    }


def cqr_scores(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """CQR conformity score: the worse of the two edge violations.

    Taking the worse violation rather than the sum is what makes the correction
    symmetric. A prediction that runs too high at the lower edge and one that runs
    too low at the upper edge are equally wrong, and a score that treated them
    differently would produce an interval that is systematically shifted.

    The score is negative when the truth falls inside the quantile bounds, because
    the maximum of two negative numbers is the one nearer zero. That is not a bug: it
    is what lets conformalisation trim a quantile interval that is wider than it
    needs to be, instead of only ever padding it.
    """
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    if low.shape != high.shape or low.shape != truth.shape:
        raise ValueError(
            f"shape mismatch: lower {low.shape}, upper {high.shape}, truth {truth.shape}"
        )
    return np.maximum(low - truth, truth - high)


class _CalibratorBase:
    """Shared plumbing: the target rate, the conformal width, and the clipping rule."""

    __slots__ = ("_target_coverage", "_width")

    def __init__(self, target_coverage: float) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(f"target coverage must be in (0, 1), got {target_coverage}")
        self._target_coverage = float(target_coverage)
        self._width: float | None = None

    @property
    def target_coverage(self) -> float:
        """The nominal coverage rate."""
        return self._target_coverage

    @property
    def width(self) -> float:
        """The conformal correction. Raises until the calibrator has been fitted."""
        if self._width is None:
            raise ValueError("calibrator has not been calibrated")
        return self._width

    def _finish(self, scores: npt.NDArray[np.float64]) -> None:
        """Store the conformal quantile of the given scores."""
        self._width = conformal_quantile(scores, self._target_coverage)


class SplitConformalCalibrator(_CalibratorBase):
    """Constant-width conformal interval from held-out absolute residuals.

    The residual is taken as an absolute value, so the interval is symmetric about
    the point estimate. That is the price of model-agnosticism, and it is a real one:
    a model that systematically over-predicts in early life gets an interval centred
    on the wrong place. `CqrCalibrator` is the alternative that does not pay it.
    """

    __slots__ = ()

    def calibrate(self, residuals: npt.NDArray[np.float64]) -> SplitConformalCalibrator:
        """Fit the half-width from held-out residuals. Returns self for chaining."""
        values = np.asarray(residuals, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot calibrate on an empty residual set")
        self._finish(np.abs(values))
        return self

    def bounds(
        self, point: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Lower and upper bounds for a batch of point estimates.

        The lower bound is clipped at zero because remaining useful life cannot be
        negative. Without the clip a point estimate near failure yields a bound
        reaching into impossible territory, and a policy acting on that bound would
        trigger on arithmetic rather than on evidence.
        """
        if self._width is None:
            raise ValueError("calibrator has not been calibrated")
        estimate = np.asarray(point, dtype=np.float64)
        width = self._width
        return (
            np.maximum(estimate - width, 0.0),
            estimate + width,
        )

    def predict(self, point: npt.NDArray[np.float64]) -> Interval:
        """Interval for a single point estimate, as a value object."""
        lower, upper = self.bounds(np.atleast_1d(np.asarray(point, dtype=np.float64)))
        index = int(np.atleast_1d(point).size) - 1
        return Interval(
            lower=float(lower[index]),
            point=float(np.atleast_1d(point)[index]),
            upper=float(upper[index]),
        )


class CqrCalibrator(_CalibratorBase):
    """Conformalised quantile regression.

    Wraps a pair of quantile regressors and widens them by a single conformal
    correction learned from held-out data. Unlike split conformal the width varies
    per prediction, because the quantile regressors supply the local spread and the
    conformal term supplies the coverage.
    """

    __slots__ = ()

    def calibrate(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
        y_true: npt.NDArray[np.float64],
    ) -> CqrCalibrator:
        """Learn the conformal correction from held-out quantile predictions."""
        scores = cqr_scores(lower, upper, y_true)
        if scores.size == 0:
            raise ValueError("cannot calibrate on an empty set of quantile predictions")
        self._finish(scores)
        return self

    def bounds(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Widen quantile-regressor bounds by the conformal correction."""
        if self._width is None:
            raise ValueError("calibrator has not been calibrated")
        low = np.maximum(np.asarray(lower, dtype=np.float64) - self._width, 0.0)
        high = np.asarray(upper, dtype=np.float64) + self._width
        return low, high

    def predict(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> Interval:
        """Interval for one pair of quantile-regressor bounds."""
        low_arr, high_arr = self.bounds(lower, upper)
        return Interval(
            lower=float(np.atleast_1d(low_arr)[-1]),
            point=float((np.atleast_1d(lower)[-1] + np.atleast_1d(upper)[-1]) / 2.0),
            upper=float(np.atleast_1d(high_arr)[-1]),
        )


def coverage_by_stage(
    y_true: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    stages: npt.NDArray[np.str_],
    all_stages: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    """Coverage and width per life stage.

    Marginal coverage can hide a model that is well calibrated on average and badly
    calibrated exactly where the maintenance decision gets made. Reporting per stage
    is what makes that visible; the aggregate hides it by construction.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    labels = np.asarray(stages)
    result: dict[str, dict[str, float | int]] = {}
    for stage in all_stages:
        mask = labels == stage
        if not mask.any():
            result[stage] = {"n": 0, "coverage": float("nan"), "mean_width": float("nan")}
            continue
        result[stage] = interval_metrics(truth[mask], low[mask], high[mask])
    return result
