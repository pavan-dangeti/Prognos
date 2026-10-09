"""Models. Each predicts RUL in cycles.

Every model here is deliberately simple and every one is an honest comparison. The
point of the baselines is to establish what a trivially cheap approach achieves, so
a more complex model has to beat a real number rather than an assumption.

`predict` takes cycle numbers alongside the windows because cycle index is a
legitimate input: it is known at inference time, it is not ground truth, and
excluding it would handicap the models that could otherwise use it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

# Smallest per-step change, as a fraction of a channel's own observed range, that
# counts as degradation rather than noise. Below this the fitted trend is flat to
# within the resolution of the window, and extrapolating it to a threshold produces
# an enormous, meaningless remaining life.
MIN_RELATIVE_SLOPE = 1e-4


class RulModel(Protocol):
    """What the evaluation harness needs from any model."""

    name: str

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
    ) -> None:
        """Fit on training windows."""

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Predict RUL for each window."""


def check_shapes(
    windows: npt.NDArray[np.float32],
    cycles: npt.NDArray[np.int64],
    targets: npt.NDArray[np.float32],
) -> None:
    """Reject empty or ragged inputs before any fitting happens."""
    n = windows.shape[0]
    if cycles.shape[0] != n or targets.shape[0] != n:
        raise ValueError(
            f"mismatched rows: windows {n}, cycles {cycles.shape[0]}, targets {targets.shape[0]}"
        )
    if n == 0:
        raise ValueError("cannot work with zero windows")


def check_predict_shapes(windows: npt.NDArray[np.float32], cycles: npt.NDArray[np.int64]) -> None:
    """Same checks at prediction time, so a bad call is not silently broadcast."""
    if windows.shape[0] == 0:
        raise ValueError("cannot predict on zero windows")
    if cycles.shape[0] != windows.shape[0]:
        raise ValueError(f"cycles {cycles.shape[0]} do not match windows {windows.shape[0]}")


@dataclass(slots=True)
class MeanBaseline:
    """Predict the training set's mean RUL for every window.

    The floor. If a model cannot beat this, it is reading noise.
    """

    name: str = "mean"
    value: float = 0.0

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
    ) -> None:
        """Store the training mean."""
        check_shapes(windows, cycles, targets)
        self.value = float(np.mean(targets))

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Return the constant."""
        check_predict_shapes(windows, cycles)
        return np.full(windows.shape[0], self.value, dtype=np.float32)


@dataclass(slots=True)
class LinearCycleBaseline:
    """Least-squares fit of RUL against cycle number.

    A naive model carrying real signal: RUL falls as the unit ages. It uses no
    sensor data at all, which makes it a useful check that cycle index alone is not
    already doing all the work in the models that do use sensors.
    """

    name: str = "linear_cycle"
    slope: float = 0.0
    intercept: float = 0.0

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
    ) -> None:
        """Fit RUL ~ intercept + slope * cycle."""
        check_shapes(windows, cycles, targets)
        design = np.column_stack([np.ones(cycles.shape[0]), cycles.astype(np.float64)])
        coefficients, *_ = np.linalg.lstsq(design, targets.astype(np.float64), rcond=None)
        self.intercept, self.slope = float(coefficients[0]), float(coefficients[1])

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Evaluate the fitted line, clipped at zero."""
        check_predict_shapes(windows, cycles)
        estimate = self.intercept + self.slope * cycles.astype(np.float64)
        return np.clip(estimate, 0.0, None).astype(np.float32)


@dataclass(slots=True)
class TrendModel:
    """Per-channel linear trend extrapolated to a degradation threshold.

    The physics-adjacent baseline the design calls for. Each sensor channel's own
    window is fitted with a least-squares line, projected forward until it crosses
    a per-channel limit, and the number of steps to that crossing is the predicted
    remaining life.

    It uses only sensor readings and it extrapolates rather than interpolates, so it
    answers "how long until this channel reaches its limit" the way an engineer
    would. It is also unstable on the flat early-life portion of a unit, which is
    exactly why the harness reports error by life stage rather than as one number.

    Per-channel limits come from the range of the training data plus a margin, so
    they adapt to each channel's units instead of being hand-tuned.
    """

    name: str = "trend"
    limits: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    spans: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
    ) -> None:
        """Derive per-channel degradation limits from the training distribution."""
        check_shapes(windows, cycles, targets)
        self.limits, self.spans = channel_limits(windows)

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Steps to the nearest threshold crossing, per window."""
        check_predict_shapes(windows, cycles)
        length, channels = int(windows.shape[1]), int(windows.shape[2])
        if self.limits.shape[0] != channels:
            raise ValueError(f"model has {self.limits.shape[0]} limits for {channels} channels")
        if length < 2:
            raise ValueError("a trend needs at least two samples")

        steps = np.arange(length, dtype=np.float64)
        centred = steps - steps.mean()
        denominator = float(np.sum(centred**2))
        block = windows.astype(np.float64)

        slopes = np.einsum("nlc,l->nc", block, centred) / denominator
        last = block[:, -1, :]

        # Work in units of each channel's own range. A channel measured in psia and
        # one measured in rpm cannot be compared on raw slope, and dividing a raw
        # slope by a raw gap is what let a near-flat channel report a life of a
        # million cycles.
        spans = self.spans[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_slope = slopes / spans
            relative_gap = (self.limits[None, :] - last) / spans
            distance = relative_gap / relative_slope

        # A channel moving less than a ten-thousandth of its own range per step is
        # noise at this resolution, not degradation. Without this floor, flat
        # early-life windows report an enormous remaining life.
        usable = (
            np.isfinite(distance)
            & (np.abs(relative_slope) >= MIN_RELATIVE_SLOPE)
            & (distance >= 0.0)
        )
        candidates = np.where(usable, distance, np.inf)

        nearest = np.min(candidates, axis=1)
        # A crossing further away than the window can resolve is not a measurement.
        # One full window is the finest resolution this input supports, and it is
        # also the fallback when nothing is trending toward a limit. Reporting a
        # million cycles of life because no channel has begun to degrade would be
        # the most dangerous possible answer here.
        horizon = np.where(np.isfinite(nearest), nearest, float(length))
        return np.clip(horizon, 0.0, float(length)).astype(np.float32)


def channel_limits(
    history: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-channel degradation limits and observed spans.

    The limit is the top of the observed range plus 25% margin, derived from the
    data rather than fixed, so the same code works per channel regardless of units.

    The span is returned alongside because a trend is only meaningful relative to
    how much a channel moves in the first place: a channel drifting by 1e-9 per step
    reaches its limit eventually and would report an enormous life. Comparing the
    per-step trend against the channel's own span is what makes the estimate
    commensurable across channels measured in different units.

    A channel with no spread gets a unit-scale limit rather than a division by zero.
    """
    block = np.asarray(history, dtype=np.float64).reshape(-1, history.shape[-1])
    high = block.max(axis=0)
    span = block.max(axis=0) - block.min(axis=0)
    safe_span = np.where(span > 1e-9, span, 1.0)
    limits = np.asarray(high + 0.25 * safe_span, dtype=np.float64)
    return limits, safe_span


def build_models() -> dict[str, RulModel]:
    """The Phase 2 model set, in reporting order.

    Ordered by increasing cost so the cheap ones establish the floor first.
    """
    return {
        "mean": MeanBaseline(),
        "linear_cycle": LinearCycleBaseline(),
        "trend": TrendModel(),
    }


def constant_feature_report(
    names: Sequence[str], features: npt.NDArray[np.float32]
) -> dict[str, object]:
    """Which channels never move, for the run report.

    C-MAPSS contains several sensors that are constant within a sub-dataset. Saying
    so is more useful than silently dropping them.
    """
    std = features.std(axis=0)
    constant = [n for n, s in zip(names, std, strict=True) if s <= 1e-12]
    return {"n_features": int(features.shape[1]), "n_constant": len(constant), "constant": constant}
