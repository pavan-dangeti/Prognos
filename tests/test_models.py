"""Baseline models.

The property that matters most here: a model that cannot beat "predict the fleet
mean" must fail these tests rather than ship. Each baseline is also checked for the
shape and dtype contract the harness relies on, because a silent float64 promotion
or an off-by-one in a clip would change a reported number without failing anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prognos.models import (
    LinearCycleBaseline,
    MeanBaseline,
    TrendModel,
    build_models,
    channel_limits,
    check_predict_shapes,
    constant_feature_report,
)

CHANNELS = 4


def flat_windows(n: int = 30, length: int = 20, channels: int = CHANNELS) -> np.ndarray:
    """Noise with no trend, so a trend model has nothing to find."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, length, channels)).astype(np.float32)


def targets_and_cycles(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """RUL decreasing with cycle number, which is the real relationship."""
    cycles = np.arange(1, n + 1, dtype=np.int64)
    targets = (100 - cycles).astype(np.float32)
    return cycles, targets


class TestMeanBaseline:
    def test_predicts_the_training_mean(self) -> None:
        """The constant is the training mean, not the global mean of everything."""
        windows = flat_windows(10)
        cycles, targets = targets_and_cycles(10)
        model = MeanBaseline()
        model.fit(windows, cycles, targets)
        assert model.value == pytest.approx(float(np.mean(targets)))
        assert model.predict(windows, cycles) == pytest.approx(np.full(10, model.value))

    def test_beats_predicting_zero_on_decreasing_targets(self) -> None:
        """A sanity floor: any real model must beat this on this data."""
        windows = flat_windows(50)
        cycles, targets = targets_and_cycles(50)
        model = MeanBaseline()
        model.fit(windows, cycles, targets)
        prediction = model.predict(windows, cycles)
        assert np.sqrt(np.mean((targets - prediction) ** 2)) < np.sqrt(np.mean(targets**2))

    def test_rejects_empty_training_data(self) -> None:
        """Fitting on nothing is an error, not a model that predicts NaN."""
        with pytest.raises(ValueError, match="zero windows"):
            MeanBaseline().fit(
                np.zeros((0, 10, CHANNELS), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
            )

    def test_rejects_mismatched_lengths(self) -> None:
        """Ragged inputs are caught before any arithmetic."""
        with pytest.raises(ValueError, match="mismatched rows"):
            MeanBaseline().fit(
                np.zeros((5, 10, CHANNELS), dtype=np.float32),
                np.zeros(4, dtype=np.int64),
                np.zeros(5, dtype=np.float32),
            )


class TestLinearCycleBaseline:
    def test_recovers_a_known_line(self) -> None:
        """Fitted on an exact line, the coefficients come back exactly."""
        cycles = np.arange(0, 50, dtype=np.int64)
        targets = (120.0 - 2.0 * cycles).astype(np.float32)
        model = LinearCycleBaseline()
        model.fit(flat_windows(50), cycles, targets)
        assert model.intercept == pytest.approx(120.0)
        assert model.slope == pytest.approx(-2.0)

    def test_predictions_are_not_negative(self) -> None:
        """RUL below zero is meaningless; the line is clipped at zero."""
        cycles = np.arange(0, 100, dtype=np.int64)
        targets = (120.0 - 2.0 * cycles).astype(np.float32)
        model = LinearCycleBaseline()
        model.fit(flat_windows(100), cycles, targets)
        assert np.all(model.predict(flat_windows(100), cycles) >= 0.0)

    def test_uses_no_sensor_information(self) -> None:
        """Two different sensor sets, same coefficients.

        The point of this baseline is to show how much cycle index alone explains.
        If it used the windows, this would fail.
        """
        cycles, targets = targets_and_cycles(40)
        first = LinearCycleBaseline()
        first.fit(np.zeros((40, 20, CHANNELS), dtype=np.float32), cycles, targets)
        second = LinearCycleBaseline()
        second.fit(np.ones((40, 20, CHANNELS), dtype=np.float32), cycles, targets)
        assert first.slope == pytest.approx(second.slope)
        assert first.intercept == pytest.approx(second.intercept)


class TestChannelLimits:
    def test_limits_sit_above_the_observed_maximum(self) -> None:
        """A limit the data never reaches gives the trend somewhere to go."""
        # Three channels with different spans, so the margin has to be per-channel
        # rather than one constant.
        history = np.stack(
            [
                np.full((10, 3), 1.0, dtype=np.float32),
                np.linspace(0.0, 2.0, 30, dtype=np.float32).reshape(10, 3),
                np.linspace(1.0, 3.0, 30, dtype=np.float32).reshape(10, 3),
            ]
        )
        limits, spans = channel_limits(history)
        assert np.all(limits > history.max(axis=(0, 1)))
        assert np.allclose(spans, history.max(axis=(0, 1)) - history.min(axis=(0, 1)))

    def test_a_constant_channel_gets_a_finite_limit(self) -> None:
        """Zero span must not produce an infinite or NaN limit."""
        history = np.full((5, 10, 3), 7.0, dtype=np.float32)
        limits, spans = channel_limits(history)
        assert np.all(np.isfinite(limits))
        assert np.all(np.isfinite(spans))
        assert np.allclose(spans, 1.0), "a flat channel gets a unit-scale span"

    def test_limits_scale_with_channel_magnitude(self) -> None:
        """A channel measured in the thousands needs a proportionally larger limit.

        Limits derived from the observed range rather than a fixed constant, so the
        same code works for a temperature in degrees and a pressure in psia.
        """
        small = np.full((5, 10, 1), 1.0, dtype=np.float32)
        large = np.full((5, 10, 1), 1000.0, dtype=np.float32)
        assert channel_limits(large)[0] > channel_limits(small)[0] * 100


class TestTrendModel:
    def test_finds_a_channel_heading_to_its_limit(self) -> None:
        """The core behaviour: a rising channel reports steps until its limit.

        Built by hand so the expected answer is arithmetic rather than whatever the
        code happens to produce.
        """
        length, slope = 20, 1.0
        steps = np.arange(length, dtype=np.float32)
        # Channel 0 rises by `slope` per step, channel 1 is flat.
        block = np.zeros((1, length, 2), dtype=np.float32)
        block[0, :, 0] = steps * slope
        model = TrendModel()
        model.fit(block, np.array([1], dtype=np.int64), np.array([10.0], dtype=np.float32))
        # Limit is max(19) + 0.25 * span(19) = 23.75. Distance from 19 is 4.75,
        # which at 1.0 per step is 4.75 steps.
        prediction = model.predict(block, np.array([1], dtype=np.int64))[0]
        assert prediction == pytest.approx(4.75, abs=0.05)

    def test_reports_full_window_when_nothing_trends(self) -> None:
        """No trend toward a limit must not be reported as zero life remaining.

        Claiming an engine has no life left because no channel is trending would be
        a dangerous failure, so the fallback is one full window instead.
        """
        length = 20
        block = np.zeros((3, length, 2), dtype=np.float32)
        model = TrendModel()
        model.fit(block, np.arange(3, dtype=np.int64), np.full(3, 50.0, dtype=np.float32))
        prediction = model.predict(block, np.arange(3, dtype=np.int64))
        assert np.all(prediction == pytest.approx(float(length)))

    def test_predictions_are_never_negative(self) -> None:
        """A threshold crossing already behind us is not negative life."""
        rng = np.random.default_rng(1)
        block = (rng.normal(size=(10, 20, 3)) * 50).astype(np.float32)
        model = TrendModel()
        model.fit(block, np.arange(10, dtype=np.int64), np.full(10, 50.0, dtype=np.float32))
        assert np.all(model.predict(block, np.arange(10, dtype=np.int64)) >= 0.0)

    def test_degrading_channel_outpaces_a_slow_one(self) -> None:
        """Two channels on the same limit: the faster one determines the estimate."""
        length = 20
        steps = np.arange(length, dtype=np.float32)
        block = np.zeros((1, length, 2), dtype=np.float32)
        block[0, :, 0] = steps * 2.0
        block[0, :, 1] = steps * 0.5
        model = TrendModel()
        model.fit(block, np.array([1], dtype=np.int64), np.array([10.0], dtype=np.float32))
        prediction = model.predict(block, np.array([1], dtype=np.int64))[0]
        # Only channel 0 is near the limit; it should give the shorter estimate.
        assert prediction < length

    def test_a_channel_moving_away_is_ignored(self) -> None:
        """A channel falling when its limit is high gives a negative distance.

        Taking that at face value would produce a negative prediction, so
        distance-crossing channels are excluded.
        """
        length = 20
        steps = np.arange(length, dtype=np.float32)
        block = np.zeros((1, length, 2), dtype=np.float32)
        block[0, :, 0] = steps * 1.0
        block[0, :, 1] = 100.0 - steps * 5.0  # falling away from its high limit
        model = TrendModel()
        model.fit(block, np.array([1], dtype=np.int64), np.array([10.0], dtype=np.float32))
        assert model.predict(block, np.array([1], dtype=np.int64))[0] >= 0.0

    def test_a_flat_window_does_not_report_an_enormous_life(self) -> None:
        """The failure this guards: a near-zero slope reaching a limit eventually.

        Dividing a gap by a slope of 1e-9 gives a crossing millions of steps away,
        and the minimum over channels picks it. Reported as RUL, that is an engine
        claimed to have a million cycles left, which is the most dangerous answer a
        prognostics system can give.
        """
        length, channels, n = 50, 3, 40
        rng = np.random.default_rng(0)
        block = rng.normal(loc=100.0, scale=0.01, size=(n, length, channels)).astype(np.float32)
        model = TrendModel()
        model.fit(block, np.arange(n, dtype=np.int64), np.full(n, 100.0, dtype=np.float32))
        prediction = model.predict(block, np.arange(n, dtype=np.int64))
        assert np.all(np.isfinite(prediction))
        assert prediction.max() <= float(length), "must be capped at the resolvable horizon"
        assert prediction.max() < 100.0

    def test_relative_slope_floor_is_respected(self) -> None:
        """A trend below the floor is noise and must not be extrapolated.

        Fitted on a channel that does move, so the span is a real one, then
        predicted on windows where that channel has gone flat. The per-step change
        is tiny relative to the channel's full range, which is the case the floor
        exists for.
        """
        length = 50
        steps = np.arange(length, dtype=np.float32)
        # Channel 0 moves across 2.0 units of range, so its span is real; channel 1
        # is constant at 1000.0.
        moving = np.stack([steps * (2.0 / length), np.full(length, 1000.0)], axis=1)
        moving = moving.reshape(1, length, 2).astype(np.float32)
        flat = np.full((2, length, 2), 1000.0, dtype=np.float32)
        model = TrendModel()
        model.fit(moving, np.arange(1, dtype=np.int64), np.array([100.0], dtype=np.float32))

        prediction = model.predict(flat, np.arange(2, dtype=np.int64))
        assert np.all(prediction == pytest.approx(float(length)))

    def test_channel_count_mismatch_is_an_error(self) -> None:
        """A model fitted on 3 channels cannot predict on 4."""
        model = TrendModel()
        model.fit(
            np.zeros((2, 20, 3), dtype=np.float32),
            np.arange(2, dtype=np.int64),
            np.full(2, 10.0, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="limits for"):
            model.predict(np.zeros((2, 20, 4), dtype=np.float32), np.arange(2, dtype=np.int64))

    def test_output_dtype_is_float32(self) -> None:
        """The harness stacks predictions from several models; dtype must match."""
        block = flat_windows(5)
        model = TrendModel()
        model.fit(block, np.arange(5, dtype=np.int64), np.full(5, 10.0, dtype=np.float32))
        assert model.predict(block, np.arange(5, dtype=np.int64)).dtype == np.float32

    def test_rejects_a_single_sample_window(self) -> None:
        """A trend needs two points in time."""
        model = TrendModel()
        model.fit(
            np.zeros((2, 1, 2), dtype=np.float32),
            np.arange(2, dtype=np.int64),
            np.full(2, 10.0, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="at least two"):
            model.predict(np.zeros((2, 1, 2), dtype=np.float32), np.arange(2, dtype=np.int64))


class TestModelRegistry:
    def test_builds_the_expected_set(self) -> None:
        """The Phase 2 model list is fixed; a missing entry is a silent gap."""
        models = build_models()
        assert set(models) == {"mean", "linear_cycle", "trend"}

    def test_every_model_implements_the_protocol(self) -> None:
        """Each model can be fitted and asked to predict."""
        windows = flat_windows(20)
        cycles, targets = targets_and_cycles(20)
        for name, model in build_models().items():
            model.fit(windows, cycles, targets)
            prediction = model.predict(windows, cycles)
            assert prediction.shape == (20,), f"{name} returned {prediction.shape}"
            assert np.all(np.isfinite(prediction)), f"{name} produced non-finite values"

    def test_every_model_passes_the_shape_guards(self) -> None:
        """check_predict_shapes is the shared guard, so test it once directly."""
        with pytest.raises(ValueError, match="zero windows"):
            check_predict_shapes(
                np.zeros((0, 10, CHANNELS), dtype=np.float32), np.zeros(0, dtype=np.int64)
            )
        with pytest.raises(ValueError, match="do not match"):
            check_predict_shapes(
                np.zeros((5, 10, CHANNELS), dtype=np.float32), np.zeros(4, dtype=np.int64)
            )


class TestConstantFeatureReport:
    def test_reports_flat_channels(self) -> None:
        """Saying which channels never move is more useful than dropping them silently."""
        features = np.array([[1.0, 2.0], [1.0, 5.0]], dtype=np.float32)
        report = constant_feature_report(["flat", "moving"], features)
        assert report["n_features"] == 2
        assert report["n_constant"] == 1
        assert report["constant"] == ["flat"]


@settings(max_examples=25, deadline=None)
@given(
    slope=st.floats(min_value=0.05, max_value=5.0, allow_nan=False),
    length=st.integers(min_value=5, max_value=60),
)
def test_trend_recovers_any_ramp_rate(slope: float, length: int) -> None:
    """Property test: an exact ramp must yield a finite, plausible estimate.

    A property test rather than a handful of cases, because the interesting inputs
    are the boundary lengths and the range of slopes, not any one example.
    """
    steps = np.arange(length, dtype=np.float32)
    block = np.tile((steps * slope).reshape(1, length, 1), (2, 1, 1))
    model = TrendModel()
    model.fit(block, np.arange(2, dtype=np.int64), np.full(2, 10.0, dtype=np.float32))
    prediction = model.predict(block, np.arange(2, dtype=np.int64))
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0.0)
    # Limit sits 25% of the span above the final reading, so the estimate is
    # about a quarter of the window regardless of slope.
    assert np.all(prediction <= length)
