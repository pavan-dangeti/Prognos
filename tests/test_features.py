"""Feature extraction for the tabular baseline.

The properties tested here are the ones that silently produce a plausible model
that is wrong: a slope computed on the wrong axis, statistics that ignore their
argument, a normaliser fitted on the wrong data.
"""

from __future__ import annotations

import numpy as np
import pytest

from prognos.features import (
    STATS,
    FeatureNormaliser,
    all_feature_names,
    build_matrix,
    channel_count,
    drop_constant,
    feature_names,
    sequence_length_context,
    window_features,
)
from prognos.window import FEATURE_COLUMNS

CHANNELS = len(FEATURE_COLUMNS)


def synthetic_windows(
    *,
    n: int = 40,
    length: int = 20,
    channels: int = CHANNELS,
    slope: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Windows whose channels all ramp at a known rate."""
    rng = np.random.default_rng(seed)
    steps = np.arange(length, dtype=np.float32)
    base = rng.normal(size=(n, length, channels)).astype(np.float32)
    ramp = (steps * slope).reshape(1, length, 1)
    return (base + ramp).astype(np.float32)


class TestWindowFeatures:
    def test_shape_is_n_times_channels_times_stats(self) -> None:
        """One row of statistics per channel, in the documented order."""
        windows = synthetic_windows(n=7, length=20)
        features = window_features(windows)
        assert features.shape == (7, CHANNELS * len(STATS))

    def test_feature_names_match_the_output_width(self) -> None:
        """A name/column mismatch would silently mislabel every feature."""
        assert len(feature_names()) == CHANNELS * len(STATS)
        assert len(all_feature_names()) == CHANNELS * len(STATS) + 2

    def test_mean_feature_is_the_window_mean(self) -> None:
        """Checked against the definition, not a second call to the code."""
        windows = synthetic_windows(n=5, length=20)
        features = window_features(windows)
        assert features[:, 0] == pytest.approx(windows.mean(axis=1)[:, 0])

    def test_last_feature_is_the_final_sample(self) -> None:
        """The most recent reading, which is what an operator would look at."""
        windows = synthetic_windows(n=5, length=20)
        features = window_features(windows)
        offset = STATS.index("last") * CHANNELS
        assert features[:, offset] == pytest.approx(windows[:, -1, 0])

    def test_slope_recovers_a_known_ramp(self) -> None:
        """A channel rising by `s` per step must have slope `s`.

        This is the only feature that carries trend, which is what degradation is.
        If the slope were computed over the channel axis instead of the time axis,
        this test would fail and the model would look fine.
        """
        length, slope, n_channels = 30, 0.75, 4
        steps = np.arange(length, dtype=np.float32)
        windows = np.tile((steps * slope).reshape(1, length, 1), (3, 1, n_channels))
        features = window_features(windows)
        offset = STATS.index("slope") * n_channels
        assert features[:, offset] == pytest.approx(slope, abs=1e-4)

    def test_slope_is_independent_of_window_position(self) -> None:
        """A centred fit must not depend on where in the life the window sits.

        Windows are centred so that the slope measures rate of change per step. An
        uncentred fit would make the same degradation look steeper late in a unit
        purely because the absolute values are larger.
        """
        steps, n_channels = np.arange(20, dtype=np.float32), 2
        low = np.tile((steps * 0.5).reshape(1, 20, 1), (2, 1, n_channels))
        high = np.tile((steps * 0.5 + 1000.0).reshape(1, 20, 1), (2, 1, n_channels))
        offset = STATS.index("slope") * n_channels
        assert window_features(low)[:, offset] == pytest.approx(
            window_features(high)[:, offset], abs=1e-2
        )

    def test_flat_window_has_zero_slope(self) -> None:
        """A constant channel has no trend."""
        windows = np.ones((3, 20, CHANNELS), dtype=np.float32)
        offset = STATS.index("slope") * CHANNELS
        assert window_features(windows)[:, offset] == pytest.approx(0.0, abs=1e-5)

    def test_rejects_a_single_sample_window(self) -> None:
        """A slope needs two points, so a length-1 window is an error."""
        with pytest.raises(ValueError, match="at least two"):
            window_features(np.ones((3, 1, CHANNELS), dtype=np.float32))

    def test_rejects_wrong_dimensionality(self) -> None:
        """A 2-D input is a bug, not something to guess about."""
        with pytest.raises(ValueError, match="expected"):
            window_features(np.ones((3, 20), dtype=np.float32))

    def test_min_and_max_bracket_the_window(self) -> None:
        """A sanity check on the spread statistics."""
        windows = synthetic_windows(n=6, length=20)
        features = window_features(windows)
        block_min = features[:, STATS.index("min") * CHANNELS : (STATS.index("min") + 1) * CHANNELS]
        block_max = features[:, STATS.index("max") * CHANNELS : (STATS.index("max") + 1) * CHANNELS]
        assert np.all(block_min <= block_max + 1e-6)


class TestCycleContext:
    def test_fraction_is_relative_to_the_longest_life(self) -> None:
        """Normalising by max life keeps units of different length comparable."""
        cycles = np.array([50, 100], dtype=np.int64)
        context = sequence_length_context(cycles, 100)
        assert context[0, 0] == pytest.approx(0.5)
        assert context[1, 0] == pytest.approx(1.0)

    def test_context_has_two_columns(self) -> None:
        """Fraction and cycles-in-hundreds."""
        context = sequence_length_context(np.array([10], dtype=np.int64), 100)
        assert context.shape == (1, 2)


class TestBuildMatrix:
    def test_matrix_is_statistics_plus_context(self) -> None:
        """Width is derived, not hard-coded, so a STATS change cannot desync it."""
        windows = synthetic_windows(n=9, length=20)
        matrix = build_matrix(windows, np.arange(9, dtype=np.int64) + 1, max_cycle=100)
        assert matrix.shape == (9, CHANNELS * len(STATS) + 2)

    def test_matrix_is_finite(self) -> None:
        """No NaN or inf reaches the model; LightGBM would train on garbage."""
        windows = synthetic_windows(n=20, length=20)
        matrix = build_matrix(windows, np.arange(20, dtype=np.int64) + 1, max_cycle=100)
        assert np.all(np.isfinite(matrix))


class TestDropConstant:
    def test_removes_a_flat_channel(self) -> None:
        """A channel that never moves cannot inform a split."""
        features = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], dtype=np.float32)
        kept, names, dropped = drop_constant(features, ["flat", "moving"])
        assert names == ["moving"]
        assert dropped == ("flat",)
        assert kept.shape == (3, 1)

    def test_keeps_everything_when_all_channels_move(self) -> None:
        """No silent dropping when there is nothing to drop."""
        features = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
        kept, names, dropped = drop_constant(features, ["a", "b"])
        assert len(names) == 2
        assert dropped == ()
        assert kept.shape == (2, 2)


class TestFeatureNormaliser:
    def test_transform_standardises(self) -> None:
        """Fitted features have roughly zero mean and unit variance."""
        rng = np.random.default_rng(0)
        features = rng.normal(5.0, 2.0, size=(200, 3)).astype(np.float32)
        normaliser = FeatureNormaliser().fit(features, ["a", "b", "c"])
        scaled = normaliser.transform(features)
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-3)

    def test_constant_column_is_recorded_not_divided_by_zero(self) -> None:
        """A zero-variance channel must not produce NaN, and must be reported."""
        features = np.array([[1.0, 7.0], [1.0, 8.0], [1.0, 9.0]], dtype=np.float32)
        normaliser = FeatureNormaliser().fit(features, ["flat", "moving"])
        scaled = normaliser.transform(features)
        assert np.all(np.isfinite(scaled))
        assert normaliser.constant_features == ("flat",)

    def test_rejects_a_width_mismatch(self) -> None:
        """Applying a fitted normaliser to the wrong matrix is an error."""
        normaliser = FeatureNormaliser().fit(np.ones((4, 3), dtype=np.float32), ["a", "b", "c"])
        with pytest.raises(ValueError, match="does not match fitted"):
            normaliser.transform(np.ones((4, 5), dtype=np.float32))

    def test_round_trips_through_json(self) -> None:
        """The saved form must reproduce the transform exactly."""
        rng = np.random.default_rng(1)
        features = rng.normal(size=(50, 4)).astype(np.float32)
        features[:, 2] = 3.0
        original = FeatureNormaliser().fit(features, ["a", "b", "c", "d"])
        restored = FeatureNormaliser.from_dict(original.to_dict())
        assert np.allclose(original.transform(features), restored.transform(features))
        assert restored.constant_features == original.constant_features

    def test_from_dict_rejects_inconsistent_widths(self) -> None:
        """A truncated artefact must fail rather than broadcast wrongly."""
        with pytest.raises(ValueError, match="disagree"):
            FeatureNormaliser.from_dict({"mean": [1.0, 2.0], "std": [1.0]})

    def test_from_dict_rejects_wrong_types(self) -> None:
        """A malformed artefact is caught at load, not at inference."""
        with pytest.raises(ValueError, match="must be a list of strings"):
            FeatureNormaliser.from_dict(
                {"mean": [1.0], "std": [1.0], "constant_features": [1], "names": ["a"]}
            )

    def test_transform_is_idempotent_through_repeated_fits(self) -> None:
        """Fitting twice on the same data gives the same transform."""
        rng = np.random.default_rng(2)
        features = rng.normal(size=(30, 3)).astype(np.float32)
        first = FeatureNormaliser().fit(features, ["a", "b", "c"]).transform(features)
        second = FeatureNormaliser().fit(features, ["a", "b", "c"]).transform(features)
        assert np.allclose(first, second)


def test_channel_count_matches_the_feature_columns() -> None:
    """The feature builder and the window builder agree on channel count."""
    assert channel_count() == CHANNELS
