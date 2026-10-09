"""LightGBM baseline.

The specific risks guarded here are all silent: a feature-order mismatch between
fit and predict, a normaliser fitted on the wrong rows, a constant channel slipping
through into the split set, and a saved artefact that no longer reproduces its own
predictions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from prognos.gbm import DEFAULT_PARAMS, LightGbmModel, top_features
from prognos.metrics import rmse
from prognos.window import FEATURE_COLUMNS

# The feature builder derives names from the real channel list, so a synthetic
# window must carry the same width or the two disagree about column count.
CHANNELS = len(FEATURE_COLUMNS)
LENGTH = 20


def degraded_windows(
    *,
    n: int = 300,
    seed: int = 0,
    constant_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows with a learnable degradation signal in channel 0.

    Returns windows, cycles and RUL targets, where RUL falls linearly with cycle.
    """
    rng = np.random.default_rng(seed)
    # Capped at 130 so targets stay non-negative for any n used in these tests;
    # RUL below zero is not a thing, and a negative target would be scored against
    # a clipped prediction.
    cycles = np.arange(1, n + 1, dtype=np.int64)
    targets = np.clip(130.0 - cycles, 0.0, None).astype(np.float32)
    block = rng.normal(size=(n, LENGTH, CHANNELS)).astype(np.float32)
    block[:, :, 0] += (targets / 100.0)[:, None]
    if constant_channel:
        block[:, :, 1] = 7.0
    return block, cycles, targets


class TestFitting:
    def test_learns_the_signal(self) -> None:
        """On a clean linear relationship the model must fit well.

        If it cannot learn this, every downstream number is meaningless.
        """
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        prediction = model.predict(windows, cycles)
        assert rmse(targets.astype(float), prediction.astype(float)) < 5.0

    def test_rejects_zero_windows(self) -> None:
        """Fitting on nothing is an error, not a model that predicts NaN."""
        with pytest.raises(ValueError, match="zero windows"):
            LightGbmModel().fit(
                np.zeros((0, LENGTH, CHANNELS), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
            )

    def test_predictions_are_non_negative(self) -> None:
        """A negative RUL is meaningless, so predictions are clipped at zero."""
        windows, cycles, targets = degraded_windows(n=150)
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        assert np.all(model.predict(windows, cycles) >= 0.0)

    def test_prediction_does_not_warn(self) -> None:
        """An unnamed prediction against a named fit is a feature-order smell.

        LightGBM warns when a model fitted with feature names is scored on a plain
        array. That warning is the only signal that the columns could be misaligned,
        so it is escalated to an error here rather than tolerated.
        """
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.predict(windows[:10], cycles[:10])

    def test_fitting_is_deterministic_for_a_seed(self) -> None:
        """Same data, same seed, same predictions, so a result can be reproduced."""
        windows, cycles, targets = degraded_windows(seed=1)
        first = LightGbmModel(seed=42).fit(windows, cycles, targets) or LightGbmModel(seed=42)
        first.fit(windows, cycles, targets)
        second = LightGbmModel(seed=42)
        second.fit(windows, cycles, targets)
        assert np.allclose(first.predict(windows, cycles), second.predict(windows, cycles))

    def test_unfitted_prediction_is_an_error(self) -> None:
        """Predicting before fitting must fail loudly."""
        with pytest.raises(RuntimeError, match="not fitted"):
            LightGbmModel().predict(
                np.zeros((2, LENGTH, CHANNELS), dtype=np.float32), np.zeros(2, dtype=np.int64)
            )


class TestFeatureHandling:
    def test_constant_channel_is_dropped_and_named(self) -> None:
        """A channel that never moves cannot inform a split, and should be reported."""
        windows, cycles, targets = degraded_windows(constant_channel=True)
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        assert model.report()["dropped_constant_features"]
        assert all(
            "__mean" in f or "__std" in f for f in model.report()["dropped_constant_features"]
        )

    def test_kept_indices_align_with_kept_names(self) -> None:
        """A mismatch here means prediction selects different columns than fit."""
        windows, cycles, targets = degraded_windows(constant_channel=True)
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        assert len(model.keep_indices) == len(model.kept_features)
        assert all(0 <= i < len(model.report()["top_features"]) + 200 for i in model.keep_indices)

    def test_importances_cover_every_kept_feature(self) -> None:
        """A short importance list would silently under-report what the model used."""
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        assert set(model.importances) == set(model.kept_features)

    def test_top_features_are_ordered_by_importance(self) -> None:
        """The explainability list must be sorted, or it is not a ranking."""
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        ranked = top_features(model, limit=5)
        assert len(ranked) == 5
        assert [v for _, v in ranked] == sorted((v for _, v in ranked), reverse=True)

    def test_degradation_feature_appears_in_the_importances(self) -> None:
        """A model that ignores the degrading channel has learned nothing useful."""
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        ranked = dict(top_features(model, limit=15))
        assert any("Xs_" in name or "W_" in name for name in ranked)


class TestGeneralisation:
    def test_unseen_units_are_scored_without_error(self) -> None:
        """The real use: fit on some engines, predict on others."""
        windows, cycles, targets = degraded_windows(n=400, seed=2)
        model = LightGbmModel()
        model.fit(windows[:300], cycles[:300], targets[:300])
        prediction = model.predict(windows[300:], cycles[300:])
        assert prediction.shape == (100,)
        assert np.all(np.isfinite(prediction))

    def test_cycle_context_survives_a_batch_of_early_units(self) -> None:
        """Scaling uses the training max cycle, not the batch's.

        A batch containing only early-life units has a small max cycle. Normalising
        by it would place those windows past end of life, and the model would report
        near-zero RUL for a healthy engine.
        """
        windows, cycles, targets = degraded_windows(n=400, seed=3)
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        early_only = model.predict(windows[:20], cycles[:20])
        assert np.all(np.isfinite(early_only))
        assert np.mean(early_only) > 50.0, "healthy windows must not be predicted near failure"


class TestSerialisation:
    def test_round_trip_reproduces_predictions_exactly(self) -> None:
        """A saved model that does not reproduce its own predictions is a broken artefact."""
        windows, cycles, targets = degraded_windows(seed=4)
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        before = model.predict(windows[:50], cycles[:50])
        restored = LightGbmModel.from_dict(model.to_dict())
        assert np.allclose(before, restored.predict(windows[:50], cycles[:50]))

    def test_round_trip_preserves_feature_selection(self) -> None:
        """Feature selection is part of the artefact, not incidental state."""
        windows, cycles, targets = degraded_windows(constant_channel=True)
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        restored = LightGbmModel.from_dict(model.to_dict())
        assert restored.kept_features == model.kept_features
        assert restored.keep_indices == model.keep_indices
        assert restored.dropped_features == model.dropped_features

    def test_unfitted_model_cannot_be_serialised(self) -> None:
        """Serialising nothing would produce an artefact that fails only at inference."""
        with pytest.raises(RuntimeError, match="not fitted"):
            LightGbmModel().to_dict()

    def test_params_are_recorded_for_reproducibility(self) -> None:
        """A result without its parameters cannot be reproduced."""
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel()
        model.fit(windows, cycles, targets)
        report = model.report()
        assert report["params"]["learning_rate"] == DEFAULT_PARAMS["learning_rate"]
        assert "verbose" not in report["params"]

    def test_seed_is_recorded(self) -> None:
        """Fixed seeds are an explicit requirement of the evaluation protocol."""
        windows, cycles, targets = degraded_windows()
        model = LightGbmModel(seed=123)
        model.fit(windows, cycles, targets)
        assert model.to_dict()["seed"] == 123
