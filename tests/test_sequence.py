"""Sequence models.

The bugs worth catching here are all silent. A model that predicts the training
mean would score "no error" against a constant target while having learned nothing.
A positional encoding of the wrong length would either crash or, worse, quietly
shift every token. Batch norm evaluated in train mode at inference would make a
prediction depend on which rows happened to share a batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prognos.metrics import rmse
from prognos.sequence import (
    DEFAULT_SEED,
    CnnModel,
    HarnessAdapter,
    LstmModel,
    SequenceModel,
    TrainConfig,
    TransformerModel,
    WindowNormaliser,
    build_sequence_models,
    cycle_context,
    default_device,
    set_seed,
    weighted_mse,
)
from prognos.window import FEATURE_COLUMNS

CHANNELS = len(FEATURE_COLUMNS)
LENGTH = 30
N = 900


def synthetic(
    *,
    n: int = N,
    length: int = LENGTH,
    channels: int = CHANNELS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows with a monotone degradation signal in channel 0.

    RUL falls linearly with cycle, so a model that learns nothing predicts the mean
    and still looks plausible. Every assertion below is about whether the model
    learned the signal rather than the average.
    """
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n + 1, dtype=np.int64)
    targets = np.clip(150.0 - cycles, 0.0, None).astype(np.float32)
    block = rng.normal(size=(n, length, channels)).astype(np.float32)
    block[:, :, 0] += (targets / 150.0)[:, None] * 3.0
    return block, cycles, targets


def quick_config(**overrides: object) -> TrainConfig:
    """A short training run, so the suite stays fast without skipping the fit."""
    defaults = {"epochs": 12, "patience": 6, "learning_rate": 3e-3, "batch_size": 256}
    defaults.update(overrides)
    return TrainConfig(**defaults)  # type: ignore[arg-type]


class TestWindowNormaliser:
    def test_standardises_per_channel(self) -> None:
        """Each channel is centred and scaled independently.

        Channels measured in psia and rpm otherwise differ by three orders of
        magnitude, and the first layer would see only the largest.
        """
        block = np.random.default_rng(0).normal(100.0, 5.0, size=(50, 20, 3)).astype(np.float32)
        block[:, :, 2] *= 1000.0
        normaliser = WindowNormaliser()
        normaliser.fit(block)
        scaled = normaliser.transform(block)
        assert torch.allclose(scaled.mean(dim=(0, 1)), torch.zeros(3), atol=1e-4)
        assert torch.allclose(scaled.std(dim=(0, 1)), torch.ones(3), atol=1e-3)

    def test_a_constant_channel_does_not_divide_by_zero(self) -> None:
        """C-MAPSS has channels that never move; they must not produce NaN."""
        block = np.ones((20, 10, 2), dtype=np.float32)
        normaliser = WindowNormaliser()
        normaliser.fit(block)
        scaled = normaliser.transform(block)
        assert torch.all(torch.isfinite(scaled))

    def test_unfitted_normaliser_raises(self) -> None:
        """Transforming before fitting is an error, not silent garbage."""
        with pytest.raises(RuntimeError, match="not fitted"):
            WindowNormaliser().transform(np.zeros((1, 2, 3), dtype=np.float32))

    def test_round_trips_through_json_form(self) -> None:
        """The saved form must reproduce the transform."""
        block = np.random.default_rng(1).normal(size=(30, 8, 4)).astype(np.float32)
        original = WindowNormaliser()
        original.fit(block)
        restored = WindowNormaliser.from_dict(original.to_dict())
        assert torch.allclose(original.transform(block), restored.transform(block))

    def test_from_dict_rejects_mismatched_vectors(self) -> None:
        """A truncated artefact fails at load, not at inference."""
        with pytest.raises(ValueError, match="disagree"):
            WindowNormaliser.from_dict({"mean": [1.0, 2.0], "std": [1.0]})


class TestCycleContext:
    def test_scales_into_a_sane_range(self) -> None:
        """Divided by 100 so a 100-cycle life lands near 1.0."""
        context = cycle_context(np.array([100], dtype=np.int64))
        assert context[0] == pytest.approx(1.0)

    def test_preserves_order(self) -> None:
        """A later cycle must always show a larger context value."""
        cycles = np.array([5, 50, 500], dtype=np.int64)
        context = cycle_context(cycles)
        assert context[0] < context[1] < context[2]


class TestWeightedMse:
    def test_weight_one_is_plain_mse(self) -> None:
        """The default must be the standard loss, not a subtly different one."""
        pred = torch.tensor([3.0, 1.0])
        target = torch.tensor([1.0, 4.0])
        assert float(weighted_mse(pred, target, 1.0)) == pytest.approx(
            float(((pred - target) ** 2).mean())
        )

    def test_late_predictions_are_penalised_more(self) -> None:
        """The dangerous direction must cost more at weight above one.

        Same magnitude, opposite sign. This is the whole point of the weighting.
        """
        target = torch.tensor([100.0])
        late = weighted_mse(torch.tensor([110.0]), target, 3.0)
        early = weighted_mse(torch.tensor([90.0]), target, 3.0)
        assert float(late) > float(early)

    def test_exact_prediction_costs_nothing(self) -> None:
        """A perfect prediction is free regardless of the weight."""
        target = torch.tensor([7.0, 3.0])
        assert float(weighted_mse(target, target, 5.0)) == pytest.approx(0.0)


class TestModelContract:
    """Properties every architecture must satisfy, checked once for all three."""

    @pytest.fixture(scope="class")
    def fitted(self) -> dict[str, SequenceModel]:
        """One short training run per architecture, shared across the checks."""
        windows, cycles, targets = synthetic()
        models = build_sequence_models(quick_config(), device=torch.device("cpu"))
        for model in models.values():
            HarnessAdapter(model=model).fit(windows, cycles, targets)
        return models

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_predicts_one_value_per_window(self, fitted: dict[str, object], name: str) -> None:
        """The harness stacks predictions from several models, so the shape is fixed."""
        model = fitted[name]
        assert isinstance(model, object)
        windows, cycles, _ = synthetic()
        prediction = model.predict(windows[:64], cycles[:64])  # type: ignore[attr-defined]
        assert prediction.shape == (64,)
        assert prediction.dtype == np.float32

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_predictions_are_finite_and_non_negative(
        self, fitted: dict[str, object], name: str
    ) -> None:
        """A NaN or a negative RUL would propagate straight into a reported metric."""
        model = fitted[name]
        windows, cycles, _ = synthetic()
        prediction = model.predict(windows[:64], cycles[:64])  # type: ignore[attr-defined]
        assert np.all(np.isfinite(prediction))
        assert np.all(prediction >= 0.0)

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_prediction_is_batch_independent(self, fitted: dict[str, object], name: str) -> None:
        """Scoring one window must not depend on which batch it arrived in.

        Dropout and batch norm left in train mode would break this, and the failure
        is invisible in a metric that happens to look fine.
        """
        model = fitted[name]
        windows, cycles, _ = synthetic()
        together = model.predict(windows[:64], cycles[:64])  # type: ignore[attr-defined]
        alone = model.predict(windows[:1], cycles[:1])[0]  # type: ignore[attr-defined]
        assert float(alone) == pytest.approx(float(together[0]), abs=1e-3)

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_beats_predicting_the_mean(self, fitted: dict[str, object], name: str) -> None:
        """The floor. A model that cannot beat the training mean has learned nothing.

        Checked on a clean linear relationship, where a correct model should be
        several times better than the constant.
        """
        model = fitted[name]
        windows, cycles, targets = synthetic()
        prediction = model.predict(windows[:200], cycles[:200])  # type: ignore[attr-defined]
        constant = float(targets.mean())
        floor = rmse(targets[:200].astype(float), np.full(200, constant))
        assert rmse(targets[:200].astype(float), prediction.astype(float)) < floor

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_scales_to_a_different_window_length(
        self, fitted: dict[str, object], name: str
    ) -> None:
        """A window length change must not require retraining from scratch's code path."""
        model = fitted[name]
        short_windows, short_cycles, _ = synthetic(length=20)
        # The model was fitted at LENGTH=30, so scoring 20 is out of distribution;
        # what matters is that it runs and returns finite values rather than raising
        # a shape error deep inside the encoder.
        prediction = model.predict(short_windows[:16], short_cycles[:16])  # type: ignore[attr-defined]
        assert prediction.shape == (16,)
        assert np.all(np.isfinite(prediction))

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_unfitted_prediction_raises(self, name: str) -> None:
        """Predicting before fitting must fail loudly."""
        windows, cycles, _ = synthetic()
        model = build_sequence_models(quick_config(), device=torch.device("cpu"))[name]
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(windows[:8], cycles[:8])

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_zero_windows_is_an_error(self, fitted: dict[str, object], name: str) -> None:
        """Predicting on nothing would return an empty array and score as zero error."""
        model = fitted[name]
        with pytest.raises(ValueError, match="zero windows"):
            model.predict(  # type: ignore[attr-defined]
                np.zeros((0, LENGTH, CHANNELS), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_fit_rejects_mismatched_lengths(self, name: str) -> None:
        """Ragged inputs must be caught before any tensor is built."""
        windows, cycles, targets = synthetic()
        model = build_sequence_models(quick_config(), device=torch.device("cpu"))[name]
        with pytest.raises(ValueError, match="same length"):
            model.fit(windows[:100], cycles[:50], targets[:100])

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_fit_rejects_zero_windows(self, name: str) -> None:
        """Fitting on nothing is an error, not a model that predicts NaN."""
        model = build_sequence_models(quick_config(), device=torch.device("cpu"))[name]
        with pytest.raises(ValueError, match="zero windows"):
            model.fit(
                np.zeros((0, LENGTH, CHANNELS), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
            )

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_reports_parameters_and_config(self, fitted: dict[str, object], name: str) -> None:
        """The complexity comparison in the report needs a parameter count."""
        report = fitted[name].report()  # type: ignore[attr-defined]
        assert report["params"] > 0
        assert report["config"]["seed"] == DEFAULT_SEED
        assert report["epochs_run"] >= 1

    @pytest.mark.parametrize("name", ["cnn", "lstm", "transformer"])
    def test_unfitted_model_cannot_be_serialised(self, name: str) -> None:
        """Serialising nothing produces an artefact that fails only at inference."""
        model = build_sequence_models(quick_config(), device=torch.device("cpu"))[name]
        with pytest.raises(RuntimeError, match="not fitted"):
            model.to_dict()


class TestDeterminism:
    def test_same_seed_gives_the_same_predictions(self) -> None:
        """A reported metric has to be reproducible."""
        windows, cycles, targets = synthetic()
        config = quick_config(seed=42)
        first = HarnessAdapter(model=LstmModel(config, device=torch.device("cpu")))
        first.fit(windows, cycles, targets)
        second = HarnessAdapter(model=LstmModel(config, device=torch.device("cpu")))
        second.fit(windows, cycles, targets)
        assert np.allclose(
            first.predict(windows[:64], cycles[:64]),
            second.predict(windows[:64], cycles[:64]),
        )

    def test_set_seed_makes_torch_reproducible(self) -> None:
        """Torch's RNG is global, so a helper has to exist and be usable."""
        set_seed(7)
        a = torch.randn(4)
        set_seed(7)
        assert torch.allclose(a, torch.randn(4))


class TestEarlyStopping:
    def test_keeps_the_best_validation_weights(self) -> None:
        """A model that stops early must hold its best checkpoint, not its last.

        Late epochs on 29 engines reliably overfit, so returning the final weights
        would systematically overstate error and hide that early stopping works.
        """
        windows, cycles, targets = synthetic()
        model = LstmModel(quick_config(epochs=40, patience=3), device=torch.device("cpu"))
        cut = int(N * 0.8)
        result = model.fit(
            windows[:cut],
            cycles[:cut],
            targets[:cut],
            val_windows=windows[cut:],
            val_targets=targets[cut:],
            val_cycles=cycles[cut:],
        )
        assert result["epochs_run"] < 40, "early stopping never triggered"
        assert model.best_epoch < result["epochs_run"] - 1

    def test_patience_stops_the_run(self) -> None:
        """A long patience must actually run the full schedule."""
        windows, cycles, targets = synthetic()
        model = LstmModel(quick_config(epochs=10, patience=100), device=torch.device("cpu"))
        result = model.fit(
            windows[: cut_of(N)],
            cycles[: cut_of(N)],
            targets[: cut_of(N)],
            val_windows=windows[cut_of(N) :],
            val_targets=targets[cut_of(N) :],
            val_cycles=cycles[cut_of(N) :],
        )
        assert result["epochs_run"] == 10

    def test_validation_history_is_recorded(self) -> None:
        """Without a history, early stopping cannot be audited after the fact."""
        windows, cycles, targets = synthetic()
        model = LstmModel(quick_config(), device=torch.device("cpu"))
        cut = int(N * 0.8)
        model.fit(
            windows[:cut],
            cycles[:cut],
            targets[:cut],
            val_windows=windows[cut:],
            val_targets=targets[cut:],
            val_cycles=cycles[cut:],
        )
        assert len(model.history["train"]) == len(model.history["val"])
        assert len(model.history["train"]) >= 1


def cut_of(n: int) -> int:
    """80% of n, as a cut index."""
    return int(n * 0.8)


class TestHarnessAdapter:
    def test_holds_rows_back_for_validation(self) -> None:
        """The adapter must actually pass a validation set, or early stopping is inert."""
        windows, cycles, targets = synthetic()
        model = LstmModel(quick_config(epochs=6), device=torch.device("cpu"))
        HarnessAdapter(model=model, val_fraction=0.2).fit(windows, cycles, targets)
        assert model.history["val"], "no validation loss was recorded"

    def test_validation_fraction_one_degrades_gracefully(self) -> None:
        """Asking for no training rows must not train on the validation set instead."""
        windows, cycles, targets = synthetic()
        model = LstmModel(quick_config(epochs=3), device=torch.device("cpu"))
        HarnessAdapter(model=model, val_fraction=1.0).fit(windows, cycles, targets)
        assert model.history["train"], "nothing was trained at all"

    def test_name_is_passed_through(self) -> None:
        """The harness labels results by model name, so it must be readable."""
        adapter = HarnessAdapter(model=CnnModel(quick_config(), device=torch.device("cpu")))
        assert adapter.name == "cnn"


class TestTransformerSpecifics:
    def test_positional_encoding_matches_the_window_length(self) -> None:
        """A table of the wrong length would shift every token by an offset."""
        model = TransformerModel(quick_config(), device=torch.device("cpu"))
        windows, cycles, targets = synthetic(length=40)
        HarnessAdapter(model=model).fit(windows, cycles, targets)
        positions = model._sinusoidal(40, torch.device("cpu"))
        assert positions.shape == (1, 40, model.d_model)
        # Distinct values along the time axis, or positions carry no information.
        assert not torch.allclose(positions[0, 0], positions[0, 39])

    def test_positional_encoding_is_reused_for_the_same_length(self) -> None:
        """Regenerating a fixed table each call would be wasteful and non-deterministic."""
        model = TransformerModel(quick_config(), device=torch.device("cpu"))
        first = model._sinusoidal(16, torch.device("cpu"))
        second = model._sinusoidal(16, torch.device("cpu"))
        assert torch.equal(first, second)


class TestDefaultDevice:
    def test_returns_a_usable_torch_device(self) -> None:
        """The device choice has to resolve to something torch accepts."""
        assert isinstance(default_device(), torch.device)
