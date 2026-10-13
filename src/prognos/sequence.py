"""Sequence models over sensor windows.

Three architectures, deliberately similar in interface so the comparison is about the
encoder rather than about training plumbing: a 1D CNN, an LSTM, and a small
Transformer encoder.

Every model takes a window of shape (batch, length, channels) plus a cycle number,
and returns one RUL value in cycles. Normalisation is fitted on training windows
only and stored with the model, for the same reason as in the tabular path: fitting
it on everything leaks the validation engines' distributions into the input.

Cycle number is an input to all three. It is known at inference time and it is not
ground truth. Leaving it out would handicap these models relative to
`linear_cycle`, which gets it for free, and would make the comparison flattering
rather than fair.

The CNN is the cheapest probe of whether window shape matters at all. The LSTM is
the literature default on this benchmark. The Transformer asks whether recurrence is
still earning its place at these window lengths. With 29 engines none of them is
expected to beat the linear floor by much, and that is a result worth reporting
rather than tuning away.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

# Deterministic seeds everywhere. A sequence model that reports a metric but cannot
# be reproduced is not reporting a fact.
DEFAULT_SEED = 20241009

# Sensor channels: 4 scenario descriptors + 14 measured signals.
N_CHANNELS = 18


def set_seed(seed: int) -> None:
    """Seed every RNG that affects a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    """MPS on Apple silicon, CUDA where available, CPU otherwise.

    MPS is a good fit here: the tensors are small and the model is a few layers deep,
    which is where the metal backend is competitive. It is not bitwise deterministic
    across runs the way CPU is, which is stated rather than papered over.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_THREADS_CAPPLIED = False


def cap_torch_threads() -> None:
    """Limit torch's intra-op pool to a single thread.

    LightGBM links against GNU OpenMP and torch runs its own thread pool. Whichever
    loads second inherits a partially-initialised OpenMP environment, and torch's
    parallel reductions then deadlock — observed as a hang inside
    `torch.mean(dim=(0, 1))` after any LightGBM fit, with `prognos evaluate` stuck
    on the first sequence model and no error.

    One thread costs nothing measurable here: these tensors are small, and a second
    thread buys no wall clock on a few hundred rows of a three-layer encoder. The cap
    is idempotent and safe to call before or after either library is imported.
    """
    global _THREADS_CAPPLIED
    if _THREADS_CAPPLIED:
        return
    torch.set_num_threads(1)
    # Interop threads can only be set before the pool starts, so this raises if any
    # work has already happened. That is acceptable: the intra-op cap above is the
    # part that prevents the deadlock.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    _THREADS_CAPPLIED = True


@dataclass(slots=True)
class TrainConfig:
    """Training hyper-parameters.

    Fixed rather than tuned. On 29 engines a search over these mostly fits the fold it
    was searched on, and the search budget is better spent on a wider comparison.
    """

    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = DEFAULT_SEED
    # Weight on late predictions in the training loss. 1.0 is plain MSE.
    late_weight: float = 1.0
    verbose: bool = False


class SequenceModel(Protocol):
    """What the harness needs from a sequence model."""

    name: str

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
        val_windows: npt.NDArray[np.float32] | None = None,
        val_targets: npt.NDArray[np.float32] | None = None,
        val_cycles: npt.NDArray[np.int64] | None = None,
    ) -> dict[str, float]:
        """Fit, optionally with a validation split for early stopping."""

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Predict RUL in cycles for each window."""


class WindowNormaliser:
    """Per-channel standardisation over the window and batch axes.

    Sensor channels have very different units — pressures in psia, speeds in rpm,
    temperatures in degrees. Without scaling, one channel's magnitude dominates the
    first layer and the rest are ignored.
    """

    def __init__(self) -> None:
        """Start unfitted; call `fit` before `transform`."""
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, windows: npt.NDArray[np.float32]) -> None:
        """Fit per channel over the time and batch axes."""
        block = torch.as_tensor(np.asarray(windows, dtype=np.float32))
        self.mean = block.mean(dim=(0, 1), keepdim=True)
        self.std = block.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

    def transform(self, windows: npt.NDArray[np.float32]) -> torch.Tensor:
        """Apply the fitted statistics."""
        if self.mean is None or self.std is None:
            raise RuntimeError("window normaliser is not fitted")
        block = torch.as_tensor(np.asarray(windows, dtype=np.float32))
        return (block - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        """Serialisable form."""
        if self.mean is None or self.std is None:
            raise RuntimeError("window normaliser is not fitted")
        return {"mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WindowNormaliser:
        """Rebuild from `to_dict`, checking both vectors survived."""
        normaliser = cls()
        normaliser.mean = torch.tensor([payload["mean"]], dtype=torch.float32)
        normaliser.std = torch.tensor([payload["std"]], dtype=torch.float32)
        if normaliser.mean.shape != normaliser.std.shape:
            raise ValueError(
                f"normaliser mean {normaliser.mean.shape} and std {normaliser.std.shape} disagree"
            )
        return normaliser


class TargetNormaliser:
    """Standardisation of the RUL target, for training only.

    The network sees a centred, unit-scale target and its output is mapped back to
    cycles. This is not cosmetic: RUL spans 0 to 150 cycles, and an unnormalised
    target makes the loss surface badly scaled for a small network, which shows up as
    a model that plateaus well short of the training mean rather than as an error.

    The statistics come from the training rows only, for the same reason as the
    window normaliser.
    """

    def __init__(self) -> None:
        """Start at the identity transform; call `fit` on training targets."""
        self.mean = 0.0
        self.std = 1.0

    def fit(self, targets: npt.NDArray[np.float32]) -> None:
        """Fit on the training targets."""
        block = np.asarray(targets, dtype=np.float32)
        self.mean = float(block.mean())
        self.std = float(max(float(block.std()), 1e-6))

    def transform(self, targets: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """To normalised units."""
        return ((np.asarray(targets, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Back to cycles."""
        return (np.asarray(values, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        """Serialisable form."""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetNormaliser:
        """Rebuild from `to_dict`."""
        normaliser = cls()
        normaliser.mean = float(payload["mean"])
        normaliser.std = float(payload["std"])
        if normaliser.std <= 0:
            raise ValueError(f"target std must be positive, got {normaliser.std}")
        return normaliser


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    late_weight: float,
) -> torch.Tensor:
    """MSE with an optional penalty on late predictions.

    A late prediction says the engine has more life left than it does, which is the
    dangerous direction. Weighting it during training pushes the model conservative
    without changing the metric it is scored on. `late_weight = 1.0` is plain MSE.
    """
    error = prediction - target
    weights = torch.where(error > 0, late_weight, 1.0)
    return (weights * error.pow(2)).mean()


def cycle_context(cycles: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Cycle number as a single normalised feature.

    Scaled by 100 so a 100-cycle life lands near 1.0. Deliberately not standardised
    against the batch's own mean and spread: a prediction would then depend on which
    other rows happened to be scored alongside it, so the same window would get a
    different value alone and in a batch. The constant scale is fixed instead, which
    costs a little precision and buys batch independence.
    """
    return cycles.astype(np.float32) / 100.0


class _BaseSequenceModel(nn.Module):
    """Shared fitting, early stopping and prediction for all three encoders.

    An `nn.Module` itself rather than a wrapper around one, so each architecture owns
    its own layers and `state_dict()` is the real parameter set. Subclasses implement
    `build` to construct their layers and `_forward` to run them; everything that is
    not architecture-specific lives here, so a difference in results is a difference
    in the encoder.
    """

    name = "base"
    network: nn.Module | None = None

    def __init__(
        self,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        cap_torch_threads()
        self.config = config or TrainConfig()
        self.device = device or default_device()
        self.normaliser = WindowNormaliser()
        self.target_normaliser = TargetNormaliser()
        self.history: dict[str, list[float]] = {"train": [], "val": []}
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.channels: int = N_CHANNELS

    def build(self, channels: int, length: int) -> None:
        """Construct this architecture's layers."""
        raise NotImplementedError

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the network, returning shape (batch,)."""
        raise NotImplementedError

    def _inputs(
        self, windows: npt.NDArray[np.float32], cycles: npt.NDArray[np.int64]
    ) -> torch.Tensor:
        """Normalise the window and append the cycle context as an extra channel.

        The cycle number is broadcast across the time axis so it enters as a channel
        rather than being concatenated to the pooled representation, which lets every
        layer see it.
        """
        block = self.normaliser.transform(windows).to(self.device)
        context = torch.as_tensor(cycle_context(cycles), device=self.device).reshape(-1, 1)
        # (batch, 1) -> (batch, length, 1): the same cycle number along every step
        # of the window, so it enters as one extra channel.
        expanded = context.unsqueeze(1).expand(-1, block.shape[1], -1)
        return torch.cat([block, expanded], dim=-1)

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
        val_windows: npt.NDArray[np.float32] | None = None,
        val_targets: npt.NDArray[np.float32] | None = None,
        val_cycles: npt.NDArray[np.int64] | None = None,
    ) -> dict[str, float]:
        """Train with early stopping on a validation split.

        The validation engines are a slice of the training group, never the final
        test group, so early stopping is not model selection on the reported set.
        """
        if windows.shape[0] == 0:
            raise ValueError("cannot fit on zero windows")
        if windows.shape[0] != cycles.shape[0] or windows.shape[0] != targets.shape[0]:
            raise ValueError("windows, cycles and targets must have the same length")

        set_seed(self.config.seed)
        self.normaliser.fit(windows)
        self.target_normaliser.fit(targets)
        self.channels = int(windows.shape[2])
        self.build(self.channels + 1, int(windows.shape[1]))
        self.to(self.device)

        inputs = self._inputs(windows, cycles)
        labels = torch.as_tensor(self.target_normaliser.transform(targets), device=self.device)

        generator = torch.Generator().manual_seed(self.config.seed)
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        val_inputs: torch.Tensor | None
        val_labels: torch.Tensor | None
        use_val = (
            val_windows is not None
            and val_targets is not None
            and val_cycles is not None
            and val_windows.shape[0] > 0
        )
        if use_val:
            assert val_windows is not None and val_targets is not None
            assert val_cycles is not None
            val_inputs = self._inputs(val_windows, val_cycles)
            val_labels = torch.as_tensor(
                self.target_normaliser.transform(val_targets), device=self.device
            )
        else:
            val_inputs = None
            val_labels = None

        n = int(inputs.shape[0])
        best_state: dict[str, torch.Tensor] | None = None
        since_improved = 0

        for epoch in range(self.config.epochs):
            self.train()
            order = torch.randperm(n, generator=generator).to(self.device)
            total = 0.0
            for start in range(0, n, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = weighted_mse(
                    self._forward(inputs[index]), labels[index], self.config.late_weight
                )
                loss.backward()  # type: ignore[no-untyped-call]
                # Clipping: one early-life batch can produce a gradient large enough
                # to undo several epochs of progress.
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimiser.step()
                total += float(loss.detach()) * int(index.numel())
            self.history["train"].append(total / n)

            if val_inputs is not None and val_labels is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = float(
                        weighted_mse(self._forward(val_inputs), val_labels, self.config.late_weight)
                    )
                self.history["val"].append(val_loss)
                if val_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                    since_improved = 0
                else:
                    since_improved += 1
                    if since_improved >= self.config.patience:
                        break
                if self.config.verbose:
                    print(
                        f"  {self.name} epoch {epoch:3d} train={self.history['train'][-1]:.3f} "
                        f"val={val_loss:.3f}",
                        flush=True,
                    )

        if best_state is not None:
            self.load_state_dict(best_state)
        return {
            "epochs_run": float(len(self.history["train"])),
            "best_epoch": float(self.best_epoch),
            "best_val_loss": self.best_val_loss,
        }

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Predict RUL in cycles, clipped at zero.

        Inference runs under `no_grad` in eval mode, so dropout cannot make a
        prediction depend on how the batch was assembled.
        """
        if not self._is_built:
            raise RuntimeError(f"{self.name} is not fitted")
        if windows.shape[0] == 0:
            raise ValueError("cannot predict on zero windows")

        self.eval()
        inputs = self._inputs(windows, cycles)
        with torch.no_grad():
            normalised = self._forward(inputs)
        prediction = self.target_normaliser.inverse(normalised.cpu().numpy().astype(np.float32))
        return np.clip(prediction, 0.0, None).astype(np.float32)

    def report(self) -> dict[str, object]:
        """Fitted-state summary for the committed result file."""
        return {
            "model": self.name,
            "params": self.count_parameters(),
            "epochs_run": len(self.history["train"]),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "device": str(self.device),
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
                "late_weight": self.config.late_weight,
                "seed": self.config.seed,
            },
        }

    def count_parameters(self) -> int:
        """Trainable parameter count, for the complexity comparison."""
        if not self._is_built:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def _is_built(self) -> bool:
        """Whether `build` has run. The presence of any child layer is the signal."""
        return any(True for _ in self.children())

    def to_dict(self) -> dict[str, Any]:
        """Serialisable fitted state for the registry."""
        if not self._is_built:
            raise RuntimeError(f"{self.name} is not fitted")
        return {
            "name": self.name,
            "channels": self.channels,
            "normaliser": self.normaliser.to_dict(),
            "target_normaliser": self.target_normaliser.to_dict(),
            "state_dict": {k: v.tolist() for k, v in self.state_dict().items()},
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
                "late_weight": self.config.late_weight,
                "seed": self.config.seed,
            },
            "history": self.history,
            "best_epoch": self.best_epoch,
        }


class CnnModel(_BaseSequenceModel):
    """1D CNN over the window, then dual pooling to one vector.

    Three strided convolutions halve the time axis each. Mean and max pooling are
    both used over what remains: mean captures a channel's overall level, max captures
    the transient peaks that degradation signals live in. Taking only one loses
    information the other has.
    """

    name = "cnn"
    conv: nn.Sequential
    head: nn.Sequential

    def build(self, channels: int, length: int) -> None:
        """Three strided convolutions followed by pooling and a linear head."""
        del length
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (batch, length, channels) -> (batch, channels, length) for Conv1d
        features: torch.Tensor = self.conv(inputs.transpose(1, 2))
        pooled: torch.Tensor = torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=-1)
        output: torch.Tensor = self.head(pooled).squeeze(-1)
        return output


class LstmModel(_BaseSequenceModel):
    """Two-layer LSTM over the window, head on the last hidden state.

    The literature default on this benchmark. Two layers because a single layer sees
    only a first-order view of a monotone trend. Not bidirectional: a live system
    predicting at the end of a window has no access to what comes after it, so a
    bidirectional model would be scoring a question nobody has to answer.
    """

    name = "lstm"
    hidden_size = 64
    lstm: nn.LSTM
    head: nn.Linear

    def build(self, channels: int, length: int) -> None:
        """Stacked LSTM plus a linear head."""
        del length
        self.lstm = nn.LSTM(channels, self.hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(self.hidden_size, 1)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # LSTM returns (output_sequence, (h_n, c_n)). h_n is
        # (num_layers, batch, hidden_size), so the last layer's final state is index -1
        # along the first axis.
        _, (hidden, _) = self.lstm(inputs)
        output: torch.Tensor = self.head(hidden[-1]).squeeze(-1)
        return output


class TransformerModel(_BaseSequenceModel):
    """Small Transformer encoder with fixed sinusoidal positional encodings.

    Sinusoidal rather than learned positions: at these window lengths a learned
    table would add parameters for no gain. Two encoder layers, four heads,
    pre-norm. No causal mask, because the whole window is available at prediction
    time.
    """

    name = "transformer"
    d_model = 64
    n_heads = 4
    n_layers = 2
    project: nn.Linear
    encoder: nn.TransformerEncoder
    norm: nn.LayerNorm
    head: nn.Linear

    def __init__(
        self,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Build the Transformer, caching its positional encodings lazily."""
        super().__init__(config=config, device=device)
        self._positions: torch.Tensor | None = None

    def build(self, channels: int, length: int) -> None:
        """Project, encode, pool, project to a scalar."""
        del length
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.project = nn.Linear(channels, self.d_model)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 1)

    def _sinusoidal(self, length: int, device: torch.device) -> torch.Tensor:
        """Fixed positional encodings, cached per length and device."""
        if self._positions is None or self._positions.shape[1] != length:
            positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
            divisor = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / self.d_model)
            )
            table = torch.zeros(1, length, self.d_model)
            table[0, :, 0::2] = torch.sin(positions * divisor)
            table[0, :, 1::2] = torch.cos(positions * divisor)
            self._positions = table.to(device)
        return self._positions

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.project(inputs) + self._sinusoidal(inputs.shape[1], inputs.device)
        encoded: torch.Tensor = self.encoder(sequence)
        pooled: torch.Tensor = self.norm(encoded.mean(dim=1))
        output: torch.Tensor = self.head(pooled).squeeze(-1)
        return output


SEQUENCE_MODELS: dict[str, type[_BaseSequenceModel]] = {
    "cnn": CnnModel,
    "lstm": LstmModel,
    "transformer": TransformerModel,
}


def build_sequence_models(
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    names: Sequence[str] = ("cnn", "lstm", "transformer"),
) -> dict[str, SequenceModel]:
    """The sequence models evaluated in Phase 3, in reporting order."""
    chosen = [SEQUENCE_MODELS[name] for name in names if name in SEQUENCE_MODELS]
    return {model.name: model(config=config, device=device) for model in chosen}


@dataclass(slots=True)
class HarnessAdapter:
    """Presents a sequence model through the harness's fit/predict interface.

    The harness calls `fit(train)` with no validation arguments, but early stopping
    needs a validation signal. This carves a fixed fraction off the *training* rows
    and passes it along. The final test group is never touched, and early stopping
    therefore never sees the engines the metric is reported on.
    """

    model: _BaseSequenceModel
    val_fraction: float = 0.15
    seed: int = DEFAULT_SEED

    @property
    def name(self) -> str:
        """The wrapped model's name, so the harness can label its output."""
        return self.model.name

    def fit(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
        targets: npt.NDArray[np.float32],
    ) -> None:
        """Fit, holding out part of the training rows for early stopping."""
        n = int(windows.shape[0])
        cut = int(n * (1.0 - self.val_fraction))
        if cut < 1 or n - cut < 1:
            # Too few rows to split; train without early stopping rather than
            # silently validating on the training set.
            self.model.fit(windows, cycles, targets)
            return
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        train_index = np.sort(order[:cut])
        val_index = np.sort(order[cut:])
        self.model.fit(
            windows[train_index],
            cycles[train_index],
            targets[train_index],
            val_windows=windows[val_index],
            val_targets=targets[val_index],
            val_cycles=cycles[val_index],
        )

    def predict(
        self,
        windows: npt.NDArray[np.float32],
        cycles: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Delegate to the wrapped model."""
        return self.model.predict(windows, cycles)
