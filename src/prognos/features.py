"""Window-level feature extraction for the tabular baseline.

The 1D-CNN and LSTM take the raw window. LightGBM cannot, so each window is
summarised into fixed-length statistics. The choice of statistics is the model:
mean and last value carry level, std and range carry spread, and the slope over the
window is the only feature that carries *trend*, which is what degradation is.

Normalisation is fitted on training units only and stored with the model. Fitting
it on everything would leak the validation units' distributions into the features,
which on 9 units is enough to shift a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .schema import assert_no_leakage
from .window import FEATURE_COLUMNS

# Statistics taken of each channel within a window.
STATS: tuple[str, ...] = ("mean", "std", "min", "max", "last", "slope")


def feature_names() -> list[str]:
    """Feature column names, in a fixed order so saved arrays stay valid."""
    return [f"{channel}__{stat}" for channel in FEATURE_COLUMNS for stat in STATS]


def channel_count() -> int:
    """Number of sensor channels in a window.

    Defined here rather than hard-coded, so the feature builder and the window
    builder cannot disagree about the shape of their input.
    """
    return len(FEATURE_COLUMNS)


def window_features(windows: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Summarise stacked windows `(n, length, n_channels)` into `(n, features)`.

    The slope is a least-squares fit over the window's time axis, centred, so it
    measures rate of change per step and does not depend on where the window sits.
    """
    if windows.ndim != 3:
        raise ValueError(f"expected (n, length, channels), got {windows.shape}")
    length = int(windows.shape[1])
    if length < 2:
        raise ValueError("a slope needs at least two samples")

    steps = np.arange(length, dtype=np.float32) - (length - 1) / 2.0
    denominator = float(np.sum(steps**2))

    blocks: list[npt.NDArray[np.float32]] = [
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows[:, -1, :],
        np.einsum("nlc,l->nc", windows, steps) / denominator,
    ]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def sequence_length_context(
    cycles: npt.NDArray[np.int64],
    max_cycle: int,
) -> npt.NDArray[np.float32]:
    """Where in its life each window sits, as a fraction and in cycles.

    Absolute cycle number is a legitimate input — it is known at inference time and
    the design allows it. Including it lets the model separate "cycle 30 of a
    100-cycle life" from "cycle 30 of a 40-cycle life" without being told the
    difference. Normalising by the longest life keeps the scale comparable across
    units of different length.
    """
    values = cycles.astype(np.float32)
    return np.stack([values / float(max_cycle), values / 100.0], axis=1).astype(np.float32)


def _vector(payload: dict[str, object], key: str) -> npt.NDArray[np.float32]:
    """Read a numeric vector out of a serialised normaliser."""
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"normaliser payload key {key!r} must be a list")
    return np.asarray(value, dtype=np.float32)


def _string_list(payload: dict[str, object], key: str) -> list[str]:
    """Read a list of strings out of a serialised normaliser."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"normaliser payload key {key!r} must be a list of strings")
    return list(value)


@dataclass(slots=True)
class FeatureNormaliser:
    """Per-feature standardisation fitted on training units only.

    `constant_features` records columns with zero variance. They carry no
    information and LightGBM would split on them arbitrarily, so they are reported
    rather than silently carried.
    """

    mean: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    std: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    constant_features: tuple[str, ...] = ()
    names: tuple[str, ...] = ()

    def fit(self, features: npt.NDArray[np.float32], names: Sequence[str]) -> FeatureNormaliser:
        """Fit on training data only."""
        if features.ndim != 2:
            raise ValueError(f"expected 2-D features, got {features.shape}")
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        # A zero-variance channel would divide by zero; leave it at unit scale and
        # record it so the report can say why it was ignored.
        constant = std <= 1e-12
        safe_std = np.where(constant, 1.0, std)
        return FeatureNormaliser(
            mean=mean.astype(np.float32),
            std=safe_std.astype(np.float32),
            constant_features=tuple(n for n, c in zip(names, constant, strict=True) if c),
            names=tuple(names),
        )

    def transform(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the fitted statistics. Raises if the width does not match."""
        expected = self.mean.shape[0]
        if features.shape[1] != expected:
            raise ValueError(f"feature width {features.shape[1]} does not match fitted {expected}")
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        """Serialisable form for the model artefact."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "constant_features": list(self.constant_features),
            "names": list(self.names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FeatureNormaliser:
        """Rebuild from `to_dict`, checking the widths are self-consistent."""
        mean = _vector(payload, "mean")
        std = _vector(payload, "std")
        if mean.shape != std.shape:
            raise ValueError(f"mean {mean.shape} and std {std.shape} disagree")
        return cls(
            mean=mean,
            std=std,
            constant_features=tuple(_string_list(payload, "constant_features")),
            names=tuple(_string_list(payload, "names")),
        )


def build_matrix(
    windows: npt.NDArray[np.float32],
    cycles: npt.NDArray[np.int64],
    max_cycle: int,
) -> npt.NDArray[np.float32]:
    """Full tabular feature matrix: window statistics plus life-stage context."""
    names = feature_names()
    assert_no_leakage(names)
    stats = window_features(windows)
    context = sequence_length_context(cycles, max_cycle)
    return np.concatenate([stats, context], axis=1).astype(np.float32)


def all_feature_names() -> list[str]:
    """Names for the full matrix, statistics plus context."""
    return [*feature_names(), "cycle_fraction", "cycle_hundreds"]


def drop_constant(
    features: npt.NDArray[np.float32],
    names: Sequence[str],
) -> tuple[npt.NDArray[np.float32], list[str], tuple[str, ...]]:
    """Remove zero-variance columns and report which were dropped.

    C-MAPSS sensors include several that never change within a sub-dataset. Keeping
    them costs training time and invites splits on floating-point noise.
    """
    std = features.std(axis=0)
    keep = std > 1e-12
    dropped = tuple(n for n, k in zip(names, keep, strict=True) if not k)
    kept_names = [n for n, k in zip(names, keep, strict=True) if k]
    return features[:, keep], kept_names, dropped
