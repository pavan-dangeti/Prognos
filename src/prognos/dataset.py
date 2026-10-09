"""Loading windows into model-ready arrays.

Held separately from `window.py` because it is the expensive step: 29 units of
N-CMAPSS is 20.9 million rows, and materialising every window for every evaluation
run would dominate the time of a hyperparameter sweep.

Two things are load-bearing here and both were found the hard way:

- **Engine identity is (sub-dataset, unit), not unit alone.** Unit 2 of DS01 and
  unit 2 of DS02 are different engines. Grouping on the bare unit number merges
  them, and a model then sees the same engine in both training and validation.
  Every grouping key in this project is `GroupKey` for that reason.
- **Life-stage fractions are per unit.** A unit that lives 63 cycles and one that
  lives 100 are not at the same point in life at cycle 40, so the fraction is
  measured against each unit's own length.

The cache is keyed by a hash of the parameters that change a window, so changing
the split seed does not invalidate it and changing the window length does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from .config import WindowConfig
from .convert import source_split_name
from .schema import assert_no_leakage
from .window import FEATURE_COLUMNS, Window, iter_windows, stack_windows


@dataclass(frozen=True, order=True, slots=True)
class GroupKey:
    """A single engine: its sub-dataset and its unit number.

    Unit numbers restart at 1 in every sub-dataset, so the bare number is not an
    identity. Using it alone would put DS01 unit 2 in the same fold as DS02 unit 2,
    which are different engines, and the reported error would stop measuring
    generalisation to a new engine.
    """

    subdataset: str
    unit: int

    def label(self) -> str:
        """Human-readable form, for logs and result files."""
        return f"{self.subdataset}/unit{self.unit:03d}"


def group_key(subdataset: str, unit: int) -> GroupKey:
    """Build a grouping key from its parts."""
    return GroupKey(subdataset=subdataset, unit=int(unit))


class WindowSource(Protocol):
    """What the evaluation harness needs from a set of windows.

    A protocol rather than the concrete `WindowSet` so the harness can be tested
    against a cheap stand-in.
    """

    @property
    def windows(self) -> npt.NDArray[np.float32]:
        """Stacked sensor windows, shape (n, length, channels)."""

    @property
    def targets(self) -> npt.NDArray[np.float32]:
        """RUL label per window, in cycles."""

    @property
    def groups(self) -> npt.NDArray[np.int64]:
        """Encoded engine identity per window. The grouping key for every split."""

    @property
    def group_labels(self) -> tuple[GroupKey, ...]:
        """The group id behind each encoded value, for reports and splits."""

    @property
    def cycles(self) -> npt.NDArray[np.int64]:
        """Cycle number per window."""

    @property
    def fractions(self) -> npt.NDArray[np.float32]:
        """Fraction of that unit's own life remaining, for the life-stage split."""

    def subset(self, mask: npt.NDArray[np.bool_]) -> WindowSource:
        """A restricted copy holding the rows where `mask` is true."""

    def summary(self) -> dict[str, object]:
        """Counts and identifiers, for the committed run report."""


@dataclass(frozen=True, slots=True)
class WindowSet:
    """Materialised windows with everything needed to split and score them."""

    windows: npt.NDArray[np.float32]
    targets: npt.NDArray[np.float32]
    groups: npt.NDArray[np.int64]
    group_labels: tuple[GroupKey, ...]
    cycles: npt.NDArray[np.int64]
    fractions: npt.NDArray[np.float32]
    splits: tuple[str, ...]
    subdatasets: tuple[str, ...]
    max_cycle: int

    def __len__(self) -> int:
        """Number of windows."""
        return int(self.windows.shape[0])

    def subset(self, mask: npt.NDArray[np.bool_]) -> WindowSet:
        """A view restricted to `mask`, for fold construction."""
        return WindowSet(
            windows=self.windows[mask],
            targets=self.targets[mask],
            groups=self.groups[mask],
            group_labels=self.group_labels,
            cycles=self.cycles[mask],
            fractions=self.fractions[mask],
            splits=tuple(s for s, m in zip(self.splits, mask, strict=True) if m),
            subdatasets=tuple(s for s, m in zip(self.subdatasets, mask, strict=True) if m),
            max_cycle=self.max_cycle,
        )

    def summary(self) -> dict[str, object]:
        """Counts for the run report."""
        return {
            "n_windows": len(self),
            "n_engines": len(self.group_labels),
            "engines": [k.label() for k in self.group_labels],
            "subdatasets": sorted(set(self.subdatasets)),
            "source_splits": sorted(set(self.splits)),
            "max_cycle": int(self.max_cycle),
            "rul_cap": int(self.targets.max()),
        }


def cache_key(config: WindowConfig, subdatasets: Iterable[str]) -> str:
    """Hash of everything that changes the windows themselves.

    The split seed is deliberately excluded: it changes which windows are in which
    fold, not what the windows are, and the array is the expensive part.
    """
    payload = {
        "length": config.length,
        "stride": config.stride,
        "downsample": config.downsample,
        "rul_cap": config.rul_cap,
        "subdatasets": sorted(subdatasets),
        "features": list(FEATURE_COLUMNS),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def life_fractions(
    groups: npt.NDArray[np.int64],
    rul_uncapped: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Fraction of each engine's own life remaining.

    Normalising by the dataset's longest life would place a 63-cycle engine and a
    100-cycle engine at different points in life at the same cycle number, and the
    life-stage breakdown would then be comparing unlike things.
    """
    lengths = {int(g): int(rul_uncapped[groups == g].max()) + 1 for g in np.unique(groups)}
    denominators = np.asarray([lengths[int(g)] for g in groups], dtype=np.float64)
    return (rul_uncapped / denominators).astype(np.float32)


def load_windows(
    parquet_dir: Path,
    config: WindowConfig,
    subdatasets: Iterable[str],
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> WindowSet:
    """Materialise all windows for the given sub-datasets, with an on-disk cache."""
    wanted = tuple(sorted(subdatasets))
    key = cache_key(config, wanted)

    if cache_dir is not None:
        cached = cache_dir / f"windows_{key}.npz"
        if cached.exists() and not rebuild:
            return _load_cache(cached)

    window_list: list[Window] = []
    for subdataset in wanted:
        directory = parquet_dir / subdataset
        if not directory.exists():
            raise FileNotFoundError(f"no converted data at {directory}; run `prognos convert`")
        for path in sorted(directory.glob("unit_*.parquet")):
            unit_id = int(path.stem.split("_")[1])
            table = pq.read_table(path)
            assert_no_leakage(table.column_names)
            split = source_split_name(_read_source_split(table))
            window_list.extend(iter_windows(table, config, unit_id, split, subdataset))

    if not window_list:
        raise ValueError(
            f"no windows produced for {wanted}; check window length {config.length} "
            f"against downsample {config.downsample}"
        )

    features, targets, meta = stack_windows(window_list)
    cycles = np.asarray(meta["cycle"], dtype=np.int64)
    rul_uncapped = np.asarray(meta["rul_uncapped"], dtype=np.int64)

    # One engine is a (sub-dataset, unit) pair, so that is what identifies it.
    engine_keys = np.asarray([group_key(w.subdataset, w.unit) for w in window_list], dtype=object)
    labels = tuple(sorted(set(engine_keys.tolist())))
    lookup = {key: index for index, key in enumerate(labels)}
    groups = np.asarray([lookup[key] for key in engine_keys], dtype=np.int64)

    result = WindowSet(
        windows=features,
        targets=targets,
        groups=groups,
        group_labels=labels,
        cycles=cycles,
        fractions=life_fractions(groups, rul_uncapped),
        splits=tuple(_string_at(meta["split"], i) for i in range(len(window_list))),
        subdatasets=tuple(w.subdataset for w in window_list),
        max_cycle=int(cycles.max()),
    )

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _save_cache(cache_dir / f"windows_{key}.npz", result)
    return result


def _string_at(values: list[int] | list[str], index: int) -> str:
    """Element access on a metadata list that holds either ints or strings.

    `stack_windows` returns one metadata dict for the whole batch, so the split
    labels have to be unpacked element by element rather than cast as a list.
    """
    value = values[index]
    return value if isinstance(value, str) else str(value)


def _read_source_split(table: pa.Table) -> int:
    """Read the dataset's own dev/test label from a converted unit file.

    Every row in a unit file carries the same value, so the first row is enough.
    Nothing here partitions on it: engine identity is the split that matters
    (docs/adr/0003). It is kept because it is useful context when comparing
    against published results, which do use it.
    """
    if "source_split" not in table.column_names:
        raise ValueError("converted table has no source_split column; re-run `prognos convert`")
    return int(table.column("source_split")[0].as_py())


def _save_cache(path: Path, data: WindowSet) -> None:
    """Write the window arrays, compressed."""
    np.savez_compressed(
        path,
        windows=data.windows,
        targets=data.targets,
        groups=data.groups,
        cycles=data.cycles,
        fractions=data.fractions,
        splits=np.asarray(data.splits),
        subdatasets=np.asarray(data.subdatasets),
        labels=np.asarray([k.label() for k in data.group_labels]),
        max_cycle=np.asarray([data.max_cycle]),
    )


def _load_cache(path: Path) -> WindowSet:
    """Read cached windows, checking the payload is complete."""
    with np.load(path, allow_pickle=False) as payload:
        # `.files` is a list property on NpzFile, not a method.
        keys = set(payload.files)
        required = {"windows", "targets", "groups", "cycles", "fractions", "labels"}
        missing = required - keys
        if missing:
            raise ValueError(f"window cache {path} is missing {sorted(missing)}")
        labels = tuple(
            GroupKey(
                subdataset=str(label).split("/")[0],
                unit=int(str(label).split("unit")[1]),
            )
            for label in payload["labels"]
        )
        return WindowSet(
            windows=payload["windows"].astype(np.float32),
            targets=payload["targets"].astype(np.float32),
            groups=payload["groups"].astype(np.int64),
            group_labels=labels,
            cycles=payload["cycles"].astype(np.int64),
            fractions=payload["fractions"].astype(np.float32),
            splits=tuple(str(s) for s in payload["splits"]),
            subdatasets=tuple(str(s) for s in payload["subdatasets"]),
            max_cycle=int(payload["max_cycle"][0]),
        )
