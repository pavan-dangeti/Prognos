"""Windowed-sample construction.

Windows run along the flight time axis within a single cycle, so a window never
spans two cycles and never leaves the operating profile it was drawn from.
Downsampling is required rather than optional: a DS02 cycle holds 5,500 to 18,000
raw rows, so a raw-row window of 50 samples is 50 seconds of flight, not a
meaningful degradation window.

RUL labels come from the dataset's own `rul` column, capped so that early-life
windows are not all labelled with the full life of the unit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from .config import WindowConfig
from .schema import ALLOWED_INPUT_COLUMNS, TARGET_COLUMN, assert_no_leakage

FEATURE_COLUMNS: tuple[str, ...] = tuple(f"W_{n}" for n in ALLOWED_INPUT_COLUMNS[:4]) + tuple(
    f"Xs_{n}" for n in ALLOWED_INPUT_COLUMNS[4:]
)


@dataclass(frozen=True, slots=True)
class Window:
    """One training sample: a fixed-length slice of a flight's sensor trace."""

    unit: int
    cycle: int
    split: str
    subdataset: str
    end_row: int
    features: npt.NDArray[np.float32]  # (length, n_features)
    rul: int
    rul_uncapped: int
    fraction_of_life_remaining: float


def cap_rul(rul: int, cap: int | None) -> int:
    """Apply the early-life RUL cap."""
    return rul if cap is None else min(rul, cap)


def _downsampled_indices(n_rows: int, downsample: int) -> npt.NDArray[np.int64]:
    return np.arange(0, n_rows, downsample, dtype=np.int64)


def iter_windows(
    table: pa.Table,
    window: WindowConfig,
    unit: int,
    split: str,
    subdataset: str,
) -> Iterator[Window]:
    """Yield every window in one unit's Parquet table.

    The table is expected to be ordered by cycle, which is how the converter
    writes it.
    """
    assert_no_leakage(table.column_names)

    cycles = table.column("cycle").to_numpy()
    rul_all = table.column(TARGET_COLUMN).to_numpy()
    if len(cycles) == 0:
        return

    feature_block = np.column_stack(
        [table.column(name).to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
    ).astype(np.float32)

    total_cycles = int(cycles.max())
    bounds = np.searchsorted(cycles, np.unique(cycles), side="left")
    bounds = np.append(bounds, len(cycles))

    for cycle_index in range(len(bounds) - 1):
        start, stop = int(bounds[cycle_index]), int(bounds[cycle_index + 1])
        cycle = int(cycles[start])
        indices = _downsampled_indices(stop - start, window.downsample)
        if len(indices) < window.length:
            continue
        block = feature_block[start:stop]
        rul_uncapped = int(rul_all[start])
        fraction = rul_uncapped / total_cycles if total_cycles else 0.0

        for offset in range(0, len(indices) - window.length + 1, window.stride):
            picked = indices[offset : offset + window.length]
            yield Window(
                unit=unit,
                cycle=cycle,
                split=split,
                subdataset=subdataset,
                end_row=start + int(picked[-1]),
                features=block[picked],
                rul=cap_rul(rul_uncapped, window.rul_cap),
                rul_uncapped=rul_uncapped,
                fraction_of_life_remaining=fraction,
            )


def stack_windows(
    windows: Sequence[Window],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], dict[str, list[int] | list[str]]]:
    """Stack windows into model-ready arrays plus the metadata needed for grouping.

    Returns features `(n, length, n_features)`, targets `(n,)`, and the unit,
    cycle and RUL metadata that the grouped split and the life-stage breakdown
    must respect.
    """
    if not windows:
        raise ValueError("no windows to stack")
    features = np.stack([w.features for w in windows]).astype(np.float32)
    targets = np.asarray([w.rul for w in windows], dtype=np.float32)
    meta: dict[str, list[int] | list[str]] = {
        "unit": [w.unit for w in windows],
        "split": [w.split for w in windows],
        "cycle": [w.cycle for w in windows],
        "rul_uncapped": [w.rul_uncapped for w in windows],
    }
    return features, targets, meta


def load_unit_table(path: str) -> pa.Table:
    """Read one per-unit Parquet file written by the converter."""
    return pq.read_table(path)
