"""HDF5 to Parquet conversion for N-CMAPSS.

Each sub-dataset file holds four splits (`_dev`, `_test`) across variable groups
(`A`, `W`, `X_s`, `X_v`, `T`, `Y`). `A` carries `unit`, `cycle`, flight class and
health state; `Y` is the RUL label. Rows are ordered by unit, then cycle, and the
row index inside a cycle is the time axis of a flight.

Output is one Parquet file per unit per sub-dataset, with `theta`/`Xv`/health
state dropped rather than filtered later. See docs/adr/0002.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import MEASURED_SENSORS, SCENARIO_DESCRIPTORS, TARGET_COLUMN

SPLITS: tuple[str, ...] = ("dev", "test")
ROW_BLOCK = 1_000_000


@dataclass(frozen=True, slots=True)
class UnitStats:
    """Per-unit summary recorded in the data manifest."""

    unit: int
    split: str
    subdataset: str
    rows: int
    cycles: int
    first_cycle: int
    last_cycle: int
    flight_classes: list[int]
    rul_at_start: int
    rul_at_end: int
    parquet_sha256: str


def assert_variable_order(handle: h5py.File, expected: tuple[str, ...], group: str) -> None:
    """Check a group's declared variable order matches what the schema expects.

    The converter indexes columns positionally, so a dataset whose columns were
    reordered or renamed would otherwise be read into the wrong fields
    silently. This fails instead.
    """
    raw: npt.NDArray[np.bytes_] = handle[f"{group}_var"][:]
    actual = tuple(name.decode("utf-8") for name in raw)
    if actual != expected:
        raise ValueError(f"{group} variable names changed: expected {expected}, got {actual}")


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    """Stream a file through SHA-256 so multi-gigabyte inputs stay off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _t_in_flight(cycles: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    """Row index within each cycle, restarting at 0 on every boundary.

    N-CMAPSS has no time column: the position of a row inside its cycle is the
    time axis of the flight. Cycles are contiguous and ordered, which the
    converter guarantees by writing them in file order.
    """
    starts = np.searchsorted(cycles, np.unique(cycles), side="left")
    starts = np.append(starts, len(cycles))
    return np.concatenate(
        [np.arange(starts[i + 1] - starts[i], dtype=np.int32) for i in range(len(starts) - 1)]
    )


def _build_unit_table(
    meta: npt.NDArray[np.float64],
    rul: npt.NDArray[np.int64],
    scenario: npt.NDArray[np.float64],
    sensors: npt.NDArray[np.float64],
    t_in_flight: npt.NDArray[np.int32],
) -> pa.Table:
    """Assemble the Arrow table for one unit from row-aligned column blocks."""
    columns: dict[str, Any] = {
        "unit": pa.array(meta[:, 0].astype(np.int32)),
        "cycle": pa.array(meta[:, 1].astype(np.int32)),
        "flight_class": pa.array(meta[:, 2].astype(np.int8)),
        "t_in_flight": pa.array(t_in_flight),
        TARGET_COLUMN: pa.array(rul.astype(np.int32)),
    }
    for index, name in enumerate(SCENARIO_DESCRIPTORS):
        columns[f"W_{name}"] = pa.array(scenario[:, index].astype(np.float32))
    for index, name in enumerate(MEASURED_SENSORS):
        columns[f"Xs_{name}"] = pa.array(sensors[:, index].astype(np.float32))
    return pa.table(columns)


def convert_subdataset(
    h5_path: Path,
    out_dir: Path,
    subdataset: str,
) -> list[UnitStats]:
    """Convert one HDF5 sub-dataset into per-unit Parquet files.

    Only `W` (scenario descriptors) and `X_s` (measured sensors) are read. `T`,
    `X_v` and the health-state flag are never loaded, so they cannot leak into the
    output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: list[UnitStats] = []

    with h5py.File(h5_path, "r") as handle:
        assert_variable_order(handle, SCENARIO_DESCRIPTORS, "W")
        assert_variable_order(handle, MEASURED_SENSORS, "X_s")

        for split in SPLITS:
            meta_ds = handle[f"A_{split}"]
            n_rows = meta_ds.shape[0]
            if n_rows == 0:
                # A sub-dataset may ship only one split; that is data, not an error.
                continue
            rul_ds = handle[f"Y_{split}"]
            scen_ds = handle[f"W_{split}"]
            sens_ds = handle[f"X_s_{split}"]

            meta = meta_ds[:]
            units = np.unique(meta[:, 0])
            for unit in units:
                start = int(np.searchsorted(meta[:, 0], unit, side="left"))
                stop = int(np.searchsorted(meta[:, 0], unit, side="right"))
                unit_id = int(unit)
                unit_meta = meta[start:stop]
                unit_rul = rul_ds[start:stop, 0]
                cycles = unit_meta[:, 1].astype(np.int64)

                table = _build_unit_table(
                    unit_meta,
                    unit_rul,
                    scen_ds[start:stop],
                    sens_ds[start:stop],
                    _t_in_flight(cycles),
                )
                target = out_dir / f"unit_{unit_id:03d}.parquet"
                pq.write_table(table, target, compression="zstd")

                unique_cycles = np.unique(cycles)
                stats.append(
                    UnitStats(
                        unit=unit_id,
                        split=split,
                        subdataset=subdataset,
                        rows=table.num_rows,
                        cycles=len(unique_cycles),
                        first_cycle=int(unique_cycles[0]),
                        last_cycle=int(unique_cycles[-1]),
                        flight_classes=sorted({int(v) for v in np.unique(unit_meta[:, 2])}),
                        rul_at_start=int(unit_rul[0]),
                        rul_at_end=int(unit_rul[-1]),
                        parquet_sha256=sha256_file(target),
                    )
                )
    return stats


def write_manifest(
    stats: list[UnitStats],
    manifest_path: Path,
    h5_digests: dict[str, str],
) -> dict[str, Any]:
    """Write the reproducibility anchor: one row per unit plus input checksums."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "inputs": h5_digests,
        "units": [
            {
                "subdataset": s.subdataset,
                "split": s.split,
                "unit": s.unit,
                "rows": s.rows,
                "cycles": s.cycles,
                "first_cycle": s.first_cycle,
                "last_cycle": s.last_cycle,
                "flight_classes": s.flight_classes,
                "rul_at_start": s.rul_at_start,
                "rul_at_end": s.rul_at_end,
                "parquet_sha256": s.parquet_sha256,
            }
            for s in stats
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
