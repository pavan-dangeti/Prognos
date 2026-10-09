"""The leakage guard.

N-CMAPSS carries the simulator's true health parameters (`T_*`), virtual sensors
computed from them (`X_v_*`), and a health-state flag (`hs`). Any of these would
let a model score well and predict nothing useful, because none of them exists on
a real engine.

These tests are the enforcement. docs/adr/0002 explains why the guarantee is
structural (the converter never writes them) and what this file is defending.
"""

from __future__ import annotations

import h5py
import numpy as np
import pyarrow as pa
import pytest

from prognos.convert import convert_subdataset
from prognos.schema import (
    ALLOWED_INPUT_COLUMNS,
    LEAKAGE_GROUPS,
    LEAKAGE_MARKERS,
    MEASURED_SENSORS,
    SCENARIO_DESCRIPTORS,
    TARGET_COLUMN,
    assert_no_leakage,
    is_leakage,
)
from prognos.window import FEATURE_COLUMNS


def test_leakage_markers_detect_every_leaking_name() -> None:
    """Every ground-truth variable name in the real files is recognised as leaking."""
    leaking = [
        "T",  # health parameters group
        "X_v",  # virtual sensors group
        "hs",  # health state flag
        "fan_eff_mod",
        "HPC_flow_mod",
        "SmFan",
        "SmLPC",
        "SmHPT",
        "SmHPC",
    ]
    for column in leaking:
        assert is_leakage(column), f"{column} should be classified as leakage"


def test_allowlist_is_clean() -> None:
    """No allowed column is classified as leakage."""
    assert_no_leakage(ALLOWED_INPUT_COLUMNS)


def test_allowlist_partitions_the_real_sensors() -> None:
    """The allowlist is exactly the 4 scenario descriptors and 14 sensors."""
    assert len(SCENARIO_DESCRIPTORS) == 4
    assert len(MEASURED_SENSORS) == 14
    assert len(ALLOWED_INPUT_COLUMNS) == 18
    assert not set(SCENARIO_DESCRIPTORS) & set(MEASURED_SENSORS)


def test_feature_columns_match_the_allowlist() -> None:
    """The model input matrix carries one column per allowed input, nothing else."""
    assert len(FEATURE_COLUMNS) == len(ALLOWED_INPUT_COLUMNS)
    assert_no_leakage(FEATURE_COLUMNS)
    for name in FEATURE_COLUMNS:
        assert name.startswith(("W_", "Xs_"))


def test_assert_no_leakage_raises_on_a_mixed_schema() -> None:
    """A schema containing ground truth is rejected, not silently filtered."""
    schema = pa.schema([("unit", pa.int32()), ("HPC_flow_mod", pa.float32())])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(schema.names)


def test_leakage_groups_cover_the_file_layout() -> None:
    """The groups we refuse to read are exactly the ones the HDF5 files carry."""
    assert set(LEAKAGE_GROUPS) == {"T", "X_v"}
    assert all(marker for marker in LEAKAGE_MARKERS)


def _synthetic_h5(path: str, *, with_leakage: bool) -> None:
    """Write a minimal HDF5 file shaped like a real N-CMAPSS sub-dataset."""
    n_rows = 400
    rows_per_unit = n_rows // 2
    rng = np.random.default_rng(0)
    # Distinct units per split, as in the real files: dev 1-2, test 3-4.
    units = {"dev": [1, 2], "test": [3, 4]}
    for split, split_units in units.items():
        unit = np.repeat(split_units, rows_per_unit).reshape(-1, 1).astype(np.float64)
        cycle = np.repeat([1.0, 2.0], rows_per_unit).reshape(-1, 1)
        meta = np.hstack([unit, cycle, np.ones((n_rows, 2))])
        with h5py.File(path, "w" if split == "dev" else "a") as handle:
            handle.create_dataset(f"A_{split}", data=meta)
            handle.create_dataset(f"W_{split}", data=rng.normal(size=(n_rows, 4)))
            handle.create_dataset(f"X_s_{split}", data=rng.normal(size=(n_rows, 14)))
            handle.create_dataset(
                f"Y_{split}", data=np.arange(n_rows, dtype=np.int64).reshape(-1, 1)
            )
            if with_leakage:
                handle.create_dataset(f"T_{split}", data=rng.normal(size=(n_rows, 10)))
                handle.create_dataset(f"X_v_{split}", data=rng.normal(size=(n_rows, 14)))

    with h5py.File(path, "a") as handle:
        handle.create_dataset("A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"]))
        handle.create_dataset("W_var", data=np.array([b"alt", b"Mach", b"TRA", b"T2"]))
        handle.create_dataset("X_s_var", data=np.array([s.encode() for s in MEASURED_SENSORS]))
        handle.create_dataset("T_var", data=np.array([b"fan_eff_mod"] * 10))
        handle.create_dataset("X_v_var", data=np.array([b"SmFan"] * 14))


def test_converter_output_contains_no_leakage(tmp_path: pytest.TempPathFactory) -> None:
    """The end-to-end path from a leaking input file to Parquet stays clean.

    The synthetic input deliberately carries `T_dev`, `X_v_dev` and the health
    flag. If any of them reached the Parquet output, this test fails.
    """
    h5_path = tmp_path / "synthetic.h5"
    _synthetic_h5(str(h5_path), with_leakage=True)
    stats = convert_subdataset(h5_path, tmp_path / "parquet", "SYNTH")

    assert stats
    expected = {
        TARGET_COLUMN,
        "unit",
        "cycle",
        "flight_class",
        "t_in_flight",
        "source_split",
    } | set(FEATURE_COLUMNS)
    for unit_stats in stats:
        parquet_path = tmp_path / "parquet" / f"unit_{unit_stats.unit:03d}.parquet"
        table = pa.parquet.read_table(parquet_path)
        assert set(table.column_names) == expected
        assert_no_leakage(table.column_names)


def test_reordered_source_columns_fail_loudly(tmp_path: pytest.TempPathFactory) -> None:
    """A source file with swapped sensor columns is rejected, not misread.

    The converter indexes positionally, so a reordered file would silently put
    T24's values under T30's name and produce plausible-looking nonsense.
    """
    h5_path = tmp_path / "reordered.h5"
    _synthetic_h5(str(h5_path), with_leakage=False)
    with h5py.File(h5_path, "a") as handle:
        swapped = [s.encode() for s in MEASURED_SENSORS]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        del handle["X_s_var"]
        handle.create_dataset("X_s_var", data=np.array(swapped))
    with pytest.raises(ValueError, match="variable names changed"):
        convert_subdataset(h5_path, tmp_path / "parquet", "REORDERED")


def test_converter_output_shape_and_ordering(tmp_path: pytest.TempPathFactory) -> None:
    """Per-unit files preserve row count and cycle ordering."""
    h5_path = tmp_path / "synthetic.h5"
    _synthetic_h5(str(h5_path), with_leakage=False)
    stats = convert_subdataset(h5_path, tmp_path / "parquet", "SYNTH")

    total_rows = sum(s.rows for s in stats)
    assert total_rows == 800, "400 rows per split across two splits"
    assert {s.split for s in stats} == {"dev", "test"}
    assert len({s.unit for s in stats}) == len(stats), "one parquet file per unit"
    for unit_stats in stats:
        table = pa.parquet.read_table(tmp_path / "parquet" / f"unit_{unit_stats.unit:03d}.parquet")
        cycles = table.column("cycle").to_numpy()
        assert np.all(np.diff(cycles) >= 0), "cycles must be non-decreasing"
        t_in_flight = table.column("t_in_flight").to_numpy()
        # t_in_flight restarts at 0 on every cycle boundary
        resets = np.flatnonzero(np.diff(cycles))
        for index in resets:
            assert t_in_flight[index + 1] == 0
