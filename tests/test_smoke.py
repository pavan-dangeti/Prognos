"""Smoke test for the CI gate.

Builds a tiny synthetic HDF5 file with the same shape as a real sub-dataset,
runs the full convert path, and asserts the outputs. Runs in CI with no NASA
data present, so it is the check that the pipeline has not broken — not a check
that the data is right. `tests/test_data_integrity.py` covers the latter when the
real data is available locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from prognos.config import WindowConfig, load_config
from prognos.convert import UnitStats, convert_subdataset, write_manifest
from prognos.schema import MEASURED_SENSORS, TARGET_COLUMN, assert_no_leakage
from prognos.split import assert_disjoint, make_splits
from prognos.window import iter_windows, load_unit_table, stack_windows

ROWS_PER_UNIT = 600
UNITS = 6
CYCLES_PER_UNIT = 6
ROWS_PER_CYCLE = ROWS_PER_UNIT // CYCLES_PER_UNIT


@pytest.fixture(scope="module")
def synthetic_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a small file with the real files' variable groups and shapes."""
    path = tmp_path_factory.mktemp("smoke") / "N-CMAPSS_SYNTH.h5"
    rng = np.random.default_rng(20241009)

    with h5py.File(path, "w") as handle:
        # Distinct units per split, as in the real files.
        for split_index, split in enumerate(("dev", "test")):
            units = split_index * UNITS + np.arange(1, UNITS + 1)
            unit_ids = np.repeat(units, ROWS_PER_UNIT).reshape(-1, 1).astype(np.float64)
            cycle_ids = np.repeat(np.arange(1, CYCLES_PER_UNIT + 1), ROWS_PER_CYCLE)
            cycle_ids = np.tile(cycle_ids, UNITS).reshape(-1, 1).astype(np.float64)
            n = len(unit_ids)
            meta = np.hstack([unit_ids, cycle_ids, rng.integers(1, 4, (n, 2))])
            rul = np.tile(
                np.repeat(np.arange(CYCLES_PER_UNIT - 1, -1, -1, dtype=np.int64), ROWS_PER_CYCLE),
                UNITS,
            )

            handle.create_dataset(f"A_{split}", data=meta)
            handle.create_dataset(f"W_{split}", data=rng.normal(size=(n, 4)))
            handle.create_dataset(f"X_s_{split}", data=rng.normal(size=(n, 14)))
            handle.create_dataset(f"Y_{split}", data=rul.reshape(-1, 1))
            handle.create_dataset(f"T_{split}", data=rng.normal(size=(n, 10)))
            handle.create_dataset(f"X_v_{split}", data=rng.normal(size=(n, 14)))
        handle.create_dataset("A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"]))
        handle.create_dataset("W_var", data=np.array([b"alt", b"Mach", b"TRA", b"T2"]))
        handle.create_dataset("X_s_var", data=np.array([s.encode() for s in MEASURED_SENSORS]))
        handle.create_dataset("T_var", data=np.array([b"fan_eff_mod"] * 10))
        handle.create_dataset("X_v_var", data=np.array([b"SmFan"] * 14))
    return path


@pytest.mark.smoke
def test_end_to_end_pipeline(synthetic_h5: Path, tmp_path: Path) -> None:
    """Convert, window, split and stack, with no leakage anywhere."""
    stats = convert_subdataset(synthetic_h5, tmp_path / "parquet", "SYNTH")
    assert len(stats) == UNITS * 2, "one parquet file per unit across both splits"
    assert len({(s.split, s.unit) for s in stats}) == len(stats), "units do not collide"

    manifest = write_manifest(stats, tmp_path / "manifest.json", {"SYNTH": "synthetic"})
    assert set(manifest["inputs"]) == {"SYNTH"}

    config = WindowConfig(length=10, stride=5, downsample=10)
    all_windows = []
    for unit_stats in stats:
        table = load_unit_table(str(tmp_path / "parquet" / f"unit_{unit_stats.unit:03d}.parquet"))
        assert_no_leakage(table.column_names)
        assert TARGET_COLUMN in table.column_names
        all_windows.extend(iter_windows(table, config, unit_stats.unit, unit_stats.split, "SYNTH"))

    assert all_windows, "windows produced"
    features, targets, meta = stack_windows(all_windows)
    assert features.shape[1:] == (10, 18)
    assert targets.shape == (len(all_windows),)

    splits = make_splits(meta["unit"], n_splits=3, test_fraction=0.34, seed=20241009)
    for split in splits:
        assert_disjoint(split.train_units, split.test_units, split.name)
        assert_disjoint(split.val_units, split.test_units, split.name)


@pytest.mark.smoke
def test_base_config_loads_and_validates() -> None:
    """The shipped config parses and its bounds hold."""
    config = load_config(Path("configs/base.yaml"))
    assert config.window.length > 0
    assert config.window.stride > 0
    assert config.split.n_splits >= 2
    assert 0.0 < config.split.test_fraction < 0.6


@pytest.mark.smoke
def test_base_config_matches_the_module_defaults() -> None:
    """The YAML and the dataclass defaults agree, so behaviour is unambiguous."""
    raw = yaml.safe_load(Path("configs/base.yaml").read_text())
    default = load_config(Path("configs/base.yaml"))
    assert raw["window"]["length"] == WindowConfig().length
    assert raw["split"]["seed"] == default.split.seed
    assert raw["split"]["n_splits"] == default.split.n_splits


@pytest.mark.smoke
def test_manifest_is_diff_friendly(tmp_path: Path) -> None:
    """The manifest is stable-sorted JSON with a trailing newline, so diffs are readable."""
    manifest_path = tmp_path / "manifest.json"
    stats = [_unit_stats(unit) for unit in sorted({int(u) for u in range(1, UNITS + 1)})]
    payload = write_manifest(stats, manifest_path, {"SYNTH": "abc123"})
    text = manifest_path.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == payload
    assert text == json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _unit_stats(unit: int) -> UnitStats:
    return UnitStats(
        unit=unit,
        split="dev",
        subdataset="SYNTH",
        rows=ROWS_PER_UNIT,
        cycles=CYCLES_PER_UNIT,
        first_cycle=1,
        last_cycle=CYCLES_PER_UNIT,
        flight_classes=[1],
        rul_at_start=CYCLES_PER_UNIT - 1,
        rul_at_end=0,
        parquet_sha256="0" * 64,
    )
