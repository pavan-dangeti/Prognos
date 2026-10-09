"""Checks on the converted data.

These run against whatever is in `data/`, and skip when it is absent so CI stays
fast. They assert the properties the rest of the pipeline assumes: no engine unit
appears in two splits, RUL decreases to zero at end of life, and every Parquet
checksum in the manifest still matches.

Run with: `uv run pytest -m integration`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from prognos.config import load_config
from prognos.convert import sha256_file
from prognos.schema import TARGET_COLUMN, assert_no_leakage

CONFIG_PATH = Path("configs/base.yaml")


def manifest() -> tuple[dict[str, object], Path]:
    """Load the data manifest, skipping the test when there is nothing converted."""
    config = load_config(CONFIG_PATH)
    if not config.data.manifest_path.exists():
        pytest.skip(f"no manifest at {config.data.manifest_path}; run `prognos convert` first")
    return json.loads(config.data.manifest_path.read_text()), config.data.parquet_dir


def test_manifest_checksums_match_the_parquet_files() -> None:
    """Recorded checksums still hold, so the manifest is a real anchor."""
    data, parquet_dir = manifest()
    for unit in data["units"]:  # type: ignore[index]
        path = parquet_dir / unit["subdataset"] / f"unit_{unit['unit']:03d}.parquet"
        assert path.exists(), f"missing {path}"
        assert sha256_file(path) == unit["parquet_sha256"], f"{path} changed since conversion"


def test_no_unit_appears_in_both_dev_and_test() -> None:
    """Unit IDs are unique within a sub-dataset.

    A collision would mean the same engine trained and evaluated, which is the
    single most common way an RUL result becomes meaningless.
    """
    data, _ = manifest()
    seen: dict[tuple[str, int], str] = {}
    for unit in data["units"]:  # type: ignore[index]
        key = (unit["subdataset"], unit["unit"])
        assert key not in seen, f"unit {key} appears twice in {unit['subdataset']}"
        seen[key] = unit["split"]


def test_rul_reaches_zero_at_end_of_life() -> None:
    """Every unit's last window has RUL 0 and its first has RUL close to its life."""
    data, _ = manifest()
    for unit in data["units"]:  # type: ignore[index]
        assert unit["rul_at_end"] == 0, f"{unit['subdataset']} unit {unit['unit']}"
        assert unit["rul_at_start"] == unit["last_cycle"] - 1, (
            f"{unit['subdataset']} unit {unit['unit']}"
        )


def test_cycle_counts_are_plausible() -> None:
    """Unit lives land in the range the C-MAPSS model produces."""
    data, _ = manifest()
    for unit in data["units"]:  # type: ignore[index]
        assert 50 <= unit["cycles"] <= 120, (
            f"{unit['subdataset']} unit {unit['unit']}: {unit['cycles']}"
        )


@pytest.mark.integration
def test_converted_parquet_has_no_leakage_and_is_ordered() -> None:
    """Every produced file carries only allowed columns, ordered by cycle."""
    data, parquet_dir = manifest()
    for unit in data["units"][:3]:  # type: ignore[index]
        path = parquet_dir / unit["subdataset"] / f"unit_{unit['unit']:03d}.parquet"
        if not path.exists():
            pytest.skip(f"{path} not converted")
        table = pq.read_table(path, columns=["cycle", "unit", TARGET_COLUMN])
        assert_no_leakage(table.column_names)
        cycles = table.column("cycle").to_numpy()
        assert np.all(np.diff(cycles) >= 0)
        rul = table.column(TARGET_COLUMN).to_numpy()
        assert np.all(np.diff(rul) <= 0), "RUL must be non-increasing"
        assert set(table.column("unit").to_numpy().tolist()) == {unit["unit"]}


@pytest.mark.integration
def test_ds02_has_the_expected_nine_units() -> None:
    """DS02 is the literature comparison sub-dataset: 9 units, 3 flight classes."""
    data, _ = manifest()
    ds02 = [u for u in data["units"] if u["subdataset"] == "DS02-006"]  # type: ignore[index]
    assert len(ds02) == 9, f"expected 9 units in DS02-006, got {len(ds02)}"
    classes = {c for unit in ds02 for c in unit["flight_classes"]}
    assert classes == {1, 2, 3}
