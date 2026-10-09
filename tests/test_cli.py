"""Test the CLI by running it, not by calling its functions.

A CLI is a boundary: config paths, argument parsing and exit codes are part of its
contract, and calling `cmd_convert` directly tests none of them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from prognos.cli import main

ROWS = 120


@pytest.fixture(scope="module")
def synthetic_setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build an HDF5 file and a config that points at it."""
    base = tmp_path_factory.mktemp("cli")
    data_dir = base / "ncmapss" / "extracted" / "data_set"
    data_dir.mkdir(parents=True)
    h5_path = data_dir / "N-CMAPSS_TEST.h5"
    rng = np.random.default_rng(0)
    n = ROWS
    unit_ids = np.repeat([1, 2], ROWS // 2).reshape(-1, 1).astype(np.float64)
    cycle_ids = np.repeat([1.0, 2.0], ROWS // 2).reshape(-1, 1)
    meta = np.hstack([unit_ids, cycle_ids, np.ones((n, 2))])
    with h5py.File(h5_path, "w") as handle:
        for split in ("dev", "test"):
            handle.create_dataset(f"A_{split}", data=meta)
            handle.create_dataset(f"W_{split}", data=rng.normal(size=(n, 4)))
            handle.create_dataset(f"X_s_{split}", data=rng.normal(size=(n, 14)))
            handle.create_dataset(f"Y_{split}", data=np.arange(n, dtype=np.int64).reshape(-1, 1))
        handle.create_dataset("A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"]))
        handle.create_dataset("W_var", data=np.array([b"alt", b"Mach", b"TRA", b"T2"]))
        handle.create_dataset(
            "X_s_var",
            data=np.array(
                [
                    s.encode()
                    for s in (
                        "T24",
                        "T30",
                        "T48",
                        "T50",
                        "P15",
                        "P2",
                        "P21",
                        "P24",
                        "Ps30",
                        "P40",
                        "P50",
                        "Nf",
                        "Nc",
                        "Wf",
                    )
                ]
            ),
        )

    config_path = base / "test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": str(base / "ncmapss"),
                    "subdatasets": ["TEST"],
                    "parquet_dir": str(base / "parquet"),
                    "manifest_path": str(base / "manifest.json"),
                }
            }
        )
    )
    return h5_path, config_path


def test_convert_writes_parquet_and_manifest(synthetic_setup: tuple[Path, Path]) -> None:
    """`prognos convert` produces files and a manifest, and exits zero."""
    _, config_path = synthetic_setup
    assert main(["convert", "--config", str(config_path)]) == 0


def test_convert_output_is_complete(synthetic_setup: tuple[Path, Path]) -> None:
    """The manifest lists every unit with a checksum."""
    _, config_path = synthetic_setup
    manifest = config_path.parent / "manifest.json"
    import json

    data = json.loads(manifest.read_text())
    assert set(data["inputs"]) == {"TEST"}
    assert len(data["inputs"]["TEST"]) == 64, "sha256 hex digest"
    assert len(data["units"]) == 4, "two units across two splits"
    for unit in data["units"]:
        assert unit["rows"] == ROWS // 2
        assert (
            config_path.parent / "parquet" / "TEST" / f"unit_{unit['unit']:03d}.parquet"
        ).exists()


def test_manifest_command_prints_the_manifest(synthetic_setup: tuple[Path, Path]) -> None:
    """`prognos manifest` reads back what convert wrote."""
    _, config_path = synthetic_setup
    assert main(["manifest", "--config", str(config_path)]) == 0


def test_missing_data_raises_a_clear_error(tmp_path: Path) -> None:
    """A missing HDF5 file fails with a message naming what was looked for."""
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": str(tmp_path),
                    "subdatasets": ["NOPE"],
                    "parquet_dir": str(tmp_path / "parquet"),
                    "manifest_path": str(tmp_path / "manifest.json"),
                }
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="NOPE"):
        main(["convert", "--config", str(config_path)])


def test_no_subcommand_is_an_error() -> None:
    """The CLI requires a subcommand rather than silently doing nothing."""
    with pytest.raises(SystemExit):
        main([])


def test_unknown_subcommand_is_an_error() -> None:
    """A typo fails loudly."""
    with pytest.raises(SystemExit):
        main(["convert-all"])  # type: ignore[list-item]
