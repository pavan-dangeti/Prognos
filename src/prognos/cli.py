"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .convert import convert_subdataset, sha256_file, write_manifest

H5_SUFFIX = ".h5"


def _subdataset_path(root: Path, subdataset: str) -> Path:
    """Locate the HDF5 file for a sub-dataset, preferring an explicit extract dir."""
    for directory in (root / "extracted" / "data_set", root / "data_set", root):
        candidate = directory / f"N-CMAPSS_{subdataset}{H5_SUFFIX}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no HDF5 file for sub-dataset {subdataset} under {root}")


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert configured sub-datasets to per-unit Parquet and write the manifest."""
    config = load_config(Path(args.config))
    h5_dir = config.data.root
    digests: dict[str, str] = {}
    all_stats = []

    for subdataset in config.data.subdatasets:
        h5_path = _subdataset_path(h5_dir, subdataset)
        print(f"converting {h5_path.name}", flush=True)
        digests[subdataset] = sha256_file(h5_path)
        out_dir = config.data.parquet_dir / subdataset
        all_stats.extend(convert_subdataset(h5_path, out_dir, subdataset))

    manifest = write_manifest(all_stats, config.data.manifest_path, digests)
    units = manifest["units"]
    total_rows = sum(int(u["rows"]) for u in units)
    print(
        f"converted {len(units)} units, {total_rows:,} rows -> {config.data.parquet_dir}",
        flush=True,
    )
    print(f"manifest: {config.data.manifest_path}", flush=True)
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    """Print the data manifest."""
    config = load_config(Path(args.config))
    print(config.data.manifest_path.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="prognos", description="Prognos pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="HDF5 to Parquet")
    convert.add_argument("--config", default="configs/base.yaml")
    convert.set_defaults(func=cmd_convert)

    manifest = sub.add_parser("manifest", help="show the data manifest")
    manifest.add_argument("--config", default="configs/base.yaml")
    manifest.set_defaults(func=cmd_manifest)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
