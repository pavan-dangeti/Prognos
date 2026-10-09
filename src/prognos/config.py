"""Config loading.

One YAML file per environment, layered over `configs/base.yaml`. Nothing in the
pipeline reads an environment variable or a hard-coded path: every value that
changes a result comes from here, so a run is reproducible from its config plus
the data manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Where the NASA data lives and which sub-datasets we convert."""

    model_config = ConfigDict(frozen=True)

    root: Path = Path("~/data/ncmapss")
    subdatasets: tuple[str, ...] = ("DS01-005", "DS02-006", "DS06")
    parquet_dir: Path = Path("data/parquet")
    manifest_path: Path = Path("data/manifest.json")


class WindowConfig(BaseModel):
    """Sliding-window sampling over the flight time axis.

    `length` and `stride` are in downsampled samples, not raw rows. N-CMAPSS
    flights hold 5,500 to 18,000 raw rows per cycle, so a raw-row window would be
    both enormous and physically meaningless at 1 Hz.
    """

    model_config = ConfigDict(frozen=True)

    length: int = Field(default=50, gt=0)
    stride: int = Field(default=10, gt=0)
    downsample: int = Field(default=10, gt=0)
    # RUL beyond this many cycles is treated as equal. Without a cap, the model
    # learns to predict total life length, which no unit in a fleet shares.
    rul_cap: int | None = 125


class SplitConfig(BaseModel):
    """Unit-level splitting. See docs/adr/0003."""

    model_config = ConfigDict(frozen=True)

    n_splits: int = Field(default=5, ge=2)
    seed: int = 20241009
    # Fraction of units held out entirely and opened once, after the model is
    # frozen.
    test_fraction: float = Field(default=0.34, gt=0.0, lt=0.6)


class Config(BaseModel):
    """Top-level configuration."""

    model_config = ConfigDict(frozen=True)

    data: DataConfig = DataConfig()
    window: WindowConfig = WindowConfig()
    split: SplitConfig = SplitConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> Config:
    """Load a YAML config, layered over a sibling `base.yaml` when present."""
    base_path = path.parent / "base.yaml"
    payload: dict[str, Any] = {}
    if base_path.exists():
        payload = yaml.safe_load(base_path.read_text()) or {}
    payload = _deep_merge(payload, yaml.safe_load(path.read_text()) or {})

    data = DataConfig.model_validate(payload.get("data", {}))
    config = Config.model_validate(
        {
            **payload,
            "data": data.model_copy(
                update={
                    "root": Path(data.root).expanduser(),
                    "parquet_dir": Path(data.parquet_dir).expanduser(),
                    "manifest_path": Path(data.manifest_path).expanduser(),
                }
            ),
        }
    )
    return config
