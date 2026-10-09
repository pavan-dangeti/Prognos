"""Column allowlist for model inputs.

The N-CMAPSS HDF5 files carry more variables than a real engine would ever
measure. `T_*` are the simulator's true health parameters, `X_v_*` are virtual
sensors computed from them, and `hs` is a health-state flag. All three are
ground truth about the degradation process, none of them exists on an aircraft,
and any of them would let a model score well while predicting nothing.

The guarantee is structural: the converter never writes them, so downstream code
cannot reach them. This module is the second line of defence, and
`tests/test_leakage.py` is what holds it in place.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

# Scenario descriptors: known on a real aircraft before the engine starts.
SCENARIO_DESCRIPTORS: Final[tuple[str, ...]] = ("alt", "Mach", "TRA", "T2")

# Measured condition-monitoring signals: what real sensors produce.
MEASURED_SENSORS: Final[tuple[str, ...]] = (
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

# Everything a model may ever see, in a fixed order.
ALLOWED_INPUT_COLUMNS: Final[tuple[str, ...]] = SCENARIO_DESCRIPTORS + MEASURED_SENSORS

# Simulator ground truth. Never model inputs.
LEAKAGE_GROUPS: Final[tuple[str, ...]] = ("T", "X_v")

# Substrings that identify a leaking column regardless of its prefix.
LEAKAGE_MARKERS: Final[tuple[str, ...]] = ("_eff_mod", "_flow_mod")

# Smoke-margin probes, one per module. All are computed from the health
# parameters rather than measured, so the whole family leaks.
_SMOKE_MARGIN_PREFIX: Final[str] = "Sm"

# Names that are ground truth on their own, with no substring to match on.
_LEAKAGE_EXACT: Final[frozenset[str]] = frozenset({"hs", "health_state"})

# Trailing tokens that mark a per-module degradation parameter.
_LEAKAGE_SUFFIXES: Final[frozenset[str]] = frozenset({"eff_mod", "flow_mod"})

# Non-sensor columns carried alongside the features: identity and target only.
META_COLUMNS: Final[tuple[str, ...]] = ("unit", "cycle", "t_in_flight", "flight_class")

TARGET_COLUMN: Final[str] = "rul"


def is_leakage(column: str) -> bool:
    """Report whether a column name identifies simulator ground truth."""
    if column in LEAKAGE_GROUPS or column in _LEAKAGE_EXACT:
        return True
    if any(marker in column for marker in LEAKAGE_MARKERS):
        return True
    if column.startswith(_SMOKE_MARGIN_PREFIX):
        return True
    return column.rsplit("_", 1)[-1] in _LEAKAGE_SUFFIXES


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise if any column name in `columns` is simulator ground truth.

    Takes any iterable of strings, so it can be called on an Arrow schema, a
    pandas frame, or a plain list.
    """
    offending = sorted({c for c in columns if is_leakage(c)})
    if offending:
        raise ValueError(f"leakage columns present in model input: {offending}")
