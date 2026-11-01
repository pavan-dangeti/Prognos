"""Choosing the operating-shift settings without choosing them on the reported engines.

The fleet envelope missed a short-haul engine suddenly flying long-haul altitudes,
and a shift past the fleet raised one alert per flight. `ShiftTracker` adds each
engine's own envelope and groups flights into events; this study picks its four
settings under the same rules as the drift study:

1. The same development and held-out split (`drift_study.split_engines`, same seed).
   The final test group is not read.
2. Three runs per engine, judged per flight on its lowest and highest flight
   conditions, which is what the live pipeline hands the tracker:
   - **clean**: any opening is a false alarm;
   - **within the fleet**: from flight 40, altitude raised by half the fleet's fitted
     range;
   - **beyond the fleet**: from flight 40, altitude raised to a tenth of the fitted
     range above the highest any training engine flew.
   A shift is detected if it opens within `HORIZON` flights; its openings over the
   rest of the engine's life count the alerts that one event raised.
3. Selected on development engines only, by `selection_key`, fixed before the run.
4. Before and after reported on held-out engines. "Before" is the Phase 6B check:
   fleet envelope only, open on one flight outside, release on one flight inside.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .dataset import GroupKey
from .drift_study import split_engines
from .pipeline import _unit_table
from .stream.faults import Envelope, ShiftTracker

START, HORIZON = 40, 15
N_DESCRIPTORS = 4

LEARN = (10, 20, 30)
MARGINS = (0.1, 0.25, 0.5)
PERSISTENCE = (1, 2, 3)
RELEASE = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class ShiftSettings:
    """The four things the study tunes. `learn=0` is the fleet envelope alone."""

    learn: int
    margin: float
    persistence: int
    release: int

    def tracker(self, fleet: Envelope) -> ShiftTracker:
        """A fresh tracker with these settings."""
        return ShiftTracker(
            unit=1,
            fleet=fleet,
            learn_cycles=self.learn,
            margin=self.margin,
            persistence=self.persistence,
            release=self.release,
        )


#: The check as Phase 6B shipped it.
SHIPPED = ShiftSettings(learn=0, margin=0.0, persistence=1, release=1)


@dataclass(frozen=True, slots=True)
class Flights:
    """One engine's flights: the lowest and highest flight conditions of each."""

    label: str
    low: npt.NDArray[np.float32]
    high: npt.NDArray[np.float32]


def flights(parquet_dir: Path, label: GroupKey) -> Flights:
    """Per-flight extremes of the flight conditions, from the engine's readings."""
    cycles, rows, _rul = _unit_table(parquet_dir, label.subdataset, label.unit)
    starts = np.flatnonzero(np.r_[True, np.diff(cycles) != 0])
    conditions = rows[:, :N_DESCRIPTORS]
    return Flights(
        label=label.label(),
        low=np.minimum.reduceat(conditions, starts, axis=0),
        high=np.maximum.reduceat(conditions, starts, axis=0),
    )


def shifted(engine: Flights, fleet: Envelope, beyond_fleet: bool) -> Flights:
    """The same engine with altitude raised from flight `START` on."""
    lower, upper = fleet
    span = float(upper[0] - lower[0])
    step = (
        float(upper[0]) - float(engine.high[START:, 0].max()) + 0.1 * span
        if beyond_fleet
        else 0.5 * span
    )
    low, high = engine.low.copy(), engine.high.copy()
    low[START:, 0] += np.float32(step)
    high[START:, 0] += np.float32(step)
    return Flights(label=engine.label, low=low, high=high)


def openings(engine: Flights, settings: ShiftSettings, fleet: Envelope) -> list[int]:
    """The flights at which an operating_shift opened."""
    tracker = settings.tracker(fleet)
    opened = []
    for index in range(engine.low.shape[0]):
        changed = tracker.observe_flight(engine.low[index], engine.high[index])
        if changed and tracker.open is not None:
            opened.append(index)
    return opened


def evaluate(engines: list[Flights], settings: ShiftSettings, fleet: Envelope) -> dict[str, Any]:
    """False alarms on clean engines, and detection and alerts per shift event."""
    alarmed = [e.label for e in engines if openings(e, settings, fleet)]
    result: dict[str, Any] = {
        "settings": asdict(settings),
        "false_alarms": {
            "engines": len(engines),
            "engines_alarmed": len(alarmed),
            "alarmed_engines": alarmed,
        },
    }
    for name, beyond in (("within_fleet", False), ("beyond_fleet", True)):
        detected, alerts = 0, []
        for engine in engines:
            opened = openings(shifted(engine, fleet, beyond), settings, fleet)
            after = [i for i in opened if i >= START]
            if after and after[0] < START + HORIZON:
                detected += 1
                alerts.append(len(after))
        result[name] = {
            "detected": detected,
            "rate": detected / len(engines) if engines else 0.0,
            "alerts_per_event": float(np.mean(alerts)) if alerts else None,
            "max_alerts_per_event": max(alerts) if alerts else None,
        }
    return result


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    """The pre-registered rule: lower sorts first.

    Fewest development engines with any false alarm; then most within-fleet shifts
    named; then most beyond-fleet shifts named; then fewest alerts per event; then the
    most sensitive setting (smallest margin, persistence and learning period) and the
    shortest release.
    """
    s = result["settings"]
    alerts = [
        result[k]["alerts_per_event"]
        for k in ("within_fleet", "beyond_fleet")
        if result[k]["alerts_per_event"] is not None
    ]
    return (
        float(result["false_alarms"]["engines_alarmed"]),
        -result["within_fleet"]["rate"],
        -result["beyond_fleet"]["rate"],
        float(np.mean(alerts)) if alerts else float("inf"),
        s["margin"],
        float(s["persistence"]),
        float(s["learn"]),
        float(s["release"]),
    )


def study(parquet_dir: Path, labels: list[GroupKey], seed: int, fleet: Envelope) -> dict[str, Any]:
    """Search on development engines, then judge the choice on held-out ones."""
    development, held_out = split_engines(labels, seed)
    dev = [flights(parquet_dir, label) for label in development]
    held = [flights(parquet_dir, label) for label in held_out]
    grid = [
        evaluate(dev, ShiftSettings(*values), fleet)
        for values in itertools.product(LEARN, MARGINS, PERSISTENCE, RELEASE)
    ]
    chosen = ShiftSettings(**min(grid, key=selection_key)["settings"])
    return {
        "development_engines": [label.label() for label in development],
        "held_out_engines": [label.label() for label in held_out],
        "fleet_envelope": {
            "lower": [float(v) for v in fleet[0]],
            "upper": [float(v) for v in fleet[1]],
        },
        "protocol": {
            "shift_from_flight": START,
            "horizon_flights": HORIZON,
            "grid": {
                "learn": list(LEARN),
                "margin": list(MARGINS),
                "persistence": list(PERSISTENCE),
                "release": list(RELEASE),
            },
            "selection": (
                "fewest development engines with any false alarm, then most within-fleet "
                "shifts named, then most beyond-fleet shifts named, then fewest alerts per "
                "event, then the most sensitive setting and the shortest release"
            ),
        },
        "shipped": asdict(SHIPPED),
        "chosen": asdict(chosen),
        "development": {
            "before": evaluate(dev, SHIPPED, fleet),
            "after": evaluate(dev, chosen, fleet),
        },
        "held_out": {
            "before": evaluate(held, SHIPPED, fleet),
            "after": evaluate(held, chosen, fleet),
        },
        "grid": grid,
    }


__all__ = ["SHIPPED", "ShiftSettings", "evaluate", "flights", "selection_key", "shifted", "study"]
