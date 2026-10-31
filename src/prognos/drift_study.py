"""Tuning the drift detector without tuning on the engines it is judged on.

The drift detector fired on healthy engines. Retuning it until a chosen control run
came out clean would produce a number that means nothing, so the procedure is fixed
here before any result exists:

1. **Split.** The cross-validation engines are divided, within each sub-dataset and by
   a seeded draw, into development and held-out halves. The final test group is not
   read at all.
2. **Measure on clean engines.** Every engine is healthy as recorded, so any
   `sensor_drift` on one is a false alarm.
3. **Measure detection.** A step is added to one measured sensor from cycle 40,
   sized as a multiple of that channel's typical cycle-to-cycle change on the
   development engines. It counts as detected if `sensor_drift` names that channel
   within `HORIZON` cycles.
4. **Select on development engines only,** by `selection_key`: fewest engines with
   any false alarm, then most detections at 5x, then at 2x, then the most sensitive
   setting.
5. **Report before and after on the held-out engines,** which played no part in the
   choice.

The detector is fed the first reading of each cycle, which is what the live pipeline
feeds it.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .dataset import GroupKey
from .pipeline import _unit_table
from .schema import SCENARIO_DESCRIPTORS
from .stream.faults import FaultDetector, channel_of
from .stream.window import N_CHANNELS

#: Measured sensors only. The scenario descriptors are flight conditions, not sensors
#: that can drift, and a step in them is an operating-condition change.
MEASURED = tuple(range(len(SCENARIO_DESCRIPTORS), N_CHANNELS))

#: Where an injected step begins, and how long the detector has to name it.
START, HORIZON = 40, 15

#: Step sizes, in multiples of the channel's typical cycle-to-cycle change.
MULTIPLES = (2.0, 5.0, 10.0)

#: The grid searched on development engines.
FACTORS = (3.0, 4.0, 6.0, 8.0, 12.0)
HALVES = (4, 6, 10)
PERSISTENCE = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class DriftSettings:
    """The three things the study tunes."""

    factor: float
    half: int
    persistence: int

    def detector(self) -> FaultDetector:
        """A fresh detector with these settings."""
        return FaultDetector(
            unit=1,
            drift_factor=self.factor,
            drift_half=self.half,
            drift_persistence=self.persistence,
        )


#: The detector as Phase 5 shipped it.
SHIPPED = DriftSettings(factor=4.0, half=6, persistence=1)


@dataclass(frozen=True, slots=True)
class CycleStream:
    """One engine, one reading per cycle."""

    label: str
    cycles: npt.NDArray[np.int64]
    rows: npt.NDArray[np.float32]


def split_engines(labels: list[GroupKey], seed: int) -> tuple[list[GroupKey], list[GroupKey]]:
    """Development and held-out engines, half of each sub-dataset to each side.

    Stratified so neither side is missing a sub-dataset: DS06's failure modes differ,
    and a detector tuned without them would be judged on engines it never resembled.
    """
    rng = np.random.default_rng(seed)
    development: list[GroupKey] = []
    held_out: list[GroupKey] = []
    for subdataset in sorted({label.subdataset for label in labels}):
        group = sorted(
            (label for label in labels if label.subdataset == subdataset),
            key=lambda key: key.unit,
        )
        order = rng.permutation(len(group))
        cut = (len(group) + 1) // 2
        development.extend(group[i] for i in order[:cut])
        held_out.extend(group[i] for i in order[cut:])
    return development, held_out


def cycle_stream(parquet_dir: Path, label: GroupKey) -> CycleStream:
    """An engine's first reading of each cycle."""
    cycles, rows, _rul = _unit_table(parquet_dir, label.subdataset, label.unit)
    firsts = np.flatnonzero(np.r_[True, np.diff(cycles) != 0])
    return CycleStream(label=label.label(), cycles=cycles[firsts], rows=rows[firsts])


def channel_scales(streams: list[CycleStream]) -> npt.NDArray[np.float64]:
    """Each channel's typical cycle-to-cycle change, pooled over the given engines.

    The size of an injected step is stated in these units, so a step means the same
    thing on core speed near 8,700 as on a pressure near 20.
    """
    differences = np.concatenate([np.diff(s.rows.astype(np.float64), axis=0) for s in streams])
    return np.asarray(np.std(differences, axis=0), dtype=np.float64)


def false_alarms(streams: list[CycleStream], settings: DriftSettings) -> dict[str, Any]:
    """`sensor_drift` raised on clean engines, by engine, cycle and channel."""
    alarmed: list[str] = []
    alarm_cycles = 0
    total_cycles = 0
    by_channel: Counter[str] = Counter()
    for stream in streams:
        detector = settings.detector()
        engine_alarms = 0
        for cycle, row in zip(stream.cycles, stream.rows, strict=True):
            detector.observe(int(cycle), row)
            drifting = [s for s in detector.active() if s.state == "sensor_drift"]
            if drifting:
                engine_alarms += 1
                by_channel.update(s.instrument for s in drifting)
        total_cycles += int(stream.cycles.size)
        alarm_cycles += engine_alarms
        if engine_alarms:
            alarmed.append(stream.label)
    return {
        "engines": len(streams),
        "engines_alarmed": len(alarmed),
        "alarmed_engines": alarmed,
        "cycles": total_cycles,
        "alarm_cycles": alarm_cycles,
        "alarm_cycles_per_100": 100.0 * alarm_cycles / total_cycles if total_cycles else 0.0,
        "by_channel": dict(by_channel.most_common()),
    }


def detects(stream: CycleStream, settings: DriftSettings, channel: int, step: float) -> int | None:
    """Cycles from the step to the first `sensor_drift` on its channel, or None."""
    detector = settings.detector()
    for index, (cycle, row) in enumerate(zip(stream.cycles, stream.rows, strict=True)):
        if index >= START + HORIZON:
            return None
        reading = row.copy()
        if index >= START:
            reading[channel] += np.float32(step)
        detector.observe(int(cycle), reading)
        if index >= START and any(
            s.state == "sensor_drift" and s.channel == channel for s in detector.active()
        ):
            return index - START
    return None


def detection(
    streams: list[CycleStream],
    settings: DriftSettings,
    scales: npt.NDArray[np.float64],
    multiple: float,
) -> dict[str, Any]:
    """Every measured channel of every engine, one step each."""
    delays: list[int] = []
    missed: Counter[str] = Counter()
    trials = 0
    for stream in streams:
        for channel in MEASURED:
            trials += 1
            delay = detects(stream, settings, channel, multiple * float(scales[channel]))
            if delay is None:
                missed[channel_of(channel)] += 1
            else:
                delays.append(delay)
    return {
        "multiple": multiple,
        "trials": trials,
        "detected": len(delays),
        "rate": len(delays) / trials if trials else 0.0,
        "median_delay_cycles": float(np.median(delays)) if delays else None,
        "missed_by_channel": dict(missed.most_common()),
    }


def evaluate(
    streams: list[CycleStream], settings: DriftSettings, scales: npt.NDArray[np.float64]
) -> dict[str, Any]:
    """False alarms and detection at every step size, for one setting."""
    return {
        "settings": asdict(settings),
        "false_alarms": false_alarms(streams, settings),
        "detection": [detection(streams, settings, scales, m) for m in MULTIPLES],
    }


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    """The pre-registered rule: lower sorts first.

    Fewest engines with any false alarm, then most steps caught at 5x, then at 2x, then
    the most sensitive setting, so a tie never goes to the setting that simply looks
    less.
    """
    rate = {d["multiple"]: d["rate"] for d in result["detection"]}
    settings = result["settings"]
    return (
        float(result["false_alarms"]["engines_alarmed"]),
        -rate[5.0],
        -rate[2.0],
        settings["factor"],
        float(settings["half"]),
        float(settings["persistence"]),
    )


def study(parquet_dir: Path, labels: list[GroupKey], seed: int) -> dict[str, Any]:
    """Search on development engines, then judge the choice on held-out ones."""
    development, held_out = split_engines(labels, seed)
    dev_streams = [cycle_stream(parquet_dir, label) for label in development]
    held_streams = [cycle_stream(parquet_dir, label) for label in held_out]
    scales = channel_scales(dev_streams)

    grid = []
    for factor, half, persistence in itertools.product(FACTORS, HALVES, PERSISTENCE):
        settings = DriftSettings(factor=factor, half=half, persistence=persistence)
        result = evaluate(dev_streams, settings, scales)
        grid.append(result)
        alarms = result["false_alarms"]
        print(
            f"  factor {factor:>4} half {half:>2} persist {persistence}: "
            f"{alarms['engines_alarmed']}/{alarms['engines']} engines alarmed, "
            + ", ".join(f"{d['multiple']:.0f}x {d['rate']:.2f}" for d in result["detection"]),
            flush=True,
        )
    chosen = DriftSettings(**min(grid, key=selection_key)["settings"])
    return {
        "development_engines": [label.label() for label in development],
        "held_out_engines": [label.label() for label in held_out],
        "channel_scales": {channel_of(c): float(scales[c]) for c in MEASURED},
        "protocol": {
            "start_cycle": START,
            "horizon_cycles": HORIZON,
            "multiples": list(MULTIPLES),
            "grid": {
                "factor": list(FACTORS),
                "half": list(HALVES),
                "persistence": list(PERSISTENCE),
            },
            "selection": (
                "fewest development engines with any false alarm, then highest detection "
                "at 5x, then at 2x, then the most sensitive setting"
            ),
        },
        "shipped": asdict(SHIPPED),
        "chosen": asdict(chosen),
        "development": {
            # Evaluated directly when the grid does not contain it, so narrowing the
            # grid cannot lose the comparison against what shipped.
            "before": next(
                (r for r in grid if DriftSettings(**r["settings"]) == SHIPPED),
                None,
            )
            or evaluate(dev_streams, SHIPPED, scales),
            "after": next(r for r in grid if DriftSettings(**r["settings"]) == chosen),
        },
        "held_out": {
            "before": evaluate(held_streams, SHIPPED, scales),
            "after": evaluate(held_streams, chosen, scales),
        },
        "grid": grid,
    }


__all__ = [
    "SHIPPED",
    "DriftSettings",
    "channel_scales",
    "detection",
    "detects",
    "false_alarms",
    "selection_key",
    "split_engines",
    "study",
]
