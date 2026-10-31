"""System failures, provoked against real engines and the served model.

The fault scenarios in `pipeline.run_fault_scenarios` break one sensor at a time and
ask the detector to name it. These break the system around the sensors: inference
goes down, the engine flies somewhere the model has never seen, a stream stops, a
stream arrives out of order. Each runs the whole pipeline over a real engine and
records what was named, what was alerted, and what was predicted while it happened.

A clean control runs first through the same path. A scenario only means something if
the same pipeline, unprovoked, names nothing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .pipeline import _unit_table
from .stream.api import build_inference_state
from .stream.inference import Predictor, RemotePredictor, RulPredictor
from .stream.pipeline import PipelineConfig, StreamPipeline
from .stream.service import build_server

Envelope = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
Engine = tuple[str, int, npt.NDArray[np.int64], npt.NDArray[np.float32]]

#: Where the outage starts and ends, as fractions of the engine's readings.
OUTAGE = (0.4, 0.6)

#: The cycle from which a shift or a late block is introduced.
EVENT_CYCLE = 40

#: Silence: seconds of stream time per reading, how long the stream stops, and the
#: watchdog's timeout.
SECONDS_PER_READING, PAUSE_SECONDS, SILENCE_TIMEOUT = 0.001, 15.0, 10.0


def _label(source: str, unit: int) -> str:
    return f"{source}/unit{unit:03d}"


def _opened(pipeline: StreamPipeline) -> dict[str, list[str]]:
    """Every degraded state that raised an alert, by engine."""
    return {
        _label(str(source), unit): [str(a["state"]) for a in state.degraded_alerts]
        for (source, unit), state in pipeline._engines.items()
        if state.degraded_alerts
    }


def _replay(
    predictor: Predictor,
    config: PipelineConfig,
    envelope: Envelope,
    engine: Engine,
    before_each: Callable[[int], None] | None = None,
) -> tuple[StreamPipeline, list[int]]:
    """One engine through the pipeline, with a hook before each reading."""
    source, unit, cycles, rows = engine
    pipeline = StreamPipeline(predictor, config, envelope=envelope)
    produced: list[int] = []
    for index in range(cycles.size):
        if before_each is not None:
            before_each(index)
        if pipeline.push(unit, int(cycles[index]), rows[index], source=source):
            produced.append(index)
    return pipeline, produced


def control(
    model: RulPredictor, config: PipelineConfig, envelope: Envelope, engine: Engine
) -> dict[str, Any]:
    """The clean run every scenario is judged against."""
    pipeline, produced = _replay(model, config, envelope, engine)
    opened = _opened(pipeline)
    return {
        "scenario": "control",
        "injected": "nothing",
        "expected": [],
        "opened": opened,
        "predictions": len(produced),
        "passed": not opened,
    }


def inference_outage(
    model_dir: Path, config: PipelineConfig, envelope: Envelope, engine: Engine
) -> dict[str, Any]:
    """Take the real inference service down partway through an engine, then restore it."""
    source, unit, cycles, _rows = engine
    servers: list[ThreadingHTTPServer] = []

    def start(port: int = 0) -> int:
        server = build_server(build_inference_state(model_dir=model_dir), port=port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return int(server.server_address[1])

    def stop() -> None:
        servers[-1].shutdown()
        servers[-1].server_close()

    port = start()
    down, up = (int(cycles.size * fraction) for fraction in OUTAGE)

    def toggle(index: int) -> None:
        if index == down:
            stop()
        elif index == up:
            start(port)

    remote = RemotePredictor(f"http://127.0.0.1:{port}", timeout=1.0)
    try:
        pipeline, produced = _replay(remote, config, envelope, engine, before_each=toggle)
    finally:
        stop()
    state = pipeline._engines[(source, unit)]
    during = sum(down <= i < up for i in produced)
    opened = _opened(pipeline)
    return {
        "scenario": "inference_outage",
        "injected": (
            f"inference service stopped for readings {down:,} to {up:,} of {cycles.size:,}"
        ),
        "expected": ["inference_unavailable"],
        "opened": opened,
        "predictions_before": sum(i < down for i in produced),
        "predictions_during": during,
        "predictions_after": sum(i >= up for i in produced),
        "missed_predictions": state.missed_predictions,
        "cleared": state.outage is None,
        "passed": (
            list(opened.values()) == [["inference_unavailable"]]
            and during == 0
            and state.outage is None
        ),
    }


def operating_shift(
    model: RulPredictor,
    config: PipelineConfig,
    envelope: Envelope,
    engine: Engine,
    beyond_fleet: bool,
) -> dict[str, Any]:
    """Altitude jumps from one cycle on, past the fleet's experience or within it.

    Only the flight condition is moved; the measured sensors are left as recorded,
    because there is no simulator here to say how they would respond. The scenario
    tests whether the system says the model is out of its depth, not what the engine
    does.

    Run twice. Beyond the fleet, the engine climbs a tenth of the fitted range above
    the highest altitude any training engine reached, and must be named. Within it,
    altitude rises by half the fitted range: new for a short-haul engine, familiar to
    a fleet that includes long-haul ones. The envelope is the fleet's, so it cannot
    see that, and the run is reported as the gap it is.
    """
    source, unit, cycles, rows = engine
    lower, upper = envelope
    span = float(upper[0] - lower[0])
    late = cycles >= EVENT_CYCLE
    if beyond_fleet:
        step = float(upper[0]) - float(rows[late, 0].max()) + 0.1 * span
        described = "to a tenth of the fitted altitude range above the training fleet's highest"
    else:
        step = 0.5 * span
        described = "by half the fitted altitude range, staying inside the training fleet's"
    shifted = rows.copy()
    shifted[late, 0] += np.float32(step)
    pipeline, produced = _replay(model, config, envelope, (source, unit, cycles, shifted))
    opened = _opened(pipeline)
    return {
        "scenario": "operating_shift" if beyond_fleet else "operating_shift_within_fleet",
        "injected": f"altitude raised from cycle {EVENT_CYCLE} {described}",
        "expected": ["operating_shift"],
        "step_feet": step,
        "engine_max_altitude_after": float(shifted[late, 0].max()),
        "envelope_altitude": [float(lower[0]), float(upper[0])],
        "opened": opened,
        "predictions": len(produced),
        # Named and only this, however often. It opens per flight that leaves the
        # envelope and releases on a flight that stays inside, so the count of
        # openings is reported rather than required to be one.
        "openings": sum(len(names) for names in opened.values()),
        "passed": {n for names in opened.values() for n in names} == {"operating_shift"},
    }


def delayed(
    model: RulPredictor,
    config: PipelineConfig,
    envelope: Envelope,
    engine: Engine,
    other: Engine,
    readings: int = 60_000,
) -> dict[str, Any]:
    """Two engines interleaved; one stops sending while the other carries on.

    Stream time is simulated, one millisecond per reading, so the scenario does not
    depend on how fast this machine is.
    """
    now = [0.0]
    pipeline = StreamPipeline(model, config, envelope=envelope, clock=lambda: now[0])
    a_source, a_unit, a_cycles, a_rows = engine
    b_source, b_unit, b_cycles, b_rows = other
    pause_at = readings // 2
    pause = int(PAUSE_SECONDS / SECONDS_PER_READING)
    detected_after: float | None = None
    for index in range(readings + pause):
        now[0] += SECONDS_PER_READING
        pipeline.push(a_unit, int(a_cycles[index]), a_rows[index], source=a_source)
        if index < pause_at or index >= pause_at + pause:
            b_index = index if index < pause_at else index - pause
            pipeline.push(b_unit, int(b_cycles[b_index]), b_rows[b_index], source=b_source)
        if index % 100 == 0 and pipeline.check_silence(SILENCE_TIMEOUT) and detected_after is None:
            detected_after = (index - pause_at) * SECONDS_PER_READING
    opened = _opened(pipeline)
    silent = _label(b_source, b_unit)
    return {
        "scenario": "delayed",
        "injected": (
            f"{silent} stopped for {PAUSE_SECONDS:.0f} s of stream time while "
            f"{_label(a_source, a_unit)} continued; watchdog timeout {SILENCE_TIMEOUT:.0f} s"
        ),
        "expected": ["stream_silent"],
        "opened": opened,
        "detected_after_seconds": detected_after,
        "cleared": pipeline._engines[(b_source, b_unit)].silent is None,
        "passed": (
            opened == {silent: ["stream_silent"]}
            and pipeline._engines[(b_source, b_unit)].silent is None
        ),
    }


def out_of_order(
    model: RulPredictor,
    config: PipelineConfig,
    envelope: Envelope,
    engine: Engine,
    late: int = 200,
) -> dict[str, Any]:
    """The last readings of one cycle arrive after the next cycle has started."""
    source, unit, cycles, rows = engine
    start = int(np.flatnonzero(cycles == EVENT_CYCLE)[0])
    order = np.r_[
        np.arange(0, start - late),
        np.arange(start, start + 1),
        np.arange(start - late, start),
        np.arange(start + 1, cycles.size),
    ]
    _clean, clean_produced = _replay(model, config, envelope, engine)
    pipeline, produced = _replay(
        model, config, envelope, (source, unit, cycles[order], rows[order])
    )
    state = pipeline._engines[(source, unit)]
    opened = _opened(pipeline)
    return {
        "scenario": "out_of_order",
        "injected": (
            f"the last {late} readings of cycle {EVENT_CYCLE - 1} delivered after the "
            f"first reading of cycle {EVENT_CYCLE}"
        ),
        "expected": ["cycle_regression"],
        "opened": opened,
        "late_dropped": state.late_dropped,
        "predictions": len(produced),
        "predictions_in_order": len(clean_produced),
        "passed": (
            [n for names in opened.values() for n in names] == ["cycle_regression"]
            and state.late_dropped == late
        ),
    }


def run_failure_scenarios(
    model_dir: Path,
    parquet_dir: Path,
    config: PipelineConfig,
    envelope: Envelope,
    engines: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Every scenario, control first, on the first two of `engines`."""
    model = RulPredictor.load(model_dir)
    tables: list[Engine] = []
    for source, unit in engines[:2]:
        cycles, rows, _rul = _unit_table(parquet_dir, source, unit)
        tables.append((source, unit, cycles, rows))
    main, second = tables
    runs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("control", lambda: control(model, config, envelope, main)),
        ("inference_outage", lambda: inference_outage(model_dir, config, envelope, main)),
        ("operating_shift", lambda: operating_shift(model, config, envelope, main, True)),
        (
            "operating_shift_within_fleet",
            lambda: operating_shift(model, config, envelope, main, False),
        ),
        ("delayed", lambda: delayed(model, config, envelope, main, second)),
        ("out_of_order", lambda: out_of_order(model, config, envelope, main)),
    ]
    scenarios = []
    for name, run in runs:
        result = run()
        scenarios.append(result)
        print(f"  {name}: opened {result['opened']} passed={result['passed']}", flush=True)
    return scenarios


__all__ = [
    "control",
    "delayed",
    "inference_outage",
    "operating_shift",
    "out_of_order",
    "run_failure_scenarios",
]
