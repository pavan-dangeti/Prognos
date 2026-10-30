"""Replaying an engine through the whole pipeline.

`docs/design.md` §4.6 describes six components. Each is separately useful and
separately tested, and a set of components that each pass can still be wired together
wrongly — the window fed the wrong cycle, the alert keyed on the point estimate instead
of the lower bound, the coverage monitor handed something other than the truth it is
supposed to score.

`StreamPipeline` is that wiring, in one process, so it can be run against real held-out
engines and produce a number that either holds up or does not. The production
deployment puts the same components behind HTTP in six containers; the arithmetic is
identical either way, which is the property worth being able to claim.

The final test in `tests/test_stream_service.py` pins the wiring: the pipeline's
coverage verdict is compared against a coverage count computed directly from the same
predictions. If the monitor ever scores a different stream than the one reported, the
two diverge.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from prognos.stream.alerting import AlertingService, MaintenanceAlert
from prognos.stream.coverage import CoverageMonitor, CoverageVerdict, Observation
from prognos.stream.events import StreamStats, TelemetryBatch, validate_batch
from prognos.stream.faults import FaultDetector, FaultInjector, InjectedFault
from prognos.stream.fleet import FleetView
from prognos.stream.inference import Prediction, RulPredictor
from prognos.stream.observability import (
    MetricsRegistry,
    record_coverage_metrics,
    record_stream_metrics,
    span,
)
from prognos.stream.window import OnlineWindow


@dataclass(slots=True)
class PipelineConfig:
    """Window and policy settings for the replay.

    Mirrors `configs/base.yaml` so a replay and an offline evaluation use the same
    window. A streaming demo on a different window than the model was trained on would
    be measuring the window, not the pipeline.
    """

    window_length: int = 50
    window_stride: int = 10
    downsample: int = 10
    rul_cap: int = 125
    alert_threshold: float = 20.0
    alert_hysteresis: float = 5.0
    target_coverage: float = 0.9
    min_coverage_samples: int = 50


@dataclass(slots=True)
class EngineReplay:
    """What one engine's replay produced."""

    unit: int
    n_windows: int
    n_rows: int
    predictions: list[Prediction] = field(default_factory=list)
    alerts: list[MaintenanceAlert] = field(default_factory=list)
    faults: list[dict[str, object]] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _EngineState:
    """One engine's live stream: its window, its detector, and what it has raised."""

    window: OnlineWindow
    detector: FaultDetector | None
    detected_cycle: int | None = None
    last_good: npt.NDArray[np.float32] | None = None
    alerts: list[MaintenanceAlert] = field(default_factory=list)


class StreamPipeline:
    """Telemetry in, calibrated predictions and alerts out.

    Holds the components as collaborators rather than constructing them, so a test can
    substitute a deliberately bad one and watch the pipeline report the damage
    instead of silently succeeding.
    """

    __slots__ = (
        "_alerting",
        "_config",
        "_coverage",
        "_detectors",
        "_engines",
        "_fleet",
        "_injectors",
        "_logger",
        "_metrics",
        "_predictor",
        "_stats",
    )

    def __init__(
        self,
        predictor: RulPredictor,
        config: PipelineConfig,
        metrics: MetricsRegistry | None = None,
        logger: logging.Logger | None = None,
        detect_faults: bool = True,
    ) -> None:
        """Wire the pipeline around a predictor.

        `detect_faults` exists so a caller can replay a clean stream without paying for
        detection. It defaults on because S6 requires any degraded state to be
        observable, and a detector that is off by default is a detector nobody turned on.
        """
        self._predictor = predictor
        self._config = config
        self._metrics = metrics or MetricsRegistry()
        self._logger = logger or logging.getLogger("prognos.pipeline")
        self._stats = StreamStats()
        self._detectors: dict[int, FaultDetector] | None = {} if detect_faults else None
        self._coverage = CoverageMonitor(
            target=config.target_coverage, min_samples=config.min_coverage_samples
        )
        self._alerting = AlertingService(
            threshold=config.alert_threshold, hysteresis=config.alert_hysteresis
        )
        self._fleet = FleetView()
        self._engines: dict[int, _EngineState] = {}
        self._injectors: dict[int, FaultInjector] = {}

    @property
    def fleet(self) -> FleetView:
        """The live fleet view, fed as predictions arrive.

        Owned by the pipeline rather than the dashboard because the pipeline is the
        only thing that sees the stream. The dashboard reads it; it does not build it.
        """
        return self._fleet

    @property
    def coverage_monitor(self) -> CoverageMonitor:
        """The live S5 monitor, for a dashboard that reports the same verdict."""
        return self._coverage

    @property
    def metrics(self) -> MetricsRegistry:
        """The registry, for the scrape endpoint."""
        return self._metrics

    @property
    def coverage_verdict(self) -> CoverageVerdict:
        """The current S5 verdict."""
        return self._coverage.verdict()

    def _detector_for(self, unit: int) -> FaultDetector | None:
        """A fresh detector for an engine whose stream is starting."""
        if self._detectors is None:
            return None
        # Replaced rather than reused: a replay restarts the engine's stream, and a
        # detector carrying the last replay's frozen baseline would judge the new one
        # against a different engine life.
        self._detectors[unit] = FaultDetector(unit=unit)
        return self._detectors[unit]

    @property
    def degraded_states(self) -> list[dict[str, object]]:
        """Every fault currently open, for the dashboard and the metrics endpoint.

        S6: any degraded state must be observable. A fault that is detected and then
        dropped on the floor is the same as one never detected.
        """
        if self._detectors is None:
            return []
        states: list[dict[str, object]] = []
        for detector in self._detectors.values():
            states.extend(state.to_dict() for state in detector.active())
        return states

    @property
    def open_alerts(self) -> dict[int, MaintenanceAlert]:
        """Every engine currently under a maintenance alert."""
        return self._alerting.alerts()

    def arm_fault(self, unit: int, fault: InjectedFault | None) -> None:
        """Inject a named fault into an engine's live stream, or disarm it.

        Applied to the rows as they arrive, before detection and windowing, so what the
        dashboard shows afterwards is the pipeline's own reaction to a broken sensor
        rather than a flag someone set.
        """
        self._injectors[unit] = FaultInjector(unit=unit, fault=fault)

    def _engine(self, unit: int) -> _EngineState:
        """Per-engine stream state, created on first sight."""
        state = self._engines.get(unit)
        if state is None:
            config = self._config
            state = _EngineState(
                window=OnlineWindow(
                    length=config.window_length,
                    downsample=config.downsample,
                    stride=config.window_stride,
                ),
                detector=self._detector_for(unit),
            )
            self._engines[unit] = state
        return state

    def push(
        self,
        unit: int,
        cycle: int,
        row: npt.NDArray[np.float32],
        truth: float | None = None,
        source: str | None = None,
    ) -> list[Prediction]:
        """Take one sensor reading and return any predictions it completed.

        The live entry point: engines interleave freely, and each keeps its own window
        and detector. Most readings complete no window and return an empty list.

        The detector sees the first reading of each cycle, not every reading. It judges
        engine behaviour cycle to cycle, which is the cadence `results/phase5_faults.json`
        was measured at. Fed every raw row it judged within-cycle scatter instead, and at
        about 11,000 rows a cycle it was the largest cost in the whole pipeline.
        """
        state = self._engine(unit)
        injector = self._injectors.get(unit)
        if injector is not None:
            row = injector.apply(cycle, row.copy())
        detector = state.detector
        if detector is not None and cycle != state.detected_cycle:
            state.detected_cycle = cycle
            detector.observe(cycle, row)
            self._fleet.set_degraded(unit, [s.to_dict() for s in detector.active()])

        # A missing channel is held at its last good value, so the engine keeps a
        # prediction while the detector names the dropout and the fleet view marks the
        # engine degraded. The window refuses NaN outright, and letting that refusal
        # escape stopped the whole replay over one dead sensor. With no good value yet
        # there is nothing honest to hold, so the reading is skipped.
        finite = np.isfinite(row)
        if finite.all():
            state.last_good = row
        elif state.last_good is None:
            return []
        else:
            row = np.where(finite, row, state.last_good).astype(np.float32)
        window = state.window
        window.push(cycle, row)

        predictions: list[Prediction] = []
        while window.has_window:
            features = window.materialise()
            interval = self._predictor.predict(
                features[None, ...], np.asarray([cycle], dtype=np.int64)
            )[0]
            prediction = Prediction(
                unit=unit,
                cycle=cycle,
                interval=interval,
                model=self._predictor.model_name,
                target_coverage=self._config.target_coverage,
            )
            predictions.append(prediction)
            self._fleet.record(unit=unit, cycle=cycle, interval=interval, source=source)
            if truth is not None:
                self._coverage.record(
                    Observation(
                        unit=unit,
                        truth=truth,
                        lower=interval.lower,
                        upper=interval.upper,
                        point=interval.point,
                    )
                )
            state.alerts.extend(
                self._alerting.evaluate(
                    unit=unit, cycle=cycle, lower=interval.lower, point=interval.point
                )
            )
            open_alert = self._alerting.alerts().get(unit)
            self._fleet.set_alert(unit, open_alert.to_dict() if open_alert is not None else None)
        return predictions

    def run_engine(
        self,
        unit: int,
        cycles: npt.NDArray[np.int64],
        rows: npt.NDArray[np.float32],
        truth_by_cycle: dict[int, float] | None = None,
        source: str | None = None,
        pace: float | None = None,
    ) -> EngineReplay:
        """Replay one engine end to end, from a fresh window.

        `pace` caps the replay at that many readings per second. Unpaced, an engine's
        whole life passes in seconds, which is right for a measurement and too fast for
        anyone watching the dashboard to see a fault arrive and clear.

        `truth` is the RUL at each window, supplied by the replay driver from the
        dataset. It is what the coverage monitor scores against. Passing it as an
        argument rather than reading it from a global is deliberate: a monitor that
        cannot be given the wrong truth is a monitor that cannot be quietly wrong.

        `source` is the sub-dataset, which is half of an engine's identity: unit
        numbers restart per sub-dataset, so keying on `unit` alone merges engines that
        are not the same engine. That mistake once produced a 0.716 coverage figure.

        `unit` is therefore the engine's own unit number and not a row or group index.
        The two differ — group 0 is DS01 unit 1 — and an operator reading a fleet view
        showing `unit000` would be reading an identifier that means nothing.
        """
        self._engines.pop(unit, None)
        replay = EngineReplay(unit=unit, n_windows=0, n_rows=int(cycles.size))
        cycle_array = np.asarray(cycles, dtype=np.int64)
        block = np.asarray(rows, dtype=np.float32)

        # Counted once per engine rather than per row. Validating row by row would add
        # a million empty batches to the counters and make the scrape useless.
        empty = TelemetryBatch(source="replay", unit=unit, rows=[])
        self._stats.record(empty, validate_batch(empty))

        started = time.monotonic()
        for index in range(cycle_array.size):
            # Checked every thousand readings: often enough to keep the rate steady, rare
            # enough that the clock read is not the cost being paid.
            if pace is not None and index % 1000 == 0:
                ahead = index / pace - (time.monotonic() - started)
                if ahead > 0:
                    time.sleep(ahead)
            cycle = int(cycle_array[index])
            truth = truth_by_cycle.get(cycle) if truth_by_cycle is not None else None
            for prediction in self.push(unit, cycle, block[index], truth=truth, source=source):
                replay.predictions.append(prediction)
                replay.coverage.append(float(prediction.interval.point))
                replay.n_windows += 1

        state = self._engines.get(unit)
        if state is not None:
            replay.alerts = state.alerts
            if state.detector is not None:
                replay.faults = [state.detector.raise_alert_for(s) for s in state.detector.history]
        return replay

    def record_metrics(self) -> None:
        """Fold the current counters into the registry, for a scrape."""
        with span(self._metrics, "pipeline.record_metrics"):
            record_stream_metrics(self._metrics, self._stats)
            record_coverage_metrics(self._metrics, self.coverage_verdict)

    def summary(self) -> dict[str, object]:
        """A status document, for the dashboard and the status endpoint."""
        verdict = self.coverage_verdict
        return {
            "stream": self._stats.to_dict(),
            "coverage": verdict.to_dict(),
            "open_alerts": {
                str(unit): alert.to_dict() for unit, alert in self._alerting.alerts().items()
            },
            "degraded_states": self.degraded_states,
            "fleet": self._fleet.rows(),
            "model": self._predictor.model_name,
            "target_coverage": self._config.target_coverage,
        }
