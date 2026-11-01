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
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from prognos.schema import SCENARIO_DESCRIPTORS
from prognos.stream.alerting import AlertingService, MaintenanceAlert
from prognos.stream.coverage import CoverageMonitor, CoverageVerdict, Observation
from prognos.stream.events import StreamStats, TelemetryBatch, validate_batch
from prognos.stream.faults import (
    DegradedState,
    FaultDetector,
    FaultInjector,
    InjectedFault,
    ShiftTracker,
    alert_payload,
)
from prognos.stream.fleet import FleetView, engine_label
from prognos.stream.inference import InferenceUnavailableError, Prediction, Predictor
from prognos.stream.observability import (
    MetricsRegistry,
    record_coverage_metrics,
    record_stream_metrics,
    span,
    tracer,
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


#: The flight conditions lead every reading; the shift tracker judges only these.
N_DESCRIPTORS = len(SCENARIO_DESCRIPTORS)


@dataclass(slots=True)
class _EngineState:
    """One engine's live stream: its window, its detector, and what it has raised."""

    window: OnlineWindow
    detector: FaultDetector | None
    detected_cycle: int | None = None
    last_good: npt.NDArray[np.float32] | None = None
    alerts: list[MaintenanceAlert] = field(default_factory=list)
    #: Pipeline-level conditions, beside the detector's sensor faults.
    outage: DegradedState | None = None
    silent: DegradedState | None = None
    shift: ShiftTracker | None = None
    #: The current flight's lowest and highest flight conditions, handed to the shift
    #: tracker when the flight ends.
    flight_low: npt.NDArray[np.float32] | None = None
    flight_high: npt.NDArray[np.float32] | None = None
    last_seen: float = 0.0
    #: Set when a replay reaches the end of an engine's recorded life, so the silence
    #: watchdog does not call an engine that has finished a stream that went quiet.
    ended: bool = False
    #: Every degraded state that opened, as the alert payload it raised.
    degraded_alerts: list[dict[str, object]] = field(default_factory=list)
    open_states: set[tuple[str, int | None]] = field(default_factory=set)
    late_dropped: int = 0
    missed_predictions: int = 0


class StreamPipeline:
    """Telemetry in, calibrated predictions and alerts out.

    Holds the components as collaborators rather than constructing them, so a test can
    substitute a deliberately bad one and watch the pipeline report the damage
    instead of silently succeeding.
    """

    __slots__ = (
        "_alerting",
        "_clock",
        "_config",
        "_coverage",
        "_detectors",
        "_engines",
        "_envelope",
        "_fleet",
        "_injectors",
        "_logger",
        "_metrics",
        "_predictor",
        "_stats",
    )

    def __init__(
        self,
        predictor: Predictor,
        config: PipelineConfig,
        metrics: MetricsRegistry | None = None,
        logger: logging.Logger | None = None,
        detect_faults: bool = True,
        envelope: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the pipeline around a predictor.

        `detect_faults` exists so a caller can replay a clean stream without paying for
        detection. It defaults on because S6 requires any degraded state to be
        observable, and a detector that is off by default is a detector nobody turned on.

        `envelope` is the range of flight conditions the model was fitted on; outside
        it an engine is named `operating_shift`. `clock` is injectable so the silence
        watchdog can be tested without waiting.
        """
        self._envelope = envelope
        self._clock = clock
        self._predictor = predictor
        self._config = config
        self._metrics = metrics or MetricsRegistry()
        self._logger = logger or logging.getLogger("prognos.pipeline")
        self._stats = StreamStats()
        self._detectors: dict[tuple[str | None, int], FaultDetector] | None = (
            {} if detect_faults else None
        )
        self._coverage = CoverageMonitor(
            target=config.target_coverage, min_samples=config.min_coverage_samples
        )
        self._alerting = AlertingService(
            threshold=config.alert_threshold, hysteresis=config.alert_hysteresis
        )
        self._fleet = FleetView()
        self._engines: dict[tuple[str | None, int], _EngineState] = {}
        self._injectors: dict[tuple[str | None, int], FaultInjector] = {}

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

    def _detector_for(self, key: tuple[str | None, int]) -> FaultDetector | None:
        """A fresh detector for an engine whose stream is starting."""
        if self._detectors is None:
            return None
        # Replaced rather than reused: a replay restarts the engine's stream, and a
        # detector carrying the last replay's frozen baseline would judge the new one
        # against a different engine life.
        self._detectors[key] = FaultDetector(unit=key[1])
        return self._detectors[key]

    @property
    def degraded_states(self) -> list[dict[str, object]]:
        """Every fault currently open, for the dashboard and the metrics endpoint.

        S6: any degraded state must be observable. A fault that is detected and then
        dropped on the floor is the same as one never detected.
        """
        states: list[dict[str, object]] = []
        for state in self._engines.values():
            if state.detector is not None:
                states.extend(s.to_dict() for s in state.detector.active())
            shift = state.shift.open if state.shift is not None else None
            pipeline_states = (state.outage, state.silent, shift)
            states.extend(s.to_dict() for s in pipeline_states if s is not None)
        return states

    @property
    def open_alerts(self) -> dict[tuple[str | None, int], MaintenanceAlert]:
        """Every engine currently under a maintenance alert."""
        return self._alerting.alerts()

    def arm_fault(self, unit: int, fault: InjectedFault | None, source: str | None = None) -> None:
        """Inject a named fault into an engine's live stream, or disarm it.

        Applied to the rows as they arrive, before detection and windowing, so what the
        dashboard shows afterwards is the pipeline's own reaction to a broken sensor
        rather than a flag someone set.
        """
        self._injectors[(source, unit)] = FaultInjector(unit=unit, fault=fault)

    def _engine(self, key: tuple[str | None, int]) -> _EngineState:
        """Per-engine stream state, created on first sight.

        Keyed by `(source, unit)` like everything else that holds per-engine state:
        unit numbers restart in every sub-dataset, so the unit alone is not an engine.
        """
        state = self._engines.get(key)
        if state is None:
            config = self._config
            state = _EngineState(
                window=OnlineWindow(
                    length=config.window_length,
                    downsample=config.downsample,
                    stride=config.window_stride,
                ),
                detector=self._detector_for(key),
                shift=(
                    ShiftTracker(unit=key[1], fleet=self._envelope)
                    if self._envelope is not None
                    else None
                ),
            )
            self._engines[key] = state
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
        key = (source, unit)
        state = self._engine(key)
        state.last_seen = self._clock()
        if state.silent is not None:
            state.silent = None
            self._publish(key, state)
        injector = self._injectors.get(key)
        if injector is not None:
            row = injector.apply(cycle, row.copy())

        # A reading from a cycle already passed is named and dropped. Folding it in
        # would change windows whose predictions have already been reported.
        if state.detected_cycle is not None and cycle < state.detected_cycle:
            state.late_dropped += 1
            if state.detector is not None:
                state.detector.observe(cycle, row)
                self._publish(key, state)
            return []
        new_cycle = cycle != state.detected_cycle
        if state.shift is not None:
            self._track_flight(key, state, row, new_cycle)
        detector = state.detector
        if detector is not None and new_cycle:
            detector.observe(cycle, row)
            self._publish(key, state)
        state.detected_cycle = cycle

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
            # One span per window, never per reading: a reading that completes no
            # window does nothing worth tracing, and there are thousands per cycle.
            with tracer.start_as_current_span(
                "pipeline.window",
                attributes={"engine": engine_label(source, unit), "cycle": cycle},
            ):
                features = window.materialise()
                try:
                    interval = self._predictor.predict(
                        features[None, ...], np.asarray([cycle], dtype=np.int64)
                    )[0]
                except InferenceUnavailableError as error:
                    # No prediction rather than the last one: a stale interval drawn as
                    # current is the silent wrong answer this state exists to prevent.
                    state.missed_predictions += 1
                    if state.outage is None:
                        state.outage = DegradedState(
                            state="inference_unavailable",
                            engine=unit,
                            detail=f"no prediction since cycle {cycle}: {error}",
                            details={"engine": unit, "cycle": cycle},
                        )
                        self._publish(key, state)
                    continue
                if state.outage is not None:
                    state.outage = None
                    self._publish(key, state)
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
                            source=source,
                        )
                    )
                with tracer.start_as_current_span("alerting.evaluate"):
                    state.alerts.extend(
                        self._alerting.evaluate(
                            unit=unit,
                            cycle=cycle,
                            lower=interval.lower,
                            point=interval.point,
                            source=source,
                        )
                    )
                open_alert = self._alerting.alerts().get(key)
                self._fleet.set_alert(
                    unit, open_alert.to_dict() if open_alert is not None else None, source=source
                )
        return predictions

    def _track_flight(
        self,
        key: tuple[str | None, int],
        state: _EngineState,
        row: npt.NDArray[np.float32],
        new_cycle: bool,
    ) -> None:
        """Extend the current flight's extremes; judge the last flight when one ends."""
        conditions = row[:N_DESCRIPTORS]
        if new_cycle:
            self._end_flight(key, state)
        if state.flight_low is None or state.flight_high is None:
            state.flight_low = conditions.copy()
            state.flight_high = conditions.copy()
        else:
            np.minimum(state.flight_low, conditions, out=state.flight_low)
            np.maximum(state.flight_high, conditions, out=state.flight_high)

    def _end_flight(self, key: tuple[str | None, int], state: _EngineState) -> None:
        """Hand a finished flight to the shift tracker."""
        if state.shift is None or state.flight_low is None or state.flight_high is None:
            return
        if state.shift.observe_flight(state.flight_low, state.flight_high):
            self._publish(key, state)
        state.flight_low = state.flight_high = None

    def _publish(self, key: tuple[str | None, int], state: _EngineState) -> None:
        """Show an engine's degraded states, and raise an alert for each that opened.

        One place for detector faults and pipeline conditions alike, so a state cannot
        reach the fleet view without its alert, or the alert without the view.
        """
        source, unit = key
        current = list(state.detector.active()) if state.detector is not None else []
        shift = state.shift.open if state.shift is not None else None
        current += [s for s in (state.outage, state.silent, shift) if s is not None]
        names = {(s.state, s.channel) for s in current}
        for degraded in current:
            if (degraded.state, degraded.channel) not in state.open_states:
                alert = {**alert_payload(degraded, state.detected_cycle or 0), "source": source}
                state.degraded_alerts.append(alert)
                self._logger.warning("degraded state opened", extra={"extra_fields": {**alert}})
        state.open_states = names
        self._fleet.set_degraded(unit, [s.to_dict() for s in current], source=source)

    def check_silence(self, timeout: float) -> list[dict[str, object]]:
        """Name every engine that has sent nothing for `timeout` seconds.

        Absence cannot be seen in the readings, only in the clock: a stream that stops
        logs nothing. Engines whose replay has finished are excluded; they are done,
        not silent.
        """
        now = self._clock()
        opened: list[dict[str, object]] = []
        for key, state in self._engines.items():
            quiet = now - state.last_seen
            if state.ended or state.silent is not None or quiet < timeout:
                continue
            state.silent = DegradedState(
                state="stream_silent",
                engine=key[1],
                detail=f"nothing received for {quiet:.1f} s",
                details={"engine": key[1], "seconds": quiet},
            )
            before = len(state.degraded_alerts)
            self._publish(key, state)
            opened.extend(state.degraded_alerts[before:])
        return opened

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
        self._engines.pop((source, unit), None)
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

        state = self._engines.get((source, unit))
        if state is not None:
            self._end_flight((source, unit), state)
            state.ended = True
            replay.alerts = state.alerts
            replay.faults = state.degraded_alerts
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
                engine_label(source, unit): alert.to_dict()
                for (source, unit), alert in self._alerting.alerts().items()
            },
            "degraded_states": self.degraded_states,
            "fleet": self._fleet.rows(),
            "model": self._predictor.model_name,
            "target_coverage": self._config.target_coverage,
        }
