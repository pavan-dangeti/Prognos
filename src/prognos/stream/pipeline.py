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
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from prognos.stream.alerting import AlertingService, MaintenanceAlert
from prognos.stream.coverage import CoverageMonitor, CoverageVerdict, Observation
from prognos.stream.events import StreamStats, TelemetryBatch, validate_batch
from prognos.stream.faults import FaultDetector
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
        "_fleet",
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

    @property
    def fleet(self) -> FleetView:
        """The live fleet view, fed as predictions arrive.

        Owned by the pipeline rather than the dashboard because the pipeline is the
        only thing that sees the stream. The dashboard reads it; it does not build it.
        """
        return self._fleet

    @property
    def metrics(self) -> MetricsRegistry:
        """The registry, for the scrape endpoint."""
        return self._metrics

    @property
    def coverage_verdict(self) -> CoverageVerdict:
        """The current S5 verdict."""
        return self._coverage.verdict()

    def _detector_for(self, unit: int) -> FaultDetector | None:
        """One detector per engine, created on first sight."""
        if self._detectors is None:
            return None
        return self._detectors.setdefault(unit, FaultDetector(unit=unit))

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

    def run_engine(
        self,
        unit: int,
        cycles: npt.NDArray[np.int64],
        rows: npt.NDArray[np.float32],
        truth_by_cycle: dict[int, float] | None = None,
        source: str | None = None,
    ) -> EngineReplay:
        """Replay one engine end to end.

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
        config = self._config
        window = OnlineWindow(
            length=config.window_length, downsample=config.downsample, stride=config.window_stride
        )
        replay = EngineReplay(unit=unit, n_windows=0, n_rows=int(cycles.size))
        cycle_array = np.asarray(cycles, dtype=np.int64)
        block = np.asarray(rows, dtype=np.float32)
        detector = self._detector_for(unit)

        # Counted once per engine rather than per row. Validating row by row would add
        # a million empty batches to the counters and make the scrape useless.
        empty = TelemetryBatch(source="replay", unit=unit, rows=[])
        self._stats.record(empty, validate_batch(empty))

        for index in range(cycle_array.size):
            cycle = int(cycle_array[index])
            if detector is not None:
                detector.observe(cycle, block[index])
                for state in detector.active():
                    replay.faults.append(detector.raise_alert_for(state))
            window.push(cycle, block[index])

            if not window.has_window:
                continue
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
                    target_coverage=config.target_coverage,
                )
                replay.predictions.append(prediction)
                replay.coverage.append(float(interval.point))
                self._fleet.record(unit=unit, cycle=cycle, interval=interval, source=source)

                if truth_by_cycle is not None and cycle in truth_by_cycle:
                    self._coverage.record(
                        Observation(
                            unit=unit,
                            truth=float(truth_by_cycle[cycle]),
                            lower=interval.lower,
                            upper=interval.upper,
                            point=interval.point,
                        )
                    )
                replay.alerts.extend(
                    self._alerting.evaluate(
                        unit=unit, cycle=cycle, lower=interval.lower, point=interval.point
                    )
                )
                open_alert = self._alerting.alerts().get(unit)
                self._fleet.set_alert(
                    unit, open_alert.to_dict() if open_alert is not None else None
                )
                if detector is not None:
                    self._fleet.set_degraded(unit, [state.to_dict() for state in detector.active()])
                replay.n_windows += 1

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
