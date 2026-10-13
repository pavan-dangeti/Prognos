"""The four component applications.

Each component answers the shared endpoints plus its own. The split exists because the
components fail differently, and a shared handler would either blur that or grow a
dispatch table that reads like a switch statement pretending to be polymorphism.

The one that matters most is `inference`'s readiness. A service serving point estimates
with no interval answers 200, looks healthy in an orchestrator, and reports nothing
wrong — while every maintenance decision drawn from it rests on a range that does not
exist. So it stays unready until a **calibrated** model is loaded, and `load` already
refuses a payload missing either half.

Error responses never carry a traceback. A stack trace in a response body leaks absolute
paths, library versions and internals to any caller that can reach the port.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .alerting import AlertingService
from .coverage import CoverageMonitor
from .events import MAX_BATCH_ROWS, Reading, StreamStats, TelemetryBatch, validate_batch
from .faults import DEGRADED_STATES, InjectedFault, channel_of
from .inference import RulPredictor
from .observability import (
    MetricsRegistry,
    record_coverage_metrics,
    record_stream_metrics,
)
from .service import ServiceState

# The four services that run as their own process.
#
# `docs/design.md` §4.6 lists six components, `features` among them. It is deliberately
# not a separate service here: the online window is a few hundred microseconds of
# arithmetic per prediction, and putting it behind HTTP would add a network hop and a
# new failure mode to every prediction to gain no isolation. It runs inside the
# inference process, which is where the windows are consumed.
COMPONENTS = ("ingestion", "inference", "alerting", "dashboard")


class HttpError(Exception):
    """An error with a status code, carried to the handler rather than raised through it."""

    def __init__(self, status: int, message: str) -> None:
        """Carry the status this error should surface as."""
        super().__init__(message)
        self.status = status
        self.message = message


def _parse_batch(payload: dict[str, Any]) -> TelemetryBatch:
    """Build a batch from a request body, refusing anything malformed.

    Raises `HttpError` rather than letting a `TypeError` escape, because a 500 with a
    stack trace hands an unauthenticated caller the layout of the container.
    """
    if not isinstance(payload, dict):
        raise HttpError(400, "body must be a JSON object")
    raw_rows = payload.get("rows")
    if raw_rows is not None and not isinstance(raw_rows, list):
        raise HttpError(400, "rows must be a list")
    rows: list[Reading] = []
    for entry in raw_rows or []:
        if not isinstance(entry, dict):
            raise HttpError(400, "each row must be an object")
        try:
            value = entry.get("value")
            rows.append(
                Reading(
                    unit=int(entry.get("unit") or payload.get("unit") or 0),
                    cycle=int(entry["cycle"]),
                    channel=int(entry["channel"]),
                    value=float("nan") if value is None else float(value),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HttpError(400, f"malformed row: {error}") from None
    try:
        unit = int(payload.get("unit", 0))
    except (TypeError, ValueError):
        raise HttpError(400, "unit must be an integer") from None
    return TelemetryBatch(source=str(payload.get("source", "unknown")), unit=unit, rows=rows)


def build_ingestion_state(
    metrics: MetricsRegistry | None = None, logger: logging.Logger | None = None
) -> ServiceState:
    """The ingestion service: validate, count, expose."""
    # Resolved once, here, and closed over. Resolving inside the handler created a fresh
    # throwaway registry per request, so /metrics served only the liveness series and
    # every ingestion counter silently went nowhere. A counter that is written to a
    # registry nobody scrapes is indistinguishable from a counter that never fired.
    registry = metrics or MetricsRegistry()
    stats = StreamStats()

    def ingest(payload: dict[str, Any]) -> dict[str, Any]:
        batch = _parse_batch(payload)
        if len(batch.rows) > MAX_BATCH_ROWS:
            raise HttpError(413, f"batch exceeds the {MAX_BATCH_ROWS} row limit")
        result = validate_batch(batch)
        stats.record(batch, result)
        record_stream_metrics(registry, stats)
        return {
            "accepted": result.accepted,
            "rejected": result.rejected,
            "reject_reasons": dict(stats.reject_reasons),
            "unit": batch.unit,
        }

    return ServiceState(
        name="ingestion",
        metrics=registry,
        ready=lambda: (True, {"accepted": stats.accepted, "rejected": stats.rejected}),
        routes={"POST /telemetry": ingest},
        logger=logger,
    )


def build_inference_state(
    model_dir: Path | None = None,
    metrics: MetricsRegistry | None = None,
    logger: logging.Logger | None = None,
) -> ServiceState:
    """The inference service: RUL with a calibrated interval, or not at all.

    `model_dir` points at a directory holding `model.pkl` and `calibrator.json`. Both
    or neither: a model without its calibrator serves bare point estimates, which look
    like answers and carry no uncertainty.
    """
    registry = metrics or MetricsRegistry()
    predictor: RulPredictor | None = None
    if model_dir is not None:
        try:
            predictor = RulPredictor.load(model_dir)
        except (FileNotFoundError, ValueError) as error:
            # Logged, not raised: a service that cannot start is worse than one that
            # starts unready and says why on /readyz.
            (logger or logging.getLogger("prognos.inference")).warning(
                "model not loaded", extra={"extra_fields": {"error": str(error)}}
            )

    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        if predictor is None:
            raise HttpError(503, "no calibrated model is loaded")
        import numpy as np

        cycles = payload.get("cycles") or []
        if not isinstance(cycles, list) or not cycles:
            raise HttpError(400, "cycles must be a non-empty list")
        windows = np.asarray(payload.get("windows"), dtype=np.float32)
        if windows.ndim != 3:
            raise HttpError(400, "windows must be a (n, length, channels) array")
        intervals = predictor.predict(windows, np.asarray(cycles, dtype=np.int64))
        return {
            "predictions": [
                {"lower": i.lower, "rul": i.point, "upper": i.upper} for i in intervals
            ],
            "model": predictor.model_name,
            "target_coverage": predictor.target_coverage,
        }

    return ServiceState(
        name="inference",
        metrics=registry,
        ready=lambda: (
            predictor is not None,
            {
                "model": predictor.model_name if predictor else None,
                "half_width": predictor.calibration_half_width if predictor else None,
            },
        ),
        routes={"POST /predict": predict},
        logger=logger,
    )


def build_alerting_state(
    threshold: float = 20.0,
    hysteresis: float = 5.0,
    metrics: MetricsRegistry | None = None,
    logger: logging.Logger | None = None,
) -> ServiceState:
    """The alerting service: raise and resolve maintenance alerts."""
    registry = metrics or MetricsRegistry()
    alerting = AlertingService(threshold=threshold, hysteresis=hysteresis)

    def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            emitted = alerting.evaluate(
                unit=int(payload["unit"]),
                cycle=int(payload["cycle"]),
                lower=float(payload["lower"]),
                point=float(payload["point"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HttpError(400, f"malformed alert evaluation: {error}") from None
        registry.gauge("alerts_open", "Open maintenance alerts", value=alerting.open_alerts())
        return {
            "emitted": [alert.to_dict() for alert in emitted],
            "open": {str(u): a.to_dict() for u, a in alerting.alerts().items()},
        }

    return ServiceState(
        name="alerting",
        metrics=registry,
        ready=lambda: (True, {"open_alerts": alerting.open_alerts()}),
        routes={"POST /alerts": evaluate},
        list_route={
            "alerts": lambda: {"open": {str(u): a.to_dict() for u, a in alerting.alerts().items()}}
        },
        logger=logger,
    )


def build_dashboard_state(
    monitor: CoverageMonitor | None = None,
    metrics: MetricsRegistry | None = None,
    logger: logging.Logger | None = None,
) -> ServiceState:
    """The dashboard service: the fleet summary an operator reads.

    Also the fault-injection control surface the design document asks for. Injection is
    exposed as a control rather than left in a test fixture, because the only way to
    check that a degraded state is observable end to end is to provoke one against a
    running system.
    """
    registry = metrics or MetricsRegistry()
    coverage = monitor or CoverageMonitor()
    injectors: dict[int, InjectedFault] = {}

    def fault_control(payload: dict[str, Any]) -> dict[str, Any]:
        """Arm or disarm a named fault on an engine."""
        kind = payload.get("fault")
        if kind in (None, "none"):
            return {"armed": {}, "available": list(DEGRADED_STATES)}
        if kind not in DEGRADED_STATES:
            raise HttpError(400, f"unknown fault {kind!r}")
        try:
            fault = InjectedFault(
                kind=kind,
                channel=int(payload.get("channel", 0)),
                from_cycle=int(payload.get("from_cycle", 1)),
                magnitude=float(payload.get("magnitude", 40.0)),
            )
        except (TypeError, ValueError) as error:
            raise HttpError(400, f"malformed fault: {error}") from None
        injectors[int(payload.get("engine", 1))] = fault
        return {"armed": {str(k): v.kind for k, v in injectors.items()}}

    def degraded() -> dict[str, Any]:
        """Every degraded state this service knows about, by name."""
        return {"states": list(DEGRADED_STATES), "instruments": [channel_of(c) for c in range(4)]}

    def coverage_payload() -> dict[str, Any]:
        """The live verdict, published as a GET so it carries no side effect."""
        record_coverage_metrics(registry, coverage.verdict())
        return coverage.verdict().to_dict()

    return ServiceState(
        name="dashboard",
        metrics=registry,
        ready=lambda: (True, {"coverage": coverage.verdict().status}),
        list_route={"coverage": coverage_payload, "degraded": degraded},
        routes={"POST /inject": fault_control},
        logger=logger,
    )


def build_component(name: str, **kwargs: Any) -> ServiceState:  # noqa: ANN401 - per-component options
    """Build a component by name, for the CLI and for compose."""
    if name == "ingestion":
        return build_ingestion_state(**kwargs)
    if name == "inference":
        return build_inference_state(**kwargs)
    if name == "alerting":
        return build_alerting_state(**kwargs)
    if name == "dashboard":
        return build_dashboard_state(**kwargs)
    raise ValueError(f"unknown component {name!r}; expected one of {', '.join(COMPONENTS)}")


def default_port(name: str) -> int:
    """The conventional port for a component, so compose and local runs agree."""
    return {"ingestion": 8081, "inference": 8082, "alerting": 8083, "dashboard": 8084}[name]


__all__ = [
    "COMPONENTS",
    "HttpError",
    "build_alerting_state",
    "build_component",
    "build_dashboard_state",
    "build_inference_state",
    "build_ingestion_state",
    "default_port",
]
