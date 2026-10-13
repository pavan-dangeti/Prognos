"""Structured logging, metrics and traces.

Three things an operator needs from a pipeline this size, and none of them are
optional once there are six components talking to each other:

- **Structured logs with a request id.** Free-text logs cannot be aggregated, and a
  prediction that cannot be traced back to the batch that produced it cannot be
  debugged. One JSON object per line, always carrying the request id.
- **Metrics in the Prometheus exposition format.** The format is a contract with the
  scraper, not a formatting preference, so `render` is tested against real output
  rather than eyeballed.
- **Spans that record failures.** A trace that swallows an exception is worse than no
  trace, because it reports the slow path as healthy.

Counters reject negative increments and reject inconsistent label sets, because a
counter that silently resets or a metric whose labels change between scrapes makes
Prometheus produce graphs that are wrong rather than missing.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

from prognos.stream.coverage import CoverageVerdict
from prognos.stream.events import StreamStats
from prognos.stream.faults import DegradedState

_REQUEST_ID: ContextVar[str] = ContextVar("prognos_request_id", default="")


class JsonLogFormatter(logging.Formatter):
    """One JSON object per log line, with the request id always present."""

    def __init__(self, service: str) -> None:
        """Format records as JSON tagged with the service that emitted them."""
        self.service = service
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as a single JSON line.

        `extra` fields are merged in at the top level. Anything that is not JSON
        serialisable is stringified rather than dropped: a log line that silently
        loses its payload is how a debugging session goes nowhere.
        """
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": (
                getattr(record, "request_id", None) or _REQUEST_ID.get() or uuid.uuid4().hex
            ),
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = _jsonable(value)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _jsonable(value: object) -> object:
    """Coerce a log field into something `json.dumps` will accept."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class RequestContext:
    """A request id for the duration of a block.

    A context variable rather than a thread local, so it survives the async boundaries
    the HTTP layer will have, and so a nested block restores its parent on exit rather
    than leaving the wrong id on every later log line. Not a frozen dataclass because
    the context token has to be stored somewhere per instance.
    """

    __slots__ = ("_token", "request_id")

    def __init__(self, request_id: str = "") -> None:
        """Use the supplied id, or mint one."""
        self.request_id = request_id or uuid.uuid4().hex
        self._token: object = None

    def __enter__(self) -> RequestContext:
        """Install this id for the block."""
        self._token = _REQUEST_ID.set(self.request_id)
        return self

    def __exit__(self, *_exc: object) -> None:
        """Restore the previous id."""
        _REQUEST_ID.reset(self._token)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Readable in a traceback."""
        return f"RequestContext(request_id={self.request_id!r})"


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,  # noqa: ANN401
) -> None:
    """Emit one structured event.

    The event name is a field rather than the message, so `event="coverage.breach"` is
    greppable even when the surrounding message text changes.
    """
    logger.log(level, event, extra={"extra_fields": {"event": event, **fields}})


class MetricsRegistry:
    """Counters and gauges in the Prometheus text exposition format."""

    __slots__ = ("_gauges", "_help", "_namespace", "_type", "_values")

    def __init__(self, namespace: str = "prognos") -> None:
        """Build an empty registry under a metric namespace."""
        self._namespace = namespace
        self._values: dict[str, float] = {}
        self._type: dict[str, str] = {}
        self._help: dict[str, str] = {}

    def counter(
        self,
        name: str,
        documentation: str,
        labels: dict[str, str] | None = None,
        increment: float = 1.0,
    ) -> None:
        """Add to a counter, creating it on first use."""
        if increment < 0:
            raise ValueError(f"a counter increment must be non-negative, got {increment}")
        key = self._key(name, labels)
        existing = self._type.get(name)
        if existing is not None and existing != "counter":
            raise ValueError(f"{name} is already a {existing}, cannot be a counter")
        self._type[name] = "counter"
        self._help.setdefault(name, documentation)
        self._values[key] = self._values.get(key, 0.0) + increment

    def gauge(
        self, name: str, documentation: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge to an absolute value."""
        key = self._key(name, labels)
        existing = self._type.get(name)
        if existing is not None and existing != "gauge":
            raise ValueError(f"{name} is already a {existing}, cannot be a gauge")
        self._type[name] = "gauge"
        self._help.setdefault(name, documentation)
        self._values[key] = float(value)

    def value(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Current value of a series, for tests and the health endpoint."""
        return self._values.get(self._key(name, labels), 0.0)

    def _key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}}"

    def render(self) -> str:
        """The full exposition text, which is what the scrape endpoint returns."""
        lines: list[str] = []
        for name in sorted(self._type):
            lines.append(f"# HELP {self._namespace}_{name} {self._help.get(name, '')}")
            lines.append(f"# TYPE {self._namespace}_{name} {self._type[name]}")
            prefix = f"{self._namespace}_{name}"
            for key, value in sorted(self._values.items()):
                if key == name:
                    lines.append(f"{prefix} {value}")
                elif key.startswith(f"{name}{{"):
                    lines.append(f"{prefix}{key[len(name) :]} {value}")
        return "\n".join(lines) + "\n"


def record_stream_metrics(registry: MetricsRegistry, stats: StreamStats) -> None:
    """Fold ingestion counters into the registry.

    Rejects are tracked per reason class as well as in total, because "we dropped 400
    rows" is not a diagnosis and "400 rows for channel-out-of-range" is.
    """
    registry.gauge("stream_batches_total", "Batches received", value=stats.batches)
    registry.gauge("stream_accepted_total", "Rows accepted", value=stats.accepted)
    registry.gauge("stream_rejected_total", "Rows rejected", value=stats.rejected)
    for reason, count in stats.reject_reasons.items():
        registry.gauge(
            "stream_rejected_by_reason_total",
            "Rows rejected, by reason class",
            value=count,
            labels={"reason": reason},
        )


def record_coverage_metrics(registry: MetricsRegistry, verdict: CoverageVerdict) -> None:
    """Expose the S5 verdict, so a breach shows on a scrape rather than in a log."""
    measured = verdict.measured if verdict.measured is not None else 0.0
    registry.gauge("interval_coverage", "Live interval coverage", value=measured)
    registry.gauge(
        "interval_coverage_samples", "Scored intervals in the window", value=verdict.n_samples
    )
    registry.gauge(
        "interval_coverage_ok",
        "1 when coverage is inside the tolerance band, 0 on breach",
        value=0.0 if verdict.is_breach else 1.0,
    )


def record_fault_metrics(registry: MetricsRegistry, states: list[DegradedState]) -> None:
    """Expose the open degraded states, so S6 is satisfied by a scrape.

    Labelled by state name rather than counted into one series, because a single
    `degraded = 3` gives an operator nothing to act on. The panel is there to say
    *which* sensor on *which* engine.
    """
    counted: Counter[str] = Counter(s.state for s in states)
    registry.gauge("degraded_states", "Degraded states open, by name", value=len(states))
    for name, count in counted.items():
        registry.gauge(
            "degraded_states_by_kind",
            "Degraded states open, by name",
            value=count,
            labels={"state": name},
        )


@contextmanager
def span(registry: MetricsRegistry, name: str, **labels: str) -> Iterator[None]:
    """Time a block and record it, including when it fails.

    The failure is re-raised: a trace that swallowed the exception would report the
    broken path as healthy, which is the opposite of the point of having one.
    """
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        registry.counter("span_errors_total", "Span failures", labels={"span": name, **labels})
        raise
    finally:
        duration = time.perf_counter() - start
        registry.counter("span_calls_total", "Span invocations", labels={"span": name, **labels})
        registry.gauge(
            "span_duration_seconds",
            "Duration of the most recent span",
            value=duration,
            labels={"span": name, **labels},
        )
