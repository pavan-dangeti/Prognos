"""The HTTP surface every component exposes.

Six containers need the same four endpoints, and they differ only in what their
readiness check means. The shared part lives here rather than being copied six times,
because a health endpoint that one service answers differently is a health endpoint
nobody can trust.

- `GET /healthz` — liveness. Answers while the process is up, no dependencies.
- `GET /readyz` — readiness. Answers only when the service can actually serve, which
  for `inference` means a model is loaded and for `ingestion` means the stream is
  being consumed.
- `GET /metrics` — Prometheus exposition.
- `GET /` — a small status document.

`healthz` and `readyz` are deliberately different. A liveness probe that fails when a
dependency is down restarts the process, which does not fix a dependency and does lose
in-flight work; a readiness probe that fails only stops traffic arriving. Conflating
them is how a pipeline ends up crash-looping through an outage it could have ridden
out.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

from prognos.stream.observability import MetricsRegistry, RequestContext, log_event


class ReadinessCheck(Protocol):
    """Something a service can be asked about before it is sent traffic."""

    def __call__(self) -> tuple[bool, dict[str, object]]:
        """Return whether the service is ready, and the detail to report."""
        ...


@dataclass(slots=True)
class ServiceState:
    """The mutable state one service exposes over HTTP."""

    name: str
    version: str = "0.1.0"
    ready: ReadinessCheck | None = None
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    status: dict[str, object] = field(default_factory=dict)
    logger: logging.Logger | None = None

    def readiness(self) -> tuple[bool, dict[str, object]]:
        """Run the readiness check, defaulting to ready when none is registered."""
        if self.ready is None:
            return True, {}
        return self.ready()


class _Handler(BaseHTTPRequestHandler):
    """Request handler bound to one `ServiceState`."""

    state: ServiceState

    # BaseHTTPRequestHandler logs every request to stderr by default, which floods
    # the structured stream. The access log is redundant next to the JSON events.
    def log_message(self, format: str, *args: object) -> None:
        """Drop the default access log."""

    def _respond(self, status: int, body: dict[str, object]) -> None:
        """Send a JSON response."""
        encoded = json.dumps(body, default=str).encode()
        self._send(status, encoded, "application/json")

    def _send(self, status: int, encoded: bytes, content_type: str) -> None:
        """Send an already-encoded body with an accurate Content-Length.

        The metrics endpoint is not JSON, so it cannot go through `_respond`: a
        response whose advertised length excludes its real body gets truncated by the
        client, and a truncated scrape looks like a quiet period.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        """Route the health, readiness, metrics and status endpoints."""
        request_id = self.headers.get("X-Request-Id") or None
        with RequestContext(request_id or ""):
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if route == "/healthz":
                self._respond(200, {"status": "ok", "service": self.state.name})
            elif route == "/readyz":
                ready, detail = self.state.readiness()
                self._respond(
                    200 if ready else 503,
                    {
                        "status": "ready" if ready else "not_ready",
                        "service": self.state.name,
                        **detail,
                    },
                )
            elif route == "/metrics":
                self._send(
                    200,
                    self.state.metrics.render().encode(),
                    "text/plain; version=0.0.4",
                )
            elif route == "/":
                self._respond(
                    200,
                    {
                        "service": self.state.name,
                        "version": self.state.version,
                        "status": self.state.status,
                        "endpoints": ["/healthz", "/readyz", "/metrics"],
                    },
                )
            else:
                self._respond(404, {"error": "not found", "path": route})


def build_server(
    state: ServiceState, host: str = "127.0.0.1", port: int = 8080
) -> ThreadingHTTPServer:
    """Bind a service's HTTP surface.

    Binds to loopback by default. A container binding to every interface by default is
    a service that is one missing `--network` flag away from being public.
    """
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def serve(state: ServiceState, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the service until interrupted."""
    logger = state.logger or logging.getLogger("prognos.service")
    server = build_server(state, host=host, port=port)
    log_event(logger, "service.listening", service=state.name, host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_event(logger, "service.stopping", service=state.name)
    finally:
        server.server_close()
