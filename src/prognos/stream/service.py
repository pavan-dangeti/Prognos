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
from typing import Any, Protocol

from prognos.stream.auth import Authorizer, RateLimiter, check_request
from prognos.stream.observability import MetricsRegistry, RequestContext, log_event


class ReadinessCheck(Protocol):
    """Something a service can be asked about before it is sent traffic."""

    def __call__(self) -> tuple[bool, dict[str, Any]]:
        """Return whether the service is ready, and the detail to report."""
        ...


@dataclass(slots=True)
class ServiceState:
    """The mutable state one service exposes over HTTP.

    `routes` maps `"METHOD /path"` to a handler taking the parsed JSON body and
    returning a JSON-serialisable dict. `list_route` covers GET-only paths that return
    state, kept separate so a GET can never be mistaken for something with a side
    effect.
    """

    name: str
    version: str = "0.1.0"
    ready: ReadinessCheck | None = None
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    status: dict[str, object] = field(default_factory=dict)
    logger: logging.Logger | None = None
    routes: dict[str, Any] = field(default_factory=dict)
    list_route: dict[str, Any] = field(default_factory=dict)
    #: Bearer-token check and rate limit, both optional. Left unset by default so a
    #: local run needs no credential; the compose file keeps every published port on
    #: loopback for the same reason. See docs/runbook.md before exposing either.
    #: Static HTML pages by path, served without the bearer check because they carry
    #: no data: the page reads every number through the JSON routes, which keep it.
    pages: dict[str, str] = field(default_factory=dict)
    authorizer: Any = None
    rate_limiter: Any = None
    trust_forwarded_for: bool = False

    def __post_init__(self) -> None:
        """Publish a liveness series, so a scrape is never a blank page.

        An exporter with no series returns an empty body, which a scraper and a human
        both read as "nothing to report". Publishing `service_up` from the start makes
        "nothing has happened yet" distinguishable from "this process is dead".
        """
        self.metrics.gauge("service_up", "1 while the service is serving", value=1.0)

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        """Run the readiness check, defaulting to ready when none is registered."""
        if self.ready is None:
            return True, {}
        return self.ready()


#: Largest request body accepted. Sized well above a legitimate telemetry batch and
#: small enough that one caller cannot make the process allocate without limit.
MAX_REQUEST_BYTES = 64 * 1024 * 1024


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

    def _authorise(self, route: str) -> bool:
        """Check the token and the rate limit. Returns False once a reply is sent.

        `/healthz` is never authenticated: an orchestrator's liveness probe has no
        credential to send, and a probe returning 401 would restart the process forever
        over a missing secret.
        """
        if route == "/healthz" or route.lstrip("/") in self.state.pages:
            return True
        authorizer = self.state.authorizer
        limiter = self.state.rate_limiter
        if authorizer is None and limiter is None:
            return True
        allowed, status, body = check_request(
            self.headers.get("Authorization"),
            authorizer if authorizer is not None else Authorizer(),
            limiter if limiter is not None else RateLimiter(),
            self._client_identity(),
        )
        if allowed:
            return True
        self._respond(status, body)
        return False

    def _client_identity(self) -> str:
        """Identify the caller for rate limiting.

        The socket peer, not `X-Forwarded-For`, unless the deployment has explicitly
        declared itself behind a trusted proxy. Trusting that header from an arbitrary
        caller would let anyone mint a fresh allowance per request, which defeats the
        limiter entirely.
        """
        peer = self.client_address[0] if self.client_address else "?"
        if self.state.trust_forwarded_for:
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return peer

    def _send(
        self,
        status: int,
        encoded: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Send an already-encoded body with an accurate Content-Length.

        The metrics endpoint is not JSON, so it cannot go through `_respond`: a
        response whose advertised length excludes its real body gets truncated by the
        client, and a truncated scrape looks like a quiet period.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        """Route the health, readiness, metrics and status endpoints."""
        from prognos.stream.api import HttpError

        request_id = self.headers.get("X-Request-Id") or None
        with RequestContext(request_id or ""):
            try:
                self._get_route(request_id)
            except HttpError as error:
                self._respond(error.status, {"error": error.message})
            except Exception:  # deliberately broad, same as do_POST; see _read_body
                # Without this a route that raises closes the connection with no status
                # line, and the caller sees a network error rather than a server error.
                # Found by running the stack: `/cost` raised because the image carried no
                # `results/`, and curl reported HTTP 000.
                self.log_error("unhandled error in GET %s", self.path)
                self._respond(500, {"error": "internal error"})

    def _get_route(self, request_id: str | None) -> None:
        """Dispatch one GET.

        Split out of `do_GET` only so the failure guard wraps the whole dispatch. An
        `HttpError` raised by a route keeps the status its author intended rather than
        being flattened to a 500 by that guard.
        """
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if not self._authorise(route):
            return
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
        elif route.lstrip("/") in self.state.pages:
            from prognos.stream.ui import CONTENT_SECURITY_POLICY

            self._send(
                200,
                self.state.pages[route.lstrip("/")].encode(),
                "text/html; charset=utf-8",
                {
                    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
                    "X-Content-Type-Options": "nosniff",
                    "Referrer-Policy": "no-referrer",
                },
            )
        elif route.lstrip("/") in self.state.list_route:
            self._respond(200, self.state.list_route[route.lstrip("/")]())
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

    def do_POST(self) -> None:
        """Dispatch a POST to its component's handler, or refuse it.

        An `HttpError` carries the status its author intended. Anything else is a bug,
        and a bug must not become a 500 with a stack trace in the body: that hands any
        caller the container's file layout.
        """
        from prognos.stream.api import HttpError

        request_id = self.headers.get("X-Request-Id") or None
        with RequestContext(request_id or ""):
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if not self._authorise(route):
                return
            key = f"POST {route}"
            handler = self.state.routes.get(key)
            if handler is None:
                self._respond(404, {"error": "not found", "path": route})
                return
            try:
                payload = self._read_body()
                self._respond(200, handler(payload))
            except HttpError as error:
                self._respond(error.status, {"error": error.message})
            except Exception:  # deliberately broad: see the comment above
                self.log_error("unhandled error in %s", key)
                self._respond(500, {"error": "internal error"})

    def _read_body(self) -> dict[str, Any]:
        """Parse the request body as JSON, with a size ceiling."""
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_REQUEST_BYTES:
            from prognos.stream.api import HttpError

            raise HttpError(413, f"request body exceeds the {MAX_REQUEST_BYTES} byte limit")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            from prognos.stream.api import HttpError

            raise HttpError(400, "body must be valid JSON") from None
        return parsed if isinstance(parsed, dict) else {"payload": parsed}


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
