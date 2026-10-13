"""Authentication and rate limiting for the HTTP surface.

Two gaps, both flagged when the compose stack first came up: there was no
authentication and no rate limit, so the only abuse control was a request size ceiling.
Neither is exotic and both are small; they are here because the alternative is a
service that should not be on a network.

**A bearer token from the environment.** Never a build argument, never a file in the
image, never a constant in the source. Compared with `hmac.compare_digest`, because `==`
on secrets leaks through timing whether a prefix matched.

**Off when unset.** A local single-operator run should not need a credential, and a
service that refuses to start without one cannot be run locally at all. That is a real
risk and it is why the compose file is loopback-only by default; it is recorded in
`docs/runbook.md`, not just here.

**A sliding-window per-client limit.** Fixed windows are trivially evaded at the
boundary. Clients are tracked with a cap, because an endpoint that accepts a
client-supplied identity is a memory leak an attacker can aim at. No identity falls back
to a shared bucket rather than getting a free pass.

**`/healthz` is never authenticated.** A liveness probe has no credential to send, and
one that returned 401 would restart the process forever over a missing secret.
"""

from __future__ import annotations

import hmac
import os
import time
from collections import deque
from collections.abc import Mapping

__all__ = [
    "Authorizer",
    "RateLimiter",
    "check_request",
    "token_from_env",
]

#: Environment variable holding the bearer token.
TOKEN_ENV = "PROGNOS_API_TOKEN"


def token_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    """The configured token, or None when unset.

    An env var set to the empty string is treated as unset. That is the usual
    misconfiguration — a compose file interpolating a variable nobody set — and
    treating it as a real token would lock every caller out of their own service.
    """
    source = os.environ if environ is None else environ
    value = source.get(TOKEN_ENV, "")
    return value or None


class Authorizer:
    """Bearer-token check, off when no token is configured."""

    __slots__ = ("_token",)

    def __init__(self, token: str | None = None) -> None:
        """Configure the expected token, or None to require nothing."""
        self._token = token or None

    @property
    def enabled(self) -> bool:
        """Whether a token is required."""
        return self._token is not None

    def permits(self, header: str | None) -> bool:
        """Whether this `Authorization` header is acceptable."""
        if self._token is None:
            return True
        if not header:
            return False
        scheme, separator, presented = header.partition(" ")
        if not separator or scheme.lower() != "bearer":
            return False
        # Compare every candidate rather than returning early, so a wrong length does
        # not fail faster than a right one.
        return hmac.compare_digest(presented.strip(), self._token)


class RateLimiter:
    """Sliding-window request limit, per client."""

    __slots__ = ("_max_clients", "_window", "clients", "limit")

    def __init__(
        self, limit: int = 120, window_seconds: float = 60.0, max_clients: int = 4096
    ) -> None:
        """Build a limiter. `max_clients` bounds the memory a caller identity can cost."""
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if max_clients < 1:
            raise ValueError(f"max_clients must be at least 1, got {max_clients}")
        self.limit = limit
        self._window = window_seconds
        self._max_clients = max_clients
        self.clients: dict[str, deque[float]] = {}

    def allow(self, client: str | None = None, now: float | None = None) -> bool:
        """Whether this caller may proceed, recording the request when it may."""
        stamp = time.monotonic() if now is None else now
        key = client or "-"
        if key not in self.clients and len(self.clients) >= self._max_clients:
            self._evict(stamp)
        bucket = self.clients.setdefault(key, deque())
        cutoff = stamp - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(stamp)
        return True

    def remaining(self, client: str | None = None, now: float | None = None) -> int:
        """Requests left in this caller's window, for a `Retry-After` hint."""
        stamp = time.monotonic() if now is None else now
        bucket = self.clients.get(client or "-", deque())
        cutoff = stamp - self._window
        return max(0, self.limit - sum(1 for t in bucket if t > cutoff))

    def _evict(self, stamp: float) -> None:
        """Drop the least recently active clients once the table is full.

        Simplest bound that works: forget the oldest bucket. An endpoint that accepts a
        caller-supplied identity can otherwise be made to allocate without bound, which
        is the same denial-of-service the limiter exists to prevent.
        """
        stale = [
            k
            for k, bucket in self.clients.items()
            if not bucket or bucket[-1] <= stamp - self._window
        ]
        for key in stale:
            del self.clients[key]
        while len(self.clients) >= self._max_clients and self.clients:
            oldest = min(
                self.clients, key=lambda k: self.clients[k][-1] if self.clients[k] else 0.0
            )
            del self.clients[oldest]


def check_request(
    header: str | None,
    authorizer: Authorizer,
    limiter: RateLimiter,
    client: str | None,
    now: float | None = None,
) -> tuple[bool, int, dict[str, object]]:
    """Authorise and rate-limit one request.

    Returns `(allowed, status, body)`. Both refusal bodies are deliberately
    uninformative: a 401 that names the header it expects is a specification for
    whoever is guessing, and a 429 that says nothing is still a correct signal.
    """
    if not authorizer.permits(header):
        return False, 401, {"error": "unauthorized"}
    if not limiter.allow(client, now=now):
        return False, 429, {"error": "rate limit exceeded"}
    return True, 200, {}
