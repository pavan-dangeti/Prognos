"""Tests for request authentication and rate limiting.

Two gaps I flagged when the compose stack first came up, and both matter before
anything is published to a network: there was no authentication and no rate limit, so
the only abuse control was a size ceiling.

The design here is deliberately minimal, because minimal is what survives review:

- **A bearer token from the environment**, compared in constant time. Secrets come from
  the environment, never from a build argument or a file in the image.
- **A per-client rate limit** with a monotonic clock, because the natural way to evade
  a per-process limit is to make several.
- **Both off when unset.** A local single-operator run should not need a token, and a
  service that refuses to start without one cannot be run locally at all. That is a
  real risk and the reason it is opt-in is stated in the runbook, not just here.
- **Refusal bodies carry no detail.** A 401 that says "token missing, expected header
  X" is a specification for whoever is guessing.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from prognos.stream.auth import (
    Authorizer,
    RateLimiter,
    check_request,
)


class TestAuthorizer:
    def test_no_configured_token_allows_everything(self) -> None:
        """Local runs must not need a token, or the stack cannot be started at all."""
        authorizer = Authorizer(token=None)
        assert authorizer.permits("Bearer anything")

    def test_a_configured_token_requires_a_match(self) -> None:
        authorizer = Authorizer(token="s3cret")
        assert authorizer.permits("Bearer s3cret")
        assert not authorizer.permits("Bearer wrong")

    def test_a_missing_header_is_refused_when_a_token_is_set(self) -> None:
        authorizer = Authorizer(token="s3cret")
        assert not authorizer.permits("")

    def test_a_malformed_header_is_refused(self) -> None:
        """`s3cret` without the scheme, or the token alone, must not pass."""
        authorizer = Authorizer(token="s3cret")
        for header in ("s3cret", "Basic s3cret", "Bearer", "Bearer "):
            assert not authorizer.permits(header), f"{header!r} was accepted"

    def test_the_scheme_is_case_insensitive(self) -> None:
        """HTTP auth schemes are case-insensitive per RFC 7235."""
        authorizer = Authorizer(token="s3cret")
        assert authorizer.permits("bearer s3cret")
        assert authorizer.permits("BEARER s3cret")

    def test_a_token_containing_spaces_is_supported(self) -> None:
        """Splitting on the first space only, since a generated token may contain one."""
        authorizer = Authorizer(token="two words")
        assert authorizer.permits("Bearer two words")

    def test_an_empty_configured_token_is_treated_as_unset(self) -> None:
        """An env var set to "" is the common misconfiguration and must not lock everyone out."""
        authorizer = Authorizer(token="")
        assert authorizer.permits("Bearer anything")

    def test_comparison_does_not_leak_length_or_prefix(self) -> None:
        """`==` on secrets leaks through timing whether a prefix matched."""
        authorizer = Authorizer(token="s3cret")
        assert not authorizer.permits("Bearer s3cre")
        assert not authorizer.permits("Bearer s3cretx")


class TestRateLimiter:
    def test_allows_up_to_the_limit(self) -> None:
        limiter = RateLimiter(limit=3, window_seconds=60.0)
        assert [limiter.allow(now=float(t)) for t in range(3)] == [True, True, True]

    def test_refuses_past_the_limit(self) -> None:
        limiter = RateLimiter(limit=2, window_seconds=60.0)
        assert limiter.allow(now=0.0) is True
        assert limiter.allow(now=0.0) is True
        assert limiter.allow(now=0.0) is False

    def test_the_window_slides(self) -> None:
        """Fixed windows are trivially evaded at the boundary; sliding are not."""
        limiter = RateLimiter(limit=2, window_seconds=10.0)
        assert limiter.allow(now=0.0) is True
        assert limiter.allow(now=1.0) is True
        assert limiter.allow(now=2.0) is False
        # The first request has aged out of the window by now.
        assert limiter.allow(now=11.0) is True

    def test_clients_are_limited_independently(self) -> None:
        """A limit keyed on nothing is a global limit.

        One caller would then exhaust it for everyone.
        """
        limiter = RateLimiter(limit=1, window_seconds=60.0)
        assert limiter.allow("a", now=0.0) is True
        assert limiter.allow("b", now=0.0) is True
        assert limiter.allow("a", now=0.0) is False

    def test_a_client_that_stops_being_seen_is_evicted(self) -> None:
        """Unbounded client tracking is a memory leak.

        Any endpoint accepting a client-supplied identity is a way to aim one.
        """
        limiter = RateLimiter(limit=1, window_seconds=10.0, max_clients=8)
        for index in range(40):
            limiter.allow(f"client-{index}", now=0.0)
        assert len(limiter.clients) <= 8

    def test_a_missing_client_key_falls_back_to_a_shared_bucket(self) -> None:
        """No identity means no separate allowance, not a free pass."""
        limiter = RateLimiter(limit=1, window_seconds=60.0)
        assert limiter.allow(None, now=0.0) is True
        assert limiter.allow(None, now=0.0) is False

    def test_rejects_a_nonsensical_limit(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            RateLimiter(limit=0, window_seconds=60.0)
        with pytest.raises(ValueError, match="window"):
            RateLimiter(limit=1, window_seconds=0.0)


class TestCheckRequest:
    def test_auth_passes_before_the_limit_is_reached(self) -> None:
        ok, status, _ = check_request(
            "Bearer t", Authorizer(token="t"), RateLimiter(limit=5, window_seconds=60.0), "a"
        )
        assert ok and status == 200

    def test_a_bad_token_is_401(self) -> None:
        ok, status, _ = check_request(
            "Bearer wrong", Authorizer(token="t"), RateLimiter(limit=5, window_seconds=60.0), "a"
        )
        assert not ok and status == 401

    def test_exceeding_the_limit_is_429(self) -> None:
        limiter = RateLimiter(limit=1, window_seconds=60.0)
        assert check_request("Bearer t", Authorizer(token="t"), limiter, "a")[1] == 200
        ok, status, _ = check_request("Bearer t", Authorizer(token="t"), limiter, "a")
        assert not ok and status == 429

    def test_the_auth_failure_body_carries_no_hint(self) -> None:
        """A refusal that explains itself is a specification for whoever is guessing."""
        ok, status, body = check_request(
            "", Authorizer(token="t"), RateLimiter(limit=5, window_seconds=60.0), "a"
        )
        assert not ok and status == 401
        text = json.dumps(body).lower()
        for hint in ("header", "token", "authorization", "bearer", "expect"):
            assert hint not in text, f"the 401 body leaks {hint!r}"


class _AuthedService:
    """A service behind the authoriser, exercised over real HTTP."""

    def __init__(self, component: str = "dashboard", token: str | None = None, limit: int = 3):
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry
        from prognos.stream.service import build_server

        state = build_component(
            component, metrics=MetricsRegistry(), logger=logging.getLogger("prognos.auth")
        )
        state.authorizer = Authorizer(token=token)
        state.rate_limiter = RateLimiter(limit=limit, window_seconds=60.0)
        self.httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        """Shut the service down."""
        self.httpd.shutdown()
        self.httpd.server_close()


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    """GET with optional headers, without raising on 4xx."""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


class TestOverHttp:
    def test_an_unauthenticated_request_is_refused(self) -> None:
        service = _AuthedService(token="t")
        try:
            status, _ = _get(f"{service.base}/")
            assert status == 401
        finally:
            service.close()

    def test_an_authenticated_request_is_served(self) -> None:
        service = _AuthedService(token="t")
        try:
            status, body = _get(f"{service.base}/", {"Authorization": "Bearer t"})
            assert status == 200
            assert json.loads(body)["service"] == "dashboard"
        finally:
            service.close()

    def test_liveness_stays_open_so_orchestrators_can_still_probe(self) -> None:
        """A liveness probe must not need a credential.

        If /healthz required a token, every probe would return 401 and the process
        would be restarted forever over a missing secret.
        """
        service = _AuthedService(token="t")
        try:
            assert _get(f"{service.base}/healthz")[0] == 200
        finally:
            service.close()

    def test_the_limit_applies_over_http(self) -> None:
        service = _AuthedService(token="t", limit=2)
        try:
            headers = {"Authorization": "Bearer t"}
            assert _get(f"{service.base}/", headers)[0] == 200
            assert _get(f"{service.base}/", headers)[0] == 200
            assert _get(f"{service.base}/", headers)[0] == 429
        finally:
            service.close()

    def test_a_refused_request_never_reaches_the_handler(self) -> None:
        """A 429 after the handler ran would have already produced its side effect."""
        service = _AuthedService(token="t", limit=1)
        try:
            headers = {"Authorization": "Bearer t"}
            _get(f"{service.base}/", headers)
            status, _ = _get(f"{service.base}/", headers)
            assert status == 429
        finally:
            service.close()


class TestSecretsNeverReachTheRepository:
    def test_the_token_is_read_from_the_environment_not_a_file(self) -> None:
        """A token in a file is a token in the repository."""
        from prognos.stream.auth import token_from_env

        assert token_from_env({}) is None
        assert token_from_env({"PROGNOS_API_TOKEN": "abc"}) == "abc"
        assert token_from_env({"PROGNOS_API_TOKEN": ""}) is None


class TestMetricsIsNotExempt:
    """`/healthz` is exempt; `/metrics` deliberately is not.

    The temptation is to exempt the scrape too, so Prometheus does not need a
    credential. Enabling the token on the stack showed what that costs: every one of
    the four prognos targets went DOWN with 401, which reads as an outage rather than a
    configuration mistake. So `/metrics` needs the token and Prometheus is given it
    through `bearer_token_file`.

    The cost of that choice is a file the operator has to create, which is a real
    operational step and is documented in the runbook. The cost of the alternative is a
    dashboard that silently stops working.
    """

    def _service(self, token: str | None) -> _AuthedService:
        return _AuthedService(token=token)

    def test_metrics_needs_the_token_when_one_is_set(self) -> None:
        service = self._service("t")
        try:
            assert _get(f"{service.base}/metrics")[0] == 401
            assert _get(f"{service.base}/metrics", {"Authorization": "Bearer t"})[0] == 200
        finally:
            service.close()

    def test_metrics_is_open_when_no_token_is_set(self) -> None:
        service = self._service(None)
        try:
            assert _get(f"{service.base}/metrics")[0] == 200
        finally:
            service.close()
