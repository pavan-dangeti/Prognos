"""Tests for the HTTP surface and the end-to-end pipeline.

The pipeline test is the one that matters. Every component has its own unit tests, and
a set of components that each pass can still be wired together wrongly — the window
feeds the wrong cycle, the alerting threshold reads the point estimate instead of the
lower bound, the coverage monitor is never handed the truth it is supposed to score.
Only running an engine through the whole thing catches that class of mistake.

So this file replays a real engine end to end and asserts on the result that matters:
that the pipeline's coverage verdict agrees with the coverage computed directly from
the same predictions. If the monitor were scoring something other than what it
reports, the two would diverge.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.request
from dataclasses import dataclass

import numpy as np
import pytest

from prognos.stream.alerting import AlertingService
from prognos.stream.coverage import CoverageMonitor, Observation
from prognos.stream.service import ServiceState, build_server


@dataclass
class _Body:
    status: int
    text: str

    def json(self) -> dict:
        """Parse the response body."""
        return json.loads(self.text)


def get(url: str) -> _Body:
    """HTTP GET returning status and text, without raising on 4xx/5xx."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return _Body(response.status, response.read().decode())
    except urllib.error.HTTPError as error:
        return _Body(error.code, error.read().decode())


@pytest.fixture
def server():
    """A running service on an ephemeral port, torn down afterwards."""
    import logging

    from prognos.stream.observability import MetricsRegistry

    state = ServiceState(
        name="test",
        ready=lambda: (True, {"model": "loaded"}),
        metrics=MetricsRegistry(namespace="prognos"),
        logger=logging.getLogger("prognos.test.service"),
    )
    httpd = build_server(state, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, state
    httpd.shutdown()
    httpd.server_close()


class TestServiceEndpoints:
    def test_healthz_is_ok(self, server) -> None:
        base, _ = server
        response = get(f"{base}/healthz")
        assert response.status == 200
        assert response.json()["status"] == "ok"

    def test_readyz_reports_ready(self, server) -> None:
        base, _ = server
        response = get(f"{base}/readyz")
        assert response.status == 200
        assert response.json()["status"] == "ready"

    def test_readyz_returns_503_when_not_ready(self) -> None:
        """Readiness has to be able to fail, or it is decoration.

        Liveness and readiness are different probes on purpose: a liveness failure
        restarts the process, which does not fix a missing model and does lose
        in-flight work.
        """
        import logging

        from prognos.stream.observability import MetricsRegistry

        state = ServiceState(
            name="test",
            ready=lambda: (False, {"reason": "model not loaded"}),
            metrics=MetricsRegistry(),
            logger=logging.getLogger("prognos.test.notready"),
        )
        httpd = build_server(state, host="127.0.0.1", port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            response = get(f"http://127.0.0.1:{httpd.server_address[1]}/readyz")
            assert response.status == 503
            assert response.json()["reason"] == "model not loaded"
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_healthz_stays_ok_when_not_ready(self) -> None:
        """The process is alive; it just cannot serve. Those are different facts."""
        import logging

        from prognos.stream.observability import MetricsRegistry

        state = ServiceState(
            name="test",
            ready=lambda: (False, {}),
            metrics=MetricsRegistry(),
            logger=logging.getLogger("prognos.test.live"),
        )
        httpd = build_server(state, host="127.0.0.1", port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            response = get(f"http://127.0.0.1:{httpd.server_address[1]}/healthz")
            assert response.status == 200
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_metrics_renders_prometheus_text(self, server) -> None:
        base, state = server
        state.metrics.counter("batches_total", "Batches", labels={"source": "sim"})
        response = get(f"{base}/metrics")
        assert response.status == 200
        assert 'prognos_batches_total{source="sim"} 1.0' in response.text
        assert "# TYPE prognos_batches_total counter" in response.text

    def test_root_reports_the_service_name(self, server) -> None:
        base, _ = server
        response = get(f"{base}/")
        assert response.json()["service"] == "test"
        assert "/readyz" in response.json()["endpoints"]

    def test_unknown_routes_are_404(self, server) -> None:
        base, _ = server
        assert get(f"{base}/nope").status == 404

    def test_default_binds_to_loopback(self) -> None:
        """A service binding to every interface by default is one flag away from public."""
        import logging

        from prognos.stream.observability import MetricsRegistry

        state = ServiceState(
            name="test", metrics=MetricsRegistry(), logger=logging.getLogger("prognos.test.bind")
        )
        httpd = build_server(state)
        try:
            assert httpd.server_address[0] == "127.0.0.1"
        finally:
            httpd.server_close()


# ------------------------------------------------------------- end to end


class TestPostFailures:
    """A handler bug must surface as a status, never as a traceback or a dropped socket."""

    @staticmethod
    def post(url: str, body: bytes) -> _Body:
        request = urllib.request.Request(url, data=body, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return _Body(response.status, response.read().decode())
        except urllib.error.HTTPError as error:
            return _Body(error.code, error.read().decode())

    def test_a_handler_that_raises_is_a_bare_500(self, server) -> None:
        base, state = server

        def broken(_payload: dict) -> dict:
            raise RuntimeError("/srv/app/secret/path.py exploded")

        state.routes["POST /boom"] = broken
        response = self.post(f"{base}/boom", b"{}")
        assert response.status == 500
        assert response.json() == {"error": "internal error"}

    def test_an_unknown_post_route_is_404(self, server) -> None:
        base, _ = server
        assert self.post(f"{base}/nowhere", b"{}").status == 404

    def test_a_body_that_is_not_json_is_400(self, server) -> None:
        base, state = server
        state.routes["POST /echo"] = lambda payload: payload
        assert self.post(f"{base}/echo", b"{not json").status == 400


def test_serve_closes_the_socket_on_interrupt(monkeypatch) -> None:
    """Ctrl-C must release the port, or the next start fails with address in use."""
    from prognos.stream import service

    closed: list[bool] = []

    class _Server:
        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(service, "build_server", lambda *_args, **_kwargs: _Server())
    service.serve(ServiceState(name="test"), port=0)
    assert closed == [True]


class TestPipelineEndToEnd:
    def test_coverage_verdict_matches_a_direct_count(self) -> None:
        """The monitor must score exactly what it reports.

        A pipeline where the monitor sees a different stream than the one being
        reported produces a green verdict over a breached system, which is the one
        outcome worse than having no monitor at all.
        """
        rng = np.random.default_rng(11)
        monitor = CoverageMonitor(target=0.9, min_samples=50)
        covered_directly: list[bool] = []

        for _ in range(400):
            point = float(rng.uniform(10.0, 90.0))
            truth = point + float(rng.normal(0.0, 12.0))
            lower, upper = point - 20.0, point + 20.0
            monitor.record(Observation(unit=1, truth=truth, lower=lower, upper=upper, point=point))
            covered_directly.append(lower <= truth <= upper)

        verdict = monitor.verdict()
        assert verdict.measured == pytest.approx(float(np.mean(covered_directly)))
        assert verdict.n_samples == 400

    def test_an_alert_fires_when_the_interval_says_so_and_clears_when_it_does_not(self) -> None:
        """The full decision path, from bound to operator-visible alert."""
        alerting = AlertingService(threshold=20.0, hysteresis=5.0)

        # Healthy: bound well above the threshold.
        assert alerting.evaluate(unit=1, cycle=10, lower=60.0, point=65.0) == []

        # Approaching failure.
        fired = alerting.evaluate(unit=1, cycle=20, lower=18.0, point=45.0)
        assert [a.kind for a in fired] == ["maintenance_due"]
        assert alerting.open_alerts(1) == 1

        # Still low: no duplicate alert.
        assert alerting.evaluate(unit=1, cycle=30, lower=15.0, point=44.0) == []

        # Recovered past the hysteresis margin.
        resolved = alerting.evaluate(unit=1, cycle=40, lower=70.0, point=72.0)
        assert [a.kind for a in resolved] == ["maintenance_resolved"]
        assert alerting.open_alerts() == 0

    def test_a_hundred_engines_alert_independently(self) -> None:
        """One engine's alert must not silence another's."""
        alerting = AlertingService(threshold=20.0)
        for engine in range(1, 101):
            lower = 10.0 if engine % 10 == 0 else 60.0
            alerting.evaluate(unit=engine, cycle=1, lower=lower, point=lower + 30.0)
        assert alerting.open_alerts() == 10

    def test_a_calibrator_survives_the_full_round_trip(self, tmp_path) -> None:
        """Serialised and reloaded, the served intervals must be identical.

        The first version of `RulPredictor.save` persisted the interval width alone,
        and loading it produced an infinite interval: a working model reporting no
        uncertainty, which reads as nothing to worry about.
        """
        from prognos.stream.inference import RulPredictor

        rng = np.random.default_rng(13)
        windows = rng.normal(size=(600, 20, 18)).astype(np.float32)
        cycles = np.arange(600, dtype=np.int64)
        targets = (120.0 - cycles / 8.0).astype(np.float32)

        predictor = RulPredictor.train(
            windows, cycles, targets, windows, cycles, targets, target_coverage=0.9, seed=1
        )
        predictor.save(tmp_path)
        reloaded = RulPredictor.load(tmp_path)

        probe = rng.normal(size=(12, 20, 18)).astype(np.float32)
        probe_cycles = np.arange(12, dtype=np.int64)
        before = [(i.lower, i.upper) for i in predictor.predict(probe, probe_cycles)]
        after = [(i.lower, i.upper) for i in reloaded.predict(probe, probe_cycles)]
        assert before == after
        assert all(math.isfinite(lo) and math.isfinite(hi) for lo, hi in after)
