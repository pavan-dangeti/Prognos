"""Tests for the per-component HTTP apps.

These are the tests that should have existed before the compose file claimed six
services were running. `prognos stream` was wired as the long-running command for all
three of them, and it is the replay CLI: it writes a result file and exits. Nothing
caught that, because every component's endpoints were tested in-process and the compose
file was only ever validated as YAML.

So these tests exercise each app over real HTTP, including the component-specific routes
that `/healthz` and `/metrics` do not reach: ingesting a batch, asking for a
prediction, reading the fleet summary.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from prognos.stream.api import build_component
from prognos.stream.faults import DEGRADED_STATES, FaultDetector
from prognos.stream.service import ServiceState, build_server
from prognos.uncertainty import Interval


def request(url: str, payload: dict[str, Any] | None = None) -> tuple[int, str]:
    """GET or POST to a local service, returning the status and raw body.

    Never raises on a 4xx or 5xx: the status *is* the thing under test in most of these
    cases, and raising would hide it behind an exception.
    """
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def post_raw(url: str, raw: bytes) -> tuple[int, str]:
    """POST a body that may not be JSON, returning the raw response."""
    req = urllib.request.Request(
        url, data=raw, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def get(url: str) -> tuple[int, dict[str, Any]]:
    """GET, parsed as JSON."""
    status, text = request(url)
    return status, json.loads(text)


def post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST JSON, parsed as JSON."""
    status, text = request(url, payload)
    return status, json.loads(text)


COMPONENTS = ("ingestion", "inference", "dashboard", "alerting")


def start(component: str) -> tuple[ServiceState, str]:
    """Build a component and start it on an ephemeral port."""
    import logging

    from prognos.stream.observability import MetricsRegistry

    state = build_component(
        component, metrics=MetricsRegistry(), logger=logging.getLogger(f"prognos.{component}")
    )
    httpd = build_server(state, host="127.0.0.1", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return state, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def service():
    """Start components on demand, closing them all at teardown."""
    started: list[Any] = []

    def factory(component: str) -> str:
        import logging

        from prognos.stream.observability import MetricsRegistry
        from prognos.stream.service import build_server

        state = build_component(
            component,
            metrics=MetricsRegistry(),
            logger=logging.getLogger(f"prognos.{component}"),
        )
        httpd = build_server(state, host="127.0.0.1", port=0)
        started.append(httpd)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        del logging
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield factory
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def a_telemetry_batch(n_cycles: int = 2, channels: int = 3) -> dict[str, Any]:
    """A well-formed telemetry payload."""
    return {
        "source": "test",
        "unit": 1,
        "rows": [
            {"unit": 1, "cycle": cycle, "channel": channel, "value": float(cycle + channel)}
            for cycle in range(1, n_cycles + 1)
            for channel in range(channels)
        ],
    }


class TestCommonEndpoints:
    @pytest.mark.parametrize("component", COMPONENTS)
    def test_every_component_answers_healthz(self, service, component: str) -> None:
        """A component without a health endpoint cannot be orchestrated."""
        base = service(component)
        status, body = get(f"{base}/healthz")
        assert status == 200
        assert body["service"] == component

    @pytest.mark.parametrize("component", COMPONENTS)
    def test_every_component_exposes_metrics(self, service, component: str) -> None:
        base = service(component)
        status, text = request(f"{base}/metrics")
        assert status == 200
        assert "prognos_" in text

    def test_an_unknown_component_is_rejected(self) -> None:
        """A typo in a compose command must fail loudly, not start a mystery service."""
        with pytest.raises(ValueError, match="unknown component"):
            build_component("ingetion")


class TestIngestion:
    def test_a_batch_is_accepted_and_counted(self, service) -> None:
        """The ingest path has to move rows, not merely answer 200."""
        status, body = post(f"{service('ingestion')}/telemetry", a_telemetry_batch())
        assert status == 200
        assert body["accepted"] == 6  # 2 cycles x 3 channels
        assert body["rejected"] == 0

    def test_bad_rows_are_counted_not_hidden(self, service) -> None:
        """A partial accept has to say what it dropped, or it reads as complete."""
        payload = a_telemetry_batch()
        payload["rows"].append({"unit": 1, "cycle": 1, "channel": 999, "value": 1.0})
        payload["rows"].append({"unit": 1, "cycle": 1, "channel": 0, "value": None})
        status, body = post(f"{service('ingestion')}/telemetry", payload)
        assert status == 200
        assert body["accepted"] == 6
        assert body["rejected"] == 2
        assert body["reject_reasons"]

    def test_an_oversized_batch_is_refused_not_truncated(self, service) -> None:
        """A truncated batch is byte-for-byte indistinguishable from a complete one."""
        payload = {
            "source": "test",
            "unit": 1,
            "rows": [{"unit": 1, "cycle": c, "channel": 0, "value": 1.0} for c in range(1, 30000)],
        }
        status, body = post(f"{service('ingestion')}/telemetry", payload)
        assert status == 413
        assert "limit" in str(body).lower()

    def test_malformed_json_is_a_400(self, service) -> None:
        """Bad JSON is the caller's problem and gets a 4xx, not a traceback."""
        base = service("ingestion")
        status, body = post_raw(f"{base}/telemetry", b"{not json")
        assert status == 400
        assert "Traceback" not in body

    def test_a_row_with_a_missing_field_is_a_400(self, service) -> None:
        """A row missing `cycle` cannot be placed on a timeline and must be refused."""
        status, _ = post(f"{service('ingestion')}/telemetry", {"rows": [{"channel": 0}]})
        assert status == 400

    def test_counters_appear_in_the_scrape(self, service) -> None:
        """Rejects are the signal that ingestion is broken; they must be scrapeable."""
        base = service("ingestion")
        post(f"{base}/telemetry", a_telemetry_batch())
        _, text = request(f"{base}/metrics")
        assert "prognos_stream_batches_total" in text
        assert "prognos_stream_accepted_total" in text

    def test_the_service_registry_is_the_one_that_gets_written(self, service) -> None:
        """Counters must land in the registry the scrape endpoint serves.

        This is invisible to a unit test of the handler and invisible in a JSON
        response: `ingest` resolved `metrics or MetricsRegistry()` *inside* the
        handler, so every request wrote to a fresh throwaway registry and `/metrics`
        served only the liveness series. The service answered 200, the response said
        `accepted: 1`, and the number went nowhere. A counter written to a registry
        nobody scrapes is indistinguishable from a counter that never fired.
        """
        base = service("ingestion")
        post(f"{base}/telemetry", a_telemetry_batch())
        _, text = request(f"{base}/metrics")
        assert "prognos_stream_batches_total 1.0" in text
        assert "prognos_stream_accepted_total 6.0" in text

    def test_the_alerting_registry_is_written_too(self, service) -> None:
        base = service("alerting")
        post(f"{base}/alerts", {"unit": 1, "cycle": 1, "lower": 1.0, "point": 30.0})
        _, text = request(f"{base}/metrics")
        assert "prognos_alerts_open 1.0" in text

    def test_metrics_are_never_a_blank_page(self, service) -> None:
        """An exporter with no series looks dead to both a scraper and a human."""
        _, text = request(f"{service('dashboard')}/metrics")
        assert "prognos_service_up 1.0" in text

    def test_readiness_needs_nothing_loaded(self, service) -> None:
        """Ingestion has no model to load, so it must be ready immediately."""
        assert get(f"{service('ingestion')}/readyz")[0] == 200


class TestInference:
    def test_inference_is_not_ready_without_a_model(self, service) -> None:
        """The dangerous readiness failure.

        A service serving point estimates with no interval answers 200, looks healthy,
        and reports nothing wrong. It has to refuse traffic until it has a calibrated
        model, which is why the model and the calibrator are loaded together.
        """
        status, body = get(f"{service('inference')}/readyz")
        assert status == 503
        assert body["status"] == "not_ready"

    def test_a_prediction_request_is_refused_while_unready(self, service) -> None:
        status, _ = post(f"{service('inference')}/predict", {"unit": 1, "cycles": [1]})
        assert status == 503

    def test_liveness_is_independent_of_readiness(self, service) -> None:
        """The process is alive; it just cannot serve. Those are different facts."""
        base = service("inference")
        assert get(f"{base}/healthz")[0] == 200
        assert get(f"{base}/readyz")[0] == 503

    def test_an_inference_service_with_a_model_is_ready(self, service, tmp_path) -> None:
        """The other half: once a calibrated model is loaded, it must serve."""
        import logging

        import numpy as np

        from prognos.stream.api import build_inference_state
        from prognos.stream.inference import RulPredictor

        rng = np.random.default_rng(0)
        windows = rng.normal(size=(200, 5, 18)).astype(np.float32)
        cycles = np.arange(200, dtype=np.int64)
        targets = (100.0 - cycles / 10.0).astype(np.float32)
        predictor = RulPredictor.train(
            windows, cycles, targets, windows, cycles, targets, target_coverage=0.9, seed=0
        )
        predictor.save(tmp_path)

        state = build_inference_state(tmp_path, logger=logging.getLogger("prognos.inference.test"))
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            assert get(f"{base}/readyz")[0] == 200
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestDashboard:
    def test_the_summary_reads_on_a_cold_start(self, service) -> None:
        """The dashboard must render an empty fleet rather than 500."""
        status, body = get(f"{service('dashboard')}/")
        assert status == 200
        assert body["service"] == "dashboard"

    def test_coverage_reads_as_undecided_before_any_data(self, service) -> None:
        """Reporting coverage of 0 on a cold start would read as a total breach."""
        status, body = get(f"{service('dashboard')}/coverage")
        assert status == 200
        assert body["status"] == "insufficient_data"


class TestAlerting:
    def test_an_alert_is_raised_then_resolved(self, service) -> None:
        """The operator path over HTTP, with the hysteresis that stops flapping."""
        base = service("alerting")
        _, fired = post(f"{base}/alerts", {"unit": 1, "cycle": 10, "lower": 5.0, "point": 40.0})
        assert fired["emitted"][0]["kind"] == "maintenance_due"

        _, repeat = post(f"{base}/alerts", {"unit": 1, "cycle": 20, "lower": 4.0, "point": 39.0})
        assert repeat["emitted"] == [], "an engine already alerting must not alert again"

        _, resolved = post(f"{base}/alerts", {"unit": 1, "cycle": 30, "lower": 90.0, "point": 95.0})
        assert resolved["emitted"][0]["kind"] == "maintenance_resolved"

    def test_open_alerts_are_listable(self, service) -> None:
        base = service("alerting")
        post(f"{base}/alerts", {"unit": 4, "cycle": 1, "lower": 1.0, "point": 30.0})
        status, body = get(f"{base}/alerts")
        assert status == 200
        assert "unit004" in body["open"]

    def test_open_alerts_name_the_sub_dataset(self, service) -> None:
        """Two engines sharing unit 4 are two alerts, listed under their full identity."""
        base = service("alerting")
        for source in ("DS01-005", "DS02-006"):
            post(
                f"{base}/alerts",
                {"unit": 4, "source": source, "cycle": 1, "lower": 1.0, "point": 30.0},
            )
        _, body = get(f"{base}/alerts")
        assert set(body["open"]) == {"DS01-005/unit004", "DS02-006/unit004"}


class TestSecurityAndLimits:
    def test_the_service_never_reflects_a_stack_trace(self, service) -> None:
        """A traceback in a response body leaks paths and internals to any caller."""
        base = service("ingestion")
        status, body = request(f"{base}/telemetry", {"source": "t", "unit": "not-an-int"})
        assert 400 <= status < 500
        assert "Traceback" not in body
        assert "prognos/" not in body

    def test_unknown_routes_are_404_not_500(self, service) -> None:
        assert get(f"{service('dashboard')}/nope")[0] == 404


def test_a_component_builds_without_overrides() -> None:
    """The bare constructor is what compose uses, so it has to work unaided."""
    assert build_component("dashboard").name == "dashboard"


def _unused_iterator_import() -> Iterator[None]:
    """Keeps the Iterator import meaningful for type checkers."""
    yield from ()


class TestInferencePaths:
    """The prediction path and the failure modes around it.

    Uncovered because the happy path needs a fitted model, which is slow. These use a
    small one and go after the edges.
    """

    def _with_model(self, tmp_path):
        """An inference service holding a real, calibrated model."""
        import numpy as np

        from prognos.stream.inference import RulPredictor

        rng = np.random.default_rng(0)
        windows = rng.normal(size=(200, 5, 18)).astype(np.float32)
        cycles = np.arange(200, dtype=np.int64)
        targets = (100.0 - cycles / 10.0).astype(np.float32)
        RulPredictor.train(
            windows, cycles, targets, windows, cycles, targets, target_coverage=0.9, seed=0
        ).save(tmp_path)
        return windows

    def test_a_prediction_carries_an_interval(self, tmp_path, service) -> None:
        """A served prediction without a range is the degradation `load` refuses."""
        from prognos.stream.api import build_inference_state
        from prognos.stream.service import build_server

        windows = self._with_model(tmp_path)
        state = build_inference_state(tmp_path)
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            status, body = post(
                f"{base}/predict",
                {"unit": 1, "cycles": [1, 2], "windows": windows[:2].tolist()},
            )
            assert status == 200
            assert len(body["predictions"]) == 2
            for prediction in body["predictions"]:
                assert prediction["lower"] <= prediction["rul"] <= prediction["upper"]
                assert prediction["upper"] > prediction["lower"]
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_windows_of_the_wrong_rank_are_refused(self, tmp_path, service) -> None:
        """A 2-D window array must not be silently reshaped into something modelable."""
        from prognos.stream.api import build_inference_state
        from prognos.stream.service import build_server

        self._with_model(tmp_path)
        state = build_inference_state(tmp_path)
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            status, _ = post(f"{base}/predict", {"unit": 1, "cycles": [1], "windows": [[1.0, 2.0]]})
            assert status == 400
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_empty_cycles_are_refused(self, tmp_path, service) -> None:
        from prognos.stream.api import build_inference_state
        from prognos.stream.service import build_server

        windows = self._with_model(tmp_path)
        state = build_inference_state(tmp_path)
        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            status, _ = post(
                f"{base}/predict", {"unit": 1, "cycles": [], "windows": windows[:1].tolist()}
            )
            assert status == 400
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_model_directory_without_a_calibrator_leaves_it_unready(self, tmp_path) -> None:
        """The half-loaded case: a model on disk, no uncertainty with it."""
        from prognos.stream.api import build_inference_state

        self._with_model(tmp_path)
        (tmp_path / "calibrator.json").unlink()
        state = build_inference_state(tmp_path)
        ready, detail = state.readiness()
        assert not ready
        assert detail["model"] is None


class TestServiceGuards:
    def test_a_body_over_the_size_limit_is_refused(self, service) -> None:
        """One caller must not be able to make the process allocate without limit.

        The guard reads `Content-Length` and refuses before allocating, so the test
        declares an enormous length without sending an enormous body: actually
        streaming 64 MB to check it gets rejected just tests the socket.
        """
        from prognos.stream.service import MAX_REQUEST_BYTES

        assert MAX_REQUEST_BYTES > 0
        base = service("ingestion")
        req = urllib.request.Request(
            f"{base}/telemetry",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(MAX_REQUEST_BYTES + 1),
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected the oversized body to be refused")
        except urllib.error.HTTPError as error:
            assert error.code == 413

    def test_a_non_json_content_type_still_gets_a_clean_error(self, service) -> None:
        base = service("ingestion")
        status, body = post_raw(f"{base}/telemetry", b"not json at all")
        assert status == 400
        assert "Traceback" not in body


class TestFaultsAreObservableThroughTheService:
    """S6: a degraded state must be observable in metrics, logs and the dashboard."""

    @staticmethod
    def _stream(n_cycles: int = 60, seed: int = 0) -> np.ndarray:
        """A healthy stream with the per-channel variance the stuck detector needs."""
        from prognos.stream.window import N_CHANNELS

        return (
            np.random.default_rng(seed)
            .normal(100.0, 5.0, size=(n_cycles, N_CHANNELS))
            .astype(np.float32)
        )

    """S6: a degraded state must be observable in metrics, logs and the dashboard.

    A detector that fires and is then dropped on the floor is the same as one that
    never existed, so these check the path all the way to a scrape and an HTTP
    response rather than the detector in isolation.
    """

    def test_the_dashboard_publishes_the_state_names(self, service) -> None:
        base = service("dashboard")
        status, body = get(f"{base}/degraded")
        assert status == 200
        assert set(body["states"]) == set(DEGRADED_STATES)

    def test_an_injected_fault_can_be_armed_over_http(self, service) -> None:
        """The injection control the design document asks the dashboard to carry."""
        base = service("dashboard")
        status, body = post(
            f"{base}/inject",
            {"engine": 2, "fault": "sensor_drift", "channel": 4, "from_cycle": 50},
        )
        assert status == 200
        assert body["armed"] == {"unit002": "sensor_drift"}

    def test_arming_nothing_clears_the_controls(self, service) -> None:
        base = service("dashboard")
        post(f"{base}/inject", {"engine": 2, "fault": "sensor_drift"})
        status, body = post(f"{base}/inject", {"engine": 2, "fault": "none"})
        assert status == 200
        assert body["armed"] == {}

    def test_an_armed_fault_is_handed_to_the_live_stream(self) -> None:
        """Without the hand-off the control recorded a fault nothing ever applied."""
        from prognos.stream.api import build_dashboard_state

        received: list[tuple[int, object, object]] = []
        state = build_dashboard_state(
            on_inject=lambda unit, fault, source: received.append((unit, fault, source))
        )
        handler = state.routes["POST /inject"]
        handler({"engine": 4, "source": "DS02-006", "fault": "sensor_stuck", "channel": 3})
        handler({"engine": 4, "source": "DS02-006", "fault": "none"})
        assert [(unit, source) for unit, _, source in received] == [(4, "DS02-006")] * 2
        assert getattr(received[0][1], "kind", None) == "sensor_stuck"
        assert received[1][1] is None

    def test_an_unknown_fault_is_a_400(self, service) -> None:
        """A typo in a dashboard control must fail, not silently inject nothing."""
        base = service("dashboard")
        status, _ = post(f"{base}/inject", {"engine": 1, "fault": "snesor_dropout"})
        assert status == 400

    def test_a_degraded_state_reaches_the_scrape(self, service) -> None:
        """The metrics endpoint is where a fault becomes visible without a human."""
        import numpy as np

        from prognos.stream.observability import MetricsRegistry, record_fault_metrics

        registry = MetricsRegistry()
        detector = FaultDetector(unit=1)
        rows = self._stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 30:
                values[7] = np.nan
            detector.observe(cycle, values)
        record_fault_metrics(registry, detector.active())

        rendered = registry.render()
        assert "prognos_degraded_states" in rendered
        assert 'state="sensor_dropout"' in rendered


class TestLogFieldCoercion:
    """Log fields arrive from numpy, from dataclasses and from call sites.

    A log line that silently drops its payload is how a debugging session goes
    nowhere, so every coercion path is exercised rather than assumed.
    """

    def _emit(self, **fields: object) -> dict:
        import io
        import logging

        from prognos.stream.observability import JsonLogFormatter, log_event

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter(service="test"))
        logger = logging.getLogger(f"prognos.test.coerce.{id(fields)}")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        log_event(logger, "test.event", **fields)
        return json.loads(stream.getvalue().splitlines()[0])

    def test_numpy_scalars_survive_as_numbers(self) -> None:
        """A numpy float is not JSON serialisable, and json.dumps would raise."""
        import numpy as np

        record = self._emit(value=np.float32(7.5), count=np.int64(3))
        assert record["value"] == pytest.approx(7.5)
        assert record["count"] == 3

    def test_arrays_become_lists(self) -> None:
        import numpy as np

        record = self._emit(values=np.array([1.0, 2.0]))
        assert record["values"] == [1.0, 2.0]

    def test_an_arbitrary_object_is_stringified_not_dropped(self) -> None:
        """Dropping an unserialisable field loses the evidence; stringifying keeps it."""
        record = self._emit(interval=Interval(1.0, 2.0, 3.0))
        assert "1.0" in str(record["interval"]) or record["interval"]

    def test_none_and_booleans_pass_through(self) -> None:
        record = self._emit(flag=False, missing=None)
        assert record["flag"] is False
        assert record["missing"] is None

    def test_a_prometheus_label_is_never_a_free_text_reason(self) -> None:
        """Metric labels must be low cardinality or the scrape breaks under load."""
        from prognos.stream.events import Reading, StreamStats, TelemetryBatch, validate_batch
        from prognos.stream.observability import MetricsRegistry, record_stream_metrics

        registry = MetricsRegistry()
        stats = StreamStats()
        for channel in (0, 999, 1000, 2000):
            batch = TelemetryBatch(source="s", unit=1, rows=[Reading(1, 0, channel, 1.0)])
            stats.record(batch, validate_batch(batch))
        record_stream_metrics(registry, stats)
        labels = {
            part.split("reason=")[1].strip('"} ')
            for part in registry.render().splitlines()
            if "reason=" in part
        }
        assert labels, "no reject labels were published"
        assert all(len(label) < 40 for label in labels)


def _get(url: str) -> tuple[int, str]:
    """GET without raising on 4xx or 5xx, so the status can be asserted directly."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


class TestAFailingRouteRepliesInsteadOfDropping:
    """A route that raises must still answer.

    Found by running the stack: `/cost` raised inside the container because the image
    did not carry `results/`, and the GET handler had no guard — so the connection was
    closed with no status line at all. `curl` reported HTTP 000. An operator sees a
    network error, not a server error, and goes looking at their own machine.

    `do_POST` already had this guard. The asymmetry was the bug.
    """

    def _serve(self, state: object) -> str:
        import threading

        from prognos.stream.service import build_server

        httpd = build_server(state, host="127.0.0.1", port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    def test_a_raising_get_route_returns_500(self) -> None:
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry

        def explode() -> dict:
            raise RuntimeError("the result file is missing")

        state = build_component(
            "dashboard", metrics=MetricsRegistry(), logger=logging.getLogger("t")
        )
        state.list_route["boom"] = explode
        base = self._serve(state)
        status, _ = _get(f"{base}/boom")
        assert status == 500, f"a raising route answered {status}, not 500"

    def test_the_500_body_leaks_no_internal_detail(self) -> None:
        """The exception text can carry the container's file layout."""
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry

        def explode() -> dict:
            raise RuntimeError("/opt/venv/lib/python3.12/site-packages/secret.py is missing")

        state = build_component(
            "dashboard", metrics=MetricsRegistry(), logger=logging.getLogger("t")
        )
        state.list_route["boom"] = explode
        base = self._serve(state)
        status, body = _get(f"{base}/boom")
        assert status == 500
        assert "site-packages" not in body
        assert "secret" not in body
        assert "internal error" in body

    def test_one_failing_route_does_not_break_the_others(self) -> None:
        import logging

        from prognos.stream.api import build_component
        from prognos.stream.observability import MetricsRegistry

        def explode() -> dict:
            raise RuntimeError("boom")

        state = build_component(
            "dashboard", metrics=MetricsRegistry(), logger=logging.getLogger("t")
        )
        state.list_route["boom"] = explode
        base = self._serve(state)
        assert _get(f"{base}/boom")[0] == 500
        assert _get(f"{base}/healthz")[0] == 200
        assert _get(f"{base}/")[0] == 200

    def test_an_http_error_still_keeps_its_own_status(self) -> None:
        """The guard must not flatten a deliberate 404 into a 500."""
        import logging

        from prognos.stream.api import HttpError, build_component
        from prognos.stream.observability import MetricsRegistry

        def refuse() -> dict:
            raise HttpError(404, "no such policy")

        state = build_component(
            "dashboard", metrics=MetricsRegistry(), logger=logging.getLogger("t")
        )
        state.list_route["nope"] = refuse
        base = self._serve(state)
        status, body = _get(f"{base}/nope")
        assert status == 404
        assert "no such policy" in body
