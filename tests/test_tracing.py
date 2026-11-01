"""OpenTelemetry tracing.

One trace across the HTTP hop, nested spans in the pipeline, failures recorded, and
trace ids in the logs.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from prognos.stream.api import build_inference_state
from prognos.stream.inference import RemotePredictor, RulPredictor
from prognos.stream.observability import (
    JsonLogFormatter,
    MetricsRegistry,
    configure_tracing,
    span,
    tracer,
)
from prognos.stream.pipeline import PipelineConfig, StreamPipeline
from prognos.stream.service import build_server
from prognos.stream.window import N_CHANNELS

WINDOW, DOWNSAMPLE, STRIDE = 5, 2, 2


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    """Capture spans.

    The global provider can be set once per process, so an existing SDK provider gets
    the in-memory processor added rather than being replaced.
    """
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()
    exporter.shutdown()


def predictor() -> RulPredictor:
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(200, WINDOW, N_CHANNELS)).astype(np.float32)
    cycles = np.arange(200, dtype=np.int64)
    targets = (100.0 - cycles / 10.0).astype(np.float32)
    return RulPredictor.train(windows, cycles, targets, windows, cycles, targets, 0.9, seed=0)


def test_a_remote_prediction_is_one_trace_across_the_socket(
    spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    predictor().save(tmp_path)
    server = build_server(build_inference_state(model_dir=tmp_path), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        remote = RemotePredictor(f"http://127.0.0.1:{server.server_address[1]}")
        remote.predict(
            np.zeros((1, WINDOW, N_CHANNELS), dtype=np.float32), np.asarray([1], np.int64)
        )
    finally:
        server.shutdown()
        server.server_close()
    finished = {s.name: s for s in spans.get_finished_spans()}
    client = finished["inference.remote_predict"]
    handler = finished["POST /predict"]
    assert client.kind == SpanKind.CLIENT and handler.kind == SpanKind.SERVER
    assert handler.context.trace_id == client.context.trace_id
    assert handler.parent is not None and handler.parent.span_id == client.context.span_id
    assert finished["inference.predict"].parent.span_id == handler.context.span_id
    assert handler.attributes["http.response.status_code"] == 200


def test_an_unreachable_service_marks_the_client_span_failed(
    spans: InMemorySpanExporter,
) -> None:
    from prognos.stream.inference import InferenceUnavailableError

    with pytest.raises(InferenceUnavailableError):
        RemotePredictor("http://127.0.0.1:9", timeout=0.5).predict(
            np.zeros((1, WINDOW, N_CHANNELS), dtype=np.float32), np.asarray([1], np.int64)
        )
    (client,) = spans.get_finished_spans()
    assert client.status.status_code == StatusCode.ERROR


def test_each_window_nests_its_prediction_and_alert(spans: InMemorySpanExporter) -> None:
    config = PipelineConfig(window_length=WINDOW, window_stride=STRIDE, downsample=DOWNSAMPLE)
    pipeline = StreamPipeline(predictor(), config)
    cycles = np.ones(30, dtype=np.int64)
    rows = np.random.default_rng(1).normal(size=(30, N_CHANNELS)).astype(np.float32)
    produced = pipeline.run_engine(1, cycles, rows, source="DS01").n_windows
    finished = spans.get_finished_spans()
    windows = [s for s in finished if s.name == "pipeline.window"]
    assert len(windows) == produced > 0
    ids = {s.context.span_id for s in windows}
    for child in ("inference.predict", "alerting.evaluate"):
        children = [s for s in finished if s.name == child]
        assert len(children) == produced
        assert all(s.parent is not None and s.parent.span_id in ids for s in children)
    assert windows[0].attributes["engine"] == "DS01/unit001"


def test_a_failing_span_records_the_error_and_reraises(spans: InMemorySpanExporter) -> None:
    registry = MetricsRegistry()
    with pytest.raises(RuntimeError), span(registry, "work"):
        raise RuntimeError("boom")
    (failed,) = spans.get_finished_spans()
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.events[0].name == "exception"
    assert registry.value("span_errors_total", {"span": "work"}) == 1.0


def test_log_lines_carry_the_trace_they_belong_to(spans: InMemorySpanExporter) -> None:
    record = logging.LogRecord("prognos", logging.INFO, __file__, 1, "hello", None, None)
    with tracer.start_as_current_span("request") as current:
        line = json.loads(JsonLogFormatter(service="test").format(record))
    assert line["trace_id"] == format(current.get_span_context().trace_id, "032x")


def test_an_exporter_is_installed_only_when_an_endpoint_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[TracerProvider] = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    configure_tracing("inference")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    configure_tracing("inference")
    quiet, exporting = installed
    assert quiet.resource.attributes["service.name"] == "prognos-inference"
    assert len(quiet._active_span_processor._span_processors) == 0
    assert len(exporting._active_span_processor._span_processors) == 1
