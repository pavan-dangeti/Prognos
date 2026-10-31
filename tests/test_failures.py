"""The failure modes the original fault suite did not cover.

An inference outage, a sudden shift in operating conditions, data that stops
arriving, and data that arrives out of order. Each must end in a named degraded state
and an alert, and never in silence or in a prediction that only looks current.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from prognos.stream.api import build_inference_state
from prognos.stream.faults import operating_envelope
from prognos.stream.inference import InferenceUnavailableError, RemotePredictor, RulPredictor
from prognos.stream.pipeline import PipelineConfig, StreamPipeline
from prognos.stream.service import build_server
from prognos.stream.window import N_CHANNELS

WINDOW, DOWNSAMPLE, STRIDE, ROWS_PER_CYCLE = 5, 2, 2, 40
CONFIG = PipelineConfig(window_length=WINDOW, window_stride=STRIDE, downsample=DOWNSAMPLE)


def predictor() -> RulPredictor:
    rng = np.random.default_rng(0)
    windows = rng.normal(100.0, 5.0, size=(300, WINDOW, N_CHANNELS)).astype(np.float32)
    cycles = np.arange(300, dtype=np.int64)
    targets = (100.0 - cycles / 10.0).astype(np.float32)
    return RulPredictor.train(
        windows, cycles, targets, windows, cycles, targets, target_coverage=0.9, seed=0
    )


def stream(n_cycles: int = 30, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Healthy readings around 100, several per cycle."""
    cycles = np.repeat(np.arange(1, n_cycles + 1), ROWS_PER_CYCLE).astype(np.int64)
    rows = np.random.default_rng(seed).normal(100.0, 5.0, size=(cycles.size, N_CHANNELS))
    return cycles, rows.astype(np.float32)


def opened(pipeline: StreamPipeline, unit: int, source: str | None = None) -> list[str]:
    """Names of every degraded state that raised an alert on one engine."""
    state = pipeline._engines[(source, unit)]
    return [str(alert["state"]) for alert in state.degraded_alerts]


class TestOperatingShift:
    @staticmethod
    def envelope() -> tuple[np.ndarray, np.ndarray]:
        """Stated outright: the synthetic noise is unbounded, real flight conditions are not."""
        return np.full(4, 50.0, dtype=np.float32), np.full(4, 150.0, dtype=np.float32)

    def test_the_envelope_bounds_only_flight_conditions_with_a_margin(self) -> None:
        windows = np.zeros((2, 3, N_CHANNELS), dtype=np.float32)
        windows[1, :, 0] = 10.0
        lower, upper = operating_envelope(windows, margin=0.1)
        assert lower.size == upper.size == 4
        assert (lower[0], upper[0]) == (pytest.approx(-1.0), pytest.approx(11.0))

    def test_a_clean_engine_stays_inside(self) -> None:
        pipeline = StreamPipeline(predictor(), CONFIG, envelope=self.envelope())
        cycles, rows = stream()
        pipeline.run_engine(1, cycles, rows)
        assert "operating_shift" not in opened(pipeline, 1)

    def test_a_shift_is_named_alerted_and_released(self) -> None:
        pipeline = StreamPipeline(predictor(), CONFIG, envelope=self.envelope())
        cycles, rows = stream()
        shifted = (cycles >= 10) & (cycles < 20)
        rows[shifted, 0] += 500.0  # altitude far outside anything fitted
        replay = pipeline.run_engine(1, cycles, rows)
        assert opened(pipeline, 1) == ["operating_shift"]
        assert replay.faults[0]["instrument"] == "alt"
        (row,) = pipeline.fleet.rows()
        assert row["status"] == "nominal", "the shift must clear once flights return"


class TestOutOfOrder:
    def test_a_late_reading_is_named_dropped_and_changes_no_window(self) -> None:
        cycles, rows = stream()
        clean = StreamPipeline(predictor(), CONFIG).run_engine(1, cycles, rows).n_windows

        pipeline = StreamPipeline(predictor(), CONFIG)
        produced = 0
        for index in range(cycles.size):
            produced += len(pipeline.push(1, int(cycles[index]), rows[index]))
            if index == 12 * ROWS_PER_CYCLE:  # the first reading of cycle 13
                assert pipeline.push(1, 11, rows[index]) == []
        state = pipeline._engines[(None, 1)]
        assert state.late_dropped == 1
        assert "cycle_regression" in opened(pipeline, 1)
        assert produced == clean


class TestSilence:
    def test_an_engine_that_stops_is_named_and_one_that_continues_is_not(self) -> None:
        now = [0.0]
        pipeline = StreamPipeline(predictor(), CONFIG, clock=lambda: now[0])
        cycles, rows = stream(n_cycles=4)
        for index in range(cycles.size):
            now[0] += 0.01
            pipeline.push(1, int(cycles[index]), rows[index])
            pipeline.push(2, int(cycles[index]), rows[index])
        for index in range(cycles.size):
            now[0] += 0.01
            pipeline.push(1, int(cycles[index]) + 4, rows[index])
        alerts = pipeline.check_silence(timeout=1.0)
        assert [a["unit"] for a in alerts] == [2]
        assert opened(pipeline, 2) == ["stream_silent"]
        assert "stream_silent" not in opened(pipeline, 1)

        pipeline.push(2, 5, rows[0])
        status = {row["unit"]: row["status"] for row in pipeline.fleet.rows()}
        assert status[2] != "degraded", "a stream that resumes is no longer silent"

    def test_a_finished_replay_is_not_silent(self) -> None:
        now = [0.0]
        pipeline = StreamPipeline(predictor(), CONFIG, clock=lambda: now[0])
        cycles, rows = stream(n_cycles=3)
        pipeline.run_engine(1, cycles, rows)
        now[0] += 3600.0
        assert pipeline.check_silence(timeout=1.0) == []


@pytest.fixture
def inference(tmp_path: Path) -> Iterator[tuple[str, Callable[[], None]]]:
    """A real inference service a test can take down and bring back on the same port."""
    predictor().save(tmp_path)
    servers: list[ThreadingHTTPServer] = []

    def start(port: int = 0) -> int:
        server = build_server(build_inference_state(model_dir=tmp_path), port=port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return int(server.server_address[1])

    def stop() -> None:
        servers[-1].shutdown()
        servers[-1].server_close()

    port = start()
    running = [True]

    def toggle() -> None:
        if running[0]:
            stop()
        else:
            start(port)
        running[0] = not running[0]

    yield f"http://127.0.0.1:{port}", toggle
    if running[0]:
        stop()


class TestInferenceOutage:
    def test_an_unreachable_service_raises_one_named_error(self) -> None:
        remote = RemotePredictor("http://127.0.0.1:9", timeout=0.5)
        windows = np.zeros((1, WINDOW, N_CHANNELS), dtype=np.float32)
        with pytest.raises(InferenceUnavailableError):
            remote.predict(windows, np.asarray([1], dtype=np.int64))

    def test_an_outage_is_named_predicts_nothing_and_recovers(
        self, inference: tuple[str, Callable[[], None]]
    ) -> None:
        url, toggle = inference
        pipeline = StreamPipeline(RemotePredictor(url, timeout=1.0), CONFIG)
        cycles, rows = stream(n_cycles=12)
        third = cycles.size // 3
        counts = [0, 0, 0]
        for index in range(cycles.size):
            if index in (third, 2 * third):
                toggle()
            counts[min(index // third, 2)] += len(pipeline.push(1, int(cycles[index]), rows[index]))
        assert counts[0] > 0 and counts[2] > 0
        assert counts[1] == 0, "nothing may be predicted while inference is down"
        assert opened(pipeline, 1) == ["inference_unavailable"]
        state = pipeline._engines[(None, 1)]
        assert state.missed_predictions > 0
        assert state.outage is None, "the state must clear once the service answers"
