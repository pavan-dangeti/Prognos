"""Tests for the streaming pipeline's own wiring.

`StreamPipeline` is the piece that can be wired wrongly while every component passes
its own tests: the alert keyed on the point estimate instead of the lower bound, the
coverage monitor handed something other than the truth, the window never advancing.
None of those show up in a unit test of any single component, so this file runs an
engine through the whole thing on data whose answers are known.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from prognos.stream.coverage import Observation
from prognos.stream.inference import RulPredictor
from prognos.stream.pipeline import PipelineConfig, StreamPipeline

WINDOW = 5
DOWNSAMPLE = 2
STRIDE = 2
ROWS_PER_CYCLE = 60


def predictor(width_hint: float = 20.0) -> RulPredictor:
    """A real predictor, fitted and calibrated on separate noise."""
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(300, WINDOW, 18)).astype(np.float32)
    cycles = np.arange(300, dtype=np.int64)
    targets = (100.0 - cycles / 10.0).astype(np.float32)
    return RulPredictor.train(
        windows, cycles, targets, windows, cycles, targets, target_coverage=0.9, seed=0
    )


def pipeline(**overrides: object) -> StreamPipeline:
    """A pipeline on a small window, so a test can push a handful of cycles."""
    settings: dict[str, object] = {
        "window_length": WINDOW,
        "window_stride": STRIDE,
        "downsample": DOWNSAMPLE,
        "alert_threshold": 30.0,
        "alert_hysteresis": 5.0,
        "target_coverage": 0.9,
        "min_coverage_samples": 5,
    }
    settings.update(overrides)
    # `detect_faults` belongs to the pipeline, not the config; accept either spelling
    # so a test asking for it does not have to know which is which.
    detect = bool(overrides.pop("detect_faults", True))
    settings.pop("detect_faults", None)
    return StreamPipeline(
        predictor(),
        PipelineConfig(**settings),
        detect_faults=detect,  # type: ignore[arg-type]
    )


def an_engine(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """One engine's raw stream: several cycle blocks of rows."""
    rng = np.random.default_rng(seed)
    cycles = np.repeat(np.arange(1, 5), ROWS_PER_CYCLE).astype(np.int64)
    rows = rng.normal(size=(cycles.size, 18)).astype(np.float32)
    return cycles, rows


def truth_for(cycles: np.ndarray, values: list[float]) -> dict[int, float]:
    """Map each cycle to its true remaining life."""
    unique = sorted({int(c) for c in cycles})
    return {cycle: values[i] for i, cycle in enumerate(unique)}


class TestStreamPipeline:
    def test_an_engine_produces_windows(self) -> None:
        cycles, rows = an_engine()
        replay = pipeline().run_engine(1, cycles, rows)
        assert replay.n_windows > 0
        assert replay.n_rows == cycles.size
        assert len(replay.predictions) == replay.n_windows

    def test_the_window_count_matches_an_independent_replay(self) -> None:
        """The incremental and batch windows must agree on the window count.

        A mismatch means the pipeline is silently dropping or duplicating windows.
        """
        from prognos.stream.window import replay_windows

        cycles, rows = an_engine(seed=2)
        replay = pipeline().run_engine(1, cycles, rows)
        batch = replay_windows(cycles, rows, length=WINDOW, stride=STRIDE, downsample=DOWNSAMPLE)
        assert replay.n_windows == batch.shape[0]

    def test_every_prediction_carries_a_calibrated_interval(self) -> None:
        """A served prediction with a zero-width interval means the calibrator is off."""
        cycles, rows = an_engine()
        replay = pipeline().run_engine(1, cycles, rows)
        assert all(p.interval.width > 0.0 for p in replay.predictions)
        assert all(p.interval.lower >= 0.0 for p in replay.predictions)

    def test_the_monitor_and_a_direct_count_agree(self) -> None:
        """The end-to-end wiring check.

        The verdict the pipeline reports must equal a coverage count computed directly
        from the same predictions. If the monitor ever scores a different stream than
        the one being reported, the two diverge — and a green verdict over a breached
        system is worse than no monitor.
        """
        cycles, rows = an_engine(seed=4)
        # Truth far outside every interval, so coverage is deterministically zero and
        # the two computations cannot coincide by accident.
        truth = truth_for(cycles, [10_000.0, 10_000.0, 10_000.0, 10_000.0])
        stream = pipeline()
        replay = stream.run_engine(1, cycles, rows, truth_by_cycle=truth)

        verdict = stream.coverage_verdict
        assert verdict.status == "breach"
        assert verdict.measured == pytest.approx(0.0)
        assert verdict.n_samples == replay.n_windows

    def test_no_truth_means_no_coverage_claims(self) -> None:
        """Without a truth the monitor must stay silent rather than guess."""
        cycles, rows = an_engine()
        stream = pipeline()
        stream.run_engine(1, cycles, rows)
        assert stream.coverage_verdict.status == "insufficient_data"

    def test_alerts_are_raised_on_the_lower_bound(self) -> None:
        """The alert keys on the interval's conservative edge.

        Phase 4 found the lower bound fires more often and costs more than the point
        estimate on this data. That is a statement about cost over a fleet, not about
        which number an operator should be shown, so the alert still keys on it.

        The threshold is set above any prediction this fixture produces, because what
        is under test is the alert logic rather than where the predictions land.
        """
        cycles, rows = an_engine(seed=6)
        stream = pipeline(alert_threshold=1e6)
        replay = stream.run_engine(1, cycles, rows)
        assert [a.kind for a in replay.alerts] == ["maintenance_due"]

    def test_an_alert_does_not_fire_when_the_bound_is_clear(self) -> None:
        """A threshold below every prediction must produce silence."""
        cycles, rows = an_engine(seed=6)
        stream = pipeline(alert_threshold=-0.0)
        replay = stream.run_engine(1, cycles, rows)
        assert all(p.interval.lower > 0.0 for p in replay.predictions)
        assert replay.alerts == []

    def test_engines_alert_independently(self) -> None:
        cycles_a, rows_a = an_engine(seed=7)
        cycles_b, rows_b = an_engine(seed=8)
        truth_for(cycles_a, [0.0, 0.0, 0.0, 0.0])
        stream = pipeline(alert_threshold=1e6)
        stream.run_engine(1, cycles_a, rows_a)
        stream.run_engine(2, cycles_b, rows_b)
        assert set(stream.open_alerts) == {(None, 1), (None, 2)}

    def test_the_same_engine_alerts_once(self) -> None:
        """One alert per engine, however many windows it produces."""
        cycles, rows = an_engine(seed=9)
        stream = pipeline(alert_threshold=1e6)
        replay = stream.run_engine(1, cycles, rows)
        assert replay.n_windows > 1
        assert len([a for a in replay.alerts if a.kind == "maintenance_due"]) == 1

    def test_metrics_record_covers_the_scrape_endpoints(self) -> None:
        cycles, rows = an_engine(seed=10)
        truth = truth_for(cycles, [0.0, 0.0, 0.0, 0.0])
        stream = pipeline()
        stream.run_engine(1, cycles, rows, truth_by_cycle=truth)
        stream.record_metrics()
        rendered = stream.metrics.render()
        assert "prognos_stream_batches_total" in rendered
        assert "prognos_interval_coverage" in rendered

    def test_summary_is_serialisable_and_names_the_model(self) -> None:
        """The dashboard reads this, so it has to survive a JSON round trip."""
        import json

        cycles, rows = an_engine(seed=11)
        stream = pipeline()
        stream.run_engine(1, cycles, rows)
        summary = stream.summary()
        assert summary["model"] == "lightgbm"
        json.dumps(summary)


class TestObservationIsAValue:
    def test_observations_are_immutable(self) -> None:
        """An edited observation is not a record of what was predicted."""
        observation = Observation(unit=1, truth=10.0, lower=5.0, upper=15.0, point=10.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            observation.truth = 99.0  # type: ignore[misc]


class TestFleetViewIsFedByTheStream:
    """The `/fleet` route is decoration unless the pipeline actually writes to it.

    An endpoint that returns an empty fleet forever is worse than no endpoint: it looks
    like a working feature, and an operator watching it sees a fleet that never sends
    data when the truth is that nothing was wired up.
    """

    def test_predictions_populate_the_fleet(self) -> None:
        cycles, rows = an_engine()
        stream = pipeline()
        stream.run_engine(4, cycles, rows, source="DS01-004")

        seen = stream.fleet.rows()
        assert [r["unit"] for r in seen] == [4]
        assert seen[0]["engine"] == "DS01-004/unit004"
        assert seen[0]["cycle"] > 0

    def test_the_fleet_carries_the_latest_cycle_not_every_one(self) -> None:
        """A view with a row per reading is a log file wearing a table's clothes."""
        cycles, rows = an_engine(seed=3)
        stream = pipeline()
        stream.run_engine(4, cycles, rows)
        assert len(stream.fleet.rows()) == 1
        assert stream.fleet.rows()[0]["cycle"] == int(cycles.max())

    def test_two_engines_are_two_rows(self) -> None:
        cycles, rows = an_engine(seed=4)
        stream = pipeline()
        stream.run_engine(1, cycles, rows, source="DS01-001")
        stream.run_engine(2, cycles, rows, source="DS01-002")
        assert len(stream.fleet.rows()) == 2

    def test_engines_from_different_subdatasets_do_not_collide(self) -> None:
        """Unit numbers restart per sub-dataset. Merging them is the coverage bug."""
        cycles, rows = an_engine(seed=5)
        stream = pipeline()
        stream.run_engine(1, cycles, rows, source="DS01-001")
        stream.run_engine(1, cycles, rows, source="DS02-001")
        assert {r["engine"] for r in stream.fleet.rows()} == {
            "DS01-001/unit001",
            "DS02-001/unit001",
        }

    def test_a_degraded_engine_is_marked_degraded(self) -> None:
        """The status column is the sort key an operator actually uses."""
        cycles, rows = an_engine(seed=6)
        # `len(rows)`, not `rows.size`: `size` is the element count, so halving it
        # indexes past the end of the array and the slice is silently empty.
        rows[len(rows) // 2 :, 5] = 1e12  # out of range for the tail of the stream
        stream = pipeline(detect_faults=True)
        stream.run_engine(7, cycles, rows)
        assert {r["status"] for r in stream.fleet.rows()} == {"degraded"}

    def test_a_clean_engine_is_nominal(self) -> None:
        cycles, rows = an_engine(seed=7)
        stream = pipeline(detect_faults=True)
        stream.run_engine(7, cycles, rows)
        assert {r["status"] for r in stream.fleet.rows()} == {"nominal"}

    def test_the_summary_exposes_the_fleet(self) -> None:
        """`summary()` has claimed "for the dashboard" since it was written."""
        cycles, rows = an_engine(seed=8)
        stream = pipeline()
        stream.run_engine(2, cycles, rows)
        assert stream.summary()["fleet"][0]["unit"] == 2


class TestLiveInjection:
    """A fault armed on a running pipeline must reach the rows the detector sees."""

    def test_an_armed_fault_turns_the_engine_degraded(self) -> None:
        from prognos.stream.faults import InjectedFault

        cycles, rows = an_engine(seed=9)
        stream = pipeline(detect_faults=True)
        stream.arm_fault(3, InjectedFault("value_out_of_range", channel=4, from_cycle=3))
        stream.run_engine(3, cycles, rows)
        (row,) = stream.fleet.rows()
        assert row["status"] == "degraded"
        assert {s["state"] for s in row["degraded"]} == {"value_out_of_range"}

    def test_disarming_lets_the_engine_recover(self) -> None:
        from prognos.stream.faults import InjectedFault

        cycles, rows = an_engine(seed=10)
        stream = pipeline(detect_faults=True)
        stream.arm_fault(3, InjectedFault("value_out_of_range", channel=4, from_cycle=1))
        stream.arm_fault(3, None)
        stream.run_engine(3, cycles, rows)
        assert {r["status"] for r in stream.fleet.rows()} == {"nominal"}

    def test_engines_interleave_without_sharing_a_window(self) -> None:
        """`push` keeps state per engine, which is what a live fleet needs."""
        cycles, rows = an_engine(seed=11)
        interleaved = pipeline()
        produced: dict[int, int] = {1: 0, 2: 0}
        for index in range(cycles.size):
            for unit in (1, 2):
                produced[unit] += len(interleaved.push(unit, int(cycles[index]), rows[index]))
        alone = pipeline().run_engine(1, cycles, rows).n_windows
        assert produced == {1: alone, 2: alone}

    def test_the_detector_sees_one_row_per_cycle(self) -> None:
        """At thousands of rows a cycle, per-row detection judged scatter, not the engine."""
        cycles, rows = an_engine(seed=12)
        stream = pipeline(detect_faults=True)
        stream.run_engine(5, cycles, rows)
        assert stream._detectors is not None
        assert stream._detectors[(None, 5)].n_observations == len(set(cycles.tolist()))

    def test_a_dropped_channel_degrades_the_engine_without_stopping_the_stream(self) -> None:
        """NaN used to escape the window and end the replay over one dead sensor."""
        from prognos.stream.faults import InjectedFault

        cycles = np.repeat(np.arange(1, 31), ROWS_PER_CYCLE).astype(np.int64)
        rows = np.random.default_rng(13).normal(size=(cycles.size, 18)).astype(np.float32)
        clean = pipeline(detect_faults=True).run_engine(3, cycles, rows).n_windows
        stream = pipeline(detect_faults=True)
        stream.arm_fault(3, InjectedFault("sensor_dropout", channel=6, from_cycle=5))
        replay = stream.run_engine(3, cycles, rows)
        assert replay.n_windows == clean
        (row,) = stream.fleet.rows()
        assert row["status"] == "degraded"
        assert {s["state"] for s in row["degraded"]} == {"sensor_dropout"}

    def test_a_reading_with_nothing_to_hold_is_skipped(self) -> None:
        stream = pipeline()
        row = np.full(18, np.nan, dtype=np.float32)
        assert stream.push(1, 1, row) == []

    def test_a_paced_replay_takes_at_least_its_budget(self) -> None:
        import time

        cycles = np.repeat(np.arange(1, 4), 1000).astype(np.int64)
        rows = np.random.default_rng(14).normal(size=(cycles.size, 18)).astype(np.float32)
        started = time.monotonic()
        paced = pipeline().run_engine(1, cycles, rows, pace=10_000.0)
        assert time.monotonic() - started >= 0.2  # 3,000 readings at 10,000 a second
        assert paced.n_windows == pipeline().run_engine(1, cycles, rows).n_windows


class TestEngineIdentity:
    """Unit numbers restart in every sub-dataset, so state must be keyed by both halves.

    Keyed by unit alone, DS02 unit 2 opened its life by resolving the alert DS01 unit 2
    had left open, and the committed replay recorded seven alerts no engine raised.
    """

    @staticmethod
    def engine_at(first_cycle: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        cycles = np.repeat(np.arange(first_cycle, first_cycle + 4), ROWS_PER_CYCLE)
        rows = np.random.default_rng(seed).normal(size=(cycles.size, 18)).astype(np.float32)
        return cycles.astype(np.int64), rows

    def test_an_engine_does_not_inherit_the_alert_of_another_sharing_its_unit(self) -> None:
        # Late-life cycles predict low RUL and early ones high, with a threshold between.
        worn_cycles, worn_rows = self.engine_at(290, seed=20)
        fresh_cycles, fresh_rows = self.engine_at(1, seed=21)
        stream = pipeline(alert_threshold=85.0, alert_hysteresis=5.0)

        worn = stream.run_engine(2, worn_cycles, worn_rows, source="DS01-005")
        assert [a.kind for a in worn.alerts][:1] == ["maintenance_due"], "fixture must alert"
        fresh = stream.run_engine(2, fresh_cycles, fresh_rows, source="DS02-006")

        assert fresh.alerts == [], "the second engine resolved an alert it never raised"
        assert set(stream.open_alerts) == {("DS01-005", 2)}
        rows = {row["engine"]: row for row in stream.fleet.rows()}
        assert rows["DS01-005/unit002"]["status"] == "alerting"
        assert rows["DS02-006/unit002"]["status"] == "nominal"

    def test_a_fault_reaches_only_the_engine_it_was_armed_on(self) -> None:
        from prognos.stream.faults import InjectedFault

        cycles, rows = an_engine(seed=22)
        stream = pipeline(detect_faults=True)
        fault = InjectedFault("value_out_of_range", channel=4, from_cycle=1)
        stream.arm_fault(2, fault, source="DS02-006")
        stream.run_engine(2, cycles, rows, source="DS01-005")
        stream.run_engine(2, cycles, rows, source="DS02-006")
        status = {row["engine"]: row["status"] for row in stream.fleet.rows()}
        assert status == {"DS01-005/unit002": "nominal", "DS02-006/unit002": "degraded"}


def test_the_alerting_service_keeps_engines_sharing_a_unit_apart() -> None:
    from prognos.stream.alerting import AlertingService

    service = AlertingService(threshold=20.0, hysteresis=5.0)
    service.evaluate(unit=2, cycle=90, lower=5.0, point=15.0, source="DS01-005")
    assert service.evaluate(unit=2, cycle=1, lower=80.0, point=95.0, source="DS02-006") == []
    assert service.open_alerts(2, source="DS01-005") == 1
    assert service.open_alerts(2, source="DS02-006") == 0


class TestLiveMetrics:
    """What a running stack's scrape shows, refreshed while the replay runs."""

    def test_publishing_fills_the_series_the_dashboards_read(self) -> None:
        cycles, rows = an_engine(seed=30)
        truth = truth_for(cycles, [40.0, 39.0, 38.0, 37.0])
        stream = pipeline(detect_faults=True)
        stream.run_engine(1, cycles, rows, truth_by_cycle=truth, source="DS01")
        stream.publish_metrics()
        text = stream.metrics.render()
        assert f"prognos_stream_accepted_total {float(cycles.size)}" in text
        for series in ("prognos_interval_coverage_samples", "prognos_fleet_rul_cycles"):
            assert series in text, series

    def test_late_readings_are_counted_as_rejected(self) -> None:
        _cycles, rows = an_engine(seed=31)
        stream = pipeline()
        stream.push(1, 3, rows[0])
        stream.push(1, 2, rows[1])
        stream.publish_metrics()
        assert stream.metrics.value("stream_rejected_total") == 1.0
        assert (
            stream.metrics.value("stream_rejected_by_reason_total", {"reason": "late_reading"})
            == 1.0
        )

    def test_rendering_while_another_thread_writes_does_not_raise(self) -> None:
        import threading

        from prognos.stream.observability import MetricsRegistry

        registry = MetricsRegistry()
        stop = threading.Event()

        def write() -> None:
            n = 0
            while not stop.is_set():
                n += 1
                registry.gauge("g", "gauge", value=n, labels={"k": str(n % 500)})

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        try:
            for _ in range(300):
                registry.render()
        finally:
            stop.set()
            writer.join()
