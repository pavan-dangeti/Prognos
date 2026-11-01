"""Tests for the operating-shift study and the tracker it tunes."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from prognos.envelope_study import (
    SHIPPED,
    Flights,
    ShiftSettings,
    evaluate,
    openings,
    selection_key,
    shifted,
)
from prognos.stream.faults import ShiftTracker

RESULT = Path("results/phase6_envelope.json")
FLEET = (np.zeros(4, dtype=np.float32), np.full(4, 1000.0, dtype=np.float32))


def engine(seed: int = 0, n: int = 70, cruise: float = 300.0) -> Flights:
    """A short-haul engine: every flight between the ground and about `cruise`."""
    rng = np.random.default_rng(seed)
    low = rng.uniform(10.0, 20.0, size=(n, 4)).astype(np.float32)
    high = (cruise + rng.uniform(-5.0, 5.0, size=(n, 4))).astype(np.float32)
    return Flights(label=f"e{seed}", low=low, high=high)


def test_a_quiet_engine_raises_nothing() -> None:
    assert openings(engine(), ShiftSettings(10, 0.1, 2, 5), FLEET) == []


def test_the_fleet_envelope_alone_misses_a_shift_inside_it() -> None:
    within = shifted(engine(), FLEET, beyond_fleet=False)
    assert float(within.high[:, 0].max()) < 1000.0
    assert openings(within, SHIPPED, FLEET) == []
    assert len(openings(within, ShiftSettings(10, 0.1, 2, 5), FLEET)) == 1


def test_one_event_is_one_alert_across_short_gaps() -> None:
    """Bursts of flights outside, with a couple of normal flights between, are one shift."""
    event = engine()
    high = event.high.copy()
    for start in range(40, 70, 5):  # three flights out, two flights in, repeated
        high[start : start + 3, 0] += np.float32(900.0)
    bursts = Flights(label="bursts", low=event.low, high=high)
    assert len(openings(bursts, SHIPPED, FLEET)) == 6
    assert len(openings(bursts, ShiftSettings(10, 0.1, 2, 5), FLEET)) == 1


def test_a_strictly_alternating_shift_is_not_named() -> None:
    """A known gap, pinned so it cannot be forgotten.

    Persistence 2 needs two flights outside in a row, so a shift on every other flight
    never opens. The study's scenarios shift every flight from one point on and never
    tested this; it is reported as a limitation rather than tuned around afterwards.
    """
    event = engine()
    high = event.high.copy()
    high[40::2, 0] += np.float32(900.0)
    alternating = Flights(label="alternating", low=event.low, high=high)
    assert openings(alternating, ShiftSettings(10, 0.1, 2, 5), FLEET) == []


def test_a_flight_the_fleet_already_calls_abnormal_does_not_teach_the_engine() -> None:
    tracker = ShiftTracker(unit=1, fleet=FLEET, learn_cycles=3, margin=0.1, persistence=1)
    absurd = np.full(4, 5000.0, dtype=np.float32)
    tracker.observe_flight(np.zeros(4, dtype=np.float32), absurd)
    for _ in range(3):
        tracker.observe_flight(np.full(4, 10.0, np.float32), np.full(4, 300.0, np.float32))
    assert tracker._own is not None
    assert float(tracker._own[1][0]) < 400.0


def test_selection_prefers_fewer_false_alarms_then_detection() -> None:
    def result(alarmed: int, within: float) -> dict[str, object]:
        return {
            "settings": {"learn": 10, "margin": 0.1, "persistence": 1, "release": 1},
            "false_alarms": {"engines_alarmed": alarmed},
            "within_fleet": {"rate": within, "alerts_per_event": 1.0},
            "beyond_fleet": {"rate": 1.0, "alerts_per_event": 1.0},
        }

    assert min([result(1, 1.0), result(0, 0.5)], key=selection_key)["false_alarms"] == {
        "engines_alarmed": 0
    }


def test_evaluate_reports_detection_and_alerts_per_event() -> None:
    report = evaluate([engine(1), engine(2)], ShiftSettings(10, 0.1, 2, 5), FLEET)
    assert report["false_alarms"]["engines_alarmed"] == 0
    assert report["within_fleet"]["rate"] == 1.0
    assert report["beyond_fleet"]["alerts_per_event"] == 1.0


def test_the_tracker_ships_with_the_settings_the_study_chose() -> None:
    if not RESULT.exists():
        pytest.skip("no committed study; run `prognos envelope-study`")
    report = json.loads(RESULT.read_text())
    chosen = report["chosen"]
    defaults = inspect.signature(ShiftTracker).parameters
    assert defaults["learn_cycles"].default == chosen["learn"]
    assert defaults["margin"].default == chosen["margin"]
    assert defaults["persistence"].default == chosen["persistence"]
    assert defaults["release"].default == chosen["release"]
    assert set(report["development_engines"]).isdisjoint(report["held_out_engines"])
    assert report["final_test_group"]["status"] == "closed"
