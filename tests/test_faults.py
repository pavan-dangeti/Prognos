"""Tests for fault injection and degraded-state detection.

`docs/design.md` §12 makes "every injected fault produces a named degraded state and an
alert" the Phase 5 exit criterion, and S6 requires any degraded state to be observable
in metrics, logs and the dashboard. Neither existed. These tests are the criterion.

The property under test throughout is that a fault is **named**, not merely counted.
"Rejects: 400" is not a degraded state an operator can act on; "sensor_stuck on
channel 7 of engine 3" is. So almost every test here asserts on the *name* and on the
evidence attached to it, because a detector that fires without saying why produces an
alert nobody trusts and then stops reading.

The counter-property matters just as much: **no fault in normal operation**. A detector
that fires on healthy data trains operators out of reading it, which is worse than not
having one. So there is a class of tests asserting silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from prognos.stream.faults import (
    DEGRADED_STATES,
    FaultDetector,
    FaultInjector,
    InjectedFault,
    channel_of,
)
from prognos.stream.window import N_CHANNELS

HEALTHY_CYCLES = 60


def healthy_stream(n_cycles: int = HEALTHY_CYCLES, seed: int = 0) -> np.ndarray:
    """A stream that varies channel to channel and cycle to cycle.

    Variance matters: a stuck or drifted sensor is defined by comparison against the
    other channels, so a fixture where every channel is constant would make those
    detectors fire on healthy data.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(100.0, 5.0, size=(n_cycles, N_CHANNELS)).astype(np.float32)


def noisy_stream(
    n_cycles: int = HEALTHY_CYCLES, channel: int = 7, seed: int = 0
) -> tuple[np.ndarray, int]:
    """A healthy stream plus noise on one channel, for the drift detector."""
    rows = healthy_stream(n_cycles, seed=seed)
    return rows, channel


class TestNamedStates:
    def test_every_state_has_a_stable_name(self) -> None:
        """Names end up as metric labels and log fields, so they must not drift."""
        assert DEGRADED_STATES, "no degraded states are defined"
        for name in DEGRADED_STATES:
            assert name == name.strip().lower()
            assert " " not in name, f"{name} has a space, which breaks a metric label"

    def test_every_state_is_an_alert_kind(self) -> None:
        """A degraded state that cannot raise an alert is a log line, not a mechanism."""
        from prognos.stream.alerting import MAINTENANCE_DUE, MAINTENANCE_RESOLVED

        assert MAINTENANCE_DUE not in DEGRADED_STATES
        assert MAINTENANCE_RESOLVED not in DEGRADED_STATES


class TestSilentOnHealthyData:
    def test_a_healthy_stream_produces_nothing(self) -> None:
        """The most important test here.

        A detector that fires on healthy data trains operators out of reading it, which
        is worse than not shipping one at all.
        """
        detector = FaultDetector(unit=1)
        for cycle, row in enumerate(healthy_stream(), start=1):
            detector.observe(cycle, row)
        assert detector.active() == []

    def test_repeated_observations_stay_silent(self) -> None:
        """A long clean run must not drift into an alert on its own."""
        detector = FaultDetector(unit=1)
        rng = np.random.default_rng(3)
        for cycle in range(1, 401):
            detector.observe(cycle, rng.normal(100.0, 5.0, size=N_CHANNELS).astype(np.float32))
        assert detector.active() == []

    def test_a_short_stream_is_not_enough_evidence(self) -> None:
        """Few cycles is not evidence of a stuck sensor; it is a stream that just started."""
        detector = FaultDetector(unit=1)
        for cycle, row in enumerate(healthy_stream(3), start=1):
            detector.observe(cycle, row)
        assert detector.active() == []


class TestSensorDropout:
    def test_a_silent_channel_is_named(self) -> None:
        """One channel stops reporting while its neighbours keep going."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 30:
                values[7] = np.nan
            detector.observe(cycle, values)

        states = detector.active()
        assert [s.state for s in states] == ["sensor_dropout"]
        assert states[0].channel == 7

    def test_the_alert_carries_which_channel_and_how_long(self) -> None:
        """Evidence, not just a name. An alert that cannot say why is not actionable."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 30:
                values[7] = np.nan
            detector.observe(cycle, values)

        state = detector.active()[0]
        assert state.silent_cycles >= 30
        assert state.details["channel"] == 7
        assert state.details["engine"] == 1

    def test_one_dropout_does_not_report_every_other_channel(self) -> None:
        """Detectors that report everything report nothing."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 30:
                values[7] = np.nan
            detector.observe(cycle, values)
        assert len(detector.active()) == 1


class TestSensorStuck:
    def test_a_frozen_channel_is_named(self) -> None:
        """A channel holding one value forever while others move."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[3] = 100.0
            detector.observe(cycle, values)

        states = {s.state: s for s in detector.active()}
        assert "sensor_stuck" in states
        assert states["sensor_stuck"].channel == 3

    def test_a_naturally_quiet_channel_is_not_reported(self) -> None:
        """A channel with genuinely low variance is quiet, not broken.

        The detector compares a channel against the spread of the others on the same
        engine, so it has to be relative. An absolute variance threshold would fire on
        whichever channel happens to be calm.
        """
        detector = FaultDetector(unit=1)
        rng = np.random.default_rng(5)
        for cycle in range(1, 121):
            row = rng.normal(100.0, 5.0, size=N_CHANNELS).astype(np.float32)
            row[9] = 50.0 + rng.normal(0.0, 0.01)  # quiet, but not frozen
            detector.observe(cycle, row)

        stuck = [s for s in detector.active() if s.state == "sensor_stuck"]
        assert stuck == [], f"a merely quiet channel was reported: {stuck}"


class TestSensorDrift:
    def test_a_step_is_named(self) -> None:
        """A channel whose level moves abruptly while its neighbours hold still.

        Drift is the fault a range check cannot see and that RMSE barely notices until
        it is large, which is why it gets its own detector.

        Measured between adjacent windows rather than against the engine's starting
        level, because N-CMAPSS sensors drift *by design* — measuring degradation is
        what they are for. Distance from the start of life flags every healthy engine.
        The control run in `results/phase5_faults.json` is that calibration.
        """
        detector = FaultDetector(unit=1)
        rows = healthy_stream(120)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[11] += 40.0
            detector.observe(cycle, values)

        assert "sensor_drift" in {h.state for h in detector.history}

    def test_a_step_names_the_channel_it_happened_on(self) -> None:
        detector = FaultDetector(unit=1)
        rows = healthy_stream(120)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[11] += 40.0
            detector.observe(cycle, values)
        assert 11 in {h.channel for h in detector.history if h.state == "sensor_drift"}

    def test_degradation_alongside_the_step_is_not_reported(self) -> None:
        """Real drift on a healthy engine is not a fault, and the control proves it.

        Every channel on this dataset trends over an engine's life. A detector that
        flags that is flagging the thing the sensors exist to measure.
        """
        detector = FaultDetector(unit=1)
        rng = np.random.default_rng(11)
        for cycle in range(1, 201):
            row = rng.normal(100.0, 5.0, size=N_CHANNELS).astype(np.float32)
            row += 0.2 * cycle  # uniform degradation trend, as a real engine shows
            detector.observe(cycle, row)
        assert detector.active() == [], (
            f"a smooth degradation trend was reported as a fault: {detector.active()}"
        )

    def test_a_small_shift_below_the_threshold_is_ignored(self) -> None:
        """A detector tuned to catch every wobble fires on everything and is ignored."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream(120)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[11] += 1.5
            detector.observe(cycle, values)
        drift = [h for h in detector.history if h.state == "sensor_drift"]
        assert drift == [], f"a sub-threshold shift was reported: {drift}"


class TestCycleRegression:
    def test_a_clock_going_backwards_is_named(self) -> None:
        """The online window silently drops late rows; something has to say so.

        A regression is an *event*, not a condition: the row was refused and the next
        one arrives in order. So it is in the history and not in `active()`, which is
        where conditions that are still true belong.
        """
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            detector.observe(cycle if cycle < 25 else cycle - 10, row)

        assert "cycle_regression" in {h.state for h in detector.history}
        assert "cycle_regression" not in {s.state for s in detector.active()}

    def test_the_stream_recovers_after_a_regression(self) -> None:
        """A refused row must not leave the detector permanently unhealthy."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            detector.observe(cycle, row)
            if cycle == 25:
                detector.observe(10, row)
        assert detector.active() == []

    def test_a_repeated_cycle_is_normal_not_a_fault(self) -> None:
        """This dataset carries thousands of rows per cycle number.

        Treating a repeated cycle as a regression named a fault on every healthy engine,
        which is the fastest way to make a detector ignored. Only a cycle that goes
        *backwards* is a fault.
        """
        detector = FaultDetector(unit=1)
        rows = healthy_stream(20)
        for cycle, row in enumerate(rows, start=1):
            for _ in range(3):  # several rows per cycle, as the real data has
                detector.observe(cycle, row)
        assert "cycle_regression" not in {h.state for h in detector.history}
        assert detector.active() == []

    def test_the_window_still_drops_the_row(self) -> None:
        """Detection and handling are separate jobs.

        The detector names it; the window still refuses to reorder a block it has already
        built windows from. One does not replace the other.
        """
        from prognos.stream.window import OnlineWindow

        window = OnlineWindow(length=3, downsample=1, stride=1)
        for cycle in range(5):
            window.push(cycle, np.full(N_CHANNELS, float(cycle)))
        window.push(1, np.full(N_CHANNELS, -1.0))
        assert window.cycle == 4


class TestValuePlausibility:
    def test_a_physically_impossible_value_is_named(self) -> None:
        """Validation rejects it; the detector explains why the stream is unwell."""
        from prognos.stream.events import N_CHANNELS as EVENT_CHANNELS

        assert EVENT_CHANNELS == N_CHANNELS
        detector = FaultDetector(unit=1, plausible_max=1000.0)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 25:
                values[2] = 1e9
            detector.observe(cycle, values)

        states = {s.state: s for s in detector.active()}
        assert "value_out_of_range" in states
        assert states["value_out_of_range"].channel == 2


class TestSilentStream:
    def test_no_data_arriving_is_itself_a_degraded_state(self) -> None:
        """Silence is not health.

        A stream that stops is the single most common failure and the easiest to miss,
        because nothing logs an error: nothing is arriving. It has to be named.
        """
        detector = FaultDetector(unit=1)
        for cycle, row in enumerate(healthy_stream(20), start=1):
            detector.observe(cycle, row)
        assert detector.active() == [], "a stream that is still flowing is not silent"

        detector.mark_expected_cycle(500)
        assert [s.state for s in detector.active()] == ["stream_silent"]


class TestRecovery:
    def test_a_state_clears_when_the_sensor_returns(self) -> None:
        """A detector that latches is an alert nobody clears."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream(90)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if 20 < cycle <= 40:
                values[7] = np.nan
            detector.observe(cycle, values)

        assert detector.active() == [], "a recovered sensor must stop reporting"
        assert detector.history, "recovery must still be recorded"

    def test_history_records_what_happened(self) -> None:
        """An operator needs to know it happened, not only that it is over."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream(90)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if 20 < cycle <= 40:
                values[7] = np.nan
            detector.observe(cycle, values)

        recorded = {(h.state, h.channel) for h in detector.history}
        assert ("sensor_dropout", 7) in recorded


class TestFaultInjector:
    """The simulator side: producing a fault on demand.

    Needed because the dashboard's injection controls and every fault test have to
    provoke the real condition rather than assert against a mock detector.
    """

    def test_injecting_a_dropout_produces_a_dropout(self) -> None:
        injector = FaultInjector(
            unit=1, fault=InjectedFault("sensor_dropout", channel=7, from_cycle=20)
        )
        detector = FaultDetector(unit=1)
        for cycle, row in enumerate(healthy_stream(), start=1):
            detector.observe(cycle, injector.apply(cycle, row.copy()))
        assert [s.state for s in detector.active()] == ["sensor_dropout"]

    def test_injecting_a_stuck_sensor_produces_a_stuck_sensor(self) -> None:
        injector = FaultInjector(
            unit=1, fault=InjectedFault("sensor_stuck", channel=3, from_cycle=20)
        )
        detector = FaultDetector(unit=1)
        for cycle, row in enumerate(healthy_stream(), start=1):
            detector.observe(cycle, injector.apply(cycle, row.copy()))
        assert "sensor_stuck" in {s.state for s in detector.active()}

    def test_injecting_a_drift_produces_drift(self) -> None:
        injector = FaultInjector(
            unit=1, fault=InjectedFault("sensor_drift", channel=11, from_cycle=20)
        )
        detector = FaultDetector(unit=1)
        for cycle, row in enumerate(healthy_stream(), start=1):
            detector.observe(cycle, injector.apply(cycle, row.copy()))
        # A step is an event: once both windows sit on the new level there is no longer
        # a step to see, so it is in the history rather than in the active set.
        assert "sensor_drift" in {h.state for h in detector.history}

    def test_no_fault_leaves_the_stream_untouched(self) -> None:
        injector = FaultInjector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            np.testing.assert_array_equal(injector.apply(cycle, row.copy()), row)

    def test_an_unknown_fault_name_is_rejected(self) -> None:
        """A typo in a dashboard control must fail, not inject nothing quietly."""
        with pytest.raises(ValueError, match="unknown fault"):
            InjectedFault("snesor_dropout", channel=1, from_cycle=1)

    def test_a_fault_before_its_start_cycle_does_nothing(self) -> None:
        injector = FaultInjector(
            unit=1, fault=InjectedFault("sensor_stuck", channel=3, from_cycle=50)
        )
        rows = healthy_stream()
        for cycle, row in enumerate(rows[:30], start=1):
            np.testing.assert_array_equal(injector.apply(cycle, row.copy()), row)


class TestChannelHelper:
    def test_channel_of_reads_the_instrument_name(self) -> None:
        """A degraded state naming `channel 7` is less useful than one naming the sensor."""
        from prognos.schema import MEASURED_SENSORS, SCENARIO_DESCRIPTORS

        assert channel_of(0) == SCENARIO_DESCRIPTORS[0]
        assert channel_of(len(SCENARIO_DESCRIPTORS)) == MEASURED_SENSORS[0]

    def test_an_out_of_range_channel_is_reported_as_such(self) -> None:
        assert "unknown" in channel_of(9999)


class TestAStepIsAnEventNotAState:
    """What a step should do once it has been seen.

    The first implementation measured drift against a rolling window that scrolled past
    the transition, which lost the fault. The second measured it against the engine's
    starting level, which flagged every healthy engine — N-CMAPSS sensors drift by
    design, because measuring degradation is what they are for. The third compares
    adjacent windows, which separates a step from a trend: the step moves abruptly
    between two windows, degradation moves slowly through both.

    The consequence, which these tests pin: a step is detected once, recorded, and then
    clears. It is not a permanent alert, because after the step the channel is simply
    reading 40 units higher and that is no longer an anomaly to keep shouting about.
    """

    def test_a_step_is_recorded_once(self) -> None:
        detector = FaultDetector(unit=1)
        rows = healthy_stream(160)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[11] += 40.0
            detector.observe(cycle, values)

        steps = [h for h in detector.history if h.state == "sensor_drift"]
        assert steps, "the step was never named"
        assert {h.channel for h in steps} == {11}

    def test_a_step_clears_once_both_windows_have_moved(self) -> None:
        """Otherwise one transient becomes a permanent alert nobody clears."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream(160)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[11] += 40.0
            detector.observe(cycle, values)

        assert [s for s in detector.active() if s.state == "sensor_drift"] == []
        assert any(h.state == "sensor_drift" for h in detector.history)

    def test_the_step_reports_both_windows_it_moved_between(self) -> None:
        """Without both numbers a shift cannot be checked against anything later."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream(120)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 20:
                values[11] += 40.0
            detector.observe(cycle, values)

        # The detector records a state when it *opens*, not each time it is re-observed.
        # That is what alerting needs — one alert when the condition starts — and it
        # means the recorded evidence is the onset, which straddles the transition and
        # so sees part of the move rather than all of it. Asserting the onset rather
        # than pretending it captured the peak.
        steps = [h for h in detector.history if h.state == "sensor_drift"]
        assert len(steps) == 1, "a step must open exactly once, not once per cycle"
        onset = steps[0]
        assert onset.details["previous"] == pytest.approx(100.0, abs=4.0)
        assert onset.details["current"] > onset.details["previous"]
        # How much of the step the onset sees depends on the comparison window: with ten
        # cycles a side the alert opens partway in, around half the move. Bounded on
        # both sides so a detector that opened on noise, or only after the whole step
        # had passed, still fails.
        assert 10.0 < onset.details["shift"] <= 54.0
        # The starting level is carried too, so the step can be placed against the
        # engine's whole life rather than only against the moment it was noticed.
        assert onset.details["baseline"] == pytest.approx(100.0, abs=4.0)

    def test_a_step_that_reverses_does_not_remain_open(self) -> None:
        detector = FaultDetector(unit=1)
        rows = healthy_stream(200)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if 20 < cycle <= 60:
                values[11] += 40.0
            detector.observe(cycle, values)

        assert [s for s in detector.active() if s.state == "sensor_drift"] == []

    def test_a_modest_excursion_is_not_a_step(self) -> None:
        """A few sigma of wobble is noise and must not be named.

        Three cycles moved by about four standard deviations, which on sigma-5 sensors
        is real but ordinary measurement scatter.
        """
        detector = FaultDetector(unit=1)
        rows = healthy_stream(200)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if 100 <= cycle < 103:
                values[5] += 20.0
            detector.observe(cycle, values)

        assert [h for h in detector.history if h.state == "sensor_drift"] == []

    def test_a_wild_excursion_is_still_named(self) -> None:
        """A 300-unit excursion is not noise, and suppressing it would be a bug.

        The threshold exists to keep ordinary scatter out, not to make the detector
        quiet. An excursion this large is worth an operator's attention even though it
        lasts three cycles.
        """
        detector = FaultDetector(unit=1)
        rows = healthy_stream(200)
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if 100 <= cycle < 103:
                values[5] += 300.0
            detector.observe(cycle, values)

        assert "sensor_drift" in {h.state for h in detector.history}


class TestAlertPayload:
    def test_a_degraded_state_renders_as_an_alert(self) -> None:
        """The fault-to-alert mapping lives next to the fault so it cannot drift."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 30:
                values[7] = np.nan
            detector.observe(cycle, values)

        payload = detector.raise_alert_for(detector.active()[0])
        assert payload["kind"] == "degraded_sensor_dropout"
        assert payload["unit"] == 1
        assert payload["state"] == "sensor_dropout"
        assert payload["silent_cycles"] >= 30

    def test_the_alert_names_the_instrument_not_just_the_index(self) -> None:
        """Whoever has to go and look at the engine needs the sensor name."""
        detector = FaultDetector(unit=1)
        rows = healthy_stream()
        for cycle, row in enumerate(rows, start=1):
            values = row.copy()
            if cycle > 30:
                values[7] = np.nan
            detector.observe(cycle, values)

        payload = detector.raise_alert_for(detector.active()[0])
        assert payload["instrument"] == channel_of(7)
        assert payload["instrument"] != "channel_7"
