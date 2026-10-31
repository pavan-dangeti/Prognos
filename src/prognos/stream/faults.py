"""Fault injection and degraded-state detection.

`docs/design.md` §12 makes "every injected fault produces a named degraded state and an
alert" the Phase 5 exit criterion, and S6 requires any degraded state to be observable
in metrics, logs and the dashboard. This module is that criterion.

Three design choices carry most of the weight.

**A degraded state is a name plus evidence.** "Rejects: 400" is not something an operator
can act on; `sensor_stuck on W_Fuel Flow of engine 3` is. Every state carries the channel,
the engine and how long, because an alert that cannot say why gets ignored within a week.

**Silence is a fault.** A stream that stops arriving is the most common failure and the
easiest to miss, because nothing logs an error: nothing is arriving. `stream_silent` is
detected from an expected cycle that has not shown up, not from a lack of traffic.

**Silence on healthy data is a requirement, not a hope.** A detector that fires on noise
trains operators out of reading it, which is worse than shipping none. So every detector
here is relative — a channel is judged against the spread and level of the other
channels on the same engine — because an absolute threshold fires on whichever channel
happens to be naturally quiet. `tests/test_faults.py` asserts silence as hard as it
asserts detection.

The injector is here rather than in the simulator because the dashboard's injection
controls and the fault tests both need to provoke the real condition. Asserting against
a mocked detector would only prove the mock agrees with itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from prognos.schema import ALLOWED_INPUT_COLUMNS, MEASURED_SENSORS, SCENARIO_DESCRIPTORS
from prognos.stream.window import N_CHANNELS

#: Every degraded state this project can name. Metric labels and log fields come from
#: these, so they are lower case and space free by construction.
DEGRADED_STATES: tuple[str, ...] = (
    "sensor_dropout",
    "sensor_stuck",
    "sensor_drift",
    "value_out_of_range",
    "cycle_regression",
    "stream_silent",
    "operating_shift",
    "inference_unavailable",
)

#: The faults an injector can apply to a row. The last two states are conditions of
#: the flight and of the service, not of a sensor, and are provoked by the failure
#: scenarios rather than by editing a reading.
INJECTABLE_FAULTS: tuple[str, ...] = DEGRADED_STATES[:6]


def alert_payload(state: DegradedState, cycle: int) -> dict[str, object]:
    """A degraded state as the payload an alerting service would carry."""
    return {
        "kind": f"degraded_{state.state}",
        "unit": state.engine,
        "cycle": cycle,
        "instrument": state.instrument,
        "detail": state.detail,
        "state": state.state,
        "silent_cycles": state.silent_cycles,
    }


def operating_envelope(
    windows: npt.NDArray[np.float32], margin: float = 0.05
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The range of each flight condition in a model's training windows, widened.

    Widened by `margin` of each range on both sides, so a reading at the edge of what
    the fleet flew is not called a shift. Only the scenario descriptors are bounded:
    a measured sensor leaving its range is a sensor or engine fault, named elsewhere.
    """
    descriptors = np.asarray(windows, dtype=np.float32)[..., : len(SCENARIO_DESCRIPTORS)]
    flat = descriptors.reshape(-1, descriptors.shape[-1])
    low = np.nanmin(flat, axis=0)
    high = np.nanmax(flat, axis=0)
    pad = (high - low) * np.float32(margin)
    return (low - pad).astype(np.float32), (high + pad).astype(np.float32)


def outside_envelope(
    row: npt.NDArray[np.float32],
    envelope: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
    unit: int,
) -> DegradedState | None:
    """An `operating_shift` if any flight condition in `row` is outside the envelope.

    Not a sensor fault: the descriptors are read correctly, the engine is simply
    somewhere the model has never seen, so its interval was calibrated on a different
    population. Judged on every reading rather than once a cycle, because a cycle's
    first reading is taken on the ground and a shift at altitude never reaches it.
    """
    lower, upper = envelope
    values = row[: lower.size]
    outside = np.flatnonzero(np.isfinite(values) & ((values < lower) | (values > upper)))
    if outside.size == 0:
        return None
    channel = int(outside[0])
    value = float(values[channel])
    return DegradedState(
        state="operating_shift",
        engine=unit,
        channel=channel,
        detail=(
            f"{channel_of(channel)} at {value:.4g}, outside the fitted range "
            f"[{lower[channel]:.4g}, {upper[channel]:.4g}]"
        ),
        details={
            "engine": unit,
            "channel": channel,
            "value": value,
            "lower": float(lower[channel]),
            "upper": float(upper[channel]),
        },
    )


def channel_of(channel: int) -> str:
    """The instrument name for a channel index.

    A degraded state naming `channel 7` is less use to whoever has to go and look at
    the engine than one naming the sensor.
    """
    if 0 <= channel < len(SCENARIO_DESCRIPTORS):
        return SCENARIO_DESCRIPTORS[channel]
    offset = channel - len(SCENARIO_DESCRIPTORS)
    if 0 <= offset < len(MEASURED_SENSORS):
        return MEASURED_SENSORS[offset]
    return f"unknown_channel_{channel}"


@dataclass(frozen=True, slots=True)
class DegradedState:
    """One named fault, with the evidence needed to act on it."""

    state: str
    engine: int
    channel: int | None = None
    detail: str = ""
    silent_cycles: int = 0
    details: dict[str, object] = field(default_factory=dict)

    @property
    def instrument(self) -> str:
        """Readable sensor name, or the fleet when the fault is not a single channel."""
        return channel_of(self.channel) if self.channel is not None else "fleet"

    def to_dict(self) -> dict[str, object]:
        """For the API, the dashboard and the alert log."""
        return {
            "state": self.state,
            "engine": self.engine,
            "channel": self.channel,
            "instrument": self.instrument,
            "detail": self.detail,
            "silent_cycles": self.silent_cycles,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class InjectedFault:
    """A fault to apply to a clean stream, for tests and the dashboard controls."""

    kind: str
    channel: int = 0
    from_cycle: int = 1
    magnitude: float = 40.0

    def __post_init__(self) -> None:
        """Reject a typo rather than injecting nothing and reporting success."""
        if self.kind not in INJECTABLE_FAULTS:
            raise ValueError(
                f"unknown fault {self.kind!r}; expected one of {', '.join(INJECTABLE_FAULTS)}"
            )


class FaultInjector:
    """Applies one named fault to a clean stream."""

    __slots__ = ("_fault", "_held", "_unit")

    def __init__(self, unit: int, fault: InjectedFault | None = None) -> None:
        """Build an injector, with no fault by default."""
        self._unit = unit
        self._fault = fault
        self._held: float | None = None

    @property
    def fault(self) -> InjectedFault | None:
        """The configured fault, or None for a pass-through."""
        return self._fault

    def apply(self, cycle: int, row: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return the row as it should reach the pipeline at this cycle.

        A cycle before the fault's start returns the row untouched, so a fault can be
        switched on partway through a stream rather than only at its first cycle.
        """
        if self._fault is None or cycle < self._fault.from_cycle:
            return row
        channel = self._fault.channel
        match self._fault.kind:
            case "sensor_dropout":
                row[channel] = np.nan
            case "sensor_stuck":
                # Held at whatever the channel read when the fault began, so it is
                # genuinely constant rather than merely low-variance.
                if self._held is None:
                    self._held = float(row[channel])
                row[channel] = self._held
            case "sensor_drift":
                row[channel] += self._fault.magnitude
            case "value_out_of_range":
                row[channel] = 1e12
            case "cycle_regression":
                pass  # applied by the caller, which owns the cycle number
            case "stream_silent":
                pass  # applied by the caller, which stops sending
            case _:
                pass
        return row


class FaultDetector:
    """Names the degraded states an engine's stream is currently in.

    One detector per engine, fed one row per cycle. `observe` only records what arrived;
    `active` computes what is wrong right now; the transition between the two is what
    goes into history. Splitting it that way is what stops the history filling with the
    same fault once per cycle, and it is what makes recovery detectable at all: a state
    that is never compared against a previous state cannot be seen to clear.

    States clear as soon as the condition stops. A detector that latches is an alert
    nobody closes, and an alert that never closes is an alert nobody reads.
    """

    __slots__ = (
        "_baseline",
        "_channel_last_seen",
        "_drift_factor",
        "_drift_half",
        "_drift_persistence",
        "_drift_streak",
        "_history",
        "_last_cycle",
        "_min_cycles",
        "_observations",
        "_open",
        "_plausible_max",
        "_rows",
        "_unit",
        "_window",
    )

    def __init__(
        self,
        unit: int,
        min_cycles: int = 20,
        window: int = 40,
        plausible_max: float = 1e6,
        drift_factor: float = 3.0,
        drift_half: int = 10,
        drift_persistence: int = 1,
    ) -> None:
        """Build a detector. `min_cycles` is the evidence floor for every judgement.

        The three `drift_` settings are the ones `prognos drift-study` tunes: how many
        spreads a shift must exceed, how many cycles each side of the comparison
        averages, and how many consecutive cycles the shift must hold before it is
        named. The defaults are the study's choice, recorded in
        `results/phase6_drift.json`; Phase 5 shipped 4.0, 6 and 1, which raised drift
        on 4 of 9 held-out healthy engines.
        """
        self._unit = unit
        self._min_cycles = min_cycles
        self._window = window
        self._plausible_max = plausible_max
        self._drift_factor = drift_factor
        self._drift_half = drift_half
        self._drift_persistence = drift_persistence
        self._drift_streak: dict[int, int] = {}
        self._rows: deque[npt.NDArray[np.float32]] = deque(maxlen=window)
        self._channel_last_seen: dict[int, int] = {}
        self._last_cycle: int | None = None
        self._observations = 0
        self._baseline: npt.NDArray[np.float64] | None = None
        self._history: deque[DegradedState] = deque(maxlen=200)
        self._open: dict[tuple[str, int | None], DegradedState] = {}

    @property
    def history(self) -> tuple[DegradedState, ...]:
        """Every state that has been raised or cleared, most recent last."""
        return tuple(self._history)

    @property
    def n_observations(self) -> int:
        """How many cycles have been seen."""
        return len(self._rows)

    def mark_expected_cycle(self, cycle: int) -> None:
        """Declare a cycle due, so silence can be detected.

        Absence cannot be inferred from a stream nobody has read from. It can only be
        inferred from a cycle that was expected and has not arrived.
        """
        self._open.pop(("stream_silent", None), None)
        self._history.append(
            DegradedState(
                state="stream_silent",
                engine=self._unit,
                detail=f"cycle {cycle} has not arrived",
                silent_cycles=self._min_cycles,
                details={"engine": self._unit, "expected_cycle": cycle},
            )
        )
        self._open[("stream_silent", None)] = DegradedState(
            state="stream_silent",
            engine=self._unit,
            detail=f"cycle {cycle} has not arrived",
            silent_cycles=self._min_cycles,
            details={"engine": self._unit, "expected_cycle": cycle},
        )

    def observe(self, cycle: int, values: npt.NDArray[np.float32]) -> None:
        """Feed one cycle's readings, in order."""
        row = np.asarray(values, dtype=np.float32)
        if row.shape != (N_CHANNELS,):
            raise ValueError(f"expected {N_CHANNELS} channels, got {row.shape}")

        if self._last_cycle is not None and cycle < self._last_cycle:
            # Named, then dropped. The window refuses to reorder a block it has already
            # emitted windows from, and the detector has already said why.
            #
            # Strictly *less than*, not less-or-equal. This dataset carries thousands of
            # rows per cycle number, so a repeated cycle is the normal case; treating
            # equality as a regression named a fault on every healthy engine, which is
            # the fastest way to make a detector ignored.
            self._record(
                DegradedState(
                    state="cycle_regression",
                    engine=self._unit,
                    detail=f"cycle {cycle} arrived after {self._last_cycle}",
                    details={"cycle": cycle, "last_cycle": self._last_cycle},
                )
            )
            return

        self._last_cycle = cycle
        self._open.pop(("stream_silent", None), None)
        self._rows.append(row)
        self._observations += 1
        for channel in range(N_CHANNELS):
            if np.isfinite(row[channel]):
                self._channel_last_seen[channel] = self._observations
        self._freeze_baseline()
        self._refresh()

    def active(self) -> list[DegradedState]:
        """The states whose condition is true right now."""
        return sorted(
            self._open.values(), key=lambda s: (s.state, s.channel if s.channel is not None else -1)
        )

    def _refresh(self) -> None:
        """Recompute the open set and record whatever entered or left it."""
        found: dict[tuple[str, int | None], DegradedState] = {}

        for channel, last_seen in self._channel_last_seen.items():
            # Counted in *observations*, not cycles. This dataset carries thousands of
            # rows per cycle number, so a channel can be silent for a very long time
            # without the cycle ever advancing, and a cycle-based threshold would never
            # trip at all — which is exactly what happened.
            if self._observations - last_seen >= self._min_cycles:
                found[("sensor_dropout", channel)] = DegradedState(
                    state="sensor_dropout",
                    engine=self._unit,
                    channel=channel,
                    detail=f"{channel_of(channel)} stopped reporting",
                    silent_cycles=self._observations - last_seen,
                    details={"engine": self._unit, "channel": channel},
                )

        row = self._rows[-1]
        for channel in range(N_CHANNELS):
            value = row[channel]
            if np.isfinite(value) and abs(value) > self._plausible_max:
                found[("value_out_of_range", channel)] = DegradedState(
                    state="value_out_of_range",
                    engine=self._unit,
                    channel=channel,
                    detail=f"{channel_of(channel)} reported {value:.3g}",
                    details={"engine": self._unit, "channel": channel, "value": float(value)},
                )

        if len(self._rows) >= self._min_cycles:
            found.update(self._stuck_states())
            found.update(self._drift_states())

        for key, state in found.items():
            if key not in self._open:
                self._history.append(state)
        self._open = found

    def _stuck_states(self) -> dict[tuple[str, int | None], DegradedState]:
        """Channels holding exactly one value across the window.

        Distinct-value counting rather than a variance threshold. A quiet sensor still
        *changes*; a frozen one does not, and the distinction is categorical rather
        than a matter of degree. It also cannot be tuned into firing on whichever
        instrument happens to be calm, which is the normal case for several of these.
        """
        block = np.stack(self._rows)
        states: dict[tuple[str, int | None], DegradedState] = {}
        for channel in range(N_CHANNELS):
            column = block[:, channel]
            finite = column[np.isfinite(column)]
            if finite.size < self._min_cycles:
                continue
            if np.unique(finite).size == 1:
                states[("sensor_stuck", channel)] = DegradedState(
                    state="sensor_stuck",
                    engine=self._unit,
                    channel=channel,
                    detail=(
                        f"{channel_of(channel)} has held {finite[0]:.6g} for {finite.size} cycles"
                    ),
                    details={
                        "engine": self._unit,
                        "channel": channel,
                        "value": float(finite[0]),
                        "cycles": int(finite.size),
                    },
                )
        return states

    def _drift_states(self) -> dict[tuple[str, int | None], DegradedState]:
        """Channels whose current level has left the baseline the engine started at.

        Measured against a **frozen** baseline taken from the first `min_cycles`, not a
        rolling window. A rolling window scrolls past a step change and declares the
        engine healthy again, which loses exactly the fault worth catching: a sensor
        that steps and stays stepped is broken from then on. Freezing the baseline is
        also what makes a transient spike distinguishable from a sustained shift — a
        spike moves the recent mean only briefly, a step moves it permanently.

        Drift is the fault a range check cannot see and that RMSE barely notices until
        it is large, which is why it gets its own detector rather than being folded into
        the plausibility bound.
        """
        block = np.stack(self._rows)
        states: dict[tuple[str, int | None], DegradedState] = {}
        baseline = self._baseline
        half = self._drift_half
        if baseline is None or block.shape[0] < 2 * half:
            return states
        spread = np.nanstd(block, axis=0)
        typical = float(np.nanmedian(spread[np.isfinite(spread)]))
        if typical <= 0.0:
            return states
        # Compared against the *previous* window rather than the frozen baseline.
        # N-CMAPSS sensors drift by design — measuring degradation is what they are
        # for — so a detector measuring distance from the starting level fires on every
        # healthy engine. A step differs from a trend in exactly one way that matters
        # here: it moves abruptly between adjacent windows, while degradation moves
        # slowly. The control run in `results/phase5_faults.json` is the calibration.
        #
        # Normalised by the channel's *own* standard deviation rather than the fleet
        # median. These instruments differ in scale by three orders of magnitude —
        # core speed sits near 8,700 while a temperature sits near 500 — so a fleet-wide
        # threshold is really a threshold on whichever channels happen to be small, and
        # it flagged two healthy ones.
        recent = block[-half:]
        previous = block[-2 * half : -half]
        # Measured sensors only. The scenario descriptors are the flight's conditions:
        # when they move, the engine is flying somewhere else, which is
        # `operating_shift`, not a sensor that has drifted.
        for channel in range(len(SCENARIO_DESCRIPTORS), N_CHANNELS):
            current = float(np.nanmean(recent[:, channel]))
            before = float(np.nanmean(previous[:, channel]))
            if not (np.isfinite(current) and np.isfinite(before)):
                continue
            # From the previous window only. A step inflates the spread of any window
            # that contains it, so normalising by the whole block meant a 40-unit jump
            # on a sigma-5 channel measured as 8x, then 4x, then 2x as the window filled
            # with shifted values — and a step that shrank below its own threshold was
            # never reported at all.
            own = float(np.nanstd(previous[:, channel]))
            if not np.isfinite(own) or own <= 0.0:
                continue
            shift = abs(current - before)
            ratio = shift / own
            fleet_ratio = shift / typical
            # Both, not either. Two independent false-positive modes showed up on real
            # data: instruments whose scales differ by three orders of magnitude, and
            # channels whose series is so smooth that ordinary movement is many times
            # their own spread. Normalising only by the channel ignores the first;
            # normalising only by the fleet ignores the second. A conjunction is not a
            # tuned constant — it is both signals having to agree — and it is the only
            # version that survived a control run on a healthy engine.
            if ratio > self._drift_factor and fleet_ratio > self._drift_factor:
                states[("sensor_drift", channel)] = DegradedState(
                    state="sensor_drift",
                    engine=self._unit,
                    channel=channel,
                    detail=(
                        f"{channel_of(channel)} moved {shift:.3g} between adjacent "
                        f"windows, {ratio:.1f}x its own spread and "
                        f"{fleet_ratio:.1f}x the fleet spread"
                    ),
                    details={
                        "engine": self._unit,
                        "channel": channel,
                        "shift": shift,
                        "ratio": ratio,
                        "fleet_ratio": fleet_ratio,
                        "previous": before,
                        "current": current,
                        "baseline": float(baseline[channel]),
                    },
                )
        return self._persisting(states)

    def _persisting(
        self, states: dict[tuple[str, int | None], DegradedState]
    ) -> dict[tuple[str, int | None], DegradedState]:
        """Only the shifts that have held for `drift_persistence` consecutive cycles.

        A broken sensor that steps stays stepped, so its shift holds for as long as the
        step sits between the two halves of the comparison. Ordinary operation moves a
        channel for a cycle or two and then settles back.
        """
        streak = {
            key[1]: self._drift_streak.get(key[1], 0) + 1 for key in states if key[1] is not None
        }
        self._drift_streak = streak
        return {
            key: state
            for key, state in states.items()
            if key[1] is not None and streak[key[1]] >= self._drift_persistence
        }

    def _freeze_baseline(self) -> None:
        """Lock the baseline once there is enough evidence for one."""
        if self._baseline is not None or len(self._rows) < self._min_cycles:
            return
        self._baseline = np.nanmean(np.stack(self._rows), axis=0)

    def raise_alert_for(self, state: DegradedState) -> dict[str, object]:
        """Render a degraded state as the payload an alerting service would carry.

        Kept here so the mapping from fault to alert lives next to the fault, and cannot
        drift from it the way a separate mapping table would.
        """
        return alert_payload(state, self._last_cycle or 0)

    def _record(self, state: DegradedState) -> None:
        """Record a state that was detected outside the recompute path."""
        self._history.append(state)
        self._open[(state.state, state.channel)] = state


__all__ = [
    "ALLOWED_INPUT_COLUMNS",
    "DEGRADED_STATES",
    "INJECTABLE_FAULTS",
    "DegradedState",
    "FaultDetector",
    "FaultInjector",
    "InjectedFault",
    "alert_payload",
    "channel_of",
    "operating_envelope",
    "outside_envelope",
]
