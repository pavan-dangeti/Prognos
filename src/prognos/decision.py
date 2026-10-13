"""Cost-based maintenance policy.

Given an engine's predicted remaining life, the question is not "what is the RUL"
but "should we replace this engine now". The two are not the same question, because
the decisions are asymmetric: replacing an engine that still has life costs a shop
visit and some downtime, while running one past its life costs an in-flight removal,
which is worse by an order of magnitude and is unbounded in the tail.

So the policy acts on the **lower bound** of the interval, not the point estimate.
An early replacement wastes a bounded, known cost. An in-flight removal does not.

**The cost model is scale-free.** Planned maintenance is normalised to 1.0 and the
failure/planned ratio is the parameter that matters, because every policy in a
comparison flies the same fleet over the same horizon, so every absolute term cancels
out of the difference. Reporting a curve against the ratio rather than one number at
a guessed ratio is what makes the conclusion survive being wrong about the guess.

**Where this departs from the design document.** §4.5 lists a
`replacement_engine_life` parameter for charging replacement cost across a
replacement's life. That number is not used here, because the simulator already knows
how long an engine lasts: it draws replacements from the pool of real observed
engine episodes. Declaring a lifetime and then not using it would be a knob that
looks load-bearing and is not, so it is gone rather than carried along. Cost is
charged in full at the event rather than amortised over the delivered life, which
matters only near the end of a horizon and is reported as a limitation rather than
papered over.

The simulator is economics, not measurement. It treats the observed engines as a
sample of engines rather than as the entire population, which is an exchangeability
assumption rather than a physical one, and it is the first thing a reader should
attack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "CostModel",
    "EngineEpisode",
    "FixedIntervalPolicy",
    "FleetOutcome",
    "Policy",
    "RunToFailurePolicy",
    "ThresholdPolicy",
    "simulate_fleet",
    "sweep_failure_ratio",
]


class CostModel:
    """Cost terms for the maintenance comparison.

    Costs are in arbitrary units with planned maintenance normalised to 1.0. That is
    not a shortcut, it is the observation that absolute scale cancels: every policy
    is compared on the same fleet over the same horizon, so only the ratio between
    terms changes the answer.
    """

    __slots__ = ("_downtime_per_cycle", "_planned", "_ratio", "_window_cycles")

    def __init__(
        self,
        planned_maintenance_cost: float,
        failure_cost_ratio: float,
        downtime_cost_per_cycle: float,
        maintenance_window_cycles: int,
    ) -> None:
        """Build a cost model, rejecting terms that would invert the comparison."""
        if planned_maintenance_cost <= 0:
            raise ValueError(
                f"planned_maintenance_cost must be positive, got {planned_maintenance_cost}"
            )
        if failure_cost_ratio <= 0:
            raise ValueError(
                f"failure_cost_ratio must be positive; zero would make an in-flight "
                f"removal free, which is incoherent. Got {failure_cost_ratio}."
            )
        if downtime_cost_per_cycle < 0:
            raise ValueError(
                f"downtime_cost_per_cycle must not be negative; a negative rate would "
                f"reward visiting more often. Got {downtime_cost_per_cycle}."
            )
        if maintenance_window_cycles < 1:
            raise ValueError(
                f"maintenance_window_cycles must be at least 1, got {maintenance_window_cycles}"
            )
        self._planned = float(planned_maintenance_cost)
        self._ratio = float(failure_cost_ratio)
        self._downtime_per_cycle = float(downtime_cost_per_cycle)
        self._window_cycles = int(maintenance_window_cycles)

    @property
    def planned_maintenance_cost(self) -> float:
        """Cost of a scheduled visit, excluding ground time."""
        return self._planned

    @property
    def failure_cost_ratio(self) -> float:
        """Unplanned removal cost as a multiple of a planned visit."""
        return self._ratio

    @property
    def downtime_cost_per_cycle(self) -> float:
        """Cost of one cycle an engine spends on the ground."""
        return self._downtime_per_cycle

    @property
    def maintenance_window_cycles(self) -> int:
        """Ground time consumed by one visit, in cycles."""
        return self._window_cycles

    def planned_visit_cost(self) -> float:
        """A scheduled replacement: the visit plus the ground time it consumes.

        Ground time is charged to planned and unplanned work at the same rate. The
        asymmetry between them is carried entirely by `failure_cost_ratio`, so the
        downtime rate does not smuggle in a second hidden assumption.
        """
        return self._planned + self._downtime_per_cycle * self._window_cycles

    def unplanned_failure_cost(self) -> float:
        """An in-flight or unplanned removal: the removal plus the ground time."""
        return self._planned * self._ratio + self._downtime_per_cycle * self._window_cycles

    def with_ratio(self, failure_cost_ratio: float) -> CostModel:
        """A copy of this model at a different failure/planned ratio."""
        return CostModel(
            planned_maintenance_cost=self._planned,
            failure_cost_ratio=failure_cost_ratio,
            downtime_cost_per_cycle=self._downtime_per_cycle,
            maintenance_window_cycles=self._window_cycles,
        )

    def as_dict(self) -> dict[str, float | int | str]:
        """Every assumption, for recording alongside any cost result.

        The absolute values are a free choice because they cancel; recording them
        anyway means a reader can see what was held constant.
        """
        return {
            "planned_maintenance_cost": self._planned,
            "failure_cost_ratio": self._ratio,
            "downtime_cost_per_cycle": self._downtime_per_cycle,
            "maintenance_window_cycles": self._window_cycles,
            "planned_visit_cost": self.planned_visit_cost(),
            "unplanned_failure_cost": self.unplanned_failure_cost(),
            "units": "arbitrary, planned maintenance normalised to 1.0",
        }


@runtime_checkable
class Policy(Protocol):
    """Decides whether an engine should be replaced this cycle."""

    @property
    def name(self) -> str:
        """Stable identifier, used as a key in the result file."""
        ...

    def should_replace(self, lower: float, point: float, cycle: int) -> bool:
        """Whether to replace now, given this cycle's interval and estimate."""
        ...


class RunToFailurePolicy:
    """Replace on failure and not before. The baseline every policy is measured against."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "run_to_failure"

    def should_replace(self, lower: float, point: float, cycle: int) -> bool:
        """Never."""
        return False


class FixedIntervalPolicy:
    """Replace every `interval` cycles, ignoring what the model predicts.

    The calendar alternative. It exists to answer "would a schedule have done as
    well", which is the comparison an interval policy has to survive.
    """

    __slots__ = ("_interval",)

    def __init__(self, interval: int) -> None:
        """Build a calendar policy with a positive replacement period."""
        if interval < 1:
            raise ValueError(f"interval must be at least 1 cycle, got {interval}")
        self._interval = int(interval)

    @property
    def name(self) -> str:
        """Stable identifier, carrying the period so results cannot be conflated."""
        return f"fixed_interval_{self._interval}"

    def should_replace(self, lower: float, point: float, cycle: int) -> bool:
        """True on every multiple of the interval."""
        return cycle % self._interval == 0


class ThresholdPolicy:
    """Replace when a prediction falls to or below a threshold.

    `use_lower_bound` selects which edge of the interval drives the decision. The
    lower-bound variant is the one under test; the point-estimate variant is the
    comparison, and the two are separate objects because conflating them is exactly
    the mistake the comparison exists to rule out.
    """

    __slots__ = ("_threshold", "_use_lower")

    def __init__(self, threshold: float, use_lower_bound: bool = True) -> None:
        """Build a threshold policy on a non-negative RUL threshold."""
        if threshold < 0:
            raise ValueError(
                f"threshold must be non-negative; remaining useful life never is, so a "
                f"negative threshold could never fire. Got {threshold}."
            )
        self._threshold = float(threshold)
        self._use_lower = bool(use_lower_bound)

    @property
    def name(self) -> str:
        """Stable identifier, carrying which edge drives the decision."""
        edge = "lower" if self._use_lower else "point"
        return f"threshold_{edge}_{self._threshold:g}"

    def should_replace(self, lower: float, point: float, cycle: int) -> bool:
        """True when the chosen edge has fallen to the threshold."""
        return (lower if self._use_lower else point) <= self._threshold


@dataclass(frozen=True, slots=True)
class EngineEpisode:
    """One engine's real life, with the predictions available along it.

    `lower` and `point` are indexed by cycles elapsed: entry `t` is the prediction
    made after `t` cycles of flight. `None` means no prediction was available, which
    is the case for cycles the window builder could not produce.

    A frozen dataclass because an episode is a record of what happened. A mutable one
    would let the simulator quietly rewrite history and then report on it.
    """

    engine: str
    true_life: int
    lower: npt.NDArray[np.float64] | None = None
    point: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate the episode's shape against its own life."""
        if self.true_life < 1:
            raise ValueError(f"engine {self.engine!r} has a non-positive true life")
        for name, values in (("lower", self.lower), ("point", self.point)):
            if values is not None and values.shape != (self.true_life,):
                raise ValueError(
                    f"engine {self.engine!r}: {name} has shape {values.shape}, "
                    f"expected ({self.true_life},) to match its true life"
                )

    def prediction_at(self, cycles_elapsed: int) -> tuple[float, float] | None:
        """The interval bound and point estimate at a given age, or None.

        None means no prediction covers this cycle: either the cycle is outside the
        episode, or no window overlapped it. The simulator skips such a cycle rather
        than inventing a prediction, which is what keeps a gap in the observation
        record from becoming a decision nobody made.
        """
        if self.lower is None or self.point is None:
            return None
        if not 0 <= cycles_elapsed < self.true_life:
            return None
        lower = float(self.lower[cycles_elapsed])
        point = float(self.point[cycles_elapsed])
        if math.isnan(lower) or math.isnan(point):
            return None
        return lower, point


@dataclass(frozen=True, slots=True)
class FleetOutcome:
    """What one policy cost over one horizon on one fleet."""

    policy: str
    horizon: int
    total_cost: float
    n_planned: int
    n_failures: int
    n_engines: int
    cost_model: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def cost_per_engine(self) -> float:
        """Cost normalised by engine count, so fleets of different size compare."""
        if self.n_engines == 0:
            raise ValueError("cannot normalise an outcome with no engines")
        return self.total_cost / self.n_engines


def simulate_fleet(
    fleet: list[EngineEpisode],
    policy: Policy,
    cost_model: CostModel,
    horizon: int,
    seed: int = 0,
) -> FleetOutcome:
    """Fly a fleet for `horizon` cycles under one policy and cost the result.

    One engine flies at a time and is replaced by drawing the next episode from the
    pool. Episodes are drawn in order, cycling through the pool, so the simulation
    is deterministic and a regression in the numbers is traceable to a change rather
    than to a different random seed.

    Replacements and failures both charge ground time at the same rate, so the only
    thing distinguishing them is `failure_cost_ratio`. That is deliberate: it means
    the sweep varies one quantity, and the curve it produces is attributable.

    Raises when the horizon is shorter than the shortest engine in the fleet. In that
    case no engine can reach the end of its life, so every policy records zero
    failures, run-to-failure costs nothing, and the comparison is vacuous. A zero
    cost is arithmetically correct and completely meaningless, so it is refused.
    """
    if not fleet:
        raise ValueError("cannot simulate an empty fleet")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 cycle, got {horizon}")
    shortest = min(episode.true_life for episode in fleet)
    if horizon < shortest:
        raise ValueError(
            f"horizon {horizon} is shorter than the shortest engine life {shortest}, "
            "so no engine can fail inside it and every policy would cost nothing"
        )

    total_cost = 0.0
    n_planned = 0
    n_failures = 0
    engines_used = 0

    cycle = 0
    while cycle < horizon:
        episode = fleet[cycle % len(fleet)]
        engines_used += 1
        age = 0
        replaced = False

        while cycle < horizon and age < episode.true_life:
            prediction = episode.prediction_at(age)
            if prediction is not None:
                lower, point = prediction
                if policy.should_replace(lower=lower, point=point, cycle=cycle + 1):
                    total_cost += cost_model.planned_visit_cost()
                    n_planned += 1
                    # The engine is on the ground for the visit, flying nothing.
                    cycle += cost_model.maintenance_window_cycles
                    replaced = True
                    break
            age += 1
            cycle += 1

        # Only a failure when the engine ran out of life rather than running out of
        # horizon. Charging on the horizon exit would bill an in-flight removal for
        # an engine that is still flying, and would make every policy look worse the
        # closer it runs to the end of the horizon.
        if not replaced and age >= episode.true_life:
            total_cost += cost_model.unplanned_failure_cost()
            n_failures += 1
            cycle += cost_model.maintenance_window_cycles

    return FleetOutcome(
        policy=policy.name,
        horizon=horizon,
        total_cost=total_cost,
        n_planned=n_planned,
        n_failures=n_failures,
        n_engines=engines_used,
        cost_model=cost_model.as_dict(),
    )


def sweep_failure_ratio(
    fleet: list[EngineEpisode],
    policy: Policy,
    cost_model: CostModel,
    horizon: int,
    ratios: list[float],
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Re-run the same policy and fleet at each failure/planned ratio.

    Each row is an independent simulation, not a rescaling of one result. Cost is
    linear in the terms, so a rescaled number would be cheaper to compute and would
    also be untestable, and the shape of the curve is the headline result.

    Savings are measured against run-to-failure **at the same ratio**. Reusing one
    baseline across the sweep would compare a policy run at ratio 100 against a
    baseline run at ratio 10, which is the difference between a large saving and a
    large loss depending only on the row.
    """
    rows: list[dict[str, float | int | str]] = []
    for ratio in ratios:
        model = cost_model.with_ratio(ratio)
        outcome = simulate_fleet(fleet, policy, model, horizon=horizon, seed=seed)
        baseline = simulate_fleet(fleet, RunToFailurePolicy(), model, horizon=horizon, seed=seed)
        rows.append(
            {
                "failure_cost_ratio": ratio,
                "policy": outcome.policy,
                "total_cost": outcome.total_cost,
                "cost_per_engine": outcome.cost_per_engine,
                "n_planned": outcome.n_planned,
                "n_failures": outcome.n_failures,
                "n_engines": outcome.n_engines,
                "run_to_failure_cost": baseline.total_cost,
                "savings_vs_run_to_failure": baseline.total_cost - outcome.total_cost,
            }
        )
    return rows
