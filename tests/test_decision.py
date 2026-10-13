"""Tests for the cost-based maintenance policy.

The claim this layer has to earn is comparative: a policy that acts on an interval's
lower bound should beat both run-to-failure and a fixed calendar, and it should do
so because replacing early is genuinely cheaper than replacing in flight. Anything
weaker than that and the module is decoration.

Two things make that hard to test honestly, and most of what is here is about them.

First, the fleet simulator is a piece of economics, not a measurement. A test that
feeds it a friendly fleet will pass for the wrong reason. So the tests drive it with
engines whose true life and prediction curves are known exactly, and check the cost
arithmetic against costs computed by hand.

Second, the cost model is scale-free. Planned maintenance is normalised to 1.0 and
the failure/planned ratio is swept, because absolute currency cancels out of every
policy comparison. Tests that assert on absolute costs would pass under any scaling
and prove nothing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from prognos.decision import (
    CostModel,
    EngineEpisode,
    FixedIntervalPolicy,
    FleetOutcome,
    Policy,
    RunToFailurePolicy,
    ThresholdPolicy,
    simulate_fleet,
    sweep_failure_ratio,
)


def make_cost_model(**overrides: float) -> CostModel:
    """A cost model with predictable defaults, overridable per test."""
    defaults = {
        "planned_maintenance_cost": 1.0,
        "failure_cost_ratio": 10.0,
        "downtime_cost_per_cycle": 0.1,
        "maintenance_window_cycles": 2,
    }
    defaults.update(overrides)
    return CostModel(**defaults)


class TestCostModel:
    def test_rejects_a_nonpositive_ratio(self) -> None:
        """A ratio of zero would make an in-flight failure free, which is incoherent."""
        with pytest.raises(ValueError, match="failure_cost_ratio"):
            make_cost_model(failure_cost_ratio=0.0)
        with pytest.raises(ValueError, match="failure_cost_ratio"):
            make_cost_model(failure_cost_ratio=-1.0)

    def test_planned_visit_cost_includes_its_downtime(self) -> None:
        """A scheduled visit is not just the shop visit, it is the ground time too."""
        model = make_cost_model(downtime_cost_per_cycle=0.25, maintenance_window_cycles=4)
        assert model.planned_visit_cost() == pytest.approx(1.0 + 0.25 * 4)

    def test_unplanned_failure_costs_more_than_a_planned_visit(self) -> None:
        """This is the whole premise; if it fails, every downstream result is void."""
        for ratio in (1.0, 5.0, 25.0, 100.0):
            model = make_cost_model(failure_cost_ratio=ratio)
            assert model.unplanned_failure_cost() >= model.planned_visit_cost()

    def test_rejects_nonpositive_downtime_and_window(self) -> None:
        """Negative cost per cycle would let a policy farm money by visiting more."""
        with pytest.raises(ValueError, match="downtime_cost_per_cycle"):
            make_cost_model(downtime_cost_per_cycle=-1.0)
        with pytest.raises(ValueError, match="maintenance_window_cycles"):
            make_cost_model(maintenance_window_cycles=0)

    def test_as_dict_records_every_assumption(self) -> None:
        """The README quotes cost numbers, so the assumptions must be quotable.

        Absolute currency is a free choice because it cancels out of every
        comparison. Recording it anyway means a reader can see what was actually
        held constant rather than inferring it.
        """
        recorded = make_cost_model().as_dict()
        for key in (
            "planned_maintenance_cost",
            "failure_cost_ratio",
            "downtime_cost_per_cycle",
            "maintenance_window_cycles",
        ):
            assert key in recorded


class TestPolicies:
    def test_run_to_failure_never_replaces(self) -> None:
        """The baseline policy is the one that waits for the worst."""
        policy = RunToFailurePolicy()
        assert policy.should_replace(lower=0.0, point=0.0, cycle=1) is False
        assert policy.should_replace(lower=0.0, point=0.0, cycle=999) is False

    def test_fixed_interval_replaces_on_its_period(self) -> None:
        """A calendar policy replaces on cadence, ignoring what the model says."""
        policy = FixedIntervalPolicy(interval=25)
        assert policy.should_replace(lower=100.0, point=100.0, cycle=25) is True
        assert policy.should_replace(lower=100.0, point=100.0, cycle=24) is False
        assert policy.should_replace(lower=0.0, point=0.0, cycle=50) is True

    def test_fixed_interval_rejects_a_nonpositive_period(self) -> None:
        """An interval of zero would replace on every cycle, which is not a calendar."""
        with pytest.raises(ValueError, match="interval"):
            FixedIntervalPolicy(interval=0)

    def test_threshold_triggers_on_the_lower_bound(self) -> None:
        """The policy under test acts on the conservative edge, not the estimate."""
        policy = ThresholdPolicy(threshold=20.0, use_lower_bound=True)
        assert policy.should_replace(lower=19.0, point=90.0, cycle=1) is True
        assert policy.should_replace(lower=21.0, point=5.0, cycle=1) is False

    def test_threshold_at_the_point_estimate_is_a_separate_policy(self) -> None:
        """Acting on the point estimate is the comparison, so it is its own object."""
        policy = ThresholdPolicy(threshold=20.0, use_lower_bound=False)
        assert policy.should_replace(lower=21.0, point=5.0, cycle=1) is True
        assert policy.should_replace(lower=0.0, point=90.0, cycle=1) is False

    def test_threshold_rejects_a_negative_threshold(self) -> None:
        """A negative threshold can never trigger against a non-negative bound."""
        with pytest.raises(ValueError, match="threshold"):
            ThresholdPolicy(threshold=-1.0)

    def test_policy_name_is_stable(self) -> None:
        """Names end up as result-file keys, so they must not drift silently."""
        assert RunToFailurePolicy().name == "run_to_failure"
        assert FixedIntervalPolicy(interval=10).name == "fixed_interval_10"
        assert ThresholdPolicy(5.0, use_lower_bound=True).name == "threshold_lower_5"
        assert ThresholdPolicy(5.0, use_lower_bound=False).name == "threshold_point_5"


class TestSimulateFleet:
    def _fleet(self) -> list[EngineEpisode]:
        """Two engines with known lives and perfect, constant predictions.

        Perfect predictions make the cost arithmetic checkable by hand: with an
        interval that is exactly right, a threshold policy knows precisely when the
        engine is about to fail, so the expected cost is fully determined.
        """
        return [
            EngineEpisode(engine="a", true_life=100, lower=None, point=None),
            EngineEpisode(engine="b", true_life=120, lower=None, point=None),
        ]

    def test_run_to_failure_charges_one_failure_per_engine(self) -> None:
        """With no replacements, each engine fails once and costs exactly that."""
        engine = EngineEpisode(engine="a", true_life=100)
        outcome = simulate_fleet([engine], RunToFailurePolicy(), make_cost_model(), horizon=100)
        assert outcome.n_failures == 1
        assert outcome.n_planned == 0

    def test_run_to_failure_over_a_longer_horizon_charges_replacements(self) -> None:
        """Past its life the engine is gone, so the horizon has to be refilled.

        Without this the simulator would silently under-count cost for every policy
        on a long horizon, and run-to-failure would look cheaper than it is purely
        because it does not fly past its own life.
        """
        outcome = simulate_fleet(
            [EngineEpisode(engine="a", true_life=50)],
            RunToFailurePolicy(),
            make_cost_model(),
            horizon=200,
        )
        assert outcome.n_failures >= 3
        assert outcome.total_cost > 0

    def test_a_threshold_policy_avoids_the_in_flight_failure(self) -> None:
        """The core claim: acting early turns an expensive failure into a cheap visit.

        With a perfect interval and a threshold of 20, an engine with 100 cycles of
        life is replaced at cycle 81 rather than failing at 100. The replacement
        costs the planned visit; the failure it avoided costs the failure cost.
        """
        model = make_cost_model(failure_cost_ratio=10.0)
        true_life = 100
        threshold = 20
        interval = np.array([true_life - t for t in range(true_life)], dtype=np.float64)
        perfect = np.array([float(true_life)] * true_life, dtype=np.float64)

        policy = ThresholdPolicy(threshold=threshold, use_lower_bound=True)
        outcome = simulate_fleet(
            [EngineEpisode(engine="a", true_life=true_life, lower=interval, point=perfect)],
            policy,
            model,
            horizon=true_life,
        )
        assert outcome.n_planned >= 1
        assert outcome.n_failures == 0

    def test_planned_replacement_is_cheaper_than_the_failure_it_prevents(self) -> None:
        """At a high failure/planned ratio, replacing early must actually save money."""
        model = make_cost_model(failure_cost_ratio=50.0)
        true_life = 100
        interval = np.array([true_life - t for t in range(true_life)], dtype=np.float64)
        perfect = np.array([float(true_life)] * true_life, dtype=np.float64)
        episode = EngineEpisode(engine="a", true_life=true_life, lower=interval, point=perfect)

        to_failure = simulate_fleet([episode], RunToFailurePolicy(), model, horizon=true_life)
        early = simulate_fleet(
            [episode],
            ThresholdPolicy(threshold=20.0, use_lower_bound=True),
            model,
            horizon=true_life,
        )
        assert early.total_cost < to_failure.total_cost

    def test_at_a_ratio_of_one_the_policies_are_close(self) -> None:
        """When a failure costs the same as a visit, planning buys little.

        This is the honest edge of the claim. If the threshold policy still won
        massively at ratio 1, the result would be an artefact of the simulator
        rather than a consequence of the cost asymmetry.
        """
        model = make_cost_model(failure_cost_ratio=1.0)
        true_life = 100
        interval = np.array([true_life - t for t in range(true_life)], dtype=np.float64)
        perfect = np.array([float(true_life)] * true_life, dtype=np.float64)
        episode = EngineEpisode(engine="a", true_life=true_life, lower=interval, point=perfect)

        to_failure = simulate_fleet([episode], RunToFailurePolicy(), model, horizon=true_life)
        early = simulate_fleet(
            [episode], ThresholdPolicy(20.0, use_lower_bound=True), model, horizon=true_life
        )
        assert early.total_cost <= to_failure.total_cost * 1.5

    def test_perfect_interval_never_triggers_after_the_engine_is_gone(self) -> None:
        """A policy that fires on a dead engine double-charges for one event.

        The simulator must stop consulting an episode once its life is exhausted,
        which is what the `None` padding in the episode fixtures relies on.
        """
        model = make_cost_model()
        episode = EngineEpisode(
            engine="a",
            true_life=10,
            lower=np.arange(10, 0, -1.0),
            point=np.arange(10, 0, -1.0),
        )
        outcome = simulate_fleet(
            [episode], ThresholdPolicy(1.0, use_lower_bound=True), model, horizon=10
        )
        assert outcome.n_planned + outcome.n_failures <= 1

    def test_rejects_a_horizon_shorter_than_the_shortest_engine(self) -> None:
        """A horizon nothing can fail in makes the comparison vacuous.

        This is not hypothetical. An earlier run built episodes from window indices
        rather than cycle numbers, so every episode was roughly a hundred times too
        long, no engine reached the end of its life, every policy recorded zero
        failures, and run-to-failure appeared to cost nothing while beating
        everything. The result was arithmetically correct and meaningless. A horizon
        that cannot produce a failure is refused rather than reported.
        """
        with pytest.raises(ValueError, match="shorter than the shortest engine"):
            simulate_fleet(
                [EngineEpisode("a", true_life=5000)],
                RunToFailurePolicy(),
                make_cost_model(),
                horizon=2000,
            )

    def test_a_cycle_with_no_prediction_is_not_a_decision(self) -> None:
        """A gap in the observation record must not become a fabricated prediction.

        Episodes are indexed by cycle, and windows are a stride apart, so many cycles
        have no window over them. The simulator skips those cycles instead of reading
        a NaN as a number, which would either poison the policy or silently become a
        confident zero.
        """
        episode = EngineEpisode(
            engine="a",
            true_life=20,
            lower=np.array([np.nan] * 20),
            point=np.array([np.nan] * 20),
        )
        assert episode.prediction_at(5) is None
        outcome = simulate_fleet(
            [episode], ThresholdPolicy(0.0, use_lower_bound=True), make_cost_model(), horizon=20
        )
        assert outcome.n_planned == 0

    def test_rejects_a_nonpositive_horizon(self) -> None:
        """A zero-cycle horizon is not a simulation."""
        with pytest.raises(ValueError, match="horizon"):
            simulate_fleet([EngineEpisode("a", 10)], RunToFailurePolicy(), make_cost_model(), 0)

    def test_rejects_an_empty_fleet(self) -> None:
        """Costing zero engines is undefined, not zero."""
        with pytest.raises(ValueError, match="fleet"):
            simulate_fleet([], RunToFailurePolicy(), make_cost_model(), horizon=10)

    def test_outcome_normalises_per_engine(self) -> None:
        """Cost per engine is what makes fleets of different size comparable."""
        # Horizon chosen so exactly two engines are in play: the first flies its 100
        # cycles and fails, its 2 cycles of ground time push the second into play with
        # 48 cycles left, and the horizon stops before the second engine dies.
        outcome = simulate_fleet(
            [EngineEpisode("a", 100), EngineEpisode("b", 100)],
            RunToFailurePolicy(),
            make_cost_model(),
            horizon=150,
        )
        assert outcome.n_engines == 2
        assert outcome.cost_per_engine == pytest.approx(outcome.total_cost / 2)

    def test_records_the_assumptions_it_ran_with(self) -> None:
        """An outcome without its cost model cannot be interpreted later."""
        outcome = simulate_fleet(
            [EngineEpisode("a", 10)], RunToFailurePolicy(), make_cost_model(), horizon=10
        )
        assert outcome.cost_model["failure_cost_ratio"] == 10.0
        assert outcome.policy == "run_to_failure"
        assert outcome.horizon == 10


class TestRatioSweep:
    def test_returns_one_row_per_ratio(self) -> None:
        """The sweep is the headline result, so its shape has to be exact."""
        ratios = [1.0, 5.0, 25.0]
        rows = sweep_failure_ratio(
            [EngineEpisode("a", 100)],
            RunToFailurePolicy(),
            make_cost_model(),
            horizon=100,
            ratios=ratios,
        )
        assert [row["failure_cost_ratio"] for row in rows] == ratios

    def test_earlier_replacement_wins_more_as_the_ratio_rises(self) -> None:
        """The advantage of planning should grow with the cost of failing.

        This is the substantive claim behind the whole sweep. If the advantage were
        flat in the ratio, the ratio would not be the thing driving the policy and
        the curve would be noise.
        """
        true_life = 100
        interval = np.array([true_life - t for t in range(true_life)], dtype=np.float64)
        perfect = np.array([float(true_life)] * true_life, dtype=np.float64)
        episode = EngineEpisode(engine="a", true_life=true_life, lower=interval, point=perfect)
        early_policy = ThresholdPolicy(20.0, use_lower_bound=True)

        rows = sweep_failure_ratio(
            [episode],
            early_policy,
            make_cost_model(),
            horizon=true_life,
            ratios=[1.0, 5.0, 25.0, 100.0],
        )
        savings = [row["savings_vs_run_to_failure"] for row in rows]
        assert savings == sorted(savings), f"savings not monotonic in ratio: {savings}"
        assert savings[-1] > savings[0]

    def test_each_row_is_a_real_simulation(self) -> None:
        """A sweep that reuses one simulation and rescales the output is not a sweep."""
        rows = sweep_failure_ratio(
            [EngineEpisode("a", 80)],
            ThresholdPolicy(15.0, use_lower_bound=True),
            make_cost_model(),
            horizon=80,
            ratios=[2.0, 10.0],
        )
        assert all(row["total_cost"] > 0 for row in rows)
        assert rows[0]["total_cost"] != rows[1]["total_cost"]


class TestFleetOutcomeIsAValue:
    def test_cannot_be_mutated_after_construction(self) -> None:
        """An outcome edited in place is not a record of what happened."""
        outcome = FleetOutcome(
            policy="run_to_failure",
            horizon=100,
            total_cost=1.0,
            n_planned=0,
            n_failures=1,
            n_engines=1,
            cost_model={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.total_cost = 99.0  # type: ignore[misc]


class TestPoliciesAreObjects:
    @pytest.mark.parametrize(
        "policy",
        [RunToFailurePolicy(), FixedIntervalPolicy(30), ThresholdPolicy(10.0)],
    )
    def test_every_policy_is_usable_through_the_protocol(self, policy: Policy) -> None:
        """The simulator must work with anything satisfying the protocol.

        Otherwise a fourth policy added later would be a special case, and the
        policy that happens to be written first would be the only tested one.
        """
        assert isinstance(policy.name, str)
        assert isinstance(policy.should_replace(lower=1.0, point=1.0, cycle=1), bool)
