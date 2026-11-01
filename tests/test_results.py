"""Guards on the committed result files.

The rule for this project is that every published number comes from a committed
script and the raw output is committed under `results/`. These tests keep that
honest: the files must parse, they must not be silently stale relative to the code,
and the figures quoted in the README must match them exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RESULTS = Path("results/phase2_baselines.json")
SEQUENCE_RESULTS = Path("results/phase3_sequence.json")
POLICY_RESULTS = Path("results/phase4_policy.json")
README = Path("README.md")

# Models whose RMSE is quoted in the README results table.
QUOTED = ("cnn", "transformer", "lightgbm", "linear_cycle", "lstm", "mean", "trend")


def load(path: Path) -> dict[str, object]:
    """Load a result file, skipping the test when it has not been produced."""
    if not path.exists():
        pytest.skip(f"{path} not present; run `prognos evaluate` to produce it")
    return json.loads(path.read_text())


class SharedResultProperties:
    """Assertions that must hold for every committed result file."""

    path: Path

    @pytest.fixture(scope="class")
    def report(self) -> dict[str, object]:
        """The result file under test."""
        return load(self.path)

    def test_records_hardware_and_config(self, report: dict[str, object]) -> None:
        """A number without the machine and config it came from is not reproducible."""
        assert report["hardware"], "hardware must be recorded"
        assert report["config"]["window"]["length"] > 0
        assert report["config"]["split"]["seed"] is not None

    def test_records_library_versions_and_a_date(self, report: dict[str, object]) -> None:
        """The result must say which libraries produced it.

        torch and LightGBM both change numerics between releases, so a result file
        that cannot name the versions it came from cannot be checked against a re-run.
        This exists because it bit: an experiment on a different torch build overwrote
        a committed result file, and nothing in the file distinguished the two runs.
        """
        versions = report["versions"]
        assert versions["torch"], "torch version must be recorded"
        for package in ("numpy", "pandas", "lightgbm", "python"):
            assert versions[package], f"{package} version must be recorded"

        generated = report["generated_at"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", generated), (
            f"generated_at must be an ISO date, got {generated!r}"
        )

    def test_final_test_group_is_marked_closed(self, report: dict[str, object]) -> None:
        """The report must state the test set was not opened."""
        assert report["final_test_group"]["status"] == "closed"
        assert report["final_test_group"]["engines"], "the held-out engines must be named"

    def test_no_fold_touches_a_test_engine(self, report: dict[str, object]) -> None:
        """The final test group must not appear in any fold, train or validation."""
        reserved = set(report["final_test_group"]["engines"])
        for fold in report["folds"]:
            used = set(fold["train_engines"]) | set(fold["val_engines"])
            assert not used & reserved, (
                f"{fold['name']} uses held-out engines {sorted(used & reserved)}"
            )

    def test_no_fold_leaks_an_engine_across_its_boundary(self, report: dict[str, object]) -> None:
        """An engine must not be both trained on and scored on within a fold."""
        for fold in report["folds"]:
            train = set(fold["train_engines"])
            val = set(fold["val_engines"])
            assert not train & val, f"{fold['name']} leaks {sorted(train & val)}"

    def test_every_engine_is_validated_exactly_once(self, report: dict[str, object]) -> None:
        """Held-out engines partition the cross-validation set.

        An engine appearing twice is scored twice; one appearing nowhere is silently
        dropped from the evaluation.
        """
        seen: list[str] = []
        for fold in report["folds"]:
            seen.extend(fold["val_engines"])
        assert len(seen) == len(set(seen)), f"an engine was validated twice: {seen}"
        reserved = set(report["final_test_group"]["engines"])
        assert set(seen) == set(report["data"]["engines"]) - reserved

    def test_models_carry_their_evidence_counts(self, report: dict[str, object]) -> None:
        """Every model summary must say how many folds and engines it rests on."""
        for name, summary in report["models"].items():
            assert summary["n_folds"] >= 2, f"{name} has too few folds"
            assert summary["n_eval_rows"] > 0, f"{name} scored nothing"
            assert summary["eval_engines"], f"{name} reports no engines"

    def test_metrics_are_finite(self, report: dict[str, object]) -> None:
        """A NaN or infinite RMSE would propagate into the README as a number."""
        for name, summary in report["models"].items():
            for metric in ("rmse", "nasa_score"):
                for key in ("mean", "ci_low", "ci_high"):
                    value = summary[metric][key]
                    assert value == value, f"{name}.{metric}.{key} is NaN"
                    assert abs(value) != float("inf"), f"{name}.{metric}.{key} is infinite"

    def test_life_stage_counts_sum_to_the_evaluation_set(self, report: dict[str, object]) -> None:
        """A dropped stage would silently under-report the breakdown."""
        for name, summary in report["models"].items():
            total = sum(int(stage["n"]) for stage in summary["by_stage"].values())
            assert total == summary["n_eval_rows"], f"{name} stage counts do not add up"

    def test_at_least_one_model_beats_the_mean_floor(self, report: dict[str, object]) -> None:
        """If nothing beats predicting the fleet mean, the pipeline is reading noise."""
        models = report["models"]
        floor = models["mean"]["rmse"]["mean"]
        best = min(summary["rmse"]["mean"] for summary in models.values())
        assert best < floor, f"best RMSE {best:.2f} does not beat the mean baseline {floor:.2f}"


class TestBaselineResults(SharedResultProperties):
    """Phase 2: the tabular baselines."""

    path = RESULTS


class TestSequenceResults(SharedResultProperties):
    """Phase 3: the sequence models, evaluated alongside the baselines."""

    path = SEQUENCE_RESULTS

    @pytest.fixture(scope="class")
    def report(self) -> dict[str, object]:
        """The phase 3 result file."""
        return load(self.path)

    def test_sequence_file_also_contains_the_baselines(self, report: dict[str, object]) -> None:
        """The comparison is the point: the same folds, the same numbers.

        If the baselines were re-run with a different seed, a sequence model's gain
        could be a split artefact rather than a model gain.
        """
        for name in ("mean", "linear_cycle", "trend", "lightgbm"):
            assert name in report["models"], f"{name} missing from the phase 3 comparison"

    def test_baseline_numbers_are_unchanged_from_phase_2(self, report: dict[str, object]) -> None:
        """Phase 3 must not have moved the Phase 2 baselines.

        They are the reference point, so a silent change would invalidate every
        comparison in the README.
        """
        phase2 = load(RESULTS)
        for name in ("mean", "linear_cycle", "trend", "lightgbm"):
            before = round(float(phase2["models"][name]["rmse"]["mean"]), 2)
            after = round(float(report["models"][name]["rmse"]["mean"]), 2)
            assert before == after, f"{name} moved: {before} -> {after}"

    def test_fit_reports_are_recorded_for_sequence_models(self, report: dict[str, object]) -> None:
        """Training cost and early stopping are part of the result.

        A sequence model that ran for 77 epochs to reach its best at epoch 29 is a
        materially different claim from one that converged, and the reader needs to
        see which.
        """
        fits = report["model_fits"]
        for name in ("cnn", "lstm", "transformer"):
            assert name in fits, f"{name} has no fit report"
            entry = fits[name]
            assert entry["params"] > 0, f"{name} reports no parameters"
            assert entry["epochs_run"] >= 1, f"{name} reports no epochs"
            assert entry["config"]["seed"], f"{name} was not seeded"

    def test_a_sequence_model_beats_the_cycle_only_floor(self, report: dict[str, object]) -> None:
        """Sensor data has to earn its place.

        `linear_cycle` uses no sensors at all. Beating it is the only evidence that
        the sensor channels are contributing anything beyond the engine's age, and it
        is the claim Phase 3 exists to test.
        """
        models = report["models"]
        floor = models["linear_cycle"]["rmse"]["mean"]
        best = min(models[name]["rmse"]["mean"] for name in ("cnn", "lstm", "transformer"))
        assert best < floor, (
            f"best sequence RMSE {best:.2f} does not beat the cycle-only floor {floor:.2f}; "
            "sensor data is not yet shown to help"
        )


class TestReadmeMatchesResults:
    """The README quotes numbers; they must come from the committed result file."""

    def _readme_rmse_table(self) -> dict[str, float]:
        """Parse the model RMSE column out of the README's RMSE table.

        The README carries more than one table and the life-stage breakdown's first
        column is also a model name. Rather than pattern-match rows, find the table
        whose header names an `rmse` column and read the rows under it, so a
        life-stage figure can never be checked against an overall one.
        """
        if not README.exists():
            pytest.skip("no README")

        rows: dict[str, float] = {}
        in_target_table = False
        for line in README.read_text().splitlines():
            stripped = line.strip()
            if not (stripped.startswith("|") and stripped.endswith("|")):
                in_target_table = False
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if re.fullmatch(r":?-{2,}:?", cells[0]):
                continue
            if any(cell.lower() == "rmse" for cell in cells):
                in_target_table = True
                continue
            if not in_target_table or len(cells) < 2:
                continue
            name = cells[0].strip("`")
            if re.fullmatch(r"[a-z_0-9]+", name) and re.fullmatch(r"[0-9]+\.[0-9]+", cells[1]):
                rows[name] = float(cells[1])
        return rows

    def test_quoted_rmse_values_match_a_committed_result_file(self) -> None:
        """A drifted README figure is a false claim, so it fails CI.

        Phase 3 supersedes phase 2 for the full table, but both files are committed, so
        a number is accepted if it matches either. A number matching neither is a
        figure nobody can reproduce.
        """
        quoted = self._readme_rmse_table()
        if not quoted:
            pytest.skip("README has no RMSE results table yet")

        known: dict[str, set[float]] = {}
        for path in (SEQUENCE_RESULTS, RESULTS):
            if path.exists():
                data = json.loads(path.read_text())
                for name, summary in data["models"].items():
                    known.setdefault(name, set()).add(round(float(summary["rmse"]["mean"]), 2))

        for name, value in quoted.items():
            if name not in known:
                pytest.fail(f"README quotes {name!r}, which is in no result file")
            assert value in known[name], (
                f"README says {name} RMSE {value:.2f}; committed results have "
                f"{sorted(known[name])}. Re-run `prognos evaluate` and update both."
            )

    def test_readme_documents_the_reproduce_command(self) -> None:
        """A results table without a command to reproduce it is not a result."""
        text = README.read_text()
        assert "prognos evaluate" in text, "README must show how to reproduce the numbers"
        assert "results/" in text, "README must link the raw output"


class TestPolicyResults:
    """Phase 4: conformal intervals and the cost comparison.

    The guards here are about the result being interpretable rather than about it
    looking good. Both of the failure modes below produced a file that read as a
    successful experiment while measuring nothing.
    """

    @pytest.fixture(scope="class")
    def report(self) -> dict[str, object]:
        """The phase 4 result file."""
        return load(POLICY_RESULTS)

    def test_records_versions_and_date(self, report: dict[str, object]) -> None:
        """Provenance, same as every other result file."""
        assert report["versions"]["torch"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", report["generated_at"])

    def test_final_test_group_stays_closed(self, report: dict[str, object]) -> None:
        """Phase 4 opened no held-out engine."""
        assert report["final_test_group"]["status"] == "closed"
        assert report["final_test_group"]["engines"]

    @pytest.mark.parametrize("method", ["split", "cqr"])
    def test_coverage_is_a_probability(self, report: dict[str, object], method: str) -> None:
        """Coverage outside [0, 1] means the interval and the truth do not line up."""
        entry = report["intervals"][method]
        assert 0.0 <= entry["empirical_coverage"] <= 1.0
        low, high = entry["coverage_ci"]
        assert 0.0 <= low <= entry["empirical_coverage"] <= high <= 1.0, (
            f"{method}: coverage {entry['empirical_coverage']:.3f} is not bracketed "
            f"by its own confidence interval [{low:.3f}, {high:.3f}]"
        )

    @pytest.mark.parametrize("method", ["split", "cqr"])
    def test_intervals_are_not_degenerate(self, report: dict[str, object], method: str) -> None:
        """A zero-width interval that hits the coverage target is a fake result."""
        assert report["intervals"][method]["mean_width"] > 0.0

    def test_meets_the_coverage_slo(self, report: dict[str, object]) -> None:
        """R6: coverage within five points of the target."""
        target = report["config"]["uncertainty"]["target_coverage"]
        for method in ("split", "cqr"):
            entry = report["intervals"][method]
            assert abs(entry["empirical_coverage"] - target) <= 0.05, (
                f"{method} coverage {entry['empirical_coverage']:.3f} is more than "
                f"5 points from the {target:.0%} SLO"
            )

    def test_calibration_engines_are_disjoint_from_evaluation(
        self, report: dict[str, object]
    ) -> None:
        """Conformal coverage is a lie if calibration engines were fitted on.

        The result file records all three engine sets so this is checkable from the
        artifact rather than trusted from the code. An engine appearing in two of
        fit, calibration and evaluation is the failure that makes a coverage number
        optimistic, and it is invisible in the coverage itself.
        """
        for method in ("split", "cqr"):
            folds = report["intervals"][method]["folds"]
            assert len(folds) >= 2, f"{method}: needs at least two folds to mean anything"
            for fold in folds:
                fit = set(fold["fit_engines"])
                calibration = set(fold["calibration_engines"])
                evaluation = set(fold["eval_engines"])
                assert fit, f"{method}/{fold['name']}: no fitting engines recorded"
                assert calibration, f"{method}/{fold['name']}: no calibration engines"
                assert evaluation, f"{method}/{fold['name']}: no evaluation engines"
                assert not fit & calibration, (
                    f"{method}/{fold['name']}: engines both fitted and calibrated: "
                    f"{sorted(fit & calibration)}"
                )
                assert not fit & evaluation, (
                    f"{method}/{fold['name']}: engines both fitted and evaluated: "
                    f"{sorted(fit & evaluation)}"
                )
                assert not calibration & evaluation, (
                    f"{method}/{fold['name']}: calibration engines were scored: "
                    f"{sorted(calibration & evaluation)}"
                )

    def test_some_policy_can_fail(self, report: dict[str, object]) -> None:
        """Run-to-failure must actually incur failures.

        This is the guard that catches the bug that produced an entire Phase 4 run in
        which every policy recorded zero failures and run-to-failure appeared free.
        Episodes were built from window indices rather than cycle numbers, so every
        engine looked roughly a hundred times longer than it was and the horizon
        ended before anything could break.
        """
        policies = {row["policy"]: row for row in report["decision"]["policies"]}
        assert policies["run_to_failure"]["n_failures"] > 0, (
            "run-to-failure recorded no failures, so no policy in this file can be "
            "cheaper than doing nothing and the whole comparison is vacuous"
        )

    def test_the_horizon_outlasts_the_shortest_engine(self, report: dict[str, object]) -> None:
        """A horizon shorter than the shortest engine life makes the comparison void."""
        horizon = report["decision"]["horizon"]
        assert horizon >= report["decision"]["episode_lives"]["min"], (
            f"horizon {horizon} is shorter than the shortest engine "
            f"({report['decision']['episode_lives']['min']}), so nothing can fail"
        )

    def test_run_to_failure_is_the_most_expensive_policy(self, report: dict[str, object]) -> None:
        """Waiting for engines to break is the worst available strategy.

        Under every cost model here an in-flight removal costs more than a planned
        visit, so a policy that prevents failures should always win. If this fails,
        the cost model is not encoding the asymmetry the whole comparison assumes.
        """
        rows = report["decision"]["policies"]
        worst = max(rows, key=lambda row: row["total_cost"])
        assert worst["policy"] == "run_to_failure", (
            f"run-to-failure costs {policies_cost(rows, 'run_to_failure'):.1f} but "
            f"{worst['policy']} costs {worst['total_cost']:.1f}"
        )

    def test_the_ratio_sweep_covers_more_than_one_ratio(self, report: dict[str, object]) -> None:
        """R7: a policy that only wins under one cost assumption is not a result."""
        sweeps = report["decision"]["ratio_sweeps"]
        assert sweeps, "no ratio sweep in the result file"
        ratios = [row["failure_cost_ratio"] for row in sweeps[0]["rows"]]
        assert len(ratios) >= 3, "a sweep needs at least three ratios to be a sweep"

    def test_savings_grow_with_the_failure_ratio(self, report: dict[str, object]) -> None:
        """The advantage of planning must grow with the cost of failing.

        If the advantage were flat in the ratio, the ratio would not be what drives
        the policy and the curve would be noise rather than a result.
        """
        for sweep in report["decision"]["ratio_sweeps"]:
            savings = [row["savings_vs_run_to_failure"] for row in sweep["rows"]]
            assert savings == sorted(savings), (
                f"threshold {sweep['threshold']}: savings are not monotonic in the "
                f"failure ratio: {savings}"
            )

    def test_policy_cost_is_stable_across_the_ratio_sweep(self, report: dict[str, object]) -> None:
        """The sweep's own cost column must not move with the ratio.

        Every failure-free row costs the same regardless of the ratio, because the
        ratio only prices failures and these policies have none. A row that moved
        would mean the baseline and the policy are being compared at different
        assumptions.
        """
        for sweep in report["decision"]["ratio_sweeps"]:
            costs = {round(row["total_cost"], 6) for row in sweep["rows"]}
            assert len(costs) == 1, (
                f"threshold {sweep['threshold']}: cost varies with the failure ratio "
                f"({sorted(costs)}) for a policy that records no failures"
            )

    def test_cost_model_is_recorded_with_every_result(self, report: dict[str, object]) -> None:
        """A cost number without its assumptions is not a result."""
        model = report["decision"]["cost_model"]
        for key in (
            "planned_maintenance_cost",
            "failure_cost_ratio",
            "downtime_cost_per_cycle",
            "maintenance_window_cycles",
        ):
            assert key in model
        assert "normalised" in str(model["units"])


def policies_cost(rows: list[dict[str, object]], name: str) -> float:
    """Total cost of one named policy, for failure messages."""
    return float(next(row["total_cost"] for row in rows if row["policy"] == name))


STREAM_RESULTS = Path("results/phase5_streaming.json")


class TestStreamingResults:
    """Phase 5: the replayed fleet and the S5 verdict.

    The guard that matters is the engine identity. The first replaying run reported
    0.716 coverage — a real breach — because the driver used the global engine index
    as a Parquet file number, so it fed the model rows from engines it had never been
    fitted on. Every other number in that file looked entirely plausible. The breach
    was the only thing that gave it away, which is the argument for having the check.
    """

    @pytest.fixture(scope="class")
    def report(self) -> dict[str, object]:
        """The phase 5 result file."""
        return load(STREAM_RESULTS)

    def test_records_provenance(self, report: dict[str, object]) -> None:
        """Same provenance contract as every other result file."""
        assert report["versions"]["torch"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", report["generated_at"])
        assert report["hardware"]

    def test_final_test_group_stays_closed(self, report: dict[str, object]) -> None:
        """Phase 5 replayed cross-validation engines, not the held-out ten."""
        assert report["final_test_group"]["status"] == "closed"

    def test_replayed_engines_exist_in_the_dataset(self, report: dict[str, object]) -> None:
        """Every replayed engine must be a real engine of the dataset.

        This is the guard for the bug above. The driver used a global engine index as
        a Parquet filename, so it read a *different* engine than the one it named, and
        the file recorded the mismatch in neither direction.
        """
        known = set(report["data"]["engines"])
        assert known, "no engines listed in the dataset summary"
        for engine in report["engines"]:
            assert engine["engine"] in known, (
                f"{engine['engine']} was replayed but is not an engine of the dataset"
            )

    def test_replayed_engines_carry_their_subdataset(self, report: dict[str, object]) -> None:
        """Engine identity is (sub-dataset, unit); a bare unit number is ambiguous."""
        for engine in report["engines"]:
            assert "/" in engine["engine"], (
                f"{engine['engine']} has no sub-dataset, so its unit number is ambiguous"
            )

    def test_every_engine_produced_windows(self, report: dict[str, object]) -> None:
        """An engine that produced no windows is a replay that failed, not a finding."""
        for engine in report["engines"]:
            assert engine["n_windows"] > 0, f"{engine['engine']} produced no windows"
            assert engine["n_rows"] >= engine["n_windows"]

    def test_coverage_is_a_probability_within_the_monitor_bounds(
        self, report: dict[str, object]
    ) -> None:
        coverage = report["coverage"]
        measured = coverage["measured_coverage"]
        assert measured is None or 0.0 <= measured <= 1.0
        assert coverage["status"] in {"ok", "breach", "insufficient_data"}

    def test_breach_status_matches_the_tolerance(self, report: dict[str, object]) -> None:
        """A verdict that says breach without being under the band is a false alarm."""
        coverage = report["coverage"]
        measured = coverage["measured_coverage"]
        if measured is None:
            assert coverage["status"] == "insufficient_data"
            return
        below = measured < coverage["target_coverage"] - coverage["tolerance"]
        assert (coverage["status"] == "breach") == below, (
            f"status {coverage['status']!r} disagrees with measured coverage "
            f"{measured:.3f} against a {coverage['target_coverage']:.0%} target"
        )

    def test_the_monitor_reached_a_verdict(self, report: dict[str, object]) -> None:
        """S5 only means something once enough predictions have been scored."""
        assert report["coverage"]["n_samples"] >= 50
        assert report["coverage"]["measured_coverage"] is not None

    def test_calibration_engines_are_disjoint_from_the_fit_set(
        self, report: dict[str, object]
    ) -> None:
        """Calibrating on an engine that was fitted on invalidates the interval."""
        calibration = set(report["calibration"]["engines"])
        fitting = set(report["calibration"]["fit_engines"])
        assert calibration, "no calibration engines recorded"
        assert not calibration & fitting, (
            f"engines both fitted and calibrated: {sorted(calibration & fitting)}"
        )

    def test_calibration_is_wide_enough_to_be_useful(self, report: dict[str, object]) -> None:
        """An interval covering everything is not a result, and its width gives it away.

        Recorded so a regression to an unusably narrow or unusably wide interval is
        visible in the diff rather than only in the coverage figure.
        """
        half_width = report["calibration"]["half_width"]
        cap = report["config"]["window"]["rul_cap"]
        assert half_width > 0.0
        assert half_width < cap, (
            f"half-width {half_width:.1f} exceeds the RUL cap {cap}, so the interval "
            "spans the whole operating range and conveys nothing"
        )

    def test_alerts_are_counted_and_open_ones_are_named(self, report: dict[str, object]) -> None:
        """An alert total without the open set is not something an operator can act on."""
        alerts = report["alerts"]
        assert alerts["total"] >= 0
        assert isinstance(alerts["open"], dict)

    def test_metrics_are_exposed_for_a_scrape(self, report: dict[str, object]) -> None:
        """The coverage verdict has to be on a scrape, not only in the result file."""
        rendered = "\n".join(report["metrics_sample"])
        assert "prognos_interval_coverage" in rendered
        assert "prognos_stream_batches_total" in rendered


FAULT_RESULTS = Path("results/phase5_faults.json")


class TestFaultScenarioResults:
    """The Phase 5 exit criterion, checked from the committed artifact.

    `docs/design.md` §12 requires that every injected fault produces a named degraded
    state. This is the committed evidence for that claim.

    Until Phase 6B two of these tests asserted a *failure*: the drift detector did not
    survive its control run, and the artifact recorded that rather than a threshold
    tuned until the control went green. The detector was then retuned on separate
    development engines and judged on held-out ones (`results/phase6_drift.json`);
    the control is clean now, and the test says why that alone is not the evidence.
    """

    @pytest.fixture(scope="class")
    def report(self) -> dict[str, object]:
        """The fault scenario result file."""
        return load(FAULT_RESULTS)

    def _scenario(self, report: dict[str, object], fault: str) -> dict[str, object]:
        return next(s for s in report["scenarios"] if s["fault"] == fault)

    def test_records_provenance(self, report: dict[str, object]) -> None:
        """Same provenance contract as every other result file."""
        assert report["versions"]["lightgbm"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", report["generated_at"])

    def test_a_control_run_is_recorded(self, report: dict[str, object]) -> None:
        """A suite that reports everything detects nothing, so the clean case is tested."""
        assert any(not s["injected"] for s in report["scenarios"])

    @pytest.mark.parametrize("fault", ["sensor_dropout", "sensor_stuck", "sensor_drift"])
    def test_each_injected_fault_is_named(self, report: dict[str, object], fault: str) -> None:
        scenario = self._scenario(report, fault)
        assert scenario["injected"], f"{fault} was not actually injected"
        assert fault in scenario["detected"], (
            f"{fault} was injected on channel {scenario['channel']} from cycle "
            f"{scenario['from_cycle']} and the detector named {scenario['detected']}"
        )

    def test_every_detected_fault_carries_evidence(self, report: dict[str, object]) -> None:
        """A name with no detail is not something an operator can act on."""
        for scenario in report["scenarios"]:
            for state in scenario.get("states", []):
                assert state["detail"], f"{state['state']} has no detail"
                assert state["engine"] is not None
                if state["channel"] is not None:
                    assert state["instrument"] != "", (
                        f"{state['state']} names channel {state['channel']} but not "
                        "the instrument, so nobody knows which sensor to go and look at"
                    )

    def test_the_control_run_is_clean_and_its_caveat_holds(self, report: dict[str, object]) -> None:
        """Clean since the drift study, and not mistaken for the evidence that it should be.

        Until Phase 6B this test asserted the control *failed*: the drift detector fired
        on healthy channels and the artifact said so. The study in
        `results/phase6_drift.json` retuned it on development engines and judged it on
        held-out ones. This control engine is one of the development engines, so its
        clean run cannot be the proof; the held-out rate is, and both are checked here.
        """
        control = self._scenario(report, "none")
        assert control["passed"] is True
        assert control["detected"] == []
        study = load(Path("results/phase6_drift.json"))
        assert report["engine"] in study["development_engines"], (
            "the caveat says the control engine tuned the detector; if that changed, "
            "the caveat must change with it"
        )
        assert study["held_out"]["after"]["false_alarms"]["engines_alarmed"] == 0

    def test_the_caveat_is_documented(self, report: dict[str, object]) -> None:
        """The artifact itself must say why the clean control is not independent."""
        text = " ".join(report["known_limitations"]).lower()
        assert "development" in text and "held-out" in text

    def test_the_replay_cadence_is_recorded(self, report: dict[str, object]) -> None:
        """The scenarios sample one row per cycle, and a reader must be able to know."""
        text = " ".join(report["known_limitations"]).lower()
        assert "one row per cycle" in text

    def test_control_clean_agrees_with_the_control_scenario(
        self, report: dict[str, object]
    ) -> None:
        """The summary flag and the scenario it summarises must not disagree."""
        control_passed = all(bool(s["passed"]) for s in report["scenarios"] if not s["injected"])
        assert report["control_clean"] == control_passed

    def test_all_passed_agrees_with_the_scenarios(self, report: dict[str, object]) -> None:
        """Same, for the headline flag."""
        assert report["all_passed"] == all(bool(s["passed"]) for s in report["scenarios"])


class TestPhase6bReadmeMatchesResults:
    """The drift, failure and published-comparison figures quoted in the README.

    Each expected line is rebuilt from its committed result file, so a figure that
    drifts from the file, or a file regenerated without the README, fails here.
    """

    @staticmethod
    def readme() -> str:
        # Whitespace collapsed, so a figure wrapped across a line still counts as quoted.
        return " ".join(README.read_text().split())

    def test_drift_study_rows(self) -> None:
        study = load(Path("results/phase6_drift.json"))
        text = self.readme()
        for side in ("held_out", "development"):
            for when in ("before", "after"):
                run = study[side][when]
                alarms = run["false_alarms"]
                rate = {d["multiple"]: d for d in run["detection"]}
                s = run["settings"]
                line = (
                    f"| {side.replace('_', '-')} | {when} "
                    f"({s['factor']:g}, {s['half']}, {s['persistence']}) "
                    f"| {alarms['engines_alarmed']}/{alarms['engines']} "
                    f"| {alarms['alarm_cycles']}/{alarms['cycles']} "
                    f"| {rate[2.0]['rate']:.1%} | {rate[5.0]['rate']:.1%} "
                    f"| {rate[5.0]['median_delay_cycles']:g} |"
                )
                assert line in text, f"README drift row drifted: {line}"
        missed = study["held_out"]["after"]["detection"][-1]["missed_by_channel"]
        for channel in missed:
            assert f"`{channel}`" in text, f"README omits blind channel {channel}"

    def test_failure_scenario_figures(self) -> None:
        report = load(Path("results/phase6_failures.json"))
        text = self.readme()
        by_name = {s["scenario"]: s for s in report["scenarios"]}
        outage = by_name["inference_outage"]
        assert outage["predictions_during"] == 0
        assert f"{outage['missed_predictions']} windows missed" in text
        assert by_name["operating_shift"]["openings"] == 1
        assert by_name["operating_shift_within_fleet"]["openings"] == 1
        assert "one alert for the event" in text
        assert f"after {by_name['delayed']['detected_after_seconds']:.1f} s" in text
        order = by_name["out_of_order"]
        assert (
            f"all {order['late_dropped']} dropped; {order['predictions']:,} predictions "
            f"against {order['predictions_in_order']:,} in order"
        ) in text
        named = sum(bool(s["passed"]) for s in report["scenarios"] if s["scenario"] != "control")
        failures = len(report["scenarios"]) - 1
        assert f"{named} of {failures} provoked failures named" in text

    def test_published_comparison_rows(self) -> None:
        report = load(Path("results/phase6_published_comparison.json"))
        text = self.readme()
        for name, result in report["ours"].items():
            rmse = result["rmse"]
            assert f"| {rmse['14']:.2f} | {rmse['15']:.2f} | this run |" in text, name
        for row in report["published"]:
            r = row["rmse"]
            assert f"| {row['method']} | {r['14']:.2f} | {r['15']:.2f} |" in text, row["method"]

    def test_envelope_study_rows(self) -> None:
        study = load(Path("results/phase6_envelope.json"))
        text = self.readme()
        for side in ("held_out", "development"):
            for when in ("before", "after"):
                run = study[side][when]
                st, fa = run["settings"], run["false_alarms"]
                within, beyond = run["within_fleet"], run["beyond_fleet"]
                name = (
                    "fleet only, open 1, release 1"
                    if st["learn"] == 0
                    else f"own from {st['learn']} flights ±{st['margin']:.0%}, "
                    f"open {st['persistence']}, release {st['release']}"
                )
                most = max(within["max_alerts_per_event"], beyond["max_alerts_per_event"])
                line = (
                    f"| {side.replace('_', '-')} | {when}: {name} "
                    f"| {fa['engines_alarmed']}/{fa['engines']} "
                    f"| {within['detected']}/{fa['engines']} "
                    f"| {beyond['detected']}/{fa['engines']} "
                    f"| {within['alerts_per_event']:.1f} | {beyond['alerts_per_event']:.1f} "
                    f"| {most} |"
                )
                assert line in text, f"README envelope row drifted: {line}"
