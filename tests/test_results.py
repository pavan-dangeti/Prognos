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
