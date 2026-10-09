"""Guards on the committed result file.

The rule for this project is that every published number comes from a committed
script and the raw output is committed under `results/`. These tests keep that
honest: the file must parse, it must not be silently stale relative to the code, and
the figures quoted in the README must match it exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RESULTS = Path("results/phase2_baselines.json")
README = Path("README.md")


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    """Load the committed result file, skipping if it has not been produced."""
    if not RESULTS.exists():
        pytest.skip(f"{RESULTS} not present; run `prognos evaluate` to produce it")
    return json.loads(RESULTS.read_text())


class TestResultFile:
    def test_records_hardware_and_config(self, report: dict[str, object]) -> None:
        """A number without the machine and config it came from is not reproducible."""
        assert report["hardware"], "hardware must be recorded"
        assert report["config"]["window"]["length"] > 0
        assert report["config"]["split"]["seed"] is not None

    def test_final_test_group_is_marked_closed(self, report: dict[str, object]) -> None:
        """The report must state the test set was not opened."""
        assert report["final_test_group"]["status"] == "closed"
        assert report["final_test_group"]["engines"], "the held-out engines must be named"

    def test_every_engine_appears_in_exactly_one_fold_role(self, report: dict[str, object]) -> None:
        """No engine may be both trained on and scored on within a fold."""
        for fold in report["folds"]:
            train = set(fold["train_engines"])
            val = set(fold["val_engines"])
            assert not train & val, f"{fold['name']} leaks {sorted(train & val)}"

    def test_test_group_engines_never_appear_in_a_fold(self, report: dict[str, object]) -> None:
        """The final test group must not be touched by cross-validation."""
        reserved = set(report["final_test_group"]["engines"])
        for fold in report["folds"]:
            used = set(fold["train_engines"]) | set(fold["val_engines"])
            assert not used & reserved, (
                f"{fold['name']} uses held-out engines {sorted(used & reserved)}"
            )

    def test_every_engine_is_evaluated_exactly_once(self, report: dict[str, object]) -> None:
        """Held-out engines partition the cross-validation set.

        Each engine appears in exactly one fold's validation set. An engine appearing
        twice would be scored twice, and one appearing nowhere would be silently
        dropped from the evaluation.
        """
        seen: list[str] = []
        for fold in report["folds"]:
            seen.extend(fold["val_engines"])
        assert len(seen) == len(set(seen)), f"an engine was validated twice: {seen}"
        reserved = set(report["final_test_group"]["engines"])
        all_engines = set(report["data"]["engines"])
        assert set(seen) == all_engines - reserved

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

    def test_best_model_beats_the_mean_floor(self, report: dict[str, object]) -> None:
        """At least one model must beat predicting the fleet mean.

        If nothing does, the pipeline is reading noise and every downstream number
        is meaningless. This is the single most important assertion in the file.
        """
        models = report["models"]
        floor = models["mean"]["rmse"]["mean"]
        best = min(summary["rmse"]["mean"] for summary in models.values())
        assert best < floor, f"best RMSE {best:.2f} does not beat the mean baseline {floor:.2f}"


class TestReadmeMatchesResults:
    """The README quotes numbers; they must come from the committed result file."""

    def _readme_rmse_table(self) -> dict[str, float]:
        """Parse `| model | rmse | ... |` rows out of the README results table."""
        if not README.exists():
            pytest.skip("no README")
        text = README.read_text()
        rows: dict[str, float] = {}
        for line in text.splitlines():
            match = re.match(r"^\|\s*`?([a-z_0-9]+)`?\s*\|\s*([0-9]+\.[0-9]+)\s*\|", line)
            if match:
                rows[match.group(1)] = float(match.group(2))
        return rows

    def test_quoted_rmse_values_match_the_result_file(self, report: dict[str, object]) -> None:
        """A drifted README figure is a false claim, so it fails CI."""
        quoted = self._readme_rmse_table()
        if not quoted:
            pytest.skip("README has no RMSE results table yet")

        models = report["models"]
        for name, value in quoted.items():
            if name not in models:
                pytest.fail(f"README quotes {name!r}, which is not in {RESULTS}")
            expected = round(float(models[name]["rmse"]["mean"]), 2)
            assert value == expected, (
                f"README says {name} RMSE {value:.2f}, "
                f"{RESULTS} says {expected:.2f}. Re-run `prognos evaluate` and update both."
            )
