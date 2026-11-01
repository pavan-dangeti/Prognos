"""Tests for the published DS02 comparison.

The comparison exists under one constraint: it may not read the closed final test
group. That is pinned here against the committed split, along with the result file's
obligation to say why its numbers are not comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prognos.dataset import GroupKey, WindowSet
from prognos.published import (
    CONCLUSION,
    PUBLISHED,
    PUBLISHED_TEST,
    PUBLISHED_TRAIN,
    TEST,
    TRAIN,
    compare,
)

RESULT = Path("results/phase6_published_comparison.json")
CLOSED_DS02 = {5, 11, 20}


def test_the_runnable_split_is_the_published_one_minus_the_closed_engines() -> None:
    assert set(TRAIN) == set(PUBLISHED_TRAIN) - CLOSED_DS02
    assert set(TEST) == set(PUBLISHED_TEST) - CLOSED_DS02


def test_the_closed_engines_match_the_committed_test_group() -> None:
    """If the closed group ever changes, this comparison's split must be revisited."""
    phase3 = json.loads(Path("results/phase3_sequence.json").read_text())
    closed = {
        int(label.split("unit")[1])
        for label in phase3["final_test_group"]["engines"]
        if label.startswith("DS02")
    }
    assert closed == CLOSED_DS02


def test_the_comparison_runs_and_reads_only_its_engines() -> None:
    rng = np.random.default_rng(0)
    labels = tuple(GroupKey("DS02-006", unit) for unit in (2, 5, 10, 11, 14, 15, 16, 18, 20))
    per_engine, length = 20, 5
    n = per_engine * len(labels)
    groups = np.repeat(np.arange(len(labels)), per_engine).astype(np.int64)
    cycles = np.tile(np.arange(1, per_engine + 1), len(labels)).astype(np.int64)
    windows = rng.normal(size=(n, length, 18)).astype(np.float32)
    # A closed engine carries an impossible target: reading it would show in the result.
    targets = np.where(np.isin(groups, [1, 3, 8]), 1e6, 80.0 - cycles).astype(np.float32)
    data = WindowSet(
        windows=windows,
        targets=targets,
        groups=groups,
        group_labels=labels,
        cycles=cycles,
        fractions=np.zeros(n, dtype=np.float32),
        splits=("dev",) * n,
        subdatasets=("DS02-006",) * n,
        max_cycle=per_engine,
    )
    report = compare(data, seed=0, epochs=1)
    assert report["n_train_windows"] == per_engine * len(TRAIN)
    for result in report["ours"].values():
        assert result["n_test_windows"] == per_engine * len(TEST)
        assert all(value < 1e3 for value in result["rmse"].values())


def test_the_result_says_why_it_is_not_comparable() -> None:
    if not RESULT.exists():
        pytest.skip("no committed comparison; run `prognos compare`")
    report = json.loads(RESULT.read_text())
    assert report["conclusion"] == CONCLUSION
    assert report["final_test_group"]["status"] == "closed"
    assert report["published"] == [dict(row) for row in PUBLISHED]
    assert any("cycle number" in text for text in report["differences"])


def test_the_final_run_compares_with_and_without_the_cycle_input() -> None:
    from prognos.published import PUBLISHED_TEST, final_comparison

    rng = np.random.default_rng(1)
    labels = tuple(GroupKey("DS02-006", unit) for unit in (2, 5, 10, 11, 14, 15, 16, 18, 20))
    per_engine = 20
    n = per_engine * len(labels)
    groups = np.repeat(np.arange(len(labels)), per_engine).astype(np.int64)
    cycles = np.tile(np.arange(1, per_engine + 1), len(labels)).astype(np.int64)
    data = WindowSet(
        windows=rng.normal(size=(n, 5, 18)).astype(np.float32),
        targets=(80.0 - cycles).astype(np.float32),
        groups=groups,
        group_labels=labels,
        cycles=cycles,
        fractions=np.zeros(n, dtype=np.float32),
        splits=("dev",) * n,
        subdatasets=("DS02-006",) * n,
        max_cycle=per_engine,
    )
    report = final_comparison(data, seed=0, epochs=1)
    assert report["n_train_windows"] == per_engine * 6
    assert set(report["ours"]) == {"with_cycle", "without_cycle"}
    assert "linear_cycle" in report["ours"]["with_cycle"]
    assert "linear_cycle" not in report["ours"]["without_cycle"]
    for models in report["ours"].values():
        for result in models.values():
            assert set(result["rmse"]) == {*map(str, PUBLISHED_TEST), "all"}
    # Targets here are a pure function of the cycle: with it LightGBM is near exact,
    # without it the same model cannot know the age, which is the point of the variant.
    with_cycle = report["ours"]["with_cycle"]["lightgbm"]["rmse"]["all"]
    without_cycle = report["ours"]["without_cycle"]["lightgbm"]["rmse"]["all"]
    assert with_cycle < without_cycle


def test_the_final_command_refuses_to_run_twice(tmp_path: Path) -> None:
    from prognos.cli import main

    out = tmp_path / "final.json"
    out.write_text("{}")
    assert main(["final-ds02", "--out", str(out)]) == 1


def test_the_readme_quotes_the_final_run() -> None:
    """Every final-run figure in the README rebuilds from the committed result."""
    result = Path("results/final_ds02.json")
    if not result.exists():
        pytest.skip("the final run has not happened")
    report = json.loads(result.read_text())
    readme = " ".join(Path("README.md").read_text().split())
    inputs = {"with_cycle": "sensors and cycle", "without_cycle": "sensors only"}
    for variant, models in report["ours"].items():
        for name, result_row in models.items():
            r = result_row["rmse"]
            label = "cycle only" if name == "linear_cycle" else inputs[variant]
            cells = f"| {r['11']:.2f} | {r['14']:.2f} | {r['15']:.2f} |"
            line = f"| `{name}` | {label} {cells}"
            assert line in readme, line
    opened = report["final_test_group"]["opened_by_this_run"]
    assert opened == ["DS02-006/unit005", "DS02-006/unit011", "DS02-006/unit020"]
    assert len(report["final_test_group"]["still_closed"]) == 7


def test_seed_summaries_are_mean_and_t_interval() -> None:
    from prognos.published import summarise_seeds

    def run(value: float) -> dict[str, object]:
        return {"ours": {"without_cycle": {"cnn": {"rmse": {"all": value}}}}}

    summary = summarise_seeds([run(v) for v in (6.0, 7.0, 8.0, 7.0, 7.0)])
    row = summary["without_cycle"]["cnn"]["all"]
    assert row["mean"] == pytest.approx(7.0)
    # std 0.7071, t(0.975, 4) = 2.776: half-width 2.776 * 0.7071 / sqrt(5) = 0.878
    assert row["ci_high"] - row["mean"] == pytest.approx(0.878, abs=1e-3)
    assert row["values"] == [6.0, 7.0, 8.0, 7.0, 7.0]


def test_the_readme_quotes_the_seeded_run() -> None:
    seeded = Path("results/final_ds02_seeds.json")
    if not seeded.exists():
        pytest.skip("the seeded run has not happened")
    summary = json.loads(seeded.read_text())["summary"]
    single = json.loads(Path("results/final_ds02.json").read_text())["ours"]
    readme = " ".join(Path("README.md").read_text().split())

    def cell(variant: str, name: str) -> str:
        a = summary[variant][name]["all"]
        return f"{a['mean']:.2f} [{a['ci_low']:.2f}, {a['ci_high']:.2f}]"

    for name in ("transformer", "cnn", "lightgbm"):
        line = (
            f"| `{name}` | {cell('without_cycle', name)} "
            f"| {single['without_cycle'][name]['rmse']['all']:.2f} "
            f"| {cell('with_cycle', name)} | {single['with_cycle'][name]['rmse']['all']:.2f} |"
        )
        assert line in readme, line


def test_the_first_seed_reproduces_the_single_seed_run() -> None:
    seeded = Path("results/final_ds02_seeds.json")
    if not seeded.exists():
        pytest.skip("the seeded run has not happened")
    first = json.loads(seeded.read_text())["runs"][0]["ours"]
    single = json.loads(Path("results/final_ds02.json").read_text())["ours"]
    assert first == single
