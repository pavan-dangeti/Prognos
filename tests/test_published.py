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
