"""Tests for the drift detector study.

The study's value is that the detector was tuned on engines it is not judged on, so
the split and the selection rule are pinned here, along with the link between the
committed result and the defaults the detector actually ships with.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from prognos.dataset import GroupKey
from prognos.drift_study import (
    SHIPPED,
    CycleStream,
    DriftSettings,
    channel_scales,
    detects,
    false_alarms,
    selection_key,
    split_engines,
)
from prognos.stream.faults import FaultDetector
from prognos.stream.window import N_CHANNELS

RESULT = Path("results/phase6_drift.json")


def labels() -> list[GroupKey]:
    return [GroupKey("DS01", u) for u in range(1, 9)] + [GroupKey("DS06", u) for u in range(1, 6)]


def stream(seed: int = 0, n: int = 70) -> CycleStream:
    """A quiet engine: level channels with small independent noise."""
    rng = np.random.default_rng(seed)
    rows = (100.0 + rng.normal(0.0, 1.0, size=(n, N_CHANNELS))).astype(np.float32)
    return CycleStream(label=f"e{seed}", cycles=np.arange(1, n + 1, dtype=np.int64), rows=rows)


def test_the_split_is_disjoint_and_covers_every_sub_dataset() -> None:
    development, held_out = split_engines(labels(), seed=7)
    assert not set(development) & set(held_out)
    assert set(development) | set(held_out) == set(labels())
    for side in (development, held_out):
        assert {key.subdataset for key in side} == {"DS01", "DS06"}


def test_the_split_is_reproducible() -> None:
    assert split_engines(labels(), seed=7) == split_engines(labels(), seed=7)


def test_a_large_step_is_detected_and_a_quiet_engine_raises_nothing() -> None:
    quiet = stream()
    scales = channel_scales([quiet])
    settings = DriftSettings(factor=3.0, half=10, persistence=1)
    assert false_alarms([quiet], settings)["engines_alarmed"] == 0
    assert detects(quiet, settings, channel=10, step=20 * float(scales[10])) is not None


def test_persistence_holds_back_a_one_cycle_blip() -> None:
    """A shift that has to hold for longer is named later, never sooner."""
    quiet = stream(seed=1)
    scales = channel_scales([quiet])
    step = 20 * float(scales[10])
    eager = detects(quiet, DriftSettings(3.0, 10, 1), channel=10, step=step)
    patient = detects(quiet, DriftSettings(3.0, 10, 3), channel=10, step=step)
    assert eager is not None and patient is not None
    assert patient >= eager


def test_selection_prefers_fewer_alarmed_engines_over_detection() -> None:
    def result(alarmed: int, at5: float, factor: float) -> dict[str, object]:
        return {
            "settings": {"factor": factor, "half": 6, "persistence": 1},
            "false_alarms": {"engines_alarmed": alarmed},
            "detection": [{"multiple": 2.0, "rate": 0.1}, {"multiple": 5.0, "rate": at5}],
        }

    noisy, quiet = result(1, 0.9, 3.0), result(0, 0.2, 8.0)
    assert min([noisy, quiet], key=selection_key) is quiet


def test_the_detector_ships_with_the_settings_the_study_chose() -> None:
    """The committed choice and the code's defaults must not drift apart."""
    if not RESULT.exists():
        pytest.skip("no committed drift study; run `prognos drift-study`")
    chosen = json.loads(RESULT.read_text())["chosen"]
    defaults = inspect.signature(FaultDetector).parameters
    assert defaults["drift_factor"].default == chosen["factor"]
    assert defaults["drift_half"].default == chosen["half"]
    assert defaults["drift_persistence"].default == chosen["persistence"]


def test_the_report_judges_on_engines_it_did_not_tune_on() -> None:
    if not RESULT.exists():
        pytest.skip("no committed drift study; run `prognos drift-study`")
    report = json.loads(RESULT.read_text())
    assert not set(report["development_engines"]) & set(report["held_out_engines"])
    held = set(report["held_out_engines"])
    for side in ("before", "after"):
        assert report["held_out"][side]["false_alarms"]["engines"] == len(held)
    assert report["shipped"] == {"factor": SHIPPED.factor, "half": SHIPPED.half, "persistence": 1}
    assert report["final_test_group"]["status"] == "closed"


def test_the_whole_study_runs_on_parquet_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command's real path, on four quiet engines and a two-setting grid."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from prognos import drift_study
    from prognos.window import FEATURE_COLUMNS

    monkeypatch.setattr(drift_study, "FACTORS", (3.0,))
    monkeypatch.setattr(drift_study, "HALVES", (10,))
    monkeypatch.setattr(drift_study, "PERSISTENCE", (1, 3))
    engines = [GroupKey("DS01", unit) for unit in range(1, 5)]
    (tmp_path / "DS01").mkdir()
    for key in engines:
        quiet = stream(seed=key.unit, n=60)
        columns = {"cycle": quiet.cycles, "rul": np.full(quiet.cycles.size, 50.0)}
        columns.update({name: quiet.rows[:, i] for i, name in enumerate(FEATURE_COLUMNS)})
        pq.write_table(pa.table(columns), tmp_path / "DS01" / f"unit_{key.unit:03d}.parquet")

    report = drift_study.study(tmp_path, engines, seed=3)
    assert len(report["grid"]) == 2
    assert report["chosen"] in [r["settings"] for r in report["grid"]]
    assert set(report["development_engines"]).isdisjoint(report["held_out_engines"])
    held = report["held_out"]["after"]
    assert held["false_alarms"]["engines"] == len(report["held_out_engines"])
    assert [d["multiple"] for d in held["detection"]] == [2.0, 5.0, 10.0]
    # The shipped setting is not in this narrowed grid, so "before" is looked up by
    # value from the held-out evaluation rather than the grid.
    assert report["held_out"]["before"]["settings"] == {"factor": 4.0, "half": 6, "persistence": 1}
