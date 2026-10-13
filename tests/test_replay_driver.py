"""Tests for the replay driver.

This is the module that fed the model rows from an engine it had never been fitted on:
the driver used the global engine index as a Parquet file number, so `GroupKey`
`DS01-005`/unit 2 was read from `unit_001.parquet`. Every number in the result file
looked reasonable; the only thing that gave it away was the coverage breach.

Engine identity is the whole subject here, so the tests build real Parquet files in
the real layout and check that the driver reads the engine it says it is reading.

The rest is deliberately cheap: a tiny synthetic dataset, because the driver's job is
wiring and the model's competence is measured elsewhere.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prognos.dataset import GroupKey, WindowSet
from prognos.pipeline import _truth_per_cycle, _unit_table, replay_pipeline
from prognos.schema import MEASURED_SENSORS, SCENARIO_DESCRIPTORS, TARGET_COLUMN
from prognos.window import FEATURE_COLUMNS

ROWS_PER_CYCLE = 800
CYCLES_PER_ENGINE = 6


def write_engine(directory, subdataset: str, unit: int, seed: int = 0) -> None:
    """Write one engine's Parquet file in the real layout.

    The real files carry thousands of rows per cycle number and `iter_windows`
    downsamples within the block, so the fixture reproduces that rather than a tidier
    one-row-per-cycle shape the production code never sees.
    """
    rng = np.random.default_rng(seed + unit)
    rows = CYCLES_PER_ENGINE * ROWS_PER_CYCLE
    cycles = np.repeat(np.arange(1, CYCLES_PER_ENGINE + 1), ROWS_PER_CYCLE).astype(np.int64)
    # RUL descends with cycle and is constant within a cycle, as the converter writes it.
    rul = (CYCLES_PER_ENGINE - cycles + 40).astype(np.int64)

    columns: dict[str, np.ndarray] = {
        "cycle": cycles,
        TARGET_COLUMN: rul,
        "unit": np.full(rows, unit),
    }
    for name in FEATURE_COLUMNS:
        columns[name] = rng.normal(size=rows).astype(np.float32)

    target = directory / subdataset / f"unit_{unit:03d}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: pa.array(v) for k, v in columns.items()}), target)


@pytest.fixture
def fleet(tmp_path):
    """Three engines across two sub-datasets, with windows built from their Parquet.

    The windows come from the production `iter_windows` rather than being synthesised,
    so the fixture exercises the same shapes and downsampling the driver will meet.
    """
    from prognos.config import WindowConfig
    from prognos.window import iter_windows, stack_windows

    engines = [("DS01-005", 1), ("DS01-005", 2), ("DS02-006", 3)]
    for index, (sub, unit) in enumerate(engines):
        write_engine(tmp_path, sub, unit, seed=index)

    config = WindowConfig(length=50, stride=10, downsample=10)
    features, targets, meta, cycles, fractions = [], [], [], [], []
    for index, (sub, unit) in enumerate(engines):
        table = pq.read_table(tmp_path / sub / f"unit_{unit:03d}.parquet")
        windows = list(iter_windows(table, config, unit=unit, split="dev", subdataset=sub))
        stacked, window_targets, stacked_meta = stack_windows(windows)
        features.append(stacked)
        targets.append(window_targets)
        meta.append(stacked_meta)
        cycles.append(np.asarray(stacked_meta["cycle"], dtype=np.int64))
        fractions.append(
            np.asarray([w.fraction_of_life_remaining for w in windows], dtype=np.float32)
        )

    count = sum(f.shape[0] for f in features)
    return {
        "dir": tmp_path,
        "data": WindowSet(
            windows=np.concatenate(features).astype(np.float32),
            targets=np.concatenate(targets).astype(np.float32),
            groups=np.repeat(
                np.arange(len(engines), dtype=np.int64),
                [f.shape[0] for f in features],
            ),
            group_labels=tuple(GroupKey(sub, unit) for sub, unit in engines),
            cycles=np.concatenate(cycles),
            fractions=np.concatenate(fractions),
            splits=tuple("dev" for _ in range(count)),
            subdatasets=tuple("SYNTH" for _ in range(count)),
            max_cycle=CYCLES_PER_ENGINE,
        ),
        "engines": engines,
    }


class TestUnitTable:
    def test_reads_the_engine_that_was_asked_for(self, fleet) -> None:
        """The bug: reading `unit_001` when asked for global index 1.

        Global index 1 is the first engine listed, which is `DS01-005` unit 1 here and
        `DS01-005` unit 2 in the real dataset. Reading by bare index silently returns a
        neighbouring engine, and nothing downstream can tell.
        """
        subdataset, unit = fleet["engines"][0]
        cycles, features, _rul = _unit_table(fleet["dir"], subdataset, unit)
        assert cycles.size == CYCLES_PER_ENGINE * ROWS_PER_CYCLE
        assert features.shape[1] == len(FEATURE_COLUMNS)
        # Every engine in the fixture was written with a distinct seed, so a mismatch
        # between them is detectable by value.
        other_sub, other_unit = fleet["engines"][1]
        _, other_features, _ = _unit_table(fleet["dir"], other_sub, other_unit)
        assert not np.allclose(features, other_features)

    def test_a_missing_file_is_a_file_not_found(self, fleet) -> None:
        with pytest.raises(FileNotFoundError):
            _unit_table(fleet["dir"], "DS01-005", 99)


class TestTruthPerCycle:
    def test_maps_each_cycle_to_its_remaining_life(self) -> None:
        cycles = np.array([1, 1, 2, 2, 3], dtype=np.int64)
        rul = np.array([70.0, 70.0, 69.0, 69.0, 68.0])
        assert _truth_per_cycle(cycles, rul) == {1: 70.0, 2: 69.0, 3: 68.0}

    def test_keys_on_cycle_not_row_position(self) -> None:
        """Row-index keying drifts out of alignment the moment windows are a subset."""
        cycles = np.array([10, 20, 30], dtype=np.int64)
        rul = np.array([1.0, 2.0, 3.0])
        truth = _truth_per_cycle(cycles, rul)
        assert truth[20] == 2.0


class TestReplayPipeline:
    def _config(self, tmp_path):
        from prognos.config import Config, DataConfig, UncertaintyConfig

        return Config(
            data=DataConfig(parquet_dir=tmp_path, subdatasets=("DS01-005", "DS02-006")),
            uncertainty=UncertaintyConfig(target_coverage=0.9, calibration_fraction=0.34),
        )

    def test_replays_every_engine_and_reports_them_by_name(self, fleet) -> None:
        """Engines must be reported as `subdataset/unit`, not as a bare index."""
        outcome = replay_pipeline(fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3)
        names = {entry["engine"] for entry in outcome.report["engines"]}  # type: ignore[index]
        expected = {GroupKey(sub, unit).label() for sub, unit in fleet["engines"]}
        assert names == expected

    def test_each_engine_produces_windows(self, fleet) -> None:
        outcome = replay_pipeline(fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3)
        for entry in outcome.report["engines"]:  # type: ignore[index]
            assert entry["n_windows"] > 0, f"{entry['engine']} produced no windows"

    def test_the_coverage_verdict_is_reported(self, fleet) -> None:
        """The SLO check has to produce a verdict object, not merely collect data."""
        outcome = replay_pipeline(fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3)
        assert outcome.report["coverage"]["status"] in {  # type: ignore[index]
            "ok",
            "breach",
            "insufficient_data",
        }

    def test_calibration_engines_are_disjoint_from_the_fit_set(self, fleet) -> None:
        outcome = replay_pipeline(fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3)
        calibration = outcome.report["calibration"]  # type: ignore[index]
        assert calibration["n_engines"] > 0
        assert not set(calibration["engines"]) & set(calibration["fit_engines"])

    def test_the_model_artifact_is_written_when_asked(self, fleet, tmp_path) -> None:
        """Inference has nothing to load unless the replay leaves one behind."""
        model_dir = tmp_path / "model"
        replay_pipeline(
            fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3, model_dir=model_dir
        )
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "calibrator.json").exists()

    def test_no_model_artifact_is_written_by_default(self, fleet, tmp_path) -> None:
        replay_pipeline(fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3)
        assert not (tmp_path / "model").exists()

    def test_the_report_carries_provenance(self, fleet) -> None:
        outcome = replay_pipeline(fleet["data"], [0, 1, 2], self._config(fleet["dir"]), seed=3)
        report = outcome.report
        assert report["generated_at"]  # type: ignore[index]
        assert report["versions"]["lightgbm"]  # type: ignore[index]
        assert report["final_test_group"]["status"] == "closed"  # type: ignore[index]


def test_schema_column_count_matches_the_fixture() -> None:
    """A fixture that silently drops channels would make every test above vacuous."""
    assert len(FEATURE_COLUMNS) == len(SCENARIO_DESCRIPTORS) + len(MEASURED_SENSORS)
