"""Window materialisation: cache identity, engine grouping and life fractions.

The two properties that matter here were both bugs first:

- **Engine identity.** Unit numbers restart at 1 in every sub-dataset, so DS01 unit
  2 and DS02 unit 2 are different engines. Grouping on the bare number merged them,
  and the reported error stopped measuring generalisation to a new engine.
- **Life fractions.** Normalising RUL by the dataset's longest life puts a 63-cycle
  engine and a 100-cycle engine at different points in life at the same cycle
  number, so the life-stage breakdown was comparing unlike things.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prognos.config import WindowConfig
from prognos.convert import convert_subdataset
from prognos.dataset import (
    GroupKey,
    _load_cache,
    _save_cache,
    cache_key,
    group_key,
    life_fractions,
    load_windows,
)
from prognos.schema import MEASURED_SENSORS
from prognos.window import FEATURE_COLUMNS

ROWS_PER_CYCLE = 400
CYCLES = 4
UNITS_PER_SUBDATASET = 3


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, tuple[str, ...]]:
    """Two sub-datasets with overlapping unit numbers, converted to Parquet."""
    import h5py

    root = tmp_path_factory.mktemp("dataset")
    parquet_dir = root / "parquet"
    rng = np.random.default_rng(0)
    subdatasets = ("SYNTH-A", "SYNTH-B")

    for index, subdataset in enumerate(subdatasets):
        directory = parquet_dir / subdataset
        directory.mkdir(parents=True)
        h5_path = directory / f"N-CMAPSS_{subdataset}.h5"
        for unit in range(1, UNITS_PER_SUBDATASET + 1):
            n = ROWS_PER_CYCLE * CYCLES
            unit_ids = np.full(n, unit, dtype=np.float64)
            cycle_ids = np.repeat(np.arange(1, CYCLES + 1, dtype=np.float64), ROWS_PER_CYCLE)
            meta = np.column_stack([unit_ids, cycle_ids, np.full(n, 1.0), np.zeros(n)])
            # Distinct base level per engine, so the two engines sharing a unit
            # number are distinguishable in the data as well as in the key.
            base = float(index * 100 + unit)
            with h5py.File(h5_path, "w") as handle:
                for split in ("dev", "test"):
                    handle.create_dataset(f"A_{split}", data=meta)
                    handle.create_dataset(f"W_{split}", data=rng.normal(size=(n, 4)))
                    sensors = rng.normal(size=(n, 14)) + base
                    handle.create_dataset(f"X_s_{split}", data=sensors)
                    handle.create_dataset(
                        f"Y_{split}",
                        data=np.repeat(
                            np.arange(CYCLES - 1, -1, -1, dtype=np.int64), ROWS_PER_CYCLE
                        ).reshape(-1, 1),
                    )
                handle.create_dataset("A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"]))
                handle.create_dataset("W_var", data=np.array([b"alt", b"Mach", b"TRA", b"T2"]))
                handle.create_dataset(
                    "X_s_var", data=np.array([s.encode() for s in MEASURED_SENSORS])
                )
            # One file per unit, as the real converter writes them.
            convert_subdataset(h5_path, directory, subdataset)
            for stale in directory.glob(f"unit_{unit:03d}.parquet"):
                if int(stale.stem.split("_")[1]) != unit:
                    stale.unlink()

    return parquet_dir, subdatasets


class TestGroupKey:
    def test_identifies_one_engine(self) -> None:
        """Identity is the (sub-dataset, unit) pair."""
        assert group_key("DS02-006", 2) == GroupKey("DS02-006", 2)

    def test_distinguishes_the_same_number_in_two_subdatasets(self) -> None:
        """The case that caused a real leak: DS01 unit 2 vs DS02 unit 2."""
        assert group_key("DS01-005", 2) != group_key("DS02-006", 2)

    def test_orders_deterministically(self) -> None:
        """Split reproducibility depends on a stable ordering."""
        keys = [GroupKey("DS02-006", 5), GroupKey("DS01-005", 5), GroupKey("DS02-006", 2)]
        assert sorted(keys) == [
            GroupKey("DS01-005", 5),
            GroupKey("DS02-006", 2),
            GroupKey("DS02-006", 5),
        ]

    def test_label_names_both_parts(self) -> None:
        """A label must identify the engine unambiguously."""
        assert GroupKey("DS02-006", 11).label() == "DS02-006/unit011"


class TestLifeFractions:
    def test_engine_at_end_of_life_is_zero(self) -> None:
        """RUL 0 is 0% of life remaining, whatever the engine's length."""
        fractions = life_fractions(np.zeros(2, dtype=np.int64), np.array([0, 0]))
        assert np.allclose(fractions, 0.0)

    def test_uses_each_engine_own_length(self) -> None:
        """Two engines at RUL 20 are at different points in life if their lengths differ.

        A 100-cycle engine at RUL 20 has 20% left; a 40-cycle engine at RUL 20 has
        50%. Using the dataset's longest life for both would place them together.
        """
        groups = np.array([0, 0, 1, 1], dtype=np.int64)
        # Engine 0 lives 100 cycles, engine 1 lives 40.
        rul = np.array([99, 20, 39, 20], dtype=np.int64)
        fractions = life_fractions(groups, rul)
        # The first window is cycle 1 of 100, so 99 of 100 cycles remain.
        assert fractions[0] == pytest.approx(0.99)
        assert fractions[1] == pytest.approx(0.20)
        assert fractions[2] == pytest.approx(0.975)
        assert fractions[3] == pytest.approx(0.50)
        # The point of the test: the same RUL is a different fraction of life.
        assert fractions[3] > fractions[1]

    def test_all_fractions_are_within_the_unit_interval(self) -> None:
        """A fraction outside [0, 1] would put a window in the wrong life stage."""
        groups = np.array([0, 1], dtype=np.int64)
        fractions = life_fractions(groups, np.array([50, 5], dtype=np.int64))
        assert np.all(fractions >= 0.0) and np.all(fractions <= 1.0)

    def test_monotone_within_an_engine(self) -> None:
        """Later cycles must always show less life remaining."""
        groups = np.zeros(10, dtype=np.int64)
        rul = np.arange(100, 90, -1, dtype=np.int64)
        fractions = life_fractions(groups, rul)
        assert np.all(np.diff(fractions) < 0)


class TestCacheKey:
    def test_changes_with_window_length(self) -> None:
        """Different windows must not share a cache entry."""
        a = cache_key(WindowConfig(length=50), ["DS02-006"])
        b = cache_key(WindowConfig(length=100), ["DS02-006"])
        assert a != b

    def test_changes_with_downsample(self) -> None:
        """Downsampling changes which rows enter a window."""
        a = cache_key(WindowConfig(downsample=5), ["DS02-006"])
        b = cache_key(WindowConfig(downsample=10), ["DS02-006"])
        assert a != b

    def test_changes_with_rul_cap(self) -> None:
        """The cap changes the labels, so a cached array is stale without it."""
        a = cache_key(WindowConfig(rul_cap=125), ["DS02-006"])
        b = cache_key(WindowConfig(rul_cap=80), ["DS02-006"])
        assert a != b

    def test_independent_of_subdataset_order(self) -> None:
        """Same set of sub-datasets, same cache entry."""
        assert cache_key(WindowConfig(), ["DS01-005", "DS02-006"]) == cache_key(
            WindowConfig(), ["DS02-006", "DS01-005"]
        )


class TestLoadWindows:
    def test_counts_every_engine_across_subdatasets(
        self, converted: tuple[Path, tuple[str, ...]]
    ) -> None:
        """Engine count is per sub-dataset, not per unit number."""
        parquet_dir, subdatasets = converted
        data = load_windows(
            parquet_dir,
            WindowConfig(length=10, stride=5, downsample=10),
            subdatasets,
            cache_dir=None,
        )
        assert len(data.group_labels) == UNITS_PER_SUBDATASET * len(subdatasets)

    def test_same_unit_number_yields_separate_groups(
        self, converted: tuple[Path, tuple[str, ...]]
    ) -> None:
        """The distinguishing property, checked on real loaded data."""
        parquet_dir, subdatasets = converted
        data = load_windows(
            parquet_dir,
            WindowConfig(length=10, stride=5, downsample=10),
            subdatasets,
            cache_dir=None,
        )
        by_number: dict[int, set[str]] = {}
        for key in data.group_labels:
            by_number.setdefault(key.unit, set()).add(key.subdataset)
        assert all(len(sub) == len(subdatasets) for sub in by_number.values())

    def test_windows_from_different_subdatasets_have_different_values(
        self, converted: tuple[Path, tuple[str, ...]]
    ) -> None:
        """Distinct engines must produce distinct sensor values, not just distinct keys."""
        parquet_dir, subdatasets = converted
        data = load_windows(
            parquet_dir,
            WindowConfig(length=10, stride=5, downsample=10),
            subdatasets,
            cache_dir=None,
        )
        per_engine = {
            int(g): float(data.windows[data.groups == g].mean()) for g in np.unique(data.groups)
        }
        assert len({round(v, 3) for v in per_engine.values()}) == len(per_engine)

    def test_cache_round_trip_preserves_everything(
        self, converted: tuple[Path, tuple[str, ...]], tmp_path: Path
    ) -> None:
        """A cached array that does not match the loaded one is worse than no cache."""
        parquet_dir, subdatasets = converted
        config = WindowConfig(length=10, stride=5, downsample=10)
        direct = load_windows(parquet_dir, config, subdatasets, cache_dir=None)
        cache_dir = tmp_path / "cache"
        cached = load_windows(parquet_dir, config, subdatasets, cache_dir=cache_dir)
        assert np.array_equal(direct.windows, cached.windows)
        assert np.array_equal(direct.groups, cached.groups)
        assert np.array_equal(direct.fractions, cached.fractions)
        assert direct.group_labels == cached.group_labels

    def test_second_load_comes_from_cache(
        self, converted: tuple[Path, tuple[str, ...]], tmp_path: Path
    ) -> None:
        """The cache is actually written and actually read."""
        parquet_dir, subdatasets = converted
        config = WindowConfig(length=10, stride=5, downsample=10)
        cache_dir = tmp_path / "cache"
        load_windows(parquet_dir, config, subdatasets, cache_dir=cache_dir)
        assert list(cache_dir.glob("windows_*.npz")), "no cache file written"
        from_cache = load_windows(parquet_dir, config, subdatasets, cache_dir=cache_dir)
        assert len(from_cache) > 0

    def test_rebuild_ignores_the_cache(
        self, converted: tuple[Path, tuple[str, ...]], tmp_path: Path
    ) -> None:
        """A stale cache must be discardable without deleting it by hand."""
        parquet_dir, subdatasets = converted
        config = WindowConfig(length=10, stride=5, downsample=10)
        cache_dir = tmp_path / "cache"
        load_windows(parquet_dir, config, subdatasets, cache_dir=cache_dir)
        fresh = load_windows(parquet_dir, config, subdatasets, cache_dir=cache_dir, rebuild=True)
        direct = load_windows(parquet_dir, config, subdatasets, cache_dir=None)
        assert np.array_equal(fresh.windows, direct.windows)

    def test_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        """Pointing at unconverted data must say so."""
        with pytest.raises(FileNotFoundError, match="run `prognos convert`"):
            load_windows(
                tmp_path / "absent",
                WindowConfig(length=10, downsample=10),
                ["NOPE"],
                cache_dir=None,
            )

    def test_window_too_long_for_a_cycle_is_an_error(
        self, converted: tuple[Path, tuple[str, ...]]
    ) -> None:
        """Downsample 1000 leaves fewer samples than the window length."""
        parquet_dir, subdatasets = converted
        with pytest.raises(ValueError, match="no windows produced"):
            load_windows(
                parquet_dir,
                WindowConfig(length=50, downsample=1000),
                subdatasets,
                cache_dir=None,
            )

    def test_summary_reports_engines_not_units(
        self, converted: tuple[Path, tuple[str, ...]]
    ) -> None:
        """The report must not call an engine a unit; the two are not the same."""
        parquet_dir, subdatasets = converted
        data = load_windows(
            parquet_dir,
            WindowConfig(length=10, stride=5, downsample=10),
            subdatasets,
            cache_dir=None,
        )
        summary = data.summary()
        assert "n_engines" in summary
        assert "n_units" not in summary
        assert all("/unit" in label for label in summary["engines"])

    def test_loaded_columns_match_the_allowlist(
        self, converted: tuple[Path, tuple[str, ...]]
    ) -> None:
        """A loaded window carries exactly the allowed channels, in order."""
        parquet_dir, subdatasets = converted
        data = load_windows(
            parquet_dir,
            WindowConfig(length=10, stride=5, downsample=10),
            subdatasets,
            cache_dir=None,
        )
        assert data.windows.shape[2] == len(FEATURE_COLUMNS)

    def test_incomplete_cache_is_rejected(self, tmp_path: Path) -> None:
        """A truncated cache file must fail rather than be read as empty."""
        path = tmp_path / "windows_bad.npz"
        np.savez_compressed(path, windows=np.zeros((1, 2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="missing"):
            _load_cache(path)


def test_save_and_load_cache_are_symmetric(tmp_path: Path) -> None:
    """Direct exercise of the cache writers, independent of any dataset."""
    from prognos.dataset import WindowSet

    data = WindowSet(
        windows=np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        targets=np.array([1.0, 2.0], dtype=np.float32),
        groups=np.array([0, 1], dtype=np.int64),
        group_labels=(GroupKey("A", 1), GroupKey("B", 2)),
        cycles=np.array([1, 2], dtype=np.int64),
        fractions=np.array([0.5, 0.25], dtype=np.float32),
        splits=("dev", "test"),
        subdatasets=("A", "B"),
        max_cycle=99,
    )
    path = tmp_path / "windows_x.npz"
    _save_cache(path, data)
    restored = _load_cache(path)
    assert np.array_equal(restored.windows, data.windows)
    assert np.array_equal(restored.fractions, data.fractions)
    assert restored.group_labels == data.group_labels
    assert restored.max_cycle == data.max_cycle
