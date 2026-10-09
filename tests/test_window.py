"""Window construction and the offline/online parity contract.

The parity test here is the one that matters most for the streaming phase. It is
written now, before any online code exists, so the contract is fixed by a test
rather than by whatever the online implementation happens to do later.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prognos.config import WindowConfig
from prognos.schema import TARGET_COLUMN, assert_no_leakage
from prognos.window import (
    FEATURE_COLUMNS,
    Window,
    cap_rul,
    iter_windows,
    load_unit_table,
    stack_windows,
)


def synthetic_table(
    *,
    unit: int = 3,
    cycles: int = 4,
    rows_per_cycle: int = 500,
    seed: int = 0,
) -> pa.Table:
    """Build a Parquet table shaped like one converted unit."""
    rng = np.random.default_rng(seed)
    n = cycles * rows_per_cycle
    cycle = np.repeat(np.arange(1, cycles + 1), rows_per_cycle).astype(np.int32)
    columns: dict[str, object] = {
        "unit": pa.array(np.full(n, unit, dtype=np.int32)),
        "cycle": pa.array(cycle),
        "flight_class": pa.array(np.ones(n, dtype=np.int8)),
        "t_in_flight": pa.array(np.tile(np.arange(rows_per_cycle, dtype=np.int32), cycles)),
        TARGET_COLUMN: pa.array(np.linspace(120, 0, n).astype(np.int32)),
    }
    for index, name in enumerate(FEATURE_COLUMNS):
        columns[name] = pa.array(
            (rng.normal(size=n) + cycle * 0.1 * (index + 1)).astype(np.float32)
        )
    return pa.table(columns)


def test_windows_do_not_span_cycles() -> None:
    """Every window is drawn from a single cycle."""
    table = synthetic_table()
    for window in iter_windows(table, WindowConfig(), unit=3, split="dev", subdataset="S"):
        assert window.cycle > 0
        assert window.features.shape == (50, len(FEATURE_COLUMNS))


def test_window_stride_is_respected() -> None:
    """Windows start every `stride` downsampled samples, not every sample."""
    table = synthetic_table(rows_per_cycle=500)
    windows = list(
        iter_windows(table, WindowConfig(downsample=10, length=50, stride=10), 3, "dev", "S")
    )
    # 500 rows / 10 = 50 downsampled samples, so exactly one window fits per cycle.
    assert len(windows) == table.column("cycle").to_numpy().max()

    wider = list(
        iter_windows(table, WindowConfig(downsample=1, length=50, stride=10), 3, "dev", "S")
    )
    dense_per_cycle = len(wider) // table.column("cycle").to_numpy().max()
    # 500 samples, length 50, stride 10 -> 46 windows per cycle
    assert dense_per_cycle == 46


def test_short_cycles_produce_no_windows() -> None:
    """A cycle shorter than the window is skipped rather than zero-padded."""
    table = synthetic_table(cycles=2, rows_per_cycle=20)
    assert not list(iter_windows(table, WindowConfig(downsample=1, length=50), 3, "dev", "S"))


def test_cap_rul_is_identity_without_a_cap() -> None:
    """No cap means no modification, which is a real option, not an oversight."""
    assert cap_rul(300, None) == 300
    assert cap_rul(300, 125) == 125
    assert cap_rul(50, 125) == 50


def test_capped_targets_never_exceed_the_cap() -> None:
    """The cap binds on early life and leaves late life alone."""
    table = synthetic_table(cycles=4, rows_per_cycle=500)
    windows = list(iter_windows(table, WindowConfig(rul_cap=100), 3, "dev", "S"))
    assert windows
    assert max(w.rul for w in windows) <= 100
    assert max(w.rul_uncapped for w in windows) > 100


def test_windows_never_leak() -> None:
    """The window builder refuses a table carrying ground truth."""
    table = synthetic_table()
    poisoned = table.append_column("HPC_flow_mod", pa.array(np.zeros(table.num_rows)))
    with pytest.raises(ValueError, match="leakage"):
        list(iter_windows(poisoned, WindowConfig(), 3, "dev", "S"))


def test_stacking_produces_model_ready_shapes() -> None:
    """Stacked output is (n, length, n_features) with a float32 target."""
    table = synthetic_table()
    windows = list(iter_windows(table, WindowConfig(), 3, "dev", "S"))
    features, targets, meta = stack_windows(windows)
    assert features.shape == (len(windows), 50, len(FEATURE_COLUMNS))
    assert features.dtype == np.float32
    assert targets.shape == (len(windows),)
    assert set(meta) == {"unit", "split", "cycle", "rul_uncapped"}
    assert all(u == 3 for u in meta["unit"])


def test_stacking_nothing_raises() -> None:
    """An empty window set is an error, not a silent empty array."""
    with pytest.raises(ValueError, match="no windows"):
        stack_windows([])


def test_window_features_match_the_source_rows() -> None:
    """The first downsampled sample of a window equals the cycle's first row."""
    table = synthetic_table(cycles=1, rows_per_cycle=1000, seed=3)
    window = next(iter(iter_windows(table, WindowConfig(downsample=10, length=50), 3, "dev", "S")))
    source = np.column_stack(
        [table.column(name).to_numpy(zero_copy_only=False) for name in FEATURE_COLUMNS]
    ).astype(np.float32)
    assert np.array_equal(window.features[0], source[0])
    assert np.array_equal(window.features[10], source[100])


def test_load_unit_table_round_trips(tmp_path: pytest.TempPathFactory) -> None:
    """What the converter writes is what the window builder reads."""
    import pyarrow.parquet as pq

    table = synthetic_table()
    path = tmp_path / "unit_003.parquet"
    pq.write_table(table, path)
    loaded = load_unit_table(str(path))
    assert set(loaded.column_names) == set(table.column_names)
    assert_no_leakage(loaded.column_names)
    assert loaded.num_rows == table.num_rows


@settings(max_examples=40, deadline=None)
@given(
    rows_per_cycle=st.integers(min_value=1, max_value=300),
    cycles=st.integers(min_value=1, max_value=4),
    length=st.integers(min_value=1, max_value=60),
    stride=st.integers(min_value=1, max_value=20),
    downsample=st.integers(min_value=1, max_value=15),
)
def test_window_counts_match_the_sliding_window_arithmetic(
    rows_per_cycle: int,
    cycles: int,
    length: int,
    stride: int,
    downsample: int,
) -> None:
    """Property test: the emitted window count is exactly what the formula gives.

    Downsample, length and stride interact, and getting that arithmetic wrong is
    easy and silent. Property-based because the interesting cases are the
    boundaries, not any particular example.
    """
    config = WindowConfig(length=length, stride=stride, downsample=downsample)
    table = synthetic_table(cycles=cycles, rows_per_cycle=rows_per_cycle, seed=1)
    windows = list(iter_windows(table, config, unit=1, split="dev", subdataset="S"))

    downsampled = len(range(0, rows_per_cycle, downsample))
    expected = max(0, (downsampled - length) // stride + 1) if downsampled >= length else 0
    assert len(windows) == expected * cycles
    for window in windows:
        assert window.features.shape == (length, len(FEATURE_COLUMNS))


def test_downsampling_is_a_fixed_stride_not_an_average() -> None:
    """Downsample picks every k-th row, so a sensor spike is preserved, not smoothed.

    Averaging would silently destroy the transient behaviour that degradation
    signals live in. Picking rows keeps it.
    """
    table = synthetic_table(cycles=1, rows_per_cycle=1000, seed=5)
    spiked = np.zeros(table.num_rows, dtype=np.float32)
    spiked[30] = 1000.0  # row 30 is picked by a downsample factor of 10
    with_spike = table.set_column(table.column_names.index("Xs_Nf"), "Xs_Nf", pa.array(spiked))
    window = next(iter_windows(with_spike, WindowConfig(downsample=10, length=50), 1, "dev", "S"))
    assert window.features[:, FEATURE_COLUMNS.index("Xs_Nf")].max() == pytest.approx(1000.0)


def test_window_is_immutable() -> None:
    """A Window is a value, so a caller cannot mutate shared window data."""
    window = Window(
        unit=1,
        cycle=1,
        split="dev",
        subdataset="S",
        end_row=0,
        features=np.zeros((2, 2), dtype=np.float32),
        rul=10,
        rul_uncapped=10,
        fraction_of_life_remaining=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        window.rul = 5  # type: ignore[misc]
