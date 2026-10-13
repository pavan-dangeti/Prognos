"""Tests for the streaming telemetry contract and the online window.

Two properties matter more than anything else here, and both are about what happens
when a stream is wrong.

A live fleet stream is not the clean offline dataset. It arrives late, out of order,
duplicated, and with the occasional garbage reading. The ingestion contract has to
have a decided answer for each of those, and the answer has to be *counted*, because
a pipeline that silently drops bad input is indistinguishable from a pipeline with no
data at all.

The online window is the other half. `docs/design.md` §4.6 requires its arithmetic to
match the offline feature builder exactly. If it drifts, the model trained offline is
being fed something else in production, and nothing errors — the predictions just get
worse. So the test here compares against the real offline builder, not a restatement
of the formula.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from prognos.stream.events import (
    MAX_BATCH_ROWS,
    N_CHANNELS,
    Reading,
    StreamStats,
    TelemetryBatch,
    ValidationError,
    validate_batch,
)
from prognos.stream.window import OnlineWindow, replay_windows

WINDOW = 50
DOWNSAMPLE = 10
STRIDE = 10
# Rows per cycle, matching the real layout: one cycle number carries thousands of rows.
# 800 rows per cycle x downsample 10 = 80 downsampled rows, enough for 4 windows.
ROWS_PER_CYCLE = 800
N_CYCLES = 4


def make_reading(unit: int, cycle: int, channel: int, value: float) -> Reading:
    """One sensor value for one channel on one engine at one cycle."""
    return Reading(unit=unit, cycle=cycle, channel=channel, value=value)


def a_batch(n_rows: int = 5, unit: int = 1) -> TelemetryBatch:
    """A small well-formed batch: `n_rows` cycles, three channels each."""
    return TelemetryBatch(
        source="simulator",
        unit=unit,
        rows=[
            make_reading(unit, cycle, channel, float(cycle * 100 + channel))
            for cycle in range(1, n_rows + 1)
            for channel in range(3)
        ],
    )


class TestValidation:
    def test_accepts_a_wellformed_batch(self) -> None:
        result = validate_batch(a_batch())
        assert result.accepted == 15  # 5 cycles x 3 channels
        assert result.rejected == 0

    def test_rejects_a_channel_outside_the_instrument_list(self) -> None:
        """An unknown channel must not become a silently accepted zero column.

        This is the classic silent-corruption path: a channel out of range written
        into the feature matrix as zero is invisible, the model predicts happily, and
        every prediction afterwards is wrong in a way nothing flags.
        """
        result = validate_batch(
            TelemetryBatch(
                source="simulator", unit=1, rows=[make_reading(1, 1, N_CHANNELS + 5, 1.0)]
            )
        )
        assert result.accepted == 0
        assert result.rejected == 1
        assert any("channel" in r.reason for r in result.rejections)

    def test_rejects_a_nonfinite_value(self) -> None:
        """NaN and infinity must be refused, not clipped or imputed."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            result = validate_batch(
                TelemetryBatch(source="s", unit=1, rows=[make_reading(1, 1, 0, bad)])
            )
            assert result.rejected == 1, f"{bad} was accepted"

    def test_rejects_a_negative_cycle(self) -> None:
        result = validate_batch(
            TelemetryBatch(source="s", unit=1, rows=[make_reading(1, -1, 0, 1.0)])
        )
        assert result.rejected == 1

    def test_accepts_an_empty_batch_as_a_no_op(self) -> None:
        """No rows is a normal state, not an error to crash on."""
        result = validate_batch(TelemetryBatch(source="s", unit=1, rows=[]))
        assert result.accepted == 0
        assert result.rejected == 0

    def test_enforces_a_row_limit_by_rejecting_the_whole_batch(self) -> None:
        """An unbounded batch is a denial-of-service vector.

        The limit rejects the whole batch rather than truncating, because a silently
        truncated batch is byte-for-byte indistinguishable from a complete one.
        """
        oversized = TelemetryBatch(
            source="s", unit=1, rows=[make_reading(1, c, 0, 1.0) for c in range(1, 5001)]
        )
        result = validate_batch(oversized, max_rows=1000)
        assert result.accepted == 0
        assert result.rejected == 5000
        assert all("too many rows" in r.reason for r in result.rejections)

    def test_strict_mode_raises_for_a_request_boundary(self) -> None:
        """Strict is for HTTP boundaries, where a partial accept would be a lie."""
        oversized = TelemetryBatch(
            source="s", unit=1, rows=[make_reading(1, c, 0, 1.0) for c in range(1, 50)]
        )
        with pytest.raises(ValidationError):
            validate_batch(oversized, max_rows=10, strict=True)

    def test_the_default_limit_is_bounded(self) -> None:
        """The shipped default has to be a number, not an open door."""
        assert 0 < MAX_BATCH_ROWS <= 1_000_000

    def test_counts_rejects_rather_than_raising(self) -> None:
        """One bad row must not discard the good rows in the same batch."""
        result = validate_batch(
            TelemetryBatch(
                source="s",
                unit=1,
                rows=[
                    make_reading(1, 1, 0, 1.0),
                    make_reading(1, 2, 0, float("nan")),
                    make_reading(1, 3, 0, 3.0),
                ],
            )
        )
        assert result.accepted == 2
        assert result.rejected == 1


class TestStreamStats:
    def test_accumulates_across_batches(self) -> None:
        stats = StreamStats()
        stats.record(a_batch(), validate_batch(a_batch()))
        stats.record(a_batch(unit=2), validate_batch(a_batch(unit=2)))
        assert stats.batches == 2
        assert stats.accepted == 30  # two batches of 15
        assert stats.rejected == 0

    def test_groups_rejects_by_reason_class(self) -> None:
        """Free text makes a terrible metric label; the counters must be low cardinality.

        A per-row reason string embeds the offending value, so label cardinality
        grows with traffic and the scrape eventually breaks.
        """
        stats = StreamStats()
        stats.record(
            a_batch(),
            validate_batch(
                TelemetryBatch(
                    source="s",
                    unit=1,
                    rows=[make_reading(1, c, 0, float("nan")) for c in range(1, 5)],
                )
            ),
        )
        payload = stats.to_dict()
        assert set(payload["reject_reasons"]) == {"non_finite_value"}
        assert all(len(label) < 40 for label in payload["reject_reasons"])

    def test_is_serialisable_for_the_metrics_endpoint(self) -> None:
        stats = StreamStats()
        stats.record(a_batch(), validate_batch(a_batch()))
        payload = stats.to_dict()
        assert payload["batches"] == 1
        assert payload["accepted"] == 15


class TestOnlineWindow:
    def test_no_window_before_the_block_is_long_enough(self) -> None:
        """A partial window must be refused, not padded.

        Padding produces a plausible feature vector from a window that has not
        happened yet, and the model predicts on it with confidence.
        """
        window = OnlineWindow(length=WINDOW, downsample=DOWNSAMPLE, stride=STRIDE)
        for row in range((WINDOW - 1) * DOWNSAMPLE):
            window.push(1, np.full(N_CHANNELS, float(row)))
        assert window.downsampled_rows == WINDOW - 1
        assert not window.has_window

    def test_a_window_appears_at_exactly_the_right_row(self) -> None:
        window = OnlineWindow(length=WINDOW, downsample=DOWNSAMPLE, stride=STRIDE)
        ready = False
        for row in range(WINDOW * DOWNSAMPLE):
            ready = window.push(1, np.full(N_CHANNELS, float(row)))
        assert ready, "no window after a full block of downsampled rows"
        assert window.has_window

    def test_the_window_holds_the_downsample_rate(self) -> None:
        """With downsample 10, the window must skip 9 rows out of every 10.

        This is the arithmetic the §4.6 contract is about. An implementation that
        kept every row would produce windows that look right and are not.
        """
        window = OnlineWindow(length=4, downsample=3, stride=1)
        for row in range(12):
            window.push(1, np.full(N_CHANNELS, float(row)))
        while window.has_window:
            materialised = window.materialise()
        expected = [0.0, 3.0, 6.0, 9.0]
        np.testing.assert_allclose(materialised[:, 0], expected)

    def test_stride_advances_between_windows(self) -> None:
        window = OnlineWindow(length=3, downsample=1, stride=2)
        for row in range(9):
            window.push(1, np.full(N_CHANNELS, float(row)))
        first = window.materialise()
        np.testing.assert_allclose(first[:, 0], [0.0, 1.0, 2.0])
        second = window.materialise()
        np.testing.assert_allclose(second[:, 0], [2.0, 3.0, 4.0])

    def test_a_new_cycle_resets_the_block(self) -> None:
        """Blocks are per cycle, as in the offline builder."""
        window = OnlineWindow(length=WINDOW, downsample=DOWNSAMPLE, stride=STRIDE)
        for row in range(WINDOW * DOWNSAMPLE):
            window.push(1, np.full(N_CHANNELS, float(row)))
        assert window.has_window
        window.push(2, np.full(N_CHANNELS, 99.0))
        assert not window.has_window, "a new cycle must start a fresh block"

    def test_late_rows_are_dropped(self) -> None:
        """Reordering a block after the fact would change already-reported predictions."""
        window = OnlineWindow(length=3, downsample=1, stride=1)
        for row in range(5):
            window.push(2, np.full(N_CHANNELS, float(row)))
        window.push(1, np.full(N_CHANNELS, -1.0))
        assert window.cycle == 2

    def test_rejects_a_wrong_width_vector(self) -> None:
        window = OnlineWindow(length=3, downsample=1, stride=1)
        with pytest.raises(ValueError, match="channels"):
            window.push(1, np.zeros(N_CHANNELS + 1))

    def test_rejects_nonfinite_values(self) -> None:
        window = OnlineWindow(length=3, downsample=1, stride=1)
        with pytest.raises(ValueError, match="non-finite"):
            window.push(1, np.full(N_CHANNELS, float("nan")))

    def test_materialising_before_ready_raises(self) -> None:
        window = OnlineWindow(length=5, downsample=2, stride=1)
        window.push(1, np.zeros(N_CHANNELS))
        with pytest.raises(ValueError, match="downsampled rows"):
            window.materialise()


class TestMatchesTheOfflineBuilder:
    """The §4.6 contract: online arithmetic must equal offline arithmetic.

    Compared against `iter_windows` and `stack_windows` themselves, not a
    restatement of the formula. A restatement would pass when both copies of the
    formula were wrong in the same way, which is exactly when it matters.
    """

    @staticmethod
    def _fixture(seed: int = 0) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """A synthetic engine laid out like the real one: many rows per cycle."""
        rng = np.random.default_rng(seed)
        cycles = np.repeat(np.arange(1, N_CYCLES + 1), ROWS_PER_CYCLE).astype(np.int64)
        rows = rng.normal(size=(cycles.size, N_CHANNELS)).astype(np.float32)
        return cycles, rows

    @staticmethod
    def _offline(
        cycles: npt.NDArray[np.int64], rows: npt.NDArray[np.float32], stride: int = STRIDE
    ) -> npt.NDArray:
        """Windows from the production offline builder."""
        import pyarrow as pa

        from prognos.schema import TARGET_COLUMN
        from prognos.window import (
            FEATURE_COLUMNS,
            WindowConfig,
            iter_windows,
            stack_windows,
        )

        columns: dict[str, npt.NDArray] = {
            "cycle": cycles,
            TARGET_COLUMN: np.zeros(cycles.size, dtype=np.int64),
        }
        for index, name in enumerate(FEATURE_COLUMNS):
            columns[name] = rows[:, index]
        table = pa.table({k: pa.array(v) for k, v in columns.items()})
        windows = list(
            iter_windows(
                table,
                WindowConfig(length=WINDOW, stride=stride, downsample=DOWNSAMPLE),
                unit=1,
                split="dev",
                subdataset="SYNTH",
            )
        )
        assert windows, "the offline fixture produced no windows; the comparison is void"
        features, _targets, _meta = stack_windows(windows)
        return features

    def test_online_replay_equals_the_offline_builder(self) -> None:
        cycles, rows = self._fixture()
        offline = self._offline(cycles, rows)
        online = replay_windows(cycles, rows, length=WINDOW, stride=STRIDE, downsample=DOWNSAMPLE)
        assert online.shape == offline.shape, (
            f"online produced {online.shape[0]} windows of {online.shape[1:]} rows, "
            f"offline produced {offline.shape[0]} of {offline.shape[1:]}"
        )
        np.testing.assert_allclose(online, offline, rtol=1e-6, atol=1e-6)

    def test_the_match_survives_a_stride_sweep(self) -> None:
        """One matching stride could be luck. The arithmetic has to hold across the sweep."""
        cycles, rows = self._fixture(seed=3)
        for stride in (1, 5, STRIDE, 25):
            offline = self._offline(cycles, rows, stride=stride)
            online = replay_windows(
                cycles, rows, length=WINDOW, stride=stride, downsample=DOWNSAMPLE
            )
            assert online.shape == offline.shape, f"mismatch at stride {stride}"
            np.testing.assert_allclose(online, offline, rtol=1e-6, atol=1e-6)

    def test_replay_rejects_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            replay_windows(
                np.arange(10, dtype=np.int64),
                np.zeros((9, N_CHANNELS), dtype=np.float32),
                length=2,
                stride=1,
                downsample=1,
            )

    def test_replay_reports_when_no_window_is_possible(self) -> None:
        """Too little data is an error with a message, not an empty array."""
        with pytest.raises(ValueError, match="not enough data"):
            replay_windows(
                np.ones(10, dtype=np.int64),
                np.zeros((10, N_CHANNELS), dtype=np.float32),
                length=50,
                stride=10,
                downsample=10,
            )
