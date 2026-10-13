"""The online sliding window.

`docs/design.md` §4.6 requires the online feature arithmetic to match the offline
builder exactly. If the two drift, the model trained offline is being fed something
else in production and nothing raises — predictions just get quietly worse, and the
only symptom looks like sensor degradation.

The offline builder does not treat a cycle as one row. In this dataset a single cycle
number carries thousands of rows, and `iter_windows` downsamples *within* that block:
every `downsample`-th row of the block, then a `length`-row window slid across the
result by `stride`. An online window that assumed one row per cycle would produce
entirely different features while looking perfectly healthy, so this mirrors the block
structure rather than approximating it.

A window that is not yet full is *not* padded. Padding produces a plausible feature
vector from a window that has not happened yet, and the model will predict on it with
confidence. "Not enough data yet" is a legitimate state and is represented as such.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from prognos.schema import ALLOWED_INPUT_COLUMNS

N_CHANNELS = len(ALLOWED_INPUT_COLUMNS)


class OnlineWindow:
    """Streaming mirror of `prognos.window.iter_windows`, for one engine.

    Rows arrive in cycle order. Rows sharing a cycle number form one block. A window
    is emitted from that block once it holds enough downsampled rows and the count has
    reached the next stride boundary — which is exactly the loop the offline builder
    runs, expressed incrementally.
    """

    __slots__ = (
        "_block_rows",
        "_cycle",
        "_downsample",
        "_downsampled",
        "_last_cycle",
        "_length",
        "_next_offset",
        "_stride",
        "n_emitted",
    )

    def __init__(self, length: int, downsample: int, stride: int = 1) -> None:
        """Build a window emitting `length`-row slices, downsample rows apart."""
        if length < 1:
            raise ValueError(f"length must be positive, got {length}")
        if downsample < 1:
            raise ValueError(f"downsample must be positive, got {downsample}")
        if stride < 1:
            raise ValueError(f"stride must be positive, got {stride}")
        self._length = int(length)
        self._downsample = int(downsample)
        self._stride = int(stride)
        self._downsampled: list[npt.NDArray[np.float32]] = []
        self._block_rows = 0
        self._cycle: int | None = None
        self._last_cycle: int | None = None
        self._next_offset = 0
        self.n_emitted = 0

    @property
    def length(self) -> int:
        """Rows in a materialised window."""
        return self._length

    @property
    def cycle(self) -> int | None:
        """The cycle whose block is currently being filled."""
        return self._cycle

    @property
    def downsampled_rows(self) -> int:
        """Downsampled rows seen in the current cycle block."""
        return len(self._downsampled)

    @property
    def has_window(self) -> bool:
        """Whether a window is ready right now."""
        return self._next_offset + self._length <= len(self._downsampled)

    def push(self, cycle: int, values: npt.NDArray[np.float64]) -> bool:
        """Add one raw row. Returns True when a window became available.

        The cycle is passed separately rather than tucked into a channel slot, because
        every one of the `N_CHANNELS` entries is a real sensor and borrowing one would
        make a missing reading indistinguishable from a cycle number.

        Late arrivals are dropped rather than appended. Reordering a block after the
        fact would change predictions that have already been reported, which is worse
        than ignoring the late row.
        """
        row = np.asarray(values, dtype=np.float64)
        if row.shape != (N_CHANNELS,):
            raise ValueError(f"expected {N_CHANNELS} channels, got {row.shape}")
        if not np.isfinite(row).all():
            raise ValueError("values contain a non-finite entry")
        if self._last_cycle is None:
            self._last_cycle = int(cycle)
        if cycle < self._last_cycle:
            return False
        if cycle != self._cycle:
            self._cycle = int(cycle)
            self._last_cycle = int(cycle)
            self._downsampled = []
            self._block_rows = 0
            self._next_offset = 0
        self._block_rows += 1
        # `iter_windows` takes arange(0, n_rows, downsample), so the block's first row
        # is always included. Off by one here and every window shifts by one row,
        # which produces features that look right and are not.
        if (self._block_rows - 1) % self._downsample == 0:
            self._downsampled.append(row.astype(np.float32))
        return self.has_window

    def materialise(self) -> npt.NDArray[np.float32]:
        """The window at the next stride offset, advancing past it."""
        if not self.has_window:
            raise ValueError(
                f"cycle {self._cycle} has {len(self._downsampled)} downsampled rows; "
                f"need {_next_offset_label(self)}"
            )
        window = np.stack(self._downsampled[self._next_offset : self._next_offset + self._length])
        self._next_offset += self._stride
        self.n_emitted += 1
        return window


def _next_offset_label(window: OnlineWindow) -> str:
    """Human-readable shortfall, for the error message."""
    need = window._next_offset + window._length
    return f"{need} rows at stride offset {window._next_offset}"


def replay_windows(
    cycles: npt.NDArray[np.int64],
    rows: npt.NDArray[np.float32],
    length: int,
    stride: int,
    downsample: int,
) -> npt.NDArray[np.float32]:
    """Every window a replayed stream produces, in the offline builder's order.

    For batch replay and tests, where the whole engine history is in hand. The live
    path pushes rows one at a time and calls `materialise`.
    """
    cycle_array = np.asarray(cycles, dtype=np.int64)
    block = np.asarray(rows, dtype=np.float32)
    if cycle_array.ndim != 1 or cycle_array.shape[0] != block.shape[0]:
        raise ValueError("cycles and rows must be the same length")

    emitted: list[npt.NDArray[np.float32]] = []
    window = OnlineWindow(length=length, downsample=downsample, stride=stride)
    for index in range(block.shape[0]):
        window.push(int(cycle_array[index]), block[index])
        while window.has_window:
            emitted.append(window.materialise())
    if not emitted:
        raise ValueError(
            f"not enough data for one window: {block.shape[0]} rows, "
            f"need {length * downsample} within a single cycle block"
        )
    return np.stack(emitted)
