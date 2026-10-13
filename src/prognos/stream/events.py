"""The telemetry contract for the streaming pipeline.

A live fleet stream is not the clean offline dataset. Readings arrive late, out of
order, duplicated, and occasionally as garbage. Every one of those needs a decided
answer, and the answer needs to be *counted*, because a pipeline that silently drops
bad input is indistinguishable from a pipeline that has stopped receiving data.

So validation never raises on content and never truncates silently. It returns the
accepted rows, counts the rejects, and records a reason per rejected row. A caller
that wants a batch rejected outright asks for `strict`, which is what a request
boundary should do, because a partially-accepted HTTP response would claim to be
something it is not.

The size limit is a hard reject rather than a truncation for the same reason: a
silently truncated batch is byte-for-byte indistinguishable from a complete one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from prognos.schema import ALLOWED_INPUT_COLUMNS

N_CHANNELS = len(ALLOWED_INPUT_COLUMNS)

#: Hard cap on rows in one batch. Sized so a legitimate batch of every channel at
#: every cycle of a short window fits comfortably, and so a hostile caller cannot
#: make the server allocate without limit. See docs/design.md §11.
MAX_BATCH_ROWS = 20_000


class ValidationError(ValueError):
    """A batch that cannot be accepted at all."""


@dataclass(frozen=True, slots=True)
class Reading:
    """One sensor value: one engine, one cycle, one channel."""

    unit: int
    cycle: int
    channel: int
    value: float

    def reason_if_invalid(self) -> str | None:
        """Why this reading is unacceptable, or None if it is fine."""
        if self.channel < 0 or self.channel >= N_CHANNELS:
            return f"channel {self.channel} is outside the instrument list [0, {N_CHANNELS})"
        if self.cycle < 0:
            return f"cycle {self.cycle} is negative"
        if not np.isfinite(self.value):
            return f"value {self.value} is not finite"
        return None


@dataclass(frozen=True, slots=True)
class Rejection:
    """A rejected row and why. Counted, never silently dropped."""

    unit: int
    cycle: int
    channel: int
    reason: str


@dataclass(slots=True)
class TelemetryBatch:
    """A batch of readings for one engine."""

    source: str
    unit: int
    rows: list[Reading]
    received_at: str | None = None

    def to_frame(self) -> npt.NDArray[np.float32]:
        """Accepted-by-construction `(cycles, channels)` array, missing rows as NaN.

        NaN rather than zero: a missing cycle is unknown, and zero is a real
        measurement that the offline feature builder would happily average in.
        """
        if not self.rows:
            return np.empty((0, N_CHANNELS), dtype=np.float32)
        max_cycle = max(r.cycle for r in self.rows)
        out = np.full((max_cycle + 1, N_CHANNELS), np.nan, dtype=np.float32)
        for row in self.rows:
            out[row.cycle, row.channel] = np.float32(row.value)
        return out


@dataclass(slots=True)
class ValidationResult:
    """What survived validation, and what did not."""

    rows: list[Reading]
    rejections: list[Rejection]

    @property
    def accepted(self) -> int:
        """Count of accepted rows."""
        return len(self.rows)

    @property
    def rejected(self) -> int:
        """Count of rejected rows."""
        return len(self.rejections)


def validate_batch(
    batch: TelemetryBatch,
    max_rows: int = MAX_BATCH_ROWS,
    strict: bool = False,
) -> ValidationResult:
    """Validate a batch, counting rejects rather than raising on them.

    `strict` is for request boundaries: it raises on an oversized batch so the
    caller can return a 4xx, rather than returning a partial accept that looks
    complete to the client.
    """
    if len(batch.rows) > max_rows:
        message = f"too many rows: batch has {len(batch.rows)}, limit is {max_rows}"
        if strict:
            raise ValidationError(message)
        return ValidationResult(
            rows=[],
            rejections=[Rejection(batch.unit, 0, 0, message) for _ in range(len(batch.rows))],
        )

    accepted: list[Reading] = []
    rejections: list[Rejection] = []
    for row in batch.rows:
        reason = row.reason_if_invalid()
        if reason is None:
            accepted.append(row)
        else:
            rejections.append(Rejection(row.unit, row.cycle, row.channel, reason))
    return ValidationResult(rows=accepted, rejections=rejections)


@dataclass(slots=True)
class StreamStats:
    """Running counters for the metrics endpoint.

    Rejects are tracked by reason rather than as a single total, because "we dropped
    400 rows" is not actionable and "we dropped 400 rows for channel-out-of-range" is
    the whole diagnosis.
    """

    batches: int = 0
    accepted: int = 0
    rejected: int = 0
    reject_reasons: Counter[str] = field(default_factory=Counter)

    def record(self, batch: TelemetryBatch, result: ValidationResult) -> None:
        """Fold one batch and its validation result into the counters."""
        self.batches += 1
        self.accepted += result.accepted
        self.rejected += result.rejected
        for rejection in result.rejections:
            self.reject_reasons[_reason_class(rejection.reason)] += 1

    def to_dict(self) -> dict[str, object]:
        """Counters as plain data, for JSON logging and the metrics endpoint."""
        return {
            "batches": self.batches,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "reject_reasons": dict(self.reject_reasons),
        }


def _reason_class(reason: str) -> str:
    """Collapse a free-text reason into a stable metric label.

    Free text makes a terrible metric label: it includes row-specific values, so the
    label cardinality grows with the traffic and the scrape breaks.
    """
    if "channel" in reason:
        return "channel_out_of_range"
    if "not finite" in reason:
        return "non_finite_value"
    if "negative" in reason:
        return "negative_cycle"
    if "limit is" in reason:
        return "batch_too_large"
    return "other"
