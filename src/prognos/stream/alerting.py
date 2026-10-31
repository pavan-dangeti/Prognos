"""Maintenance alerting from the interval's lower bound.

An engine is due for maintenance when the **lower bound** of its interval falls to the
threshold, not the point estimate. The reasoning is the asymmetry from
`docs/design.md` §4.5: an early replacement costs a bounded, known visit, while running
an engine past its life costs an in-flight removal, which is larger and unbounded in
the tail. Acting on the optimistic edge would fire late precisely when the interval is
widest and the model is least sure.

Phase 4 found that on this data the lower bound fires more often than the point
estimate and costs more without preventing a single extra failure — the point-estimate
policy already prevented all of them. That is a statement about this data's very wide
intervals, not an argument for ignoring uncertainty, and it does not change what an
alert should be keyed on. The two live at different granularities: the policy is
costed over a fleet over a horizon, the alert is what an operator sees on one engine
at one cycle.

Two behaviours matter more than the threshold choice.

**Alerts fire once, not once per window.** A window produces a prediction every few
cycles, so an alert per window is a stream nobody reads. Each engine holds at most one
open maintenance alert.

**Recovery needs hysteresis.** A bound flapping across the threshold would otherwise
flap the alert with it, and a flapping alert is trained out of operators faster than
a wrong one. Recovery must clear the threshold by a margin before the alert resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAINTENANCE_DUE = "maintenance_due"
MAINTENANCE_RESOLVED = "maintenance_resolved"


@dataclass(frozen=True, slots=True)
class MaintenanceAlert:
    """One alert, with the numbers an operator needs to act on it."""

    kind: str
    unit: int
    cycle: int
    lower: float
    point: float
    threshold: float
    #: The engine's sub-dataset, the other half of its identity. Unit numbers restart in
    #: every sub-dataset, so an alert carrying only the unit can be closed by a
    #: different engine that happens to share the number.
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        """For the API, the dashboard and the alert log."""
        return {
            "kind": self.kind,
            "unit": self.unit,
            "cycle": self.cycle,
            "lower": self.lower,
            "point": self.point,
            "threshold": self.threshold,
            "source": self.source,
        }


@dataclass(slots=True)
class AlertingService:
    """Raises and resolves maintenance alerts per engine."""

    threshold: float
    hysteresis: float = 5.0
    _open: dict[tuple[str | None, int], MaintenanceAlert] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a configuration that cannot fire or cannot recover."""
        if self.threshold < 0:
            raise ValueError(
                f"threshold must be non-negative; remaining useful life never is, so a "
                f"negative threshold could never fire. Got {self.threshold}."
            )
        if self.hysteresis < 0:
            raise ValueError(f"hysteresis must be non-negative, got {self.hysteresis}")

    def evaluate(
        self, unit: int, cycle: int, lower: float, point: float, source: str | None = None
    ) -> list[MaintenanceAlert]:
        """Decide what, if anything, to emit for one engine at one cycle.

        Keyed by `(source, unit)`. Keyed by unit alone, DS02 unit 2 opened its life by
        resolving the alert DS01 unit 2 had left open, and every engine that shared a
        unit number with an earlier one recorded an alert it never raised.
        """
        key = (source, unit)
        existing = self._open.get(key)
        due = lower <= self.threshold

        if existing is None:
            if due:
                alert = MaintenanceAlert(
                    kind=MAINTENANCE_DUE,
                    unit=unit,
                    cycle=cycle,
                    lower=lower,
                    point=point,
                    threshold=self.threshold,
                    source=source,
                )
                self._open[key] = alert
                return [alert]
            return []

        # Already under alert: only a clear recovery, and only past the hysteresis
        # margin, closes it.
        if lower >= self.threshold + self.hysteresis:
            del self._open[key]
            return [
                MaintenanceAlert(
                    kind=MAINTENANCE_RESOLVED,
                    unit=unit,
                    cycle=cycle,
                    lower=lower,
                    point=point,
                    threshold=self.threshold,
                    source=source,
                )
            ]
        return []

    def open_alerts(self, unit: int | None = None, source: str | None = None) -> int:
        """How many alerts are open, fleet-wide or for one engine."""
        if unit is None:
            return len(self._open)
        return 1 if (source, unit) in self._open else 0

    def alerts(self) -> dict[tuple[str | None, int], MaintenanceAlert]:
        """Every open alert, keyed by `(source, unit)`."""
        return dict(self._open)
