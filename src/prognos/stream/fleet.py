"""The fleet view and the cost panel — the two things an operator opens the dashboard for.

Neither number here is computed on request. The fleet view reads live state the
streaming pipeline maintains as readings arrive, because that is the only source that
reflects the stream in front of you. The cost panel reads the committed Phase 4 result
file, because a figure shown to a reader has to match a figure that is in `results/`
and can be re-derived by the command printed next to it.

The cost panel is a *comparison*, so the interesting content is not the winning policy
but the ratio between planned and unplanned work. Phase 4 measured that the
threshold policy loses to the point estimate and that planning only pays above a
failure-to-planned cost ratio; a panel showing a bare winner would hide both.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from prognos.dataset import GroupKey
from prognos.uncertainty import Interval

__all__ = ["FleetView", "build_cost_panel", "engine_label"]

#: Where the committed Phase 4 comparison lives.
POLICY_RESULT = Path("results/phase4_policy.json")

#: Worst first. A degraded engine is one nobody is watching; an alerting one is
#: already known, so it ranks below.
_RANK = {"degraded": 0, "alerting": 1, "nominal": 2}


def engine_label(source: str | None, unit: int) -> str:
    """Human-readable engine identity.

    `GroupKey.label()` is reused rather than reimplemented: unit numbers restart per
    sub-dataset, so the bare number is not an identity, and a second copy of the rule
    would eventually disagree with the first. Where the sub-dataset is unknown the row
    says `unit007` alone, which is honest about the gap rather than guessing one.
    """
    return GroupKey(subdataset=source, unit=unit).label() if source else f"unit{unit:03d}"


class FleetView:
    """One row per engine, overwritten as readings arrive.

    Locked because it is written by the stream thread and read by the HTTP handler, in
    the same process, for the entire life of the service.
    """

    __slots__ = ("_alerts", "_degraded", "_lock", "_rows")

    def __init__(self) -> None:
        """An empty fleet."""
        self._lock = threading.Lock()
        self._rows: dict[tuple[str | None, int], dict[str, Any]] = {}
        # Keyed like the rows, by `(source, unit)`. Keyed by unit alone, one engine's
        # alert showed on every row that shared its unit number.
        self._alerts: dict[tuple[str | None, int], dict[str, Any]] = {}
        self._degraded: dict[tuple[str | None, int], list[dict[str, Any]]] = {}

    def record(
        self,
        unit: int,
        cycle: int,
        interval: Interval,
        source: str | None = None,
    ) -> None:
        """Note an engine's latest prediction, replacing any row it already had."""
        with self._lock:
            self._rows[(source, unit)] = {
                "unit": unit,
                "source": source,
                "engine": engine_label(source, unit),
                "cycle": cycle,
                "rul": interval.point,
                "lower": interval.lower,
                "upper": interval.upper,
                # The interval's own width. Not a confidence score and not comparable
                # across models, but on one model over time a rising width is real, and
                # hiding it would make the panel calmer than the data is.
                "width": interval.width,
            }

    def set_alert(self, unit: int, alert: dict[str, Any] | None, source: str | None = None) -> None:
        """Attach or clear an engine's alert."""
        with self._lock:
            if alert is None:
                self._alerts.pop((source, unit), None)
            else:
                self._alerts[(source, unit)] = alert

    def set_degraded(
        self, unit: int, states: list[dict[str, Any]], source: str | None = None
    ) -> None:
        """Attach or clear an engine's degraded states."""
        with self._lock:
            self._degraded[(source, unit)] = list(states)

    def rows(self) -> list[dict[str, Any]]:
        """The fleet, worst first, then by unit so the view does not jump."""
        with self._lock:
            merged = []
            for row in self._rows.values():
                key = (row["source"], row["unit"])
                alert = self._alerts.get(key)
                degraded = self._degraded.get(key, [])
                merged.append(
                    {
                        **row,
                        "alert": alert.get("kind") if alert else None,
                        "alert_detail": alert,
                        "degraded": degraded,
                        "status": (
                            "degraded" if degraded else ("alerting" if alert else "nominal")
                        ),
                    }
                )
        merged.sort(key=lambda r: (_RANK[r["status"]], r["unit"]))
        return merged


def build_cost_panel(path: Path | str = POLICY_RESULT) -> dict[str, Any]:
    """The measured policy comparison, read from a committed result file.

    Raises rather than returning an empty panel: a dashboard showing no policies looks
    identical to one showing that no policy wins, and only one of those is a finding.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"no policy result at {source}; run `prognos policy --out {source}` first"
        )
    # `.get` rather than `[]`: a result file from a different phase, or a truncated
    # write, should say what is missing instead of raising KeyError at a reader.
    decision = json.loads(source.read_text()).get("decision", {})
    policies = decision.get("policies") or []
    if not policies:
        raise ValueError(
            f"{source} records no policies under 'decision', so there is nothing to "
            "compare; regenerate it with `prognos policy`"
        )

    ranked = sorted(policies, key=lambda p: p["cost_per_engine"])
    return {
        "policies": ranked,
        "best": ranked[0]["policy"],
        "cost_model": {
            **decision["cost_model"],
            "horizon": decision["horizon"],
            "n_episodes": decision["n_episodes"],
        },
        "ratio_sweeps": decision.get("ratio_sweeps", []),
        "source": str(source),
    }
