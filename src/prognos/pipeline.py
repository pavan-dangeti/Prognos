"""The replay driver: real held-out engines, through the real pipeline.

Phase 5's claim is that the streaming pipeline works and that the S5 coverage check
catches a calibration that has stopped holding. Both are only worth anything if the
replay uses real data, so this module reads the Parquet tables directly and feeds each
engine's raw sensor rows through the online window, rather than reusing the
materialised windows the offline evaluation produced.

Two distinctions that matter:

**The rows are replayed, not the windows.** `iter_windows` produces windows in one
pass; the online path produces the same windows one row at a time. Feeding the
pre-built windows into the online path would test nothing, because the window logic is
exactly what is in question.

**The engine's true RUL is the score.** The coverage monitor needs the truth at each
cycle, which only the dataset has. It is read from the same Parquet row the
prediction was made from, so the truth cannot drift out of alignment with the
prediction by an index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from .config import Config
from .dataset import WindowSource
from .evaluate import build_folds, library_versions
from .policy_eval import split_fit_and_calibration
from .stream.alerting import MaintenanceAlert
from .stream.coverage import CoverageVerdict
from .stream.inference import RulPredictor
from .stream.observability import MetricsRegistry
from .stream.pipeline import PipelineConfig, StreamPipeline
from .stream.window import replay_windows
from .window import FEATURE_COLUMNS


@dataclass(slots=True)
class ReplayOutcome:
    """Everything the replay produced, including the committed report."""

    report: dict[str, Any]
    verdict: CoverageVerdict
    alerts: dict[int, MaintenanceAlert]

    @property
    def total_alerts(self) -> int:
        """How many alerts fired over the whole replay."""
        return int(self.report["alerts"]["total"])


def _unit_table(
    parquet_dir: Path, subdataset: str, unit: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Read one engine's cycles, feature rows and true RUL from its Parquet table."""
    path = parquet_dir / subdataset / f"unit_{unit:03d}.parquet"
    table = pq.read_table(path, columns=["cycle", "rul", *FEATURE_COLUMNS])
    cycles = table.column("cycle").to_numpy().astype(np.int64)
    rul = table.column("rul").to_numpy().astype(np.float64)
    features = np.column_stack(
        [table.column(name).to_numpy().astype(np.float32) for name in FEATURE_COLUMNS]
    )
    return cycles, features, rul


def _truth_per_cycle(
    cycles: npt.NDArray[np.int64], rul: npt.NDArray[np.float64]
) -> dict[int, float]:
    """Map each cycle to the engine's true remaining life at that cycle."""
    return {int(c): float(r) for c, r in zip(cycles, rul, strict=True)}


def replay_pipeline(
    data: WindowSource,
    engines: list[int],
    config: Config,
    seed: int,
    calibration_fraction: float | None = None,
    logger: logging.Logger | None = None,
) -> ReplayOutcome:
    """Train the serving predictor, then replay each engine through the pipeline.

    The predictor is fitted on the cross-validation engines and calibrated on a
    disjoint subset of them, mirroring Phase 4, so the live coverage check is measuring
    the same thing the offline check measured.
    """
    log = logger or logging.getLogger("prognos.replay")
    fraction = (
        calibration_fraction
        if calibration_fraction is not None
        else config.uncertainty.calibration_fraction
    )
    parquet_dir = Path(config.data.parquet_dir)
    labels = data.group_labels

    # Fit and calibrate on disjoint engine sets, split by the same tested helper Phase 4
    # uses. Calibrating on a single engine gave a half-width of 6.3 against 17.5 for a
    # proper split, and live coverage of 0.716 against a 0.90 target: one engine's
    # residual distribution is not the fleet's, and the conformal quantile inherits
    # that engine's narrowness. The breadth of the calibration set is what the coverage
    # guarantee rests on, and one engine is not a breadth.
    groups = np.asarray(data.groups)
    in_scope = np.isin(groups, np.asarray(engines, dtype=np.int64))
    fit_mask, calibration_mask, calibration_units = split_fit_and_calibration(
        groups, in_scope, seed=seed, fraction=fraction
    )

    def subset_for(
        selector: npt.NDArray[np.bool_],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        subset = data.subset(selector)
        if subset.windows.shape[0] == 0:
            raise ValueError("a fit/calibration split produced no rows")
        return (
            subset.windows,
            subset.cycles,
            np.asarray(subset.targets, dtype=np.float32),
        )

    fit_windows, fit_cycles, fit_targets = subset_for(fit_mask)
    calibration_windows, calibration_cycles, calibration_targets = subset_for(calibration_mask)

    print("fitting the serving model", flush=True)
    predictor = RulPredictor.train(
        fit_windows,
        fit_cycles,
        fit_targets,
        calibration_windows,
        calibration_cycles,
        calibration_targets,
        target_coverage=config.uncertainty.target_coverage,
        seed=seed,
    )

    pipeline_config = PipelineConfig(
        window_length=config.window.length,
        window_stride=config.window.stride,
        downsample=config.window.downsample,
        rul_cap=config.window.rul_cap or 125,
        alert_threshold=config.cost.thresholds[0] if config.cost.thresholds else 20.0,
        target_coverage=config.uncertainty.target_coverage,
    )
    metrics = MetricsRegistry()
    pipeline = StreamPipeline(predictor, pipeline_config, metrics=metrics, logger=log)

    per_engine: list[dict[str, Any]] = []
    for group in engines:
        key = labels[group]
        try:
            cycles, features, rul = _unit_table(parquet_dir, key.subdataset, key.unit)
        except FileNotFoundError:
            print(f"  engine {group}: no Parquet table for {key.label()}, skipping", flush=True)
            continue
        truth = _truth_per_cycle(cycles, rul)
        replay = pipeline.run_engine(group, cycles, features, truth_by_cycle=truth)
        per_engine.append(
            {
                "engine": key.label(),
                "group": int(group),
                "n_rows": replay.n_rows,
                "n_windows": replay.n_windows,
                "n_alerts": len(replay.alerts),
                "mean_interval_width": float(
                    np.mean([p.interval.width for p in replay.predictions])
                )
                if replay.predictions
                else 0.0,
            }
        )
        print(
            f"  {key.label()}: {replay.n_windows} windows from {replay.n_rows} rows, "
            f"{len(replay.alerts)} alerts",
            flush=True,
        )

    pipeline.record_metrics()
    verdict = pipeline.coverage_verdict
    folds = build_folds(
        data,
        n_splits=config.split.n_splits,
        test_fraction=config.split.test_fraction,
        seed=seed,
    )

    report: dict[str, Any] = {
        "config": {
            "window": config.window.model_dump(),
            "uncertainty": config.uncertainty.model_dump(),
            "split": config.split.model_dump(),
            "subdatasets": list(config.data.subdatasets),
        },
        "hardware": _hardware(),
        "versions": library_versions(),
        "generated_at": date.today().isoformat(),
        "data": data.summary(),
        "final_test_group": {
            "engines": sorted(folds[0].split.test_units) if folds else [],
            "status": "closed",
            "note": (
                "Replayed engines are cross-validation engines. The final test group "
                "is still not opened in Phase 5."
            ),
        },
        "calibration": {
            "fraction": fraction,
            "engines": sorted(int(u) for u in calibration_units),
            "n_engines": len(calibration_units),
            "fit_engines": sorted(int(u) for u in groups[fit_mask]),
            "half_width": float(predictor.calibration_half_width),
        },
        "coverage": verdict.to_dict(),
        "engines": per_engine,
        "alerts": {
            "total": int(sum(int(entry["n_alerts"]) for entry in per_engine)),
            "open": {str(u): a.to_dict() for u, a in pipeline.open_alerts.items()},
        },
        "metrics_sample": metrics.render().splitlines()[:40],
    }
    return ReplayOutcome(report=report, verdict=verdict, alerts=pipeline.open_alerts)


def _hardware() -> str:
    """Platform description, matching the offline result files."""
    import platform

    return f"{platform.system()} {platform.machine()} / Python {platform.python_version()}"


__all__ = ["ReplayOutcome", "replay_pipeline", "replay_windows"]
