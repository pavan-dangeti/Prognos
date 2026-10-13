"""Phase 4 evaluation: calibrated intervals and the cost-based maintenance policy.

Both live in one command because the policy needs the intervals, and comparing
their widths across a shared fold structure needs the same fits.

**Calibration gets its own engines.** Within each fold the training engines are split
again into a fitting set and a calibration set. The model is fitted on the first and
the interval is calibrated on the second, which is held out of the fit entirely.
Calibration engines, evaluation engines and the final test group are three disjoint
sets, so the coverage number is not inflated by residuals from engines the model
memorised.

That nested split costs training engines, so the point RMSE under this harness is
slightly worse than the Phase 3 figure for the same model. Both are reported rather
than the better one quoted, because an interval is only interpretable next to the
accuracy of the model it came from.

**Split conformal and CQR come out of one fold pass.** Three models are fitted per
fold: the mean regressor for the point estimate, and two quantile regressors that
supply CQR's local width. Split conformal uses only the mean regressor's residuals.
Fitting them together is not an optimisation, it is what makes the width comparison
attributable: same folds, same calibration engines, same data, different construction.

**The final test group stays closed.** The same ten engines reserved since Phase 1
are not opened here. Nothing in Phase 4 has looked at them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupShuffleSplit

from .dataset import WindowSource
from .decision import (
    CostModel,
    EngineEpisode,
    FixedIntervalPolicy,
    FleetOutcome,
    RunToFailurePolicy,
    ThresholdPolicy,
    simulate_fleet,
    sweep_failure_ratio,
)
from .evaluate import Fold, library_versions
from .gbm import DEFAULT_PARAMS, LightGbmModel
from .metrics import LIFE_STAGES, grouped_bootstrap_ci, life_stages_for, rmse
from .models import RulModel
from .uncertainty import (
    CqrCalibrator,
    SplitConformalCalibrator,
    coverage_by_stage,
    interval_metrics,
)

CALIBRATION_FRACTION = 0.3
QUANTILE_ALPHAS = (0.05, 0.95)


@dataclass(slots=True)
class CalibratedFold:
    """One fold's predictions and intervals on its validation engines."""

    name: str
    y_true: npt.NDArray[np.float64]
    point: npt.NDArray[np.float64]
    cycles: npt.NDArray[np.int64]
    split_lower: npt.NDArray[np.float64]
    split_upper: npt.NDArray[np.float64]
    cqr_lower: npt.NDArray[np.float64]
    cqr_upper: npt.NDArray[np.float64]
    groups: npt.NDArray[np.int64]
    stages: npt.NDArray[np.str_]
    eval_engines: tuple[int, ...]
    calibration_engines: tuple[int, ...]
    fit_engines: tuple[int, ...]
    split_width: float
    cqr_width: float

    @property
    def n(self) -> int:
        """Number of scored windows."""
        return int(self.y_true.size)


def split_fit_and_calibration(
    groups: npt.NDArray[np.int64],
    train_mask: npt.NDArray[np.bool_],
    seed: int,
    fraction: float = CALIBRATION_FRACTION,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], tuple[int, ...]]:
    """Split a fold's training engines into fitting and calibration sets.

    Split by engine, not by window. A window-level split would put windows from the
    same engine on both sides, and since overlapping windows share most of their rows
    the calibration residuals would be optimistically small, which is the failure
    mode that turns a conformal coverage claim into a lie.
    """
    train_groups = np.asarray(groups)[train_mask]
    unique = np.asarray(sorted({int(u) for u in train_groups}))
    if unique.size < 3:
        raise ValueError(
            f"need at least 3 training engines to hold out a calibration set, got {unique.size}"
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    picked, _ = next(splitter.split(unique, np.zeros(unique.size), unique))
    calibration_units = tuple(sorted(int(u) for u in unique[picked]))
    fit_units = tuple(sorted(int(u) for u in unique if int(u) not in set(calibration_units)))
    calibration_mask = train_mask & np.isin(groups, np.asarray(calibration_units))
    fit_mask = train_mask & np.isin(groups, np.asarray(fit_units))
    return fit_mask, calibration_mask, calibration_units


def _fit(model: RulModel, data: WindowSource, mask: npt.NDArray[np.bool_]) -> RulModel:
    """Fit a model on a masked subset and return its predictions everywhere."""
    subset = data.subset(mask)
    model.fit(subset.windows, subset.cycles, subset.targets)
    return model


def _predict(
    model: RulModel, data: WindowSource, mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Point predictions over a masked subset."""
    subset = data.subset(mask)
    return np.asarray(model.predict(subset.windows, subset.cycles), dtype=np.float64)


def evaluate_fold(
    data: WindowSource,
    fold: Fold,
    seed: int,
    target_coverage: float,
) -> CalibratedFold:
    """Fit three models on a fold and produce both interval constructions.

    The mean regressor supplies the point estimate and split conformal's residuals.
    The two quantile regressors supply CQR's local width. All three are fitted on the
    same fitting engines and calibrated on the same calibration engines, so the two
    widths are comparable and the comparison is about the construction rather than
    about the data.
    """
    groups = np.asarray(data.groups)
    fit_mask, calibration_mask, calibration_units = split_fit_and_calibration(
        groups, fold.train_mask, seed
    )

    point_model = _fit(LightGbmModel(seed=seed), data, fit_mask)
    lower_model = _fit(
        LightGbmModel(
            params={**DEFAULT_PARAMS, "objective": "quantile", "alpha": QUANTILE_ALPHAS[0]},
            seed=seed,
        ),
        data,
        fit_mask,
    )
    upper_model = _fit(
        LightGbmModel(
            params={**DEFAULT_PARAMS, "objective": "quantile", "alpha": QUANTILE_ALPHAS[1]},
            seed=seed,
        ),
        data,
        fit_mask,
    )

    calibration_true = np.asarray(data.subset(calibration_mask).targets, dtype=np.float64)
    split_calibrator = SplitConformalCalibrator(target_coverage).calibrate(
        _predict(point_model, data, calibration_mask) - calibration_true
    )
    cqr_calibrator = CqrCalibrator(target_coverage).calibrate(
        _predict(lower_model, data, calibration_mask),
        _predict(upper_model, data, calibration_mask),
        calibration_true,
    )

    evaluation = data.subset(fold.val_mask)
    point = _predict(point_model, data, fold.val_mask)
    split_lower, split_upper = split_calibrator.bounds(point)
    cqr_lower, cqr_upper = cqr_calibrator.bounds(
        _predict(lower_model, data, fold.val_mask),
        _predict(upper_model, data, fold.val_mask),
    )

    fit_engines = tuple(sorted({int(u) for u in groups[fit_mask]}))
    return CalibratedFold(
        name=fold.split.name,
        y_true=np.asarray(evaluation.targets, dtype=np.float64),
        point=point,
        cycles=np.asarray(data.cycles)[fold.val_mask],
        split_lower=split_lower,
        split_upper=split_upper,
        cqr_lower=cqr_lower,
        cqr_upper=cqr_upper,
        groups=groups[fold.val_mask],
        stages=life_stages_for(np.asarray(data.fractions, dtype=np.float64)[fold.val_mask]),
        eval_engines=fold.split.val_units,
        calibration_engines=calibration_units,
        fit_engines=fit_engines,
        split_width=float(split_calibrator.width),
        cqr_width=float(cqr_calibrator.width),
    )


def summarise(
    folds: list[CalibratedFold],
    method: str,
    target_coverage: float,
    seed: int,
) -> dict[str, object]:
    """Coverage, width and stage breakdown over all folds' validation windows.

    The coverage confidence interval resamples engines, not windows, for the same
    reason every other interval in this project does.
    """
    low_key, high_key = f"{method}_lower", f"{method}_upper"
    truth = np.concatenate([f.y_true for f in folds])
    lower = np.concatenate([getattr(f, low_key) for f in folds])
    upper = np.concatenate([getattr(f, high_key) for f in folds])
    point = np.concatenate([f.point for f in folds])
    groups = np.concatenate([f.groups for f in folds])
    stages = np.concatenate([f.stages for f in folds])

    covered = ((truth >= lower) & (truth <= upper)).astype(np.float64)
    # The bootstrap statistic signature is (y_true, y_pred), and coverage is a mean
    # over the first argument alone. The indicator is passed twice so the call
    # matches the shared signature instead of quietly averaging the bounds, which is
    # what it did the first time and produced a "confidence interval" of [18.0, 20.7]
    # around a coverage of 0.93.
    _, coverage_low, coverage_high = grouped_bootstrap_ci(
        covered,
        covered,
        groups,
        statistic="coverage",
        n_resamples=500,
        seed=seed,
    )
    metrics = interval_metrics(truth, lower, upper)
    return {
        "method": method,
        "target_coverage": target_coverage,
        "empirical_coverage": metrics["coverage"],
        "coverage_ci": [coverage_low, coverage_high],
        "mean_width": metrics["mean_width"],
        "point_rmse": rmse(truth, point),
        "by_stage": coverage_by_stage(truth, lower, upper, stages, tuple(LIFE_STAGES)),
        "n_eval_rows": metrics["n"],
        "n_eval_engines": len({int(g) for g in groups}),
        "eval_engines": sorted({int(g) for g in groups}),
        "calibration_engines": sorted({int(u) for f in folds for u in f.calibration_engines}),
        "fit_engines": sorted({int(u) for f in folds for u in f.fit_engines}),
        # Recorded per fold because that is the granularity the claim holds at. Each
        # fold fits a different training set, so the union across folds legitimately
        # overlaps: a set-level check would report a leak that is not one, and miss a
        # real leak inside a single fold.
        "folds": [
            {
                "name": f.name,
                "fit_engines": sorted(int(u) for u in f.fit_engines),
                "calibration_engines": sorted(int(u) for u in f.calibration_engines),
                "eval_engines": sorted(int(u) for u in f.eval_engines),
            }
            for f in folds
        ],
        "calibration_widths": [getattr(f, f"{method}_width") for f in folds],
        "coverage_within_target": bool(abs(metrics["coverage"] - target_coverage) <= 0.05),
    }


def build_episodes(folds: list[CalibratedFold]) -> list[EngineEpisode]:
    """One episode per validation engine, in **cycle** units.

    Episodes are indexed by real cycle number, not by window rank. Consecutive
    windows for one engine are a stride apart and each covers a window length, so a
    window index is not a cycle and an episode built from window ranks is roughly two
    orders of magnitude too long. That mistake is invisible in the output: the
    simulator simply never reaches the end of an engine, every policy records zero
    failures, and run-to-failure appears to cost nothing at all.

    Cycles with no window covering them are left without a prediction, and the
    simulator does not consult an episode there, so a gap reads as "no decision
    available this cycle" rather than as a fabricated one.
    """
    cycles = np.concatenate([f.cycles for f in folds])
    point = np.concatenate([f.point for f in folds])
    lower = np.concatenate([f.split_lower for f in folds])
    groups = np.concatenate([f.groups for f in folds])

    episodes: list[EngineEpisode] = []
    for engine in sorted({int(g) for g in groups}):
        mask = groups == engine
        life = int(cycles[mask].max()) + 1
        lower_by_cycle = np.full(life, np.nan)
        point_by_cycle = np.full(life, np.nan)
        lower_by_cycle[cycles[mask]] = np.clip(lower[mask], 0.0, None)
        point_by_cycle[cycles[mask]] = np.clip(point[mask], 0.0, None)
        episodes.append(
            EngineEpisode(
                engine=f"engine_{engine}",
                true_life=life,
                lower=lower_by_cycle,
                point=point_by_cycle,
            )
        )
    return episodes


def _policy_row(outcome: FleetOutcome, baseline_cost: float) -> dict[str, object]:
    """One policy's row, with savings measured against the run-to-failure baseline."""
    return {
        "policy": outcome.policy,
        "total_cost": outcome.total_cost,
        "cost_per_engine": outcome.cost_per_engine,
        "n_planned": outcome.n_planned,
        "n_failures": outcome.n_failures,
        "n_engines": outcome.n_engines,
        "savings_vs_run_to_failure": baseline_cost - outcome.total_cost,
    }


def evaluate_policies(
    episodes: list[EngineEpisode],
    cost_model: CostModel,
    horizon: int,
    thresholds: list[float],
    fixed_intervals: list[int],
    ratios: list[float],
    seed: int,
) -> dict[str, object]:
    """Cost every policy on the same fleet, then sweep the failure/planned ratio.

    The lower-bound threshold policy is the one under test. The point-estimate
    variant and the fixed calendar are there to be beaten, and run-to-failure is
    there to be the floor.
    """
    baseline = simulate_fleet(
        episodes, RunToFailurePolicy(), cost_model, horizon=horizon, seed=seed
    )
    rows: list[dict[str, object]] = [_policy_row(baseline, baseline.total_cost)]
    for interval in fixed_intervals:
        outcome = simulate_fleet(
            episodes, FixedIntervalPolicy(interval), cost_model, horizon=horizon, seed=seed
        )
        rows.append(_policy_row(outcome, baseline.total_cost))
    for threshold in thresholds:
        for use_lower in (True, False):
            outcome = simulate_fleet(
                episodes,
                ThresholdPolicy(threshold, use_lower_bound=use_lower),
                cost_model,
                horizon=horizon,
                seed=seed,
            )
            rows.append(_policy_row(outcome, baseline.total_cost))

    return {
        "cost_model": cost_model.as_dict(),
        "horizon": horizon,
        "n_episodes": len(episodes),
        "episode_lives": {
            "min": min(e.true_life for e in episodes),
            "median": int(np.median([e.true_life for e in episodes])),
            "max": max(e.true_life for e in episodes),
        },
        "policies": rows,
        "ratio_sweeps": [
            {
                "threshold": threshold,
                "rows": sweep_failure_ratio(
                    episodes,
                    ThresholdPolicy(threshold, use_lower_bound=True),
                    cost_model,
                    horizon=horizon,
                    ratios=ratios,
                    seed=seed,
                ),
            }
            for threshold in thresholds
        ],
    }


def report_phase4(
    intervals: dict[str, dict[str, object]],
    decision: dict[str, Any],
    config: dict[str, object],
    data: WindowSource,
    folds: list[Fold],
    hardware: str,
) -> dict[str, object]:
    """Assemble the committed Phase 4 result file.

    Carries the same provenance fields as every other result file, because a cost
    curve without the cost model behind it is not a result.
    """
    return {
        "config": config,
        "hardware": hardware,
        "versions": library_versions(),
        "generated_at": date.today().isoformat(),
        "data": data.summary(),
        "final_test_group": {
            "engines": sorted(folds[0].split.test_units) if folds else [],
            "status": "closed",
            "note": "Not opened in Phase 4. These results are cross-validated only.",
        },
        "intervals": intervals,
        "decision": decision,
    }
