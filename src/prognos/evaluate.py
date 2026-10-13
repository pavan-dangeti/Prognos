"""The evaluation harness.

This is where a result is actually produced, so the guarantees in the design
document are enforced here rather than trusted:

- folds partition by engine unit, and a unit is never on both sides
- the final test group is closed during cross-validation
- metrics carry the number of folds and units behind them
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from .dataset import GroupKey, WindowSource
from .metrics import by_life_stage, grouped_bootstrap_ci, rmse
from .split import UnitSplit, make_splits


@dataclass(slots=True)
class Fold:
    """One cross-validation fold, with row masks resolved against a dataset."""

    split: UnitSplit
    train_mask: npt.NDArray[np.bool_]
    val_mask: npt.NDArray[np.bool_]
    test_mask: npt.NDArray[np.bool_]


@dataclass(slots=True)
class FoldResult:
    """Metrics for one model on one fold."""

    model: str
    fold: str
    rmse: float
    nasa_score: float
    rmse_ci: tuple[float, float, float]
    nasa_ci: tuple[float, float, float]
    n_train: int
    n_eval: int
    train_units: tuple[int, ...]
    eval_units: tuple[int, ...]
    by_stage: dict[str, dict[str, float | int]]
    mean_signed_error: float
    late_predictions: float
    seed: int = 0


FitFn = Callable[[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.float32]], None]
PredictFn = Callable[[npt.NDArray[np.float32], npt.NDArray[np.int64]], npt.NDArray[np.float32]]


def build_folds(
    data: WindowSource,
    n_splits: int,
    test_fraction: float,
    seed: int,
) -> list[Fold]:
    """Construct row masks for each cross-validation fold.

    Grouping is by engine, where an engine is a (sub-dataset, unit) pair. Unit
    numbers restart per sub-dataset, so splitting on the bare number would put two
    different engines in one fold.

    The final test group is closed here and stays closed. It is carried on the
    `Fold` only so `evaluate_model` can assert it was not used.
    """
    groups = np.asarray(data.groups, dtype=np.int64)
    folds: list[Fold] = []
    for split in make_splits(
        groups.tolist(), n_splits=n_splits, test_fraction=test_fraction, seed=seed
    ):
        folds.append(
            Fold(
                split=split,
                train_mask=np.isin(groups, np.asarray(split.train_units)),
                val_mask=np.isin(groups, np.asarray(split.val_units)),
                test_mask=np.isin(groups, np.asarray(split.test_units)),
            )
        )
    return folds


def _check_masks(
    data: WindowSource,
    train_mask: npt.NDArray[np.bool_],
    eval_mask: npt.NDArray[np.bool_],
) -> tuple[np.ndarray, np.ndarray]:
    """Reject any split that leaks or is empty."""
    n_rows = int(np.asarray(data.groups).size)
    if train_mask.shape != (n_rows,) or eval_mask.shape != (n_rows,):
        raise ValueError(f"masks must have length {n_rows}")
    if np.any(train_mask & eval_mask):
        raise ValueError("training and evaluation masks overlap")
    if not train_mask.any():
        raise ValueError("no training rows in this fold")
    if not eval_mask.any():
        raise ValueError("no evaluation rows in this fold")

    return np.asarray(data.groups)[train_mask], np.asarray(data.groups)[eval_mask]


def evaluate_model(
    data: WindowSource,
    train_mask: npt.NDArray[np.bool_],
    eval_mask: npt.NDArray[np.bool_],
    model_name: str,
    fit: FitFn,
    predict: PredictFn,
    *,
    test_units: tuple[int, ...] = (),
    fold_name: str = "fold",
    seed: int = 0,
    n_bootstrap: int = 500,
) -> FoldResult:
    """Fit on the training rows, score on the evaluation rows.

    `train_mask` and `eval_mask` must not contain any engine unit from the final
    test group. That is checked by the caller, which knows the group; this function
    only verifies the masks do not overlap each other and are non-empty.
    """
    _check_masks(data, train_mask, eval_mask)

    train = data.subset(train_mask)
    evaluation = data.subset(eval_mask)

    # A fold's train and eval rows must exclude the final test group. That group is
    # opened once, after the model is frozen; scoring it here would make the
    # cross-validation number optimistic and the test set worthless.
    if test_units:
        ensure_groups_exclude_test_group(np.asarray(train.groups), test_units, "training rows")
        ensure_groups_exclude_test_group(
            np.asarray(evaluation.groups), test_units, "evaluation rows"
        )

    fit(train.windows, train.cycles, train.targets)
    prediction = predict(evaluation.windows, evaluation.cycles)

    if prediction.shape != evaluation.targets.shape:
        raise ValueError(
            f"model returned {prediction.shape} predictions for {evaluation.targets.shape} rows"
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"{model_name} produced non-finite predictions")

    truth = evaluation.targets.astype(np.float64)
    predicted = prediction.astype(np.float64)
    groups = np.asarray(evaluation.groups, dtype=np.int64)
    # Per-engine life fraction, already computed at load time.
    fractions = np.asarray(evaluation.fractions, dtype=np.float64)

    return FoldResult(
        model=model_name,
        fold=fold_name,
        rmse=rmse(truth, predicted),
        nasa_score=float(
            grouped_bootstrap_ci(truth, predicted, groups, "nasa_score", n_bootstrap, seed=seed)[0]
        ),
        rmse_ci=grouped_bootstrap_ci(truth, predicted, groups, "rmse", n_bootstrap, seed=seed),
        nasa_ci=grouped_bootstrap_ci(
            truth, predicted, groups, "nasa_score", n_bootstrap, seed=seed
        ),
        n_train=int(train_mask.sum()),
        n_eval=int(eval_mask.sum()),
        train_units=tuple(sorted(int(u) for u in np.unique(train.groups))),
        eval_units=tuple(sorted(int(u) for u in np.unique(groups))),
        by_stage=by_life_stage(truth, predicted, fractions),
        mean_signed_error=float(np.mean(predicted - truth)),
        # Fraction of predictions that are late, i.e. the dangerous direction.
        late_predictions=float(np.mean(predicted > truth)),
        seed=seed,
    )


def aggregate_folds(
    model_name: str,
    results: list[FoldResult],
    target_coverage: float = 0.90,
    engine_labels: Sequence[GroupKey] = (),
) -> dict[str, object]:
    """Summarise one model's folds, carrying the evidence count with the mean.

    A mean over 5 folds invites over-reading, so `n_folds` and the engine counts are
    part of the summary rather than something the reader has to go and count.
    Engine ids are integers; `engine_labels` turns them back into readable
    sub-dataset/unit names so a reader of results/ can check the split by eye.
    """
    from .metrics import mean_ci

    if not results:
        raise ValueError("cannot aggregate zero folds")

    rmse_mean, rmse_low, rmse_high = mean_ci(np.array([r.rmse for r in results]))
    nasa_mean, nasa_low, nasa_high = mean_ci(np.array([r.nasa_score for r in results]))

    stage_totals: dict[str, dict[str, float | int]] = {}
    for stage in ("early", "mid", "late"):
        per_fold = [r.by_stage[stage]["rmse"] for r in results]
        counts = [int(r.by_stage[stage]["n"]) for r in results]
        mean, low, high = mean_ci(np.asarray(per_fold, dtype=np.float64))
        stage_totals[stage] = {"n": sum(counts), "rmse": mean, "ci_low": low, "ci_high": high}

    return {
        "model": model_name,
        "n_folds": len(results),
        "n_eval_rows": sum(r.n_eval for r in results),
        "eval_engines": _engine_names(
            sorted({u for r in results for u in r.eval_units}), engine_labels
        ),
        "rmse": {"mean": rmse_mean, "ci_low": rmse_low, "ci_high": rmse_high},
        "nasa_score": {"mean": nasa_mean, "ci_low": nasa_low, "ci_high": nasa_high},
        "mean_signed_error": float(np.mean([r.mean_signed_error for r in results])),
        "late_prediction_rate": float(np.mean([r.late_predictions for r in results])),
        "by_stage": stage_totals,
        "target_coverage": target_coverage,
    }


def library_versions() -> dict[str, str]:
    """The library versions these numbers were produced on.

    A RMSE without the library versions behind it is not reproducible. This is not
    decoration: torch and LightGBM both change numerics between releases, and a
    result file that cannot say which versions produced it cannot be checked against
    a re-run.
    """
    import importlib.metadata as metadata
    import platform

    versions = {"python": platform.python_version()}
    for package in ("torch", "numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow", "h5py"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:  # pragma: no cover - environment dependent
            versions[package] = "not installed"
    return versions


def report_run(
    config: dict[str, object],
    data: WindowSource,
    folds: list[Fold],
    models: dict[str, object],
    hardware: str,
    model_reports: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the committed result file.

    Includes the split, the closed status of the final test group, the hardware the
    numbers came from, and per-model fit reports, so a reader of results/ never has
    to ask how they were produced.
    """
    labels = tuple(data.group_labels)
    test_units = sorted(folds[0].split.test_units) if folds else []
    return {
        "config": config,
        "hardware": hardware,
        "versions": library_versions(),
        "generated_at": date.today().isoformat(),
        "data": data.summary(),
        "final_test_group": {
            "engines": _engine_names(test_units, labels),
            "engine_ids": test_units,
            "status": "closed",
            "note": (
                "Not scored during cross-validation. Opened once, after the model "
                "and its configuration were frozen."
            ),
        },
        "folds": [
            {
                "name": fold.split.name,
                "train_engines": _engine_names(fold.split.train_units, labels),
                "val_engines": _engine_names(fold.split.val_units, labels),
            }
            for fold in folds
        ],
        "models": models,
        "model_fits": model_reports or {},
    }


def _engine_names(engine_ids: Sequence[int], labels: Sequence[GroupKey]) -> list[str]:
    """Turn integer engine ids back into readable sub-dataset/unit names.

    Falls back to the bare id when no labels were supplied, so a caller that has
    not built a label table still gets something readable rather than an error.
    """
    if not labels:
        return [str(i) for i in engine_ids]
    return [labels[i].label() if i < len(labels) else str(i) for i in engine_ids]


def ensure_groups_exclude_test_group(
    groups: npt.NDArray[np.int64],
    test_groups: tuple[int, ...],
    label: str,
) -> None:
    """Raise if any engine from the final test group appears in `groups`."""
    overlap = sorted({int(g) for g in groups} & {int(g) for g in test_groups})
    if overlap:
        raise ValueError(f"{label} contains engines from the final test group: {overlap}")
