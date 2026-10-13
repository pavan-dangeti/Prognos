"""Evaluation metrics.

Two things matter here and they are not the same thing.

RMSE treats every error equally. The NASA asymmetric score does not: predicting an
engine has more life left than it does is penalised roughly 1.8x more steeply than
predicting less, which is the correct operational asymmetry. A model can look
excellent on RMSE and still be dangerous, so both are reported for every model.

The scoring function is implemented from Saxena et al. (2008), *Metrics for
Evaluating Performance of Prognostic Methods*. Written test-first:
`tests/test_metrics.py` was written against the published formula before this
module existed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

# Asymmetric coefficients from Saxena et al. (2008). d < 0 is an early prediction
# (the model says less life remains than actually does) and d >= 0 is a late one.
# The smaller denominator on the late branch is what makes it more expensive.
EXPONENT_EARLY = 13.0
EXPONENT_LATE = 10.0

LIFE_STAGES: tuple[str, ...] = ("early", "mid", "late")


def rmse(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Root mean squared error in cycles."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot score an empty prediction set")
    return float(np.sqrt(np.mean(np.square(true - pred))))


def mae(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Mean absolute error in cycles."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot score an empty prediction set")
    return float(np.mean(np.abs(true - pred)))


def coverage_rate(_y_true: npt.NDArray[np.float64], covered: npt.NDArray[np.float64]) -> float:
    """Mean of a 0/1 indicator, registered as a bootstrap statistic.

    Registered so interval coverage can be resampled by engine like every other
    reported metric. Coverage is a rate over windows, so the resampling has to happen
    at the engine level to be meaningful: resampling windows would let one heavily
    sampled engine dominate the interval.

    The indicator is the **second** argument because the shared bootstrap signature
    is `statistic(y_true, y_pred)`. Naming the first parameter `y_true` and reading
    it would silently average the wrong array, which is exactly what happened: it
    produced a "95% confidence interval" of [18.0, 20.7] around an empirical coverage
    of 0.926. So the first parameter is underscored and unused, and callers pass the
    indicator twice.
    """
    flag = np.asarray(covered, dtype=np.float64)
    if flag.size == 0:
        raise ValueError("cannot score an empty prediction set")
    return float(np.mean(flag))


def nasa_score(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    """NASA asymmetric scoring function, summed over predictions. Lower is better.

    For each prediction error d = y_pred - y_true:

        d < 0  (early, conservative)  ->  exp(-d / 13) - 1
        d >= 0 (late, dangerous)      ->  exp(d / 10) - 1

    Both branches are non-negative, so the score is a sum of penalties and zero is
    only achievable by a perfect prediction.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("cannot score an empty prediction set")

    d = pred - true
    penalties = np.where(
        d < 0,
        np.exp(-d / EXPONENT_EARLY) - 1.0,
        np.exp(d / EXPONENT_LATE) - 1.0,
    )
    return float(np.sum(penalties))


def life_stage(fraction_remaining: float) -> str:
    """Bucket a window by how much life it has left.

    The cut points are thirds of the unit's life. Late-life accuracy is what drives
    maintenance decisions; a model that is excellent early and useless near failure
    is not useful, so the harness reports these separately.
    """
    if fraction_remaining <= 1.0 / 3.0:
        return "late"
    if fraction_remaining <= 2.0 / 3.0:
        return "mid"
    return "early"


def by_life_stage(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    fractions: npt.NDArray[np.float64],
) -> dict[str, dict[str, float | int]]:
    """Score each life stage separately.

    A stage with no windows is reported with `n = 0` and null metrics rather than
    being dropped, so a reader can see it was empty rather than assume it was fine.
    """
    stages = life_stages_for(fractions)
    result: dict[str, dict[str, float | int]] = {}
    for stage in LIFE_STAGES:
        mask = stages == stage
        count = int(mask.sum())
        if count == 0:
            result[stage] = {"n": 0, "rmse": float("nan"), "nasa_score": float("nan")}
            continue
        result[stage] = {
            "n": count,
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "nasa_score": nasa_score(y_true[mask], y_pred[mask]),
        }
    return result


def life_stages_for(fractions: npt.NDArray[np.float64]) -> npt.NDArray[np.str_]:
    """Vectorised life-stage labels."""
    values = np.asarray(fractions, dtype=np.float64)
    return np.where(
        values <= 1.0 / 3.0,
        "late",
        np.where(values <= 2.0 / 3.0, "mid", "early"),
    ).astype(np.str_)


def mean_ci(
    values: npt.NDArray[np.float64], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Mean and a normal-approximation confidence interval.

    Used for aggregating per-unit metrics, where unit count is in the single digits
    and an exact t-interval is not available. The caller is responsible for
    reporting n alongside the interval, because at n = 5 a normal approximation is
    a rough guide and should not be read as precise.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, float("nan"), float("nan")
    half_width = float(
        stats.t.ppf((1 + confidence) / 2, arr.size - 1) * arr.std(ddof=1) / np.sqrt(arr.size)
    )
    return mean, mean - half_width, mean + half_width


def grouped_bootstrap_ci(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    groups: npt.NDArray[np.int64],
    statistic: str = "rmse",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a metric by resampling engine units, not windows.

    Two windows from one engine share most of their rows, so resampling windows
    would report a far narrower interval than the evidence supports. The engine unit
    is the independent unit of observation, so it is what gets resampled. See
    docs/adr/0003.
    """
    func = {"rmse": rmse, "mae": mae, "nasa_score": nasa_score, "coverage": coverage_rate}[
        statistic
    ]
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    group_ids = np.unique(groups)

    # Precompute row indices per group: a unit can be drawn more than once in a
    # resample, so rows are gathered by index rather than by a boolean mask.
    indices_by_group = {int(g): np.flatnonzero(groups == g) for g in group_ids}

    observed = func(true, pred)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        picked = rng.choice(group_ids, size=group_ids.size, replace=True)
        rows = np.concatenate([indices_by_group[int(g)] for g in picked])
        samples[i] = func(true[rows], pred[rows])

    alpha = (1 - confidence) / 2
    return observed, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))
