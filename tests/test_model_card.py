"""The model card quotes numbers; each must match the result file it names."""

from __future__ import annotations

import json
from pathlib import Path

CARD = Path("docs/model-card.md")


def card() -> str:
    # Whitespace collapsed, so a figure wrapped across a line still counts as quoted.
    return " ".join(CARD.read_text().split())


def load(path: str) -> dict:  # type: ignore[type-arg]
    return json.loads(Path(path).read_text())


def test_the_rmse_table_matches_the_cross_validation_result() -> None:
    models = load("results/phase3_sequence.json")["models"]
    text = card()
    for name in ("lightgbm", "cnn", "linear_cycle"):
        rmse, nasa = models[name]["rmse"], models[name]["nasa_score"]
        expected = (
            f"| {rmse['mean']:.2f} | {rmse['ci_low']:.2f}\u2013{rmse['ci_high']:.2f} "
            f"| {nasa['mean']:,.0f} |"
        )
        assert expected in text, f"{name}: {expected}"
    stages = models["lightgbm"]["by_stage"]
    assert (
        f"early {stages['early']['rmse']:.2f}, mid {stages['mid']['rmse']:.2f}, "
        f"late {stages['late']['rmse']:.2f}"
    ) in text
    assert f"{100 * models['lightgbm']['late_prediction_rate']:.1f}% of its" in text


def test_the_coverage_table_matches_the_interval_results() -> None:
    split = load("results/phase4_policy.json")["intervals"]["split"]
    text = card()
    assert f"| {split['empirical_coverage']:.3f} | {split['mean_width']:.1f} |" in text
    for stage, row in split["by_stage"].items():
        assert f"| {stage} life | {row['coverage']:.3f} | {row['mean_width']:.1f} |" in text
    live = load("results/phase5_streaming.json")
    assert f"| {live['coverage']['measured_coverage']:.3f} |" in text
    assert f"half-width {live['calibration']['half_width']:.1f}" in text


def test_the_age_comparison_matches_the_published_run() -> None:
    ours = load("results/phase6_published_comparison.json")["ours"]
    text = card()
    assert f"LightGBM scores {ours['lightgbm']['rmse']['14+15']:.2f}" in text
    assert f"no sensors {ours['linear_cycle']['rmse']['14+15']:.2f}" in text
