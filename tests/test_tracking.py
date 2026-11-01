"""Tests for experiment tracking and the model registry.

Each test gets its own SQLite store, so registered versions start at 1 and nothing
reaches the working tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prognos.tracking import (
    REGISTRY_FILE,
    artifact_root,
    export_model,
    metrics_of,
    params_of,
    record_run,
    resolve_model,
)


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    return tmp_path


def model_dir(root: Path, marker: str) -> Path:
    directory = root / f"model-{marker}"
    directory.mkdir()
    (directory / "model.pkl").write_text(marker)
    (directory / "calibrator.json").write_text(json.dumps({"marker": marker}))
    return directory


def result(root: Path) -> Path:
    path = root / "result.json"
    path.write_text("{}")
    return path


def test_metrics_are_the_numeric_leaves_with_safe_names() -> None:
    report = {
        "config": {"seed": 7},
        "ours": {"cnn": {"rmse": {"14+15": 3.77}}},
        "flag": True,
        "engines": [1, 2],
        "missing": float("nan"),
    }
    assert metrics_of(report) == {"ours.cnn.rmse.14_15": 3.77}
    assert params_of(report) == {"seed": "7"}


def test_artifacts_live_beside_the_store(store: Path) -> None:
    assert artifact_root() == (store / "artifacts").resolve()


def test_a_run_is_recorded_without_a_model(store: Path) -> None:
    recorded = record_run("evaluate", {"rmse": 1.0}, result(store))
    assert recorded["run_id"] and "model_version" not in recorded


def test_each_registered_model_becomes_the_serving_version(store: Path) -> None:
    first = record_run("stream", {}, result(store), model_dir=model_dir(store, "one"))
    second = record_run("stream", {}, result(store), model_dir=model_dir(store, "two"))
    assert (first["model_version"], second["model_version"]) == ("1", "2")
    served = resolve_model("models:/prognos-rul@serving")
    assert (served / "model.pkl").read_text() == "two"
    pinned = resolve_model("models:/prognos-rul/1")
    assert (pinned / "model.pkl").read_text() == "one"


def test_a_plain_path_is_taken_as_it_is(store: Path) -> None:
    assert resolve_model(str(store)) == store


def test_export_writes_the_version_it_copied(store: Path) -> None:
    record_run("stream", {}, result(store), model_dir=model_dir(store, "one"))
    out = store / "exported"
    record = export_model("models:/prognos-rul@serving", out)
    assert record["version"] == "1"
    assert (out / "model.pkl").read_text() == "one"
    assert json.loads((out / REGISTRY_FILE).read_text())["version"] == "1"
    by_number = export_model("models:/prognos-rul/1", store / "again")
    assert by_number["version"] == "1"


def test_export_refuses_a_path(store: Path) -> None:
    with pytest.raises(ValueError):
        export_model(str(store), store / "out")


def test_the_inference_service_reports_the_registry_version(store: Path) -> None:
    import numpy as np

    from prognos.stream.api import build_inference_state
    from prognos.stream.inference import RulPredictor

    rng = np.random.default_rng(0)
    windows = rng.normal(size=(60, 5, 18)).astype(np.float32)
    cycles = np.arange(60, dtype=np.int64)
    targets = (60.0 - cycles).astype(np.float32)
    served = store / "served"
    RulPredictor.train(windows, cycles, targets, windows, cycles, targets, 0.9, seed=0).save(served)
    (served / REGISTRY_FILE).write_text(json.dumps({"name": "prognos-rul", "version": "4"}))
    ready, detail = build_inference_state(model_dir=served).readiness()
    assert ready and detail["registry"]["version"] == "4"


def test_loading_by_reference_records_the_version_it_resolved(store: Path) -> None:
    """An alias can move after loading; the service must report what it loaded."""
    record_run("stream", {}, result(store), model_dir=model_dir(store, "one"))
    loaded = resolve_model("models:/prognos-rul@serving")
    record_run("stream", {}, result(store), model_dir=model_dir(store, "two"))
    assert json.loads((loaded / REGISTRY_FILE).read_text())["version"] == "1"
