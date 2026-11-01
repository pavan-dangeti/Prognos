"""Experiment tracking and the model registry, on MLflow.

Every command that writes a result file also records a run: the configuration as
parameters, every numeric figure in the result as a metric, the git commit as a tag,
and the result file itself as an artifact. The committed result file stays the source
of truth for anything quoted; the run is how two of them are compared and traced back.

`prognos stream` registers the model it trains as a version of `prognos-rul` and
points the `serving` alias at it. The inference service loads by reference
(`models:/prognos-rul@serving` or `models:/prognos-rul/3`). A container cannot follow
the registry's absolute artifact paths, so for one `prognos export-model` copies the
version into the directory the container mounts, with `registry.json` saying which
version it is. Rollback is moving the alias back and exporting again.

The store is SQLite under `mlruns/`, local and free; `MLFLOW_TRACKING_URI` overrides it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

REGISTERED_MODEL = "prognos-rul"
SERVING_ALIAS = "serving"
EXPERIMENT = "prognos"
REGISTRY_FILE = "registry.json"
DEFAULT_ROOT = Path("mlruns")

#: MLflow allows letters, digits and `_ - . / space` in a metric or parameter name.
_INVALID = re.compile(r"[^A-Za-z0-9_\-./ ]")


def tracking_uri() -> str:
    """The configured store, or SQLite under `mlruns/`."""
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    DEFAULT_ROOT.mkdir(exist_ok=True)
    return f"sqlite:///{DEFAULT_ROOT / 'mlflow.db'}"


def artifact_root() -> Path:
    """Where run artifacts go: beside the SQLite store.

    So a test's throwaway store keeps its artifacts in its own directory instead of
    writing them into the working tree.
    """
    uri = tracking_uri()
    if uri.startswith("sqlite:///"):
        return Path(uri.removeprefix("sqlite:///")).resolve().parent / "artifacts"
    return (DEFAULT_ROOT / "artifacts").resolve()


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:  # noqa: ANN401
    """Leaf values of nested dicts, as dotted names.

    Lists are skipped: they are per-engine or per-fold detail, which the result file
    artifact already carries in full.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif not isinstance(value, list):
        yield _INVALID.sub("_", prefix)[:250], value


def metrics_of(report: Mapping[str, Any]) -> dict[str, float]:
    """Every finite numeric leaf, except the configuration, which is logged as params."""
    found: dict[str, float] = {}
    for name, value in _flatten({k: v for k, v in report.items() if k != "config"}):
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if value == value and abs(value) != float("inf"):
            found[name] = float(value)
    return found


def params_of(report: Mapping[str, Any]) -> dict[str, str]:
    """The run's configuration, flattened, as MLflow parameters."""
    config = report.get("config")
    if not isinstance(config, Mapping):
        return {}
    return {name: str(value)[:500] for name, value in _flatten(config) if value is not None}


def _git_commit() -> str:
    """The commit the run came from, or `unknown` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.strip()


def _experiment_id(client: Any) -> str:  # noqa: ANN401 - MlflowClient
    """The project experiment, created with its artifacts beside the store."""
    existing = client.get_experiment_by_name(EXPERIMENT)
    if existing is not None:
        return str(existing.experiment_id)
    return str(client.create_experiment(EXPERIMENT, artifact_location=artifact_root().as_uri()))


def record_run(
    command: str,
    report: Mapping[str, Any],
    result_path: Path,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Log one command's run; register `model_dir` as the serving model if given."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    experiment_id = _experiment_id(client)
    with mlflow.start_run(experiment_id=experiment_id, run_name=command) as run:
        mlflow.set_tags({"command": command, "git_commit": _git_commit()})
        params = params_of(report)
        if params:
            mlflow.log_params(params)
        metrics = metrics_of(report)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(result_path), artifact_path="result")
        recorded: dict[str, Any] = {"run_id": run.info.run_id}
        if model_dir is not None:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            if not client.search_registered_models(f"name = '{REGISTERED_MODEL}'"):
                client.create_registered_model(REGISTERED_MODEL)
            version = client.create_model_version(
                REGISTERED_MODEL, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
            )
            client.set_registered_model_alias(REGISTERED_MODEL, SERVING_ALIAS, version.version)
            recorded["model_version"] = str(version.version)
    return recorded


def resolve_model(reference: str) -> Path:
    """A local model directory for `reference`: a path, or a `models:/` registry URI."""
    if not reference.startswith("models:/"):
        return Path(reference)
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    return Path(mlflow.artifacts.download_artifacts(reference))


def export_model(reference: str, out_dir: Path) -> dict[str, str]:
    """Copy a registered version into `out_dir`, recording which version it is."""
    import mlflow
    from mlflow import MlflowClient

    if not reference.startswith("models:/"):
        raise ValueError(f"{reference!r} is not a registry reference (models:/name/version)")
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    name, _, alias = reference.removeprefix("models:/").partition("@")
    if alias:
        version = client.get_model_version_by_alias(name, alias)
    else:
        name, _, number = name.partition("/")
        version = client.get_model_version(name, number)
    source = resolve_model(f"models:/{name}/{version.version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        shutil.copy2(item, out_dir / item.name)
    record = {
        "name": name,
        "version": str(version.version),
        "run_id": str(version.run_id),
        "reference": reference,
    }
    (out_dir / REGISTRY_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


__all__ = [
    "REGISTERED_MODEL",
    "REGISTRY_FILE",
    "SERVING_ALIAS",
    "artifact_root",
    "export_model",
    "metrics_of",
    "params_of",
    "record_run",
    "resolve_model",
    "tracking_uri",
]
