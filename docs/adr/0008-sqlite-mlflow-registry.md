# ADR-0008: MLflow on a local SQLite store, with the client only

Date: 2026-11-01
Status: Accepted

## Context

Every training run needs tracking, and the served model needs a registry so the
inference service loads a named version and a rollback is a single, checkable action.
The budget is $0. Hosted MLflow and Weights & Biases either cost money or cap free use.

MLflow 3's file store (`./mlruns`) is in maintenance mode and refuses to run without an
opt-out. Its registry also rejects `register_model` for an artifact that is not one of
its model flavours, and the model here is a LightGBM model plus a conformal calibrator
that only make sense together.

## Decision

- `mlflow-skinny` (the client, no server) plus SQLAlchemy and Alembic, pinned, with a
  SQLite store under `mlruns/`, gitignored; `MLFLOW_TRACKING_URI` overrides it.
- Every command that writes a result records a run: config as parameters, every
  numeric result as a metric, the git commit as a tag, the result file as an artifact.
- `prognos stream` registers the model directory as a version of `prognos-rul` with
  `create_model_version`, and moves the `serving` alias to it.
- The inference service loads by reference (`models:/prognos-rul@serving`), records
  the version it resolved, and reports it on `/readyz`.
- Containers cannot follow the registry's absolute host paths, so
  `prognos export-model` copies a version into the mounted directory with
  `registry.json` naming it.

## Rationale

SQLite is the supported local backend, needs no server, and costs nothing. The skinny
client keeps the image small and the vulnerability scan short; the UI, when wanted,
runs from a throwaway `uvx mlflow ui`. Registering the directory rather than a flavour
keeps model and calibrator one artifact, which is the property the inference service
already enforces. The committed result files remain the source of truth for every
figure quoted; the runs are how they are compared and traced.

## Consequences

- Rollback is: move the alias, export, restart, and confirm the version on `/readyz`
  (`docs/runbook.md`).
- The store is local to one machine. A shared team registry would need a database
  server and an artifact store, which this budget does not cover.
- An exported directory is a copy: if the alias moves, nothing changes in a running
  container until it is exported again and restarted, by design.
