"""Test session setup.

LightGBM links against GNU OpenMP and torch runs its own thread pool. Loading both
in one process can deadlock torch's parallel reductions. `cap_torch_threads` handles
it at the source; this calls it before any test imports either library, so the whole
suite runs in the configuration that production uses.
"""

from __future__ import annotations

import os
import tempfile

from prognos.sequence import cap_torch_threads

cap_torch_threads()

# Every command records an MLflow run. Tests get a throwaway store, so a test run never
# writes into the working tree's `mlruns/`.
os.environ["MLFLOW_TRACKING_URI"] = (
    f"sqlite:///{tempfile.mkdtemp(prefix='prognos-mlflow-')}/mlflow.db"
)
