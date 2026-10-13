"""Regression test for the OpenMP deadlock between LightGBM and torch.

This is the bug that made `prognos evaluate` hang with no output on the first
sequence model, and the full test suite hang in torch's parallel reduction. Both are
the same root cause: LightGBM's GNU OpenMP pool and torch's thread pool cannot
coexist in one process.

The test fits LightGBM first and then trains torch, which is the order that hung.
Without the fix it never returns, so it runs in a subprocess with a timeout: a test
that hangs the whole suite is worse than no test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

SCRIPT = """
import pathlib, numpy as np, torch
from prognos.dataset import load_windows
from prognos.config import WindowConfig
from prognos.gbm import DEFAULT_PARAMS, LightGbmModel

d = load_windows(
    pathlib.Path("data/parquet"),
    WindowConfig(length=20, downsample=10),
    ["DS01-005", "DS02-006", "DS06"],
    cache_dir=pathlib.Path("data/cache"),
)
windows = d.windows[:2000]
cycles = d.cycles[:2000]
targets = d.targets[:2000]

# LightGBM first: this is the order that deadlocked.
LightGbmModel(params={**DEFAULT_PARAMS, "n_estimators": 10}).fit(windows, cycles, targets)

from prognos.sequence import CnnModel, TrainConfig, cap_torch_threads
cap_torch_threads()

model = CnnModel(TrainConfig(epochs=1, batch_size=256), device=torch.device("cpu"))
model.fit(windows, cycles, targets)
prediction = model.predict(windows[:16], cycles[:16])
assert prediction.shape == (16,), prediction.shape
print("OK")
"""


class TestOpenmpInterop:
    def test_torch_survives_lightgbm_loading_first(self) -> None:
        """LightGBM then torch must not hang.

        Run in a subprocess with a hard timeout: without the fix this deadlocks, and a
        hung test is indistinguishable from a broken suite.
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK" in result.stdout

    def test_cap_is_idempotent(self) -> None:
        """Calling the cap twice must not raise or change the setting."""
        import torch

        from prognos.sequence import cap_torch_threads

        cap_torch_threads()
        cap_torch_threads()
        assert torch.get_num_threads() == 1

    def test_cap_is_applied_on_model_construction(self) -> None:
        """Constructing a model applies the cap, so no caller has to remember.

        This is the reason the fix lives in `__init__` rather than in the CLI: any
        path that builds a sequence model gets the safe configuration.
        """
        import torch

        from prognos.sequence import CnnModel, TrainConfig

        CnnModel(TrainConfig(epochs=1), device=torch.device("cpu"))
        assert torch.get_num_threads() == 1
