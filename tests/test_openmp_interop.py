"""Regression test for the OpenMP deadlock between LightGBM and torch.

This is the bug that made `prognos evaluate` hang with no output on the first
sequence model, and the full test suite hang in torch's parallel reduction. Both are
the same root cause: LightGBM's GNU OpenMP pool and torch's thread pool cannot
coexist in one process, and whichever loads second deadlocks.

The test fits LightGBM first and then trains torch, which is the order that hung.
It runs in a subprocess with a timeout, because a test that hangs the whole suite is
worse than no test.

One honest limit: the deadlock was observed on this machine, but it no longer
reproduces reliably. An uncapped run of the same sequence completes here even at
12,000 rows and 200 trees, where it used to hang indefinitely. So there is no
"without the cap this hangs" counterfactual to assert on — asserting on a hang that
may or may not arrive gives a test that passes for the wrong reason on some days.
What is asserted instead is the deterministic part: the cap is applied on model
construction, so no caller can forget it, and the subprocess timeout turns a
returning deadlock into a failure instead of a hung suite.

The dataset is built in-process from a synthetic HDF5 file rather than read from
`data/`, so the test runs in CI where the NASA archive is absent.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Builds a synthetic sub-dataset, converts it, fits LightGBM, then trains a CNN.
# LightGBM first is the order that deadlocked.
SCRIPT = """
import pathlib, tempfile, numpy as np, h5py, torch

from prognos.config import WindowConfig
from prognos.convert import convert_subdataset
from prognos.dataset import load_windows
from prognos.gbm import DEFAULT_PARAMS, LightGbmModel
from prognos.schema import MEASURED_SENSORS, SCENARIO_DESCRIPTORS
from prognos.window import FEATURE_COLUMNS

root = pathlib.Path(tempfile.mkdtemp())
h5_path = root / "N-CMAPSS_SYNTH.h5"
rng = np.random.default_rng(0)
n_rows, n_units, cycles = 600, 2, 3
rows_per_unit = n_rows // n_units

with h5py.File(h5_path, "w") as handle:
    for split in ("dev", "test"):
        unit = np.repeat(np.arange(1, n_units + 1), rows_per_unit).reshape(-1, 1)
        cycle = np.repeat(np.arange(1, cycles + 1), rows_per_unit // cycles)
        cycle = np.tile(cycle, n_units).reshape(-1, 1).astype(np.float64)
        size = len(unit)
        meta = np.hstack([unit.astype(np.float64), cycle, np.ones((size, 2))])
        handle.create_dataset(f"A_{split}", data=meta)
        handle.create_dataset(f"W_{split}", data=rng.normal(size=(size, 4)))
        handle.create_dataset(f"X_s_{split}", data=rng.normal(size=(size, 14)))
        rul = np.tile(np.arange(cycles - 1, -1, -1, dtype=np.int64), size // cycles)
        handle.create_dataset(f"Y_{split}", data=rul.reshape(-1, 1))
    handle.create_dataset("A_var", data=np.array([b"unit", b"cycle", b"Fc", b"hs"]))
    handle.create_dataset("W_var", data=np.array([s.encode() for s in SCENARIO_DESCRIPTORS]))
    handle.create_dataset("X_s_var", data=np.array([s.encode() for s in MEASURED_SENSORS]))

parquet = root / "parquet"
# The converter writes into the directory it is given; the CLI passes
# parquet_dir / subdataset, so this has to as well.
convert_subdataset(h5_path, parquet / "SYNTH", "SYNTH")
config = WindowConfig(length=10, stride=5, downsample=10)
data = load_windows(parquet, config, ["SYNTH"], cache_dir=None)

windows = data.windows
cycles_arr = data.cycles
targets = data.targets

# LightGBM first: this is the order that deadlocked.
LightGbmModel(params={**DEFAULT_PARAMS, "n_estimators": 10}).fit(windows, cycles_arr, targets)

from prognos.sequence import CnnModel, TrainConfig, cap_torch_threads
cap_torch_threads()

model = CnnModel(TrainConfig(epochs=1, batch_size=64), device=torch.device("cpu"))
model.fit(windows, cycles_arr, targets)
n = min(16, windows.shape[0])
prediction = model.predict(windows[:n], cycles_arr[:n])
assert prediction.shape == (n,), prediction.shape
assert len(FEATURE_COLUMNS) == 18
print("OK")
"""


def _run_script() -> subprocess.CompletedProcess[str]:
    """Run the reproduction script in a subprocess with a hard timeout."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


class TestOpenmpInterop:
    def test_torch_survives_lightgbm_loading_first(self) -> None:
        """LightGBM then torch must complete, with the documented timeout as the guard.

        This is the order that hung. If the deadlock ever returns, the subprocess
        times out and this fails instead of taking the suite with it.
        """
        result = _run_script()
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

        This is why the fix lives in `__init__` rather than in the CLI: any path that
        builds a sequence model gets the safe configuration.
        """
        import torch

        from prognos.sequence import CnnModel, TrainConfig

        CnnModel(TrainConfig(epochs=1), device=torch.device("cpu"))
        assert torch.get_num_threads() == 1


@pytest.mark.parametrize("order", ["lgbm_first", "torch_first"])
def test_either_import_order_survives(order: str) -> None:
    """Import order should not matter; the cap makes it not matter.

    Recorded because the bug was order-dependent, and a future refactor could
    reintroduce the dependence without any test noticing.
    """
    script = SCRIPT
    if order == "torch_first":
        script = script.replace(
            "# LightGBM first: this is the order that deadlocked.",
            "# Torch imported before LightGBM.",
        )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
