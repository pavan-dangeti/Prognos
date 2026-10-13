"""Test session setup.

LightGBM links against GNU OpenMP and torch runs its own thread pool. Loading both
in one process can deadlock torch's parallel reductions. `cap_torch_threads` handles
it at the source; this calls it before any test imports either library, so the whole
suite runs in the configuration that production uses.
"""

from __future__ import annotations

from prognos.sequence import cap_torch_threads

cap_torch_threads()
