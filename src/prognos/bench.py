"""Streaming latency and throughput, measured on the pipeline as it ships.

Two numbers, kept apart because they answer different questions:

- **End to end**, from a sensor reading handed to `StreamPipeline.push` to the updated
  prediction it returns: fault detection, the online window, features, the model, the
  conformal interval, alerting and the fleet view. This is the path the live replay
  runs, so it is the path the dashboard's numbers come from.
- **The inference hop**, one window over loopback HTTP to the inference service. The
  deployed stack puts this behind a socket, and the socket is a cost the in-process
  number does not include.

Fleets are built from real cross-validation engines, interleaved one reading per engine
in turn, so every engine keeps its own window and detector live at once. The final test
group stays closed: the benchmark has no reason to read it, so it does not.

Readings are pushed as fast as the pipeline accepts them, so latency here is service
time with no queueing. Throughput is readings per second; N-CMAPSS records at 1 Hz, so
the same figure is how many engines one process keeps up with in real time.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .dataset import GroupKey
from .pipeline import _unit_table
from .stream.inference import RulPredictor
from .stream.pipeline import PipelineConfig, StreamPipeline

#: Percentiles reported for every latency distribution.
PERCENTILES = (50, 95, 99)

#: N-CMAPSS flight data is recorded at 1 Hz (Arias Chao et al., 2021), which is what
#: turns readings per second into engines per second.
SAMPLE_RATE_HZ = 1.0


@dataclass(frozen=True, slots=True)
class Engine:
    """One engine's opening readings, the slice the benchmark replays."""

    cycles: npt.NDArray[np.int64]
    rows: npt.NDArray[np.float32]


def machine() -> dict[str, Any]:
    """The hardware a figure was measured on, stated rather than implied."""
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        cpu = brand or cpu
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "memory_gib": round(memory / 2**30, 1),
        "python": platform.python_version(),
    }


def load_engines(parquet_dir: Path, labels: list[GroupKey], rows_per_engine: int) -> list[Engine]:
    """The first `rows_per_engine` readings of each engine."""
    engines = []
    for label in labels:
        if label.subdataset is None:
            raise ValueError(f"engine {label.unit} has no sub-dataset, so no table to read")
        cycles, features, _rul = _unit_table(parquet_dir, label.subdataset, label.unit)
        engines.append(Engine(cycles=cycles[:rows_per_engine], rows=features[:rows_per_engine]))
    return engines


def percentiles(samples_ns: list[int]) -> dict[str, float]:
    """Latency percentiles in milliseconds."""
    if not samples_ns:
        raise ValueError("no samples: nothing produced a prediction to time")
    values = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


def run_fleet(
    predictor: RulPredictor, config: PipelineConfig, engines: list[Engine], fleet_size: int
) -> dict[str, Any]:
    """Replay a fleet of `fleet_size` engines, interleaved, and time every reading.

    Fleet member `i` replays source engine `i mod len(engines)` under its own unit id,
    so a fleet larger than the dataset is many independent streams, not one stream
    counted several times.
    """
    pipeline = StreamPipeline(predictor, config)
    members = [engines[i % len(engines)] for i in range(fleet_size)]
    length = min(e.cycles.size for e in members)
    updates: list[int] = []
    readings = 0
    started = time.perf_counter_ns()
    for index in range(length):
        for unit, engine in enumerate(members):
            begin = time.perf_counter_ns()
            produced = pipeline.push(unit, int(engine.cycles[index]), engine.rows[index])
            elapsed = time.perf_counter_ns() - begin
            readings += 1
            if produced:
                updates.append(elapsed)
    wall = (time.perf_counter_ns() - started) / 1e9
    return {
        "fleet_size": fleet_size,
        "readings": readings,
        "predictions": len(updates),
        "wall_seconds": wall,
        "readings_per_second": readings / wall,
        "predictions_per_second": len(updates) / wall,
        "engines_in_real_time": readings / wall / SAMPLE_RATE_HZ,
        "latency_ms": percentiles(updates),
    }


def run_http(predictor_dir: Path, window: npt.NDArray[np.float32], requests: int) -> dict[str, Any]:
    """Time `requests` single-window predictions through the real inference service."""
    from .stream.api import build_inference_state
    from .stream.service import build_server

    server = build_server(build_inference_state(model_dir=predictor_dir), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/predict"
    body = json.dumps({"windows": [window.tolist()], "cycles": [1]}).encode()
    samples: list[int] = []
    try:
        for _ in range(requests):
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            begin = time.perf_counter_ns()
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
            samples.append(time.perf_counter_ns() - begin)
    finally:
        server.shutdown()
        server.server_close()
    return {"requests": requests, "payload_bytes": len(body), "latency_ms": percentiles(samples)}


def benchmark(
    predictor_dir: Path,
    parquet_dir: Path,
    labels: list[GroupKey],
    config: PipelineConfig,
    fleet_sizes: list[int],
    rows_per_engine: int,
    repeats: int,
    http_requests: int,
) -> dict[str, Any]:
    """Every fleet size `repeats` times, interleaved, then the inference hop.

    Interleaved rather than size by size. On the reference laptop the same run varies
    about threefold from round to round, in two clusters, which is consistent with macOS
    moving an unpinned process between performance and efficiency cores. Run size by
    size, that drift landed on whichever size was running and read as a scaling
    effect; interleaved, it falls on every size alike.
    """
    predictor = RulPredictor.load(predictor_dir)
    engines = load_engines(parquet_dir, labels, rows_per_engine)
    # Warm the model before timing, so the first fleet does not pay for allocations
    # the others get for free.
    run_fleet(predictor, config, engines[:1], 1)

    runs: dict[int, list[dict[str, Any]]] = {size: [] for size in fleet_sizes}
    for round_ in range(repeats):
        for size in fleet_sizes:
            run = run_fleet(predictor, config, engines, size)
            runs[size].append(run)
            print(
                f"  round {round_ + 1} {size:>5} engines: "
                f"{run['readings_per_second']:,.0f} readings/s "
                f"p99 {run['latency_ms']['p99']:.2f} ms",
                flush=True,
            )
    fleets = [{"fleet_size": size, "runs": runs[size]} for size in fleet_sizes]

    first = engines[0]
    window = first.rows[: config.window_length * config.downsample : config.downsample]
    http = run_http(predictor_dir, window, http_requests)
    return {
        "fleets": fleets,
        "inference_http": http,
        "rows_per_engine": rows_per_engine,
        "source_engines": [label.label() for label in labels],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "model": predictor.model_name,
        "hardware": machine(),
    }


def summarise(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per fleet size: the median run, plus the throughput range across runs.

    The median run rather than the best: on a laptop the first run at a size often
    pays for a background process or a cold cache, and reporting the best of three
    would quote a number the machine does not usually reach.
    """
    rows = []
    for fleet in report["fleets"]:
        runs = sorted(fleet["runs"], key=lambda run: run["readings_per_second"])
        median = runs[len(runs) // 2]
        rows.append(
            {
                "fleet_size": fleet["fleet_size"],
                "readings_per_second": median["readings_per_second"],
                "readings_per_second_min": runs[0]["readings_per_second"],
                "readings_per_second_max": runs[-1]["readings_per_second"],
                "predictions_per_second": median["predictions_per_second"],
                "latency_ms": median["latency_ms"],
                "latency_p99_worst_ms": max(run["latency_ms"]["p99"] for run in runs),
                "n_runs": len(runs),
            }
        )
    return rows


__all__ = ["benchmark", "machine", "percentiles", "run_fleet", "run_http", "summarise"]
