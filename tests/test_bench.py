"""Tests for the streaming benchmark.

The numbers it writes go into the README, so the arithmetic behind them is pinned here:
which readings count as updates, how fleet members map to source engines, and that the
HTTP hop actually reaches a served model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prognos.bench import Engine, machine, percentiles, run_fleet, run_http, summarise
from prognos.stream.inference import RulPredictor
from prognos.stream.pipeline import PipelineConfig
from prognos.stream.window import N_CHANNELS

WINDOW, DOWNSAMPLE, STRIDE = 5, 2, 2


def predictor() -> RulPredictor:
    """A real predictor on noise, small enough to train in a test."""
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(200, WINDOW, N_CHANNELS)).astype(np.float32)
    cycles = np.arange(200, dtype=np.int64)
    targets = (100.0 - cycles / 10.0).astype(np.float32)
    return RulPredictor.train(
        windows, cycles, targets, windows, cycles, targets, target_coverage=0.9, seed=0
    )


def engine(n_rows: int = 40, seed: int = 1) -> Engine:
    """One cycle block, long enough for several windows."""
    rows = np.random.default_rng(seed).normal(size=(n_rows, N_CHANNELS)).astype(np.float32)
    return Engine(cycles=np.ones(n_rows, dtype=np.int64), rows=rows)


CONFIG = PipelineConfig(window_length=WINDOW, window_stride=STRIDE, downsample=DOWNSAMPLE)


def test_percentiles_are_milliseconds() -> None:
    assert percentiles([1_000_000] * 10) == {"p50": 1.0, "p95": 1.0, "p99": 1.0}


def test_percentiles_refuse_an_empty_sample() -> None:
    """A latency figure computed from nothing must not come out as zero."""
    with pytest.raises(ValueError):
        percentiles([])


def test_every_fleet_member_is_its_own_stream() -> None:
    """Two members replaying one engine produce twice the predictions, not the same ones."""
    model = predictor()
    one = run_fleet(model, CONFIG, [engine()], 1)
    three = run_fleet(model, CONFIG, [engine()], 3)
    assert one["readings"] == 40
    assert three["readings"] == 120
    assert three["predictions"] == 3 * one["predictions"] > 0
    assert three["engines_in_real_time"] == pytest.approx(three["readings_per_second"])


def test_a_fleet_runs_as_long_as_its_shortest_member() -> None:
    model = predictor()
    report = run_fleet(model, CONFIG, [engine(40), engine(30, seed=2)], 2)
    assert report["readings"] == 60


def test_the_http_hop_reaches_a_served_model(tmp_path: Path) -> None:
    model = predictor()
    model.save(tmp_path)
    window = engine().rows[: WINDOW * DOWNSAMPLE : DOWNSAMPLE]
    report = run_http(tmp_path, window, requests=5)
    assert report["requests"] == 5
    assert report["latency_ms"]["p50"] > 0


def test_the_hardware_is_stated() -> None:
    stated = machine()
    assert {"cpu", "cpu_count", "memory_gib", "python", "system"} <= set(stated)
    assert stated["memory_gib"] > 0


def test_the_summary_reports_the_median_run_and_the_range() -> None:
    def run(rate: float, p99: float) -> dict[str, object]:
        return {
            "readings_per_second": rate,
            "predictions_per_second": rate / 100,
            "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": p99},
        }

    report = {
        "fleets": [{"fleet_size": 10, "runs": [run(30.0, 9.0), run(50.0, 5.0), run(40.0, 6.0)]}]
    }
    (row,) = summarise(report)
    assert row["readings_per_second"] == 40.0
    assert row["latency_ms"]["p99"] == 6.0
    assert row["latency_p99_worst_ms"] == 9.0
    assert (row["readings_per_second_min"], row["readings_per_second_max"]) == (30.0, 50.0)


def test_the_readme_quotes_the_committed_benchmark() -> None:
    """Every streaming figure in the README must be re-derivable from the result file."""
    import json

    result = Path("results/phase6_streaming_bench.json")
    # Whitespace collapsed, so a figure wrapped across a line still counts as quoted.
    readme = " ".join(Path("README.md").read_text().split())
    if not result.exists():
        pytest.skip("no committed benchmark; run `prognos bench`")
    report = json.loads(result.read_text())
    rows = summarise(report)
    for row in rows:
        lat = row["latency_ms"]
        line = (
            f"| {row['fleet_size']:,} "
            f"| {row['readings_per_second']:,.0f} "
            f"({row['readings_per_second_min']:,.0f}\u2013{row['readings_per_second_max']:,.0f}) "
            f"| {row['predictions_per_second']:,.0f} "
            f"| {lat['p50']:.2f} | {lat['p95']:.2f} | {lat['p99']:.2f} "
            f"| {row['latency_p99_worst_ms']:.2f} |"
        )
        assert line in readme, f"README streaming row drifted from {result}: {line}"
    slowest = min(row["readings_per_second_min"] for row in rows)
    assert f"{slowest:,.0f} readings" in readme, "the conservative throughput figure drifted"
    http = report["inference_http"]["latency_ms"]
    assert f"p50 {http['p50']:.2f} ms, p95 {http['p95']:.2f} ms, p99 {http['p99']:.2f} ms" in readme


def test_the_whole_benchmark_runs_on_parquet_engines(tmp_path: Path) -> None:
    """The command's real path: tables on disk, a saved model, every size and round."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from prognos.bench import benchmark
    from prognos.dataset import GroupKey
    from prognos.window import FEATURE_COLUMNS

    parquet_dir = tmp_path / "parquet"
    (parquet_dir / "DS01").mkdir(parents=True)
    rows = engine(60).rows
    columns = {"cycle": np.ones(60, dtype=np.int64), "rul": np.full(60, 50.0)}
    columns.update({name: rows[:, i] for i, name in enumerate(FEATURE_COLUMNS)})
    pq.write_table(pa.table(columns), parquet_dir / "DS01" / "unit_001.parquet")
    model_dir = tmp_path / "model"
    predictor().save(model_dir)

    report = benchmark(
        model_dir,
        parquet_dir,
        [GroupKey(subdataset="DS01", unit=1)],
        CONFIG,
        fleet_sizes=[1, 3],
        rows_per_engine=40,
        repeats=2,
        http_requests=3,
    )
    assert [fleet["fleet_size"] for fleet in report["fleets"]] == [1, 3]
    assert all(len(fleet["runs"]) == 2 for fleet in report["fleets"])
    assert report["source_engines"] == ["DS01/unit001"]
    assert report["inference_http"]["requests"] == 3
