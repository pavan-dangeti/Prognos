"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, TypedDict, cast

from .config import Config, load_config
from .convert import convert_subdataset, sha256_file, write_manifest
from .dataset import WindowSource, load_windows

H5_SUFFIX = ".h5"


def _subdataset_path(root: Path, subdataset: str) -> Path:
    """Locate the HDF5 file for a sub-dataset, preferring an explicit extract dir."""
    for directory in (root / "extracted" / "data_set", root / "data_set", root):
        candidate = directory / f"N-CMAPSS_{subdataset}{H5_SUFFIX}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no HDF5 file for sub-dataset {subdataset} under {root}")


def hardware_description() -> str:
    """A short hardware string, so a committed result says where it came from.

    Deliberately plain: platform, processor and Python version. Enough to tell two
    runs apart, not enough to identify a machine.
    """
    processor = platform.processor() or platform.machine()
    return (
        f"{platform.system()} {platform.machine()} / {processor} / Python {sys.version.split()[0]}"
    )


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert configured sub-datasets to per-unit Parquet and write the manifest."""
    config = load_config(Path(args.config))
    h5_dir = config.data.root
    digests: dict[str, str] = {}
    all_stats = []

    for subdataset in config.data.subdatasets:
        h5_path = _subdataset_path(h5_dir, subdataset)
        print(f"converting {h5_path.name}", flush=True)
        digests[subdataset] = sha256_file(h5_path)
        out_dir = config.data.parquet_dir / subdataset
        all_stats.extend(convert_subdataset(h5_path, out_dir, subdataset))

    manifest = write_manifest(all_stats, config.data.manifest_path, digests)
    units = manifest["units"]
    total_rows = sum(int(u["rows"]) for u in units)
    print(
        f"converted {len(units)} units, {total_rows:,} rows -> {config.data.parquet_dir}",
        flush=True,
    )
    print(f"manifest: {config.data.manifest_path}", flush=True)
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    """Print the data manifest."""
    config = load_config(Path(args.config))
    print(config.data.manifest_path.read_text())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Cross-validate every baseline on grouped folds and write a result file."""
    from .evaluate import aggregate_folds, build_folds, evaluate_model, report_run
    from .gbm import LightGbmModel
    from .models import build_models

    config = load_config(Path(args.config))
    cache_dir = None if args.no_cache else config.data.parquet_dir.parent / "cache"

    print("loading windows", flush=True)
    data = load_windows(
        config.data.parquet_dir,
        config.window,
        config.data.subdatasets,
        cache_dir=cache_dir,
        rebuild=args.rebuild_cache,
    )
    print(f"  {len(data):,} windows across {data.summary()['n_engines']} engines", flush=True)

    folds = build_folds(
        data,
        n_splits=config.split.n_splits,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )
    test_units = folds[0].split.test_units
    print(f"  final test group held closed: {test_units}", flush=True)

    models: dict[str, Any] = dict(build_models())
    if not args.baselines_only:
        models["lightgbm"] = LightGbmModel(seed=config.split.seed)

    # Sequence models are refitted per fold from scratch. Torch's optimiser state
    # and any adapter validation split belong to one fit, so reusing an object across
    # folds would leak the previous fold's training into the next.
    sequence_models: dict[str, Any] = {}
    if args.sequence:
        from .sequence import HarnessAdapter, TrainConfig, _BaseSequenceModel, build_sequence_models

        train_config = TrainConfig(
            epochs=args.epochs,
            seed=config.split.seed,
            late_weight=args.late_weight,
            verbose=args.verbose,
        )
        sequence_models = {
            name: HarnessAdapter(model=cast("_BaseSequenceModel", model), seed=config.split.seed)
            for name, model in build_sequence_models(train_config).items()
        }
        for name, model in sequence_models.items():
            print(
                f"  {name}: {model.model.count_parameters():,} parameters on {model.model.device}"
            )

    summaries: dict[str, object] = {}
    model_reports: dict[str, object] = {}
    for name, model in {**models, **sequence_models}.items():
        print(f"evaluating {name}", flush=True)
        results = []
        for fold in folds:
            result = evaluate_model(
                data,
                fold.train_mask,
                fold.val_mask,
                model_name=name,
                fit=model.fit,
                predict=model.predict,
                test_units=test_units,
                fold_name=fold.split.name,
                seed=config.split.seed,
                n_bootstrap=args.bootstrap,
            )
            results.append(result)
            print(
                f"  {fold.split.name}: rmse={result.rmse:.2f} score={result.nasa_score:.1f}",
                flush=True,
            )
        summaries[name] = aggregate_folds(
            name,
            results,
            target_coverage=args.coverage,
            engine_labels=data.group_labels,
        )
        # A sequence model's own fit report, so the training side is auditable
        # alongside the metrics.
        inner = getattr(model, "model", model)
        if hasattr(inner, "report"):
            model_reports[name] = inner.report()
        print(
            f"  mean rmse={summaries[name]['rmse']['mean']:.2f} "  # type: ignore[index]
            f"mean score={summaries[name]['nasa_score']['mean']:.1f}",  # type: ignore[index]
            flush=True,
        )

    report = report_run(
        config={
            "window": config.window.model_dump(),
            "split": config.split.model_dump(),
            "subdatasets": list(config.data.subdatasets),
            "bootstrap_resamples": args.bootstrap,
        },
        data=data,
        folds=folds,
        models=summaries,
        hardware=hardware_description(),
        model_reports=model_reports,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out_path}", flush=True)

    ranking = sorted(summaries, key=lambda k: summaries[k]["rmse"]["mean"])  # type: ignore[index]
    print("\nranking by cross-validated RMSE:")
    for name in ranking:
        entry = summaries[name]
        print(f"  {name:15s} rmse={entry['rmse']['mean']:.2f}")  # type: ignore[index]
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    """Calibrate conformal intervals and cost every maintenance policy on them."""
    from .decision import CostModel
    from .evaluate import build_folds
    from .policy_eval import (
        build_episodes,
        evaluate_fold,
        evaluate_policies,
        report_phase4,
        summarise,
    )

    config = load_config(Path(args.config))
    cache_dir = None if args.no_cache else config.data.parquet_dir.parent / "cache"

    print("loading windows", flush=True)
    data = load_windows(
        config.data.parquet_dir,
        config.window,
        config.data.subdatasets,
        cache_dir=cache_dir,
        rebuild=args.rebuild_cache,
    )
    folds = build_folds(
        data,
        n_splits=config.split.n_splits,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )
    target = config.uncertainty.target_coverage
    print(f"calibrating to {target:.0%} coverage, final test group held closed", flush=True)

    calibrated = []
    for fold in folds:
        result = evaluate_fold(data, fold, config.split.seed, target)
        calibrated.append(result)
        print(
            f"  {fold.split.name}: split width={result.split_width:.2f} "
            f"cqr width={result.cqr_width:.2f} "
            f"on {len(result.calibration_engines)} calibration engines",
            flush=True,
        )

    intervals = {
        method: summarise(calibrated, method, target, config.split.seed)
        for method in ("split", "cqr")
    }
    for method, entry in intervals.items():
        print(
            f"  {method}: coverage={entry['empirical_coverage']:.3f} "
            f"width={entry['mean_width']:.2f}",
            flush=True,
        )

    cost = config.cost
    model = CostModel(
        planned_maintenance_cost=cost.planned_maintenance_cost,
        failure_cost_ratio=cost.failure_cost_ratio,
        downtime_cost_per_cycle=cost.downtime_cost_per_cycle,
        maintenance_window_cycles=cost.maintenance_window_cycles,
    )
    episodes = build_episodes(calibrated)
    print(
        f"costing {len(episodes)} engine episodes over a {cost.horizon}-cycle horizon", flush=True
    )
    decision = evaluate_policies(
        episodes,
        model,
        horizon=cost.horizon,
        thresholds=list(cost.thresholds),
        fixed_intervals=list(cost.fixed_intervals),
        ratios=list(cost.ratio_sweep),
        seed=config.split.seed,
    )

    report = report_phase4(
        intervals=intervals,
        decision=decision,
        config={
            "window": config.window.model_dump(),
            "split": config.split.model_dump(),
            "uncertainty": config.uncertainty.model_dump(),
            "cost": config.cost.model_dump(),
            "subdatasets": list(config.data.subdatasets),
        },
        data=data,
        folds=folds,
        hardware=hardware_description(),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out_path}", flush=True)

    print("\npolicies by cost, cheapest first:")
    for row in _policy_rows_sorted(decision):
        print(
            f"  {row['policy']:28s} cost={row['total_cost']:9.1f} "
            f"planned={row['n_planned']:3d} failures={row['n_failures']:3d}"
        )
    return 0


class PolicyRow(TypedDict):
    """One policy's costing row, with the fields the CLI prints."""

    policy: str
    total_cost: float
    cost_per_engine: float
    n_planned: int
    n_failures: int
    n_engines: int
    savings_vs_run_to_failure: float


def _policy_rows_sorted(decision: dict[str, Any]) -> list[PolicyRow]:
    """The policy rows, cheapest first.

    Typed rather than reached into as `dict[str, object]` so the print statement
    gets real field types instead of a cast at every access.
    """
    rows: list[PolicyRow] = list(decision["policies"])
    return sorted(rows, key=lambda row: row["total_cost"])


def cmd_stream(args: argparse.Namespace) -> int:
    """Replay held-out engines through the streaming pipeline and check S5.

    With `--faults`, skips the replay entirely and runs the fault scenarios instead:
    each named fault injected into a clean stream, recording what the detector named.
    That is the Phase 5 exit criterion from `docs/design.md` §12.
    """
    from prognos.evaluate import build_folds
    from prognos.pipeline import replay_pipeline

    config = load_config(Path(args.config))
    cache_dir = None if args.no_cache else config.data.parquet_dir.parent / "cache"

    print("loading windows", flush=True)
    data = load_windows(
        config.data.parquet_dir,
        config.window,
        config.data.subdatasets,
        cache_dir=cache_dir,
        rebuild=args.rebuild_cache,
    )
    if args.faults:
        return _run_fault_scenarios(data, config, args)

    folds = build_folds(
        data,
        n_splits=config.split.n_splits,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )
    cv_units = sorted({int(u) for f in folds for u in f.split.val_units})
    engines = cv_units[: args.engines] if args.engines else cv_units
    print(
        f"replaying {len(engines)} cross-validation engines; "
        f"final test group of {len(folds[0].split.test_units)} stays closed",
        flush=True,
    )

    holder: dict[str, Any] = {}

    def run() -> None:
        """The replay, on its own thread when a dashboard is watching it.

        A failure is carried back to the caller rather than dying on the thread, where
        it would surface as a bare exit code with no explanation.
        """
        try:
            holder["outcome"] = replay_pipeline(
                data,
                engines,
                config,
                seed=config.split.seed,
                calibration_fraction=args.calibration_fraction,
                pace=args.pace,
                model_dir=Path(args.model_dir),
                on_pipeline=lambda p: holder.update(pipeline=p),
            )
        except BaseException as error:
            holder["error"] = error

    if args.serve:
        # The dashboard reads the pipeline the replay is feeding, so the fleet view is
        # live while the replay runs rather than appearing all at once at the end.
        worker = threading.Thread(target=run, name="prognos-replay", daemon=True)
        worker.start()
        _serve_until_done(holder, worker, args)
        if "error" in holder:
            raise holder["error"]
        if "outcome" not in holder:
            print("the replay produced no result", file=sys.stderr)
            return 1
    else:
        run()
    outcome = holder["outcome"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outcome.report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out_path}", flush=True)

    verdict = outcome.verdict
    if verdict.measured is None:
        print(f"coverage: not enough data ({verdict.n_samples} scored predictions)")
    else:
        print(
            f"coverage: {verdict.measured:.3f} against a {verdict.target:.0%} target "
            f"({verdict.status}, {verdict.n_samples} predictions)"
        )
    print(f"alerts opened: {outcome.total_alerts}")
    return 0


def _serve_until_done(
    holder: dict[str, Any], worker: threading.Thread, args: argparse.Namespace
) -> None:
    """Serve the dashboard until the replay finishes.

    The server is started as soon as the pipeline exists and stopped when the worker
    ends, so `Ctrl-C` during a replay does not leave a port bound.
    """
    from prognos.stream.api import build_component
    from prognos.stream.auth import Authorizer, RateLimiter, token_from_env
    from prognos.stream.observability import MetricsRegistry
    from prognos.stream.service import build_server

    logger = logging.getLogger("prognos.dashboard")
    while "pipeline" not in holder and worker.is_alive():
        time.sleep(0.1)
    if "pipeline" not in holder:
        return  # the replay failed before it built one; the caller re-raises

    def state() -> Any:  # noqa: ANN401
        """The live pipeline's fleet, with the same auth the other services use."""
        pipeline = holder["pipeline"]
        built = build_component(
            "dashboard",
            metrics=MetricsRegistry(),
            logger=logger,
            fleet=pipeline.fleet,
            monitor=pipeline.coverage_monitor,
            on_inject=pipeline.arm_fault,
        )
        built.authorizer = Authorizer(token=token_from_env())
        built.rate_limiter = RateLimiter(limit=args.rate_limit, window_seconds=args.rate_window)
        return built

    httpd = build_server(state(), host=args.host, port=args.port)
    print(f"dashboard on http://{args.host}:{httpd.server_address[1]}/fleet", flush=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    worker.join()
    httpd.shutdown()
    httpd.server_close()


def _run_fault_scenarios(data: WindowSource, config: Config, args: argparse.Namespace) -> int:
    """Inject each named fault into one clean engine and record what was detected.

    This is the Phase 5 exit criterion executed rather than asserted: each named fault
    goes into an otherwise clean stream, and what the detector named is written to a
    committed file so the claim can be checked later.
    """
    from prognos.evaluate import build_folds, library_versions
    from prognos.pipeline import _unit_table, run_fault_scenarios
    from prognos.stream.faults import DEGRADED_STATES, InjectedFault

    folds = build_folds(
        data,
        n_splits=config.split.n_splits,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )
    unit = sorted({int(u) for f in folds for u in f.split.val_units})[0]
    label = data.group_labels[unit]
    cycles, features, _rul = _unit_table(
        Path(config.data.parquet_dir), label.subdataset, label.unit
    )

    faults = [
        InjectedFault("sensor_dropout", channel=7, from_cycle=40),
        InjectedFault("sensor_stuck", channel=3, from_cycle=40),
        InjectedFault("sensor_drift", channel=11, from_cycle=40),
        InjectedFault("value_out_of_range", channel=2, from_cycle=40),
    ]
    print(f"injecting {len(faults)} named faults into {label.label()}", flush=True)
    scenarios = run_fault_scenarios(cycles, features, faults)

    report: dict[str, Any] = {
        "config": {
            "window": config.window.model_dump(),
            "uncertainty": config.uncertainty.model_dump(),
            "subdatasets": list(config.data.subdatasets),
        },
        "hardware": hardware_description(),
        "versions": library_versions(),
        "generated_at": date.today().isoformat(),
        "final_test_group": {
            "engines": sorted(folds[0].split.test_units),
            "status": "closed",
            "note": "Fault scenarios replay a cross-validation engine, not a held-out one.",
        },
        "degraded_states": list(DEGRADED_STATES),
        "engine": label.label(),
        "n_rows": len(cycles),
        "scenarios": scenarios,
        "all_passed": all(bool(s["passed"]) for s in scenarios),
        "control_clean": all(bool(s["passed"]) for s in scenarios if not s["injected"]),
        "known_limitations": [
            (
                "sensor_drift does not survive the control run on healthy real data. "
                "Sampled one row per cycle, three channels (Nf, Nc, T48) move about "
                "20 units between adjacent windows as ordinary operation, and no "
                "normalisation tried separates that from a genuine step: scaling by "
                "the fleet ignores that the instruments differ by three orders of "
                "magnitude, scaling by the channel ignores that its series is smooth "
                "enough that ordinary movement is many times its own spread. Reported "
                "as failing rather than tuned until the control passes, which is the "
                "one thing that would have made the number meaningless."
            ),
            (
                "The scenarios replay one row per cycle, not the raw row stream. The "
                "dataset carries thousands of rows per cycle number, and a raw replay "
                "would leave the whole scenario inside a single cycle."
            ),
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {out_path}", flush=True)
    for scenario in scenarios:
        print(
            f"  {scenario['fault']:20s} detected={scenario['detected']} "
            f"{'ok' if scenario['passed'] else 'MISSED'}"
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Measure streaming latency and throughput at several fleet sizes."""
    from prognos.bench import benchmark
    from prognos.evaluate import build_folds, library_versions
    from prognos.stream.pipeline import PipelineConfig

    config = load_config(Path(args.config))
    cache_dir = None if args.no_cache else config.data.parquet_dir.parent / "cache"
    data = load_windows(
        config.data.parquet_dir, config.window, config.data.subdatasets, cache_dir=cache_dir
    )
    folds = build_folds(
        data,
        n_splits=config.split.n_splits,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )
    cv_units = sorted({int(u) for f in folds for u in f.split.val_units})
    labels = [data.group_labels[u] for u in cv_units]
    pipeline_config = PipelineConfig(
        window_length=config.window.length,
        window_stride=config.window.stride,
        downsample=config.window.downsample,
        rul_cap=config.window.rul_cap or 125,
        alert_threshold=config.cost.thresholds[0] if config.cost.thresholds else 20.0,
        target_coverage=config.uncertainty.target_coverage,
    )
    sizes = [int(size) for size in args.fleet_sizes.split(",")]
    print(f"benchmarking fleets of {sizes} from {len(labels)} cross-validation engines", flush=True)
    report = benchmark(
        Path(args.model_dir),
        Path(config.data.parquet_dir),
        labels,
        pipeline_config,
        fleet_sizes=sizes,
        rows_per_engine=args.rows_per_engine,
        repeats=args.repeats,
        http_requests=args.http_requests,
    )
    report.update(
        {
            "command": " ".join(["prognos", *sys.argv[1:]]),
            "versions": library_versions(),
            "generated_at": date.today().isoformat(),
            "final_test_group": {
                "engines": sorted(folds[0].split.test_units),
                "status": "closed",
            },
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve one streaming component over HTTP until interrupted."""
    import logging

    from prognos.stream.api import build_component, default_port
    from prognos.stream.auth import Authorizer, RateLimiter, token_from_env
    from prognos.stream.observability import JsonLogFormatter, log_event
    from prognos.stream.service import serve

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(message)s",
        force=True,
    )
    logger = logging.getLogger(f"prognos.{args.component}")
    for handler in handlers:
        handler.setFormatter(JsonLogFormatter(service=args.component))

    options: dict[str, Any] = {"logger": logger}
    if args.component == "inference":
        options["model_dir"] = Path(args.model_dir) if args.model_dir else None
    if args.component == "alerting":
        options["threshold"] = args.alert_threshold
        options["hysteresis"] = args.alert_hysteresis

    state = build_component(args.component, **options)

    # Read the token here rather than inside the service, so the decision is visible in
    # one place and a component built in-process can deliberately stay unauthenticated.
    state.authorizer = Authorizer(token=token_from_env())
    state.rate_limiter = RateLimiter(limit=args.rate_limit, window_seconds=args.rate_window)
    if state.authorizer.enabled:
        log_event(
            logger,
            "service.auth_enabled",
            service=args.component,
            host=args.host,
            rate_limit=args.rate_limit,
        )

    port = args.port or default_port(args.component)
    serve(state, host=args.host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="prognos", description="Prognos pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="HDF5 to Parquet")
    convert.add_argument("--config", default="configs/base.yaml")
    convert.set_defaults(func=cmd_convert)

    from prognos.stream.api import COMPONENTS

    manifest = sub.add_parser("manifest", help="show the data manifest")
    manifest.add_argument("--config", default="configs/base.yaml")
    manifest.set_defaults(func=cmd_manifest)

    evaluate = sub.add_parser("evaluate", help="cross-validate the baselines")
    evaluate.add_argument("--config", default="configs/base.yaml")
    evaluate.add_argument("--out", default="results/phase2_baselines.json")
    evaluate.add_argument("--bootstrap", type=int, default=500)
    evaluate.add_argument("--coverage", type=float, default=0.90)
    evaluate.add_argument("--no-cache", action="store_true")
    evaluate.add_argument("--rebuild-cache", action="store_true")
    evaluate.add_argument("--baselines-only", action="store_true")
    evaluate.add_argument(
        "--sequence", action="store_true", help="also evaluate the CNN, LSTM and Transformer"
    )
    evaluate.add_argument("--epochs", type=int, default=30)
    evaluate.add_argument(
        "--late-weight",
        type=float,
        default=1.0,
        help="weight on late predictions in the training loss; 1.0 is plain MSE",
    )
    evaluate.add_argument("--verbose", action="store_true", help="print per-epoch training")
    evaluate.set_defaults(func=cmd_evaluate)

    policy = sub.add_parser(
        "policy", help="calibrated prediction intervals and the maintenance policy"
    )
    policy.add_argument("--config", default="configs/base.yaml")
    policy.add_argument("--out", default="results/phase4_policy.json")
    policy.add_argument("--bootstrap", type=int, default=500)
    policy.add_argument("--no-cache", action="store_true")
    policy.add_argument("--rebuild-cache", action="store_true")
    policy.set_defaults(func=cmd_policy)

    stream = sub.add_parser("stream", help="replay held-out engines through the streaming pipeline")
    stream.add_argument("--config", default="configs/base.yaml")
    stream.add_argument("--out", default="results/phase5_streaming.json")
    stream.add_argument("--engines", type=int, default=0, help="0 means every CV engine")
    stream.add_argument(
        "--calibration-fraction",
        type=float,
        default=None,
        help="fraction of engines held out to calibrate; drives interval width",
    )
    stream.add_argument(
        "--model-dir",
        default="artifacts/model",
        help="where the fitted model and its calibrator are written for inference to load",
    )
    stream.add_argument(
        "--faults",
        action="store_true",
        help="inject each named sensor fault into a clean stream and record what is detected",
    )
    stream.add_argument(
        "--serve",
        action="store_true",
        help="serve the dashboard from the replay's own pipeline, so the fleet view is live",
    )
    stream.add_argument(
        "--pace",
        type=float,
        default=None,
        help="cap the replay at this many readings per second, for watching the dashboard",
    )
    stream.add_argument("--host", default="127.0.0.1")
    stream.add_argument("--port", type=int, default=8090)
    stream.add_argument("--rate-limit", type=int, default=600)
    stream.add_argument("--rate-window", type=float, default=60.0)
    stream.add_argument("--no-cache", action="store_true")
    stream.add_argument("--rebuild-cache", action="store_true")
    stream.set_defaults(func=cmd_stream)

    bench = sub.add_parser("bench", help="measure streaming latency and throughput")
    bench.add_argument("--config", default="configs/base.yaml")
    bench.add_argument("--out", default="results/phase6_streaming_bench.json")
    bench.add_argument("--model-dir", default="artifacts/model")
    bench.add_argument("--fleet-sizes", default="10,100,1000")
    bench.add_argument("--rows-per-engine", type=int, default=3000)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--http-requests", type=int, default=2000)
    bench.add_argument("--no-cache", action="store_true")
    bench.set_defaults(func=cmd_bench)

    serve_cmd = sub.add_parser("serve", help="serve one streaming component over HTTP")
    serve_cmd.add_argument("--component", required=True, choices=list(COMPONENTS))
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=0)
    serve_cmd.add_argument("--model-dir", default="artifacts/model")
    serve_cmd.add_argument("--alert-threshold", type=float, default=20.0)
    serve_cmd.add_argument("--alert-hysteresis", type=float, default=5.0)
    serve_cmd.add_argument(
        "--rate-limit", type=int, default=120, help="requests per window, per client"
    )
    serve_cmd.add_argument("--rate-window", type=float, default=60.0)
    serve_cmd.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
