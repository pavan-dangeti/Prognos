"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

from .config import load_config
from .convert import convert_subdataset, sha256_file, write_manifest
from .dataset import load_windows

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
    """Replay held-out engines through the streaming pipeline and check S5."""
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

    outcome = replay_pipeline(
        data,
        engines,
        config,
        seed=config.split.seed,
        calibration_fraction=args.calibration_fraction,
    )

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


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="prognos", description="Prognos pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="HDF5 to Parquet")
    convert.add_argument("--config", default="configs/base.yaml")
    convert.set_defaults(func=cmd_convert)

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
    stream.add_argument("--no-cache", action="store_true")
    stream.add_argument("--rebuild-cache", action="store_true")
    stream.set_defaults(func=cmd_stream)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
