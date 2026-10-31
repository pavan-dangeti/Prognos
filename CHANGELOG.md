# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [1.0.0] - 2026-10-31

The first release: remaining-useful-life prediction with calibrated intervals, a
cost-based maintenance policy, and a live replayed fleet with fault handling, a
dashboard and measured streaming performance.

### Added

- **Fleet dashboard** at `/ui` on the dashboard service: engines worst first with their
  90% interval bands, the RUL history of the engine streaming now, maintenance alerts
  and degraded states as they happen, the Phase 4 cost comparison, and controls to
  inject and clear a sensor fault on a live engine. The page loads nothing from other
  origins and runs under a hash-pinned Content-Security-Policy.
- **Live fault injection.** `POST /inject` now reaches the rows the detector and the
  model read. Before, it recorded the request and nothing applied it.
- **`prognos bench`**: end-to-end latency (p50, p95, p99) and throughput at fleets of
  10, 100 and 1,000 engines, interleaved over five rounds, plus the inference
  service's HTTP round trip. Results in `results/phase6_streaming_bench.json`.
- **`prognos stream --pace`** caps the replay at a number of readings per second, so a
  fault can be watched arriving and clearing.
- `StreamPipeline.push`, one reading at a time, so engines interleave as they would
  on a real stream.
- A demo video and GIF under `docs/media/`.
- A second CI coverage gate at 85% over the streaming services, the decision layer
  and the benchmark.

### Fixed

- The live path fed the fault detector every raw reading, about 11,000 per cycle,
  although it judges an engine cycle to cycle and its results were measured that
  way. It was the largest cost in the pipeline. It now sees one reading per cycle.
- A dropped sensor (NaN) stopped the whole replay, because the window refuses
  non-finite values. The channel is now held at its last good value, and the
  detector names the dropout.
- `/coverage` returned 500 intermittently during a replay: the dashboard iterated the
  coverage window while the replay thread appended to it.
- The dashboard's `/coverage` scored an empty monitor of its own instead of the
  replay's.
- The fleet view's `alert` column was always empty, because it read a key that alerts
  do not carry.
- Alerting, fault and fleet-view state were keyed by unit number, which restarts in
  every sub-dataset. A later engine sharing a unit number resolved the earlier
  engine's open alert and could show it on the wrong row: 98 alerts were recorded
  where 91 were raised. State is now keyed by `(sub-dataset, unit)`, alerts carry their
  sub-dataset, and `POST /inject` and `POST /alerts` accept a `source`.
- `results/phase5_streaming.json` regenerated on the fixed code: 91 alerts, every
  engine's count matching the original Phase 5 run, coverage unchanged at 0.991.

### Known issues

- The drift detector fires on healthy channels; its false-alarm rate is not yet
  measured.
- The coverage monitor's `worst_engine` names a bare unit number.
- No hosted demo, MLflow registry, model card or OpenTelemetry tracing yet.

## Earlier phases

Merged as pull requests before the first release:

- **Phase 5**: streaming pipeline, online window, inference, alerting, live coverage
  monitoring, fault injection and degraded-state detection, fleet view, cost panel,
  bearer-token auth and rate limiting, Docker Compose with Prometheus and Grafana.
- **Phase 4**: split conformal and conformalised quantile regression intervals, and a
  cost-based maintenance policy compared against run-to-failure and fixed intervals.
- **Phase 3**: CNN, LSTM and Transformer models with life-stage analysis.
- **Phase 2**: mean, linear-cycle, trend and LightGBM baselines and the evaluation
  harness.
- **Phase 1**: HDF5 to Parquet conversion, windowing, engine-level splits and the
  leakage test.
- **Phase 0**: design document and architecture decision records.

[1.0.0]: https://github.com/pavan-dangeti/Prognos/releases/tag/v1.0.0
