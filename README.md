# Prognos

A turbofan engine's sensors drift as it wears, long before it fails, but a single
remaining-life estimate gives a maintenance planner no idea how far to trust it.
Prognos predicts each engine's remaining useful life with a calibrated 90% interval
from NASA N-CMAPSS sensor data, turns that interval into maintenance decisions priced
against simple policies, and runs it all on a live replayed fleet that names its own
sensor faults instead of predicting through them.

![The fleet dashboard: engines sorted worst first with interval bands, a sensor fault injected on one engine and named as low confidence, then cleared](docs/media/demo.gif)

[Watch the 2-minute demo (MP4)](docs/media/demo.mp4). Live hosted demo: not yet
deployed. Research demonstration on simulated engines; read the
[Limitations](#limitations) before drawing any operational conclusion.

## Results at a glance

Every figure comes from a committed result file, produced by the command next to it,
on an Apple M5 with 16 GB (macOS, Python 3.12.13). `tests/test_results.py` and
`tests/test_bench.py` fail CI if a quoted figure drifts from its file.

| Question | Result | Baseline | How it was measured | Reproduce |
| --- | --- | --- | --- | --- |
| How accurate is the RUL point estimate? | CNN 7.40 cycles RMSE (95% CI 4.61–10.20) | 8.90 from cycle number alone, no sensors; 23.76 predicting the mean | 5-fold CV grouped by engine, 29 engines, 10-engine test group never opened | `prognos evaluate --sequence` → [`phase3_sequence.json`](results/phase3_sequence.json) |
| Do sensors help where it matters, late in life? | Transformer 3.76 RMSE in late life | 8.84 from cycle number alone | Same folds, error split by life stage | same |
| Do the intervals hold? | 0.926 offline, 0.991 live, target 0.90 | — | Split conformal, calibrated on engines held out of the fit | `prognos policy`, `prognos stream` |
| What does it save? | 30.8 total cost for the best policy | 577.3 run-to-failure, 44.0 best fixed interval | 19 engines, 2,000 cycles, failure 25× a planned visit, ratio swept 1–100 | `prognos policy` → [`phase4_policy.json`](results/phase4_policy.json) |
| How fast is the live path? | p99 5.16–9.49 ms per updated prediction; at least 43,141 readings/s | — | Real engines interleaved at fleets of 10, 100 and 1,000, 5 rounds | `prognos bench` → [`phase6_streaming_bench.json`](results/phase6_streaming_bench.json) |
| Are sensor faults named? | 4 of 4 injected faults named | Control run must name nothing: **fails** | Each fault injected alone into a clean engine | `prognos stream --faults` → [`phase5_faults.json`](results/phase5_faults.json) |

Two results go against the design document and are reported as they came out:
acting on the interval's lower bound costs **more** than acting on the point estimate
on this data, and the drift detector fires on healthy engines. Both are explained
below.

## Architecture

```mermaid
flowchart LR
    subgraph offline["Offline (laptop)"]
        H5[N-CMAPSS HDF5] -->|prognos convert| PQ[Parquet per engine<br/>health parameters never read]
        PQ --> WIN[Windows, split by engine]
        WIN -->|prognos evaluate| CV[Grouped CV:<br/>baselines, CNN, LSTM, Transformer]
        WIN -->|prognos policy| CAL[Conformal calibration<br/>+ cost simulation]
        CAL --> ART[(model + calibrator)]
    end

    subgraph live["Live replay (prognos stream)"]
        SIM[Replay of held-out engines<br/>at accelerated time] --> ING[Validation:<br/>size, range, finiteness]
        ING --> FD[Fault detector<br/>one row per cycle]
        ING --> OW[Online window<br/>identical to offline]
        OW --> INF[LightGBM + 90% interval]
        ART --> INF
        INF --> AL[Alerting on the lower bound]
        INF --> COV[Live coverage monitor]
        FD --> FV[Fleet view]
        AL --> FV
        INF --> FV
    end

    FV --> UI[Dashboard /ui<br/>fault injection]
    UI -->|POST /inject| SIM
    COV --> PROM[Prometheus] --> GRAF[Grafana]
    FV --> PROM
```

Design: [`docs/design.md`](docs/design.md), with decisions in [`docs/adr/`](docs/adr/).

## Run it locally

Needs [uv](https://docs.astral.sh/uv/) and about 2.5 GB of N-CMAPSS HDF5; where to
get it and its terms of use are in [`docs/data.md`](docs/data.md).

```bash
uv sync --locked
uv run prognos convert --config configs/base.yaml          # HDF5 -> Parquet, ~1 min
uv run prognos evaluate --config configs/base.yaml --out results/phase3_sequence.json --sequence --epochs 30
uv run prognos policy --config configs/base.yaml           # intervals and costs, ~5 min

# The live dashboard: trains the serving model, then replays the fleet.
uv run prognos stream --config configs/base.yaml --serve --pace 10000 --out /tmp/replay.json
open http://127.0.0.1:8090/ui

# Latency and throughput, after `prognos stream` has written artifacts/model.
uv run prognos bench

# The six-service stack with Prometheus and Grafana; see docs/runbook.md.
docker compose up -d --build
```

The test suite needs no NASA data: `uv run pytest -m smoke` runs the whole pipeline on
a synthetic subset, which is what CI does.

## Streaming latency and throughput

```bash
uv run prognos bench    # fleets of 10, 100 and 1,000 engines, 5 interleaved rounds
```

Latency is from a sensor reading handed to the pipeline to the updated prediction it
returns: fault detection, the online window, features, LightGBM, the conformal
interval, alerting and the fleet view. It is timed only for readings that complete a
window, which is about one in a hundred. Throughput is readings per second through one
Python process. N-CMAPSS records at 1 Hz, so readings per second is also the number of
engines one process keeps up with in real time. Fleets are built from the 19
cross-validation engines, interleaved one reading per engine in turn, so every engine
keeps its own window and detector live at once.

| fleet size | readings/s, median (min–max of 5) | predictions/s | p50 ms | p95 ms | p99 ms | worst p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 51,075 (43,141–161,353) | 443 | 1.50 | 3.13 | 5.16 | 10.88 |
| 100 | 82,241 (51,099–165,102) | 700 | 0.67 | 1.57 | 8.55 | 8.55 |
| 1,000 | 51,037 (50,162–165,015) | 434 | 1.10 | 4.17 | 9.49 | 9.49 |

The inference service over loopback HTTP, one window per request: p50 3.44 ms, p95
3.94 ms, p99 8.93 ms over 2,000 requests. That hop is not in the figures above.

Hardware: Apple M5 (10 cores, one used), 16 GB, macOS 26, Python 3.12.13, LightGBM
4.6.0. Raw runs: [`results/phase6_streaming_bench.json`](results/phase6_streaming_bench.json).

**Fleet size does not change the cost per reading between 10 and 1,000 engines.**
The variation in the table is not a scaling effect. Rounds fall into two clusters,
about 50,000 and about 160,000 readings a second, and which cluster a round lands in
depends on when it ran, not on its size: round 2 was fast at every size, rounds 4 and
5 slow at every size. That is consistent with macOS moving an unpinned process
between performance and efficiency cores, which I cannot control on this machine. So
the figure to rely on is the slowest round, 43,141 readings a second, and the sizes
are interleaved so the drift cannot pose as a trend.

**Measuring this found the detector being fed the wrong data.** As Phase 5 shipped
it, the live path handed the fault detector every raw reading, about 11,000 per
cycle, although the detector is designed to judge an engine cycle to cycle and the
fault results were measured that way. It was by far the largest cost in the pipeline.
It now sees the first reading of each cycle, and the live fleet coverage is unchanged
by the fix.

## Limitations

What this project is not:

- **No physics-based degradation model.** Everything here is learned from sensor
  data, and nothing claims a physics model is unnecessary. A hybrid would likely be
  more robust to operating regimes the training engines never saw.
- **No real aircraft data.** Every engine is simulated by NASA's C-MAPSS model under
  realistic flight profiles. No aircraft flew them, and no result transfers to a real
  fleet without being re-measured on one.
- **No safety or airworthiness claim.** This is a research demonstration. It must not
  be used to decide whether an engine or an aircraft is safe to operate.

What the evidence does not support:

- **Small data.** 29 engines in total, 19 in cross-validation. The confidence
  intervals on model comparisons are wide; the CNN and Transformer intervals overlap
  the no-sensor baseline's, and the life-stage split is the stronger evidence.
- **Three of nine sub-datasets.** DS01, DS02 and DS06. Claims about failure-mode
  generality rest on DS06 alone.
- **The cost model is an assumption set,** not airline economics. Planned maintenance
  is normalised to 1 and failure is swept from 1× to 100×; planning loses at 1× and
  wins from 5× up.
- **Intervals are wide and only marginally valid.** Mean width 38 cycles on a scale
  capped at 125. Conformal prediction guarantees coverage on average over
  exchangeable data, not per engine, per life stage, or under a shift in operating
  conditions. Live coverage over-covers at 0.991, outside the 0.90 ± 0.05 band on the
  safe side.
- **The lower-bound policy is not the cheapest here.** The point estimate already
  prevents every failure on these engines, so the lower bound buys insurance nothing
  needed.

What the system does not yet do:

- **The drift detector has false positives.** It fires on healthy channels (`Nf`,
  `Nc`, `T48`) of a healthy engine, so every fault scenario reports `sensor_drift`
  alongside the fault actually injected. Its false-alarm rate across all engines has
  not been measured yet.
- **The coverage monitor still names engines by unit number alone.** Its
  `worst_engine` field reports a bare unit, which is ambiguous across sub-datasets.
  Alerting and fault state are keyed by the full `(sub-dataset, unit)` identity.
- **Not yet built:** an inference-service outage test, an operating-condition shift
  test, MLflow tracking and a model registry, a model card, OpenTelemetry tracing, and
  a hosted demo.
- **Latency is measured on a laptop against a replay,** at accelerated time, not on an
  aircraft datalink, and the HTTP figure is loopback only.
- **Maintenance is assumed schedulable at any threshold.** Real maintenance has
  calendar, hangar and crew constraints this simulation does not model.

## Detailed results

5-fold grouped cross-validation, 29 engines from DS01, DS02 and DS06, 199,095
windows. Intervals are 95% bootstrap resampling **by engine**, not by window. The
final test group of 10 engines was held closed throughout, for every model.

| model | rmse | nasa_score | rmse 95% CI | params | how measured |
| --- | --- | --- | --- | --- | --- |
| `cnn` | 7.40 | 24,343 | 4.61–10.20 | 113,665 | grouped CV, 1D CNN over the window |
| `transformer` | 7.43 | 24,396 | 5.39–9.47 | 68,417 | grouped CV, 2-layer encoder |
| `lightgbm` | 8.15 | 30,027 | 4.23–12.07 | — | grouped CV, window statistics + cycle context |
| `linear_cycle` | 8.90 | 34,950 | 4.62–13.18 | — | grouped CV, RUL ~ cycle number, **no sensors** |
| `lstm` | 13.00 | 65,732 | 10.58–15.41 | 55,105 | grouped CV, 2-layer LSTM |
| `mean` | 23.76 | 283,059 | 22.08–25.44 | — | grouped CV, training-set mean RUL |
| `trend` | 25.87 | 557,375 | 25.09–26.65 | — | grouped CV, per-channel trend to threshold |

Reproduce:

```bash
uv run prognos convert --config configs/base.yaml
uv run prognos evaluate --config configs/base.yaml --out results/phase3_sequence.json --sequence --epochs 30
```

Raw output: [`results/phase3_sequence.json`](results/phase3_sequence.json).
`tests/test_results.py` fails CI if any figure in this table drifts from a committed
result file.

Hardware for every number: Apple M5, 16 GB, macOS, Python 3.12.13, torch 2.10.0 on
MPS. The full seven-model run takes about 55 minutes, dominated by the Transformer.
~45 s of that is materialising 199,095 windows.

Every committed result file records the machine, the library versions and the date it
was produced in, so a number can be traced to the software that made it. That field
was added after an experiment on a different torch build silently overwrote a result
file — nothing in the file distinguished the two runs, which is exactly the kind of
gap that makes a result unfalsifiable.

The run is deterministic: re-running both commands on the same machine reproduces
every figure in the table to the last printed digit, sequence models included. CNN
7.40, Transformer 7.43, LSTM 13.00, LightGBM 8.15.

### Does sensor data actually help?

This is the question Phase 3 exists to answer, and `linear_cycle` is the control for
it. That model uses **no sensor data at all** — just cycle number — and reaches 8.90
RMSE. A sequence model that cannot beat it has not shown that sensor channels
contribute anything beyond knowing the engine's age.

The CNN at 7.40 does beat it, by 1.5 cycles. So do the life-stage numbers, which are
more informative than the headline:

| model | early | mid | late |
| --- | --- | --- | --- |
| `transformer` | 9.38 | 7.80 | **3.76** |
| `cnn` | 9.30 | 7.64 | **4.16** |
| `lightgbm` | 9.04 | 8.71 | 6.37 |
| `linear_cycle` | 8.93 | 8.77 | 8.84 |

`linear_cycle` is flat across life stages, which is exactly what a model that only
knows the engine's age should be. The sequence models are worse early — where there
is little to detect yet — and roughly **half** as good late, where degradation is
visible and where maintenance decisions are actually made. That asymmetry is the
evidence that the sensors are carrying degradation information, and it is the part
that matters operationally.

Honest caveat: the CNN/Transformer confidence intervals overlap the `linear_cycle`
interval. At 29 engines, 1.5 cycles of separation is suggestive, not conclusive. The
life-stage split is the stronger evidence because it is a within-model comparison
against a model that cannot improve by learning the sensors.

### The two results worth arguing about

**The LSTM failed.** 13.00 RMSE, clearly worse than the cycle-only floor. Its
life-stage pattern is the diagnosis: 14.06 early, 7.63 mid, 15.02 late. The middle
is the best of any model in the table; the ends are the worst. It is reading the
trajectory only where the trajectory is locally informative and falling back
toward the mean elsewhere. The architectural suspect is the readout — it uses only
the final hidden state, discarding the intermediate states, while both the CNN and
the Transformer pool over the whole window. Pooling is the obvious fix and it has
**not** been applied, because validating it on these folds would be tuning on the set
being reported. It belongs in a run with a different seed and a fresh fold assignment.

**The degradation-trend baseline got worse, as predicted.** Phase 2 recorded RMSE
37.73 in late life and attributed it to the window being too short to estimate a
channel's slope. That prediction held: it is 37.73 again here, unchanged, because
nothing about the window changed. The method is sound; the input does not support it
at downsample 10 and length 50.

## Prediction intervals and the maintenance policy

Phase 4. Conformal intervals calibrated on engines held out of the fit, and a cost
comparison of four maintenance policies on the same 19 engines over a 2,000-cycle
horizon.

Calibration gets its own engines: within each fold the training engines split again
into a fitting set and a calibration set, so the coverage below is not inflated by
residuals from engines the model memorised. `tests/test_results.py` checks that
disjointness per fold from the committed artifact. The 10-engine final test group is
still closed.

```bash
uv run prognos policy --config configs/base.yaml --out results/phase4_policy.json
```

### Intervals

LightGBM, 5-fold grouped CV, target 90% coverage. ~5 minutes.

| method | coverage | 95% CI | mean width | point RMSE |
| --- | --- | --- | --- | --- |
| split conformal | 0.926 | 0.811–0.998 | 38.0 | 10.00 |
| CQR (0.05/0.95 quantiles) | 0.929 | 0.813–1.000 | 41.8 | 10.00 |

Both clear the R6 SLO. Three things are worth being blunt about:

**The intervals are wide.** Mean width 38 on an RUL scale capped at 125 is about 30%
of the entire range. This is a valid instrument and a weak one: it tells you the
model is uncertain across a third of the operating range, which is not sharp enough
to schedule against.

**CQR did not beat split conformal, and the design document expected it to.** §4.4
argued CQR should be narrower because it adapts width to where the model is
uncertain. It is wider: 41.8 against 38.0. The quantile regressors at 0.05/0.95
already produce a broad band, and the conformal correction on top of it makes the
result broader still. Reporting it as the plan did would have been the easier
story and the false one.

**Coverage is marginal, and conditional coverage is not uniform.** By life stage:
early 0.908, mid 0.908, late 0.961. The aggregate clears 90% partly because late-life
over-covers. A conformal interval promises marginal coverage over exchangeable
points — not per engine, not per stage, not under drift. That is why S5 exists.

The point RMSE is 10.00 here against 8.15 in Phase 3. That is the cost of the nested
split: this harness fits on roughly 70% of each fold's training engines instead of
all of them. Both are reported; the better number would not have been comparable.

### Cost of the policies

Costs are normalised so a planned visit is 1.0, because absolute scale cancels out of
every comparison — all policies fly the same fleet over the same horizon. The only
term that changes the answer is the failure/planned ratio, so that is what is swept.
Downtime is charged at the same rate to planned and unplanned work, leaving the ratio
as the single asymmetry.

Failure/planned ratio 25, the headline row:

| policy | total cost | planned visits | failures | vs run-to-failure |
| --- | --- | --- | --- | --- |
| `threshold_point_10` | **30.8** | 28 | 0 | **−546.5** |
| `fixed_interval_50` | 44.0 | 40 | 0 | −533.3 |
| `threshold_lower_10` | 46.2 | 42 | 0 | −531.1 |
| `threshold_lower_40` | 115.5 | 105 | 0 | −461.8 |
| `fixed_interval_125` | 419.2 | 16 | 16 | −158.1 |
| `run_to_failure` | 577.3 | 0 | 23 | 0 |

Every policy that plans beats waiting for engines to break, and the cheapest calendar
is 125 cycles, not 50 — replacing on a short schedule spends more on visits than it
saves on failures.

**The design document's hypothesis is wrong, and the lower bound loses.** §4.5 says
the lower-bound policy is the one under test. It is the *most expensive* threshold
policy at every threshold: 46.2 against 30.8 at a threshold of 10. The reason is
visible in the counts — 42 planned visits instead of 28. The point-estimate policy
already prevents every failure, so the lower bound buys insurance that nothing
needed. Acting on the conservative edge is the right instinct in general; on this
data, at this threshold, it is strictly worse.

### The result that does not depend on my cost guess

Savings of the lower-bound policy against run-to-failure, across the ratio sweep:

| threshold | 1 | 5 | 10 | 25 | 50 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | **−20.9** | 71.1 | 186.1 | 531.1 | 1106.1 | 2256.1 |
| 20 | **−31.9** | 60.1 | 175.1 | 520.1 | 1095.1 | 2245.1 |
| 30 | **−57.2** | 34.8 | 149.8 | 494.8 | 1069.8 | 2219.8 |
| 40 | **−90.2** | 1.8 | 116.8 | 461.8 | 1036.8 | 2186.8 |

At a ratio of 1, where an in-flight removal costs the same as a shop visit, planning
**loses** — −20.9 to −90.2 depending on threshold. Above that it wins, and the
savings grow monotonically. The crossover is the result, and it does not require
anyone to agree with the cost figures I picked. The columns are flat left to right
for each policy's own cost, because these policies record no failures: the ratio only
prices failures, and there are none.

### Two bugs this phase produced, both of which read as success

**Episodes in windows instead of cycles.** The first full run reported
`run_to_failure` at cost 0.0 with zero failures and every threshold policy in single
digits — apparently a crushing win. Episodes were built from window indices, and
because windows are a stride apart and each covers a window length, every engine
looked about a hundred times longer than it was. No engine reached the end of its life
inside the horizon, so nothing could fail and nothing could cost anything. The fix
was to index episodes by real cycle number. `simulate_fleet` now refuses a horizon
shorter than the shortest engine, and `tests/test_results.py` fails if run-to-failure
records no failures.

**A confidence interval of [18.0, 20.7] around a coverage of 0.926.** The bootstrap
statistic signature is `(y_true, y_pred)` and coverage is a mean over the first
argument, so passing the bounds as the second argument silently averaged the bounds.
The reported interval was a real number and was not a confidence interval at all.
Fixed, and pinned by a test that the interval must bracket its own point estimate.

## The streaming pipeline

Phase 5. Six components over `docker compose`, replaying held-out engines as a live
fleet, with the S5 coverage check actually implemented rather than merely specified.

```bash
uv run prognos stream --config configs/base.yaml --out results/phase5_streaming.json
docker compose up -d        # ingestion, inference, dashboard, Prometheus, Grafana
```

All 19 cross-validation engines: 100,000 scored predictions, 91 maintenance alerts. The
10-engine final test group is still closed.

**The alert count was briefly wrong.** Alerting and fault state were keyed by unit
number, which restarts in every sub-dataset. When DS02 unit 2 followed DS01 unit 2 in
the replay, its first prediction "resolved" the alert DS01 unit 2 had left open, and
the fleet view could show one engine's alert on the other's row. Rerun on the code as
it stood before the fix, the replay recorded 98 alerts, the 7 extra all on engines
whose unit number repeats an earlier one. State is now keyed by `(sub-dataset, unit)`,
the count is back to 91 with every engine's figure matching the original Phase 5 run,
and coverage did not move. `tests/test_stream_pipeline.py` replays two engines sharing
unit 2 and fails on the old keying.

### The online window reproduces the offline builder exactly

`docs/design.md` §4.6 requires it, and nothing enforces that requirement except a test.
Compared against `iter_windows` and `stack_windows` themselves — not a restatement of
the formula — on a real engine: **8,204 windows, maximum feature difference 0.0.**

This is not a formality. The first implementation assumed one row per cycle. This
dataset has ~11,000 rows per cycle number, and `iter_windows` downsamples *within* the
cycle block. The wrong implementation produces plausible features and a quietly worse
model. A second off-by-one, taking downsampled indices `d-1, 2d-1, …` instead of
`0, d, 2d, …`, shifted every window by one row and was caught only by that comparison.

### Live coverage: 0.991 against a 0.90 target

| | Phase 4 (offline, nested CV) | Phase 5 (live fleet replay) |
| --- | --- | --- |
| coverage | 0.926 | **0.991** |
| interval width | 38.0 | 35.0 |
| conformal half-width | 19.0 | 17.5 |
| scored predictions | 138,076 | 100,000 |

The SLO band is 0.90 ± 5 points. **0.991 is outside that band**, over-covering by 9
points. The monitor reports `ok`, and that is a deliberate asymmetry worth stating
plainly: under-covering is dangerous, because a decision is being taken on a range that
does not hold, while over-covering is merely wasteful. But it is not an SLO pass, and
an interval of half-width 17.5 on an RUL scale capped at 125 is not sharp enough to
schedule against. The honest conclusion is that the point model, not the interval
width, is the binding constraint.

### The check caught a real bug, which is the best evidence it works

The first replaying run reported **coverage 0.716 — a breach**. Every other number in
that result file looked entirely reasonable: 19 engines, sensible window counts,
plausible alert counts. The breach was the only thing that gave it away.

The cause: the replay driver used the **global engine index as a Parquet file number**.
Engine `GroupKey` is `(sub-dataset, unit)`; global index 1 is `DS01-005` unit 2, but the
driver read `unit_001.parquet` — a different engine the model had never been fitted on.
The coverage check flagged it automatically. A test now asserts that every replayed
engine exists in the dataset and carries its sub-dataset.

### What did not change

Calibration breadth barely moves coverage on this fleet, which contradicts the obvious
guess that a narrow calibration set is the problem:

| calibration fraction | engines | live coverage |
| --- | --- | --- |
| 0.05 | 1 | 0.986 |
| 0.15 | 3 | 0.995 |
| 0.30 | 13 | 0.991 |

All three over-cover. The interval is wide because the model's error is large, not
because the calibration is thin — which is the same conclusion Phase 4 reached about
the 38-wide intervals, now confirmed on the live path.

### Serving LightGBM, not the CNN

The CNN is 2.4 cycles better on cross-validated RMSE. torch costs roughly 800 MB
resident per process, and the free Always Free target is 12 GB for the whole stack.
Shipping it for a demo that does not need it is not justified. The architecture does
not prevent shipping it later — it is a different payload in the same slot — and the
trade-off is recorded here rather than buried.

### One image, six services, and actually running

Six images would mean six builds and six vulnerability scans for the same code. The
compose file runs all six components from one image with different commands. Grafana
and Prometheus are pinned by real digests, as the base image is.

```bash
uv run prognos stream --config configs/base.yaml   # trains and writes artifacts/model
docker compose up -d --build
docker compose ps        # six services healthy
```

Verified running, not just validating: six containers healthy, 5/5 Prometheus scrape
targets up, `prognos_service_up` scraped from all four services, the Grafana dashboard
provisioned with 7 panels, and ingestion counters moving on real POSTs.

**Bringing it up found four things no test found.** The compose file had
`prognos stream` as the long-running command for all three services, and that is the
replay CLI — it writes a file and exits. Every service would have died instantly and
every healthcheck would have failed. `service.py` existed and was never wired to a CLI
entry point.

Then, in order:

1. **Nothing served HTTP at all.** No `prognos serve` command existed.
2. **Every port was unreachable.** Services bound to `127.0.0.1` *inside* the
   container, so the host's port forward reached nothing. A container-internal bind must
   be `0.0.0.0`; the security boundary is the host-side mapping, which stays on
   127.0.0.1. Two different boundaries, two different settings.
3. **Inference reported unhealthy forever.** Its healthcheck hit `/readyz`, which
   correctly returns 503 with no model mounted. That is not a fault in the process, and
   pointing a liveness probe at readiness conflates the two probes deliberately kept
   apart.
4. **Every metric went nowhere.** `ingest()` resolved `metrics or MetricsRegistry()`
   *inside* the handler, so each request wrote to a fresh throwaway registry. The
   service answered 200, the response said `accepted: 1`, and `/metrics` served only
   the liveness series. A counter written to a registry nobody scrapes is
   indistinguishable from a counter that never fired.

Four bugs, zero of which any existing test could see. They were all only visible by
running the thing, which is the argument for running it.

`prognos stream` now also writes `artifacts/model` — the model and its calibrator — and
inference mounts it. Without it the service stays unready, which is the right
behaviour: no calibrator means no interval, and an interval-free prediction service is
worse than one that refuses traffic.

## The fleet view, and a cost panel that does not recompute anything

`StreamPipeline.summary()` has carried the docstring "for the dashboard and the status
endpoint" since the day it was written. Nothing read it. That is now closed, and the
route below is populated by the stream itself rather than by a fixture.

```bash
uv run prognos stream --config configs/base.yaml --engines 4 --serve --port 8090
curl -s http://127.0.0.1:8090/fleet | jq '.fleet'
```

```
DS01-005/unit001  cycle 23  rul 71.4  interval [51.5, 91.4]  width 39.9  nominal
```

Three decisions worth stating, because each looks like a detail:

**The roster is what the stream has seen.** A manifest listing 29 engines would show 29
rows on an idle dashboard, which is a lie. Engines appear as they send data.

**Engine identity is `subdataset/unit`, and the unit is the engine's own.** The first
version of this passed the window-source *group index*, which rendered `DS01-005/unit000`
— group 0 is DS01 unit 1. Caught by running it and reading the output, which is the only
way this class of mistake surfaces.

**Rows are overwritten, not appended.** One row per engine, sorted worst-first
(degraded, then alerting, then nominal) so the engine nobody is watching is at the top.
A view with a row per reading is a log file wearing a table's clothes.

The same view is published as `prognos_fleet_rul_cycles`, `prognos_fleet_interval_width`
and `prognos_fleet_engines{status}`, which is how Grafana sees it — Prometheus cannot
read a JSON route.

### The cost panel reads the committed result

`GET /cost` serves `results/phase4_policy.json` rather than recomputing anything:

```
threshold_point_10   1.06/engine   28 planned,  0 failures
threshold_lower_10   1.07/engine   42 planned,  0 failures
threshold_point_30   1.07/engine   42 planned,  0 failures
run_to_failure      24.05/engine   0 planned, 23 failures   (ratio 25, horizon 2000)
```

The cost model travels with the numbers. A cost figure without the ratio it was measured
under is not arguable, and the sweep that shows planning only pays above a ratio is part
of the panel rather than something a reader has to already know.

### Authentication and rate limiting

Both were gaps I flagged when the compose stack first came up, and both were real: the
only abuse control was a request size ceiling.

A bearer token from the environment, compared in constant time, and a sliding-window rate
limit per client keyed on the socket peer. `X-Forwarded-For` is ignored unless
explicitly trusted, because trusting it from an arbitrary caller lets anyone mint a fresh
allowance per request. Refusal bodies carry no detail — a 401 that names the header it
wants is a specification for whoever is guessing. `/healthz` is never authenticated,
because an orchestrator's probe has no credential and a 401 there restarts the process
forever over a missing secret.

`/healthz` is exempt so orchestrator probes keep working; `/metrics` deliberately is
**not**, which means Prometheus scrapes with `bearer_token_file` and
`ops/prometheus-token.txt` must exist once a token is set. Exempting the scrape would be
tidier and is what I tried first — all four prognos targets went DOWN with 401, which
reads as an outage rather than a configuration mistake. So the token file is a required
step and the runbook has the command.

Off when the variable is unset, so a local run needs no credential. That is only safe
because every published port stays on `127.0.0.1`, and `docs/runbook.md` says what to do
before that changes. A token is a shared secret: it tells you the caller was holding the
right string, not who they were. It is not TLS and it is not per-user identity.

## Fault injection, and a detector I could not make trustworthy

`docs/design.md` §12 sets the Phase 5 exit criterion: *every injected fault produces a
named degraded state and an alert, with results committed.* The Phase 5 PR did not meet
it — there was no fault detection at all, and I marked the phase complete anyway. This
is the missing work.

```bash
uv run prognos stream --config configs/base.yaml --faults --out results/phase5_faults.json
```

Five faults injected into a clean stream, one real engine, sampled one row per cycle:

| injected | everything detected | expected fault named |
| --- | --- | --- |
| none (control) | `sensor_drift` | **failed**: nothing should fire |
| `sensor_dropout` ch 7 | `sensor_drift`, `sensor_dropout` | yes |
| `sensor_stuck` ch 3 | `sensor_drift`, `sensor_stuck` | yes |
| `sensor_drift` ch 11 | `sensor_drift` | yes |
| `value_out_of_range` ch 2 | `sensor_drift`, `sensor_stuck`, `value_out_of_range` | yes |

`sensor_drift` appears in every row, including the control, because it fires on healthy
channels of this engine. Each expected fault is named; none is named alone.

**The control run fails, and that is the finding.** On a *healthy* engine, three
channels — `Nf`, `Nc`, `T48` — move about 20 units between adjacent windows as ordinary
operation. `Nf` and `Nc` are fan and core speed; they genuinely move.

I tried three normalisations and each fixed one false-positive mode while creating
another:

- **Against the fleet's spread.** Flags healthy channels, because these instruments
  differ by three orders of magnitude — core speed sits near 8,700, a temperature near
  500 — so a fleet threshold is really a threshold on whichever channels are small.
- **Against the channel's own spread.** Flags healthy channels, because at one row per
  cycle the series is so smooth that ordinary movement is many times its own standard
  deviation.
- **Both together.** Still flags them.

So `sensor_drift` is shipped as implemented and reported as failing, with the reason
recorded in `results/phase5_faults.json` under `known_limitations`. A test asserts the
control run **fails**, so the artifact cannot quietly go green. Tuning until it passed
would have produced a number that means nothing, and that is the one thing this project
exists not to do.

Two other detectors were wrong in ways only real data exposed:

- **`cycle_regression` fired on every healthy engine.** It treated a repeated cycle
  number as a backwards clock. This dataset carries thousands of rows per cycle number,
  so repetition is the normal case.
- **Dropout could never fire.** It counted cycles, so on data where a cycle holds
  thousands of rows the threshold was never reached.

### Named states, not counts

A degraded state is a name plus evidence, because "rejects: 400" is not something an
operator can act on:

```
sensor_dropout on W_Fuel Flow, engine 3, silent 1284 rows
sensor_drift    on Nf, engine 3: moved 18.8 between adjacent windows, 4.1x its own spread
```

Exposed through the dashboard (`GET /degraded`), the scrape
(`prognos_degraded_states_by_kind`), and injectable over HTTP (`POST /inject`), because
the only way to check a degraded state is observable end to end is to provoke one.

## Documentation

- [Runbook](docs/runbook.md) — triage by verdict, ingestion failures, calibration
- [Deploying to Oracle Always Free A1](docs/deploy-oci.md) — the demo host
- [Design document](docs/design.md)
- [Architecture decision records](docs/adr/)
- [Data setup](docs/data.md)