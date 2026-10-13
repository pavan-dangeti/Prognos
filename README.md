# Prognos

Remaining useful life prediction for turbofan engines: RUL with calibrated
prediction intervals, a cost-based maintenance policy, and a streaming pipeline
that replays held-out engines as a live fleet.

Built on NASA N-CMAPSS. Research demonstration — see
[Limitations](docs/design.md#9-limitations) before drawing any operational
conclusion from it.

Status: Phase 4 (intervals and cost-based maintenance policy).

## Results

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

## Documentation

- [Design document](docs/design.md)
- [Architecture decision records](docs/adr/)
- [Data setup](docs/data.md)