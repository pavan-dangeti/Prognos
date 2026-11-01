# Model card: prognos-rul

The model the inference service serves: LightGBM on window statistics plus the cycle
number, paired with a split conformal calibrator that turns each point estimate into a
90% interval. Registered in MLflow as `prognos-rul`; the version being served carries
the `serving` alias and is reported on the inference service's `/readyz`.

Every figure here is taken from a committed result file, named next to it, and
`tests/test_model_card.py` fails CI if one drifts from its file.

## Intended use

- A research demonstration of remaining-useful-life prediction with calibrated
  uncertainty, on NASA's simulated N-CMAPSS turbofan data.
- Showing how an interval, rather than a point, can drive a maintenance decision, and
  how a streaming system can say when its own prediction should not be trusted.
- Teaching and interview discussion of the trade-offs recorded in `docs/design.md`.

## Out of scope

- **Any decision about a real engine or aircraft.** No airworthiness or safety claim is
  made or implied.
- **Real fleet data.** Every engine is simulated. Nothing here has been checked against
  a real engine, and no figure transfers to one without being re-measured.
- **Engines, failure modes or flight regimes outside N-CMAPSS DS01, DS02 and DS06.**
- **A physics-based degradation model.** None is used, and none is claimed unnecessary.

## Model

| | |
| --- | --- |
| Point model | LightGBM on per-window channel statistics plus the cycle number |
| Inputs | 4 flight conditions (alt, Mach, TRA, T2) and 14 measured sensors, in windows of 50 readings taken every 10th reading within one cycle. The simulator's health parameters are never read (ADR-0002, `tests/test_leakage.py`). |
| Output | Remaining useful life in cycles, with a 90% interval |
| Interval | Split conformal, calibrated on engines held out of the fit |
| Why not the CNN | The CNN is more accurate (below), but torch costs about 800 MB per process on a 12 GB free host. The trade-off is recorded in the README. |

## Data

NASA N-CMAPSS sub-datasets DS01, DS02 and DS06: 29 engines, 199,095 windows. Engine
identity is the sub-dataset plus the unit number, because unit numbers restart in each
sub-dataset. A final test group of 10 engines was held out of every figure here. Three
of its DS02 engines (units 5, 11 and 20) were opened once, with the code frozen, for the
published comparison in `results/final_ds02.json`; the other seven have not been scored.
Access and terms are in `docs/data.md`.

## Metrics

5-fold cross-validation grouped by engine over the 19 non-test engines, 138,076 scored
windows. Intervals are 95% bootstrap intervals resampled by engine, not by window.
Source: `results/phase3_sequence.json`, produced by `prognos evaluate --sequence`.

| model | RMSE, cycles | 95% CI | NASA score |
| --- | --- | --- | --- |
| `lightgbm` (served) | 8.15 | 4.23–12.07 | 30,027 |
| `cnn` | 7.40 | 4.61–10.20 | 24,343 |
| `linear_cycle`, no sensors | 8.90 | 4.62–13.18 | 34,950 |

The served model's error by life stage: early 9.04, mid 8.71, late 6.37 cycles. It is
most accurate late in life, which is where maintenance decisions are made.

The confidence intervals of all three overlap. With 19 engines, the served model is not
shown to beat the cycle-only baseline on RMSE alone.

## Interval coverage

Target 90%. Source: `results/phase4_policy.json` (`prognos policy`) and
`results/phase5_streaming.json` (`prognos stream`).

| | coverage | mean width, cycles |
| --- | --- | --- |
| Offline, nested cross-validation | 0.926 | 38.0 |
| early life | 0.908 | 41.2 |
| mid life | 0.908 | 41.1 |
| late life | 0.961 | 31.8 |
| Live replay, 100,000 predictions | 0.991 | half-width 17.5 |

Coverage is marginal: it holds on average over exchangeable data, not per engine, per
life stage, or under a change in operating conditions. Live coverage over-covers, which
is safe but means the interval is wider than it needs to be.

## Known failure modes

- **It errs late more often than early.** Averaged over the folds, 57.8% of its
  predictions overestimate the remaining life, which is the dangerous direction: an
  engine judged to have more life than it has. The NASA score penalises this and is
  reported for that reason.
- **Wide intervals.** A mean width of 38 cycles on a scale capped at 125 is too wide to
  schedule maintenance against precisely.
- **Outside the fitted flight envelope it still predicts.** The pipeline names
  `operating_shift` and marks the engine low confidence; the number itself should not be
  used.
- **Sensor faults.** A dropped channel is held at its last good value while
  `sensor_dropout` is raised; the prediction during it is degraded. Drift on `P15`,
  `P2`, `P21`, `P24`, `P50` and `Wf` is not detected.
- **Engine age carries much of the signal, and misleads on an unusual engine.** The
  cycle number is an input, and engines in a sub-dataset live similar spans. On DS02
  units 14 and 15, LightGBM scores 5.38 RMSE and a model with no sensors 5.59
  (`results/phase6_published_comparison.json`). On the full published DS02 split, with
  unit 11 whose life differs from the training engines', LightGBM scores 10.89 with the
  cycle and 10.44 without it (`results/final_ds02.json`): the age input hurts.

## Non-goals

No physics-based degradation model, no real aircraft data, and no certification-grade
safety claim. See the README's Limitations for the full list.

## Reproduce

```bash
uv run prognos evaluate --config configs/base.yaml --out results/phase3_sequence.json --sequence --epochs 30
uv run prognos policy --config configs/base.yaml
uv run prognos stream --config configs/base.yaml    # trains, registers prognos-rul, sets @serving
uv run prognos export-model --ref models:/prognos-rul@serving --out artifacts/model
```
