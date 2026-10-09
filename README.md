# Prognos

Remaining useful life prediction for turbofan engines: RUL with calibrated
prediction intervals, a cost-based maintenance policy, and a streaming pipeline
that replays held-out engines as a live fleet.

Built on NASA N-CMAPSS. Research demonstration — see
[Limitations](docs/design.md#9-limitations) before drawing any operational
conclusion from it.

Status: Phase 2 (baselines and evaluation harness).

## Results

5-fold grouped cross-validation, 29 engines from DS01, DS02 and DS06, 199,095
windows. Intervals are 95% bootstrap resampling **by engine**, not by window. The
final test group of 10 engines was held closed throughout.

| model | rmse | nasa_score | rmse 95% CI | how measured |
| --- | --- | --- | --- | --- |
| `lightgbm` | 8.15 | 30,027 | 4.23–12.07 | grouped CV, window statistics + cycle context |
| `linear_cycle` | 8.90 | 34,950 | 4.62–13.18 | grouped CV, RUL ~ cycle number, no sensors |
| `mean` | 23.76 | 283,059 | 22.08–25.44 | grouped CV, training-set mean RUL |
| `trend` | 25.87 | 557,375 | 25.09–26.65 | grouped CV, per-channel trend to threshold |

Reproduce from a converted dataset:

```bash
uv run prognos convert --config configs/base.yaml
uv run prognos evaluate --config configs/base.yaml --out results/phase2_baselines.json
```

Raw output: [`results/phase2_baselines.json`](results/phase2_baselines.json).
`tests/test_results.py` fails CI if the figures above drift from that file.

`nasa_score` is the asymmetric scoring function from Saxena et al. (2008); lower is
better and late predictions cost more than early ones. It is a sum over
predictions, so it is not comparable across datasets of different size.

### What the numbers say

`lightgbm` at 8.15 RMSE is the bar a sequence model has to beat, and the interval
is wide: 4.23 to 12.07 across folds. At 29 engines, that interval is the honest
state of the evidence, not a formality.

`linear_cycle` at 8.90 uses **no sensor data at all** — just cycle number. It lands
within 0.75 cycles of the gradient boosting model, which says most of the
predictable signal in this window is how old the engine is, not how its sensors
have moved. That is the most useful thing Phase 2 produced, and it sets the bar
that matters: a 1D CNN or LSTM has to beat 8.90, not beat 23.76.

`trend` at 25.87 is worse than the mean floor. The degradation-trend approach fails
here for a specific reason: with a 50-sample window at downsample 10, a channel's
slope is estimated from 500 seconds of a single flight, and extrapolating that to a
threshold crossing is dominated by noise. Its life-stage breakdown shows where —
RMSE 19.5 early, 37.7 late. The method is sound; the window is too short to support
it. Phase 3's longer windows are the natural test of that.

## Documentation

- [Design document](docs/design.md)
- [Architecture decision records](docs/adr/)
- [Data setup](docs/data.md)