# Prognos

Remaining useful life prediction for turbofan engines: RUL with calibrated
prediction intervals, a cost-based maintenance policy, and a streaming pipeline
that replays held-out engines as a live fleet.

Built on NASA N-CMAPSS. Research demonstration — see
[Limitations](docs/design.md#9-limitations) before drawing any operational
conclusion from it.

Status: Phase 3 (sequence models).

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

## Documentation

- [Design document](docs/design.md)
- [Architecture decision records](docs/adr/)
- [Data setup](docs/data.md)