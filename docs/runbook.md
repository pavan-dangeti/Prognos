# Runbook

What to do when something is wrong, in the order the failures actually happen. Every
command here is one you can run from a clean checkout on a laptop; nothing assumes
cloud access.

## The one check that matters most

```bash
uv run prognos stream --config configs/base.yaml --out /tmp/live.json
```

It prints a coverage verdict. `coverage: 0.90` against a 0.90 target is healthy. A
number below **0.85** is a breach: the interval is no longer covering, and every
maintenance decision taken from it is resting on a range that does not hold.

This is the S5 SLO from `docs/design.md` §4.4, and it is the one metric worth alerting
on. Coverage falling below target means the model is confidently wrong in a way RMSE
does not show: the point estimate can drift without moving RMSE much, but the interval
stops covering immediately.

## Triage by verdict

### `coverage: 0.71` or lower — breach

The interval is under-covering. Real causes, in the order they have actually occurred
in this project:

1. **The model is serving data it was never fitted on.** This happened: the replay
   driver used the global engine index as a Parquet filename, so it fed the model rows
   from an engine it had never seen. Coverage read 0.716 and every other number in the
   result file looked entirely reasonable. Check that `calibration.fit_engines` and the
   replayed engines are drawn from the same, correct set.
2. **The calibration set is too narrow.** One engine is not a breadth. Check
   `calibration.n_engines` in the result file; below about five the interval will not
   generalise across a fleet.
3. **Real drift.** Sensors have changed. This is the case the SLO exists for and there
   is no shortcut: re-run the calibration against recent data and re-freeze.

### `coverage: 0.99` — over-covering

Not a breach, and deliberately not alarmed on: a wide interval is wasteful, not
dangerous. It does mean the interval is too wide to be useful for scheduling, because it
spans a large part of the operating range. Check `calibration.half_width` against
`config.window.rul_cap`; if the half-width is a large fraction of the cap, the interval
carries little information and a better point model is the fix, not a tighter interval.

### `insufficient_data`

Fewer than 50 scored predictions. Not a fault. The monitor refuses to judge below that
floor because three predictions give a coin flip and declaring a breach would page
someone about noise.

## Ingestion

```bash
curl -s localhost:8081/metrics | grep prognos_stream
```

`prognos_stream_rejected_by_reason_total{reason="..."}` breaks rejections down by
cause. A total is not a diagnosis.

| reason | meaning | action |
| --- | --- | --- |
| `non_finite_value` | NaN or infinity in a reading | Check the sensor. Do not impute silently. |
| `channel_out_of_range` | A channel index outside the instrument list | A producer is sending a channel this build does not know. It is rejected, not written as zero. |
| `batch_too_large` | A batch over `MAX_BATCH_ROWS` | A producer is batching wrongly. The whole batch is rejected rather than truncated, because a truncated batch is indistinguishable from a complete one. |
| `negative_cycle` | A cycle number below zero | Producer clock or counter bug. |

A rising accepted count with a flat rejected count is the healthy shape. Flat accepted
with zero rejected means the stream stopped, not that the pipeline is idle.

## Containers

```bash
docker compose ps
docker compose logs -f ingestion
docker compose restart inference
docker compose down          # stops; results in ./results survive
```

Every service exposes `/healthz`, `/readyz` and `/metrics`.

- `/healthz` answers while the process is up. A liveness failure restarts the process.
- `/readyz` answers only when the service can serve. A readiness failure stops traffic
  arriving and does **not** restart anything.

Do not point a liveness probe at a dependency. Restarting a process because a database
is down does not fix the database and does lose in-flight work.

## Before you expose any port

Every published port is bound to `127.0.0.1`, which is the only reason an unset token is
not yet a problem. It stops being the only reason the moment a port is published, a
reverse proxy is added, or the host is reachable from somewhere else.

Set a token before any of that happens:

```bash
echo "PROGNOS_API_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" >> .env
docker compose up -d
curl -H "Authorization: Bearer $PROGNOS_API_TOKEN" http://127.0.0.1:8084/fleet
```

`PROGNOS_API_TOKEN` is read from the environment only. It is never a build argument, a
file in the image, or a constant in the source. An empty value is treated as unset, not
as an empty token, because an env var nobody set is the usual way this goes wrong.

What that buys, and what it does not:

- **Bearer token, compared in constant time.** `==` on a secret leaks through timing
  whether a prefix matched.
- **Sliding-window rate limit, per client, keyed on the socket peer.** Fixed windows are
  trivially evaded at the boundary. `X-Forwarded-For` is ignored unless
  `trust_forwarded_for` is set, because trusting it from an arbitrary caller lets anyone
  mint a fresh allowance per request.
- **The 401 body says nothing.** A refusal that names the header it expects is a
  specification for whoever is guessing.
- **`/healthz` is never authenticated.** An orchestrator's probe has no credential, and
  a probe returning 401 restarts the process forever over a missing secret.

It does **not** add TLS, per-user identity, or audit of who called what. A token is a
shared secret: it tells you the caller was holding the right string, not who they were.
Put TLS in front of it, or keep it on loopback.

### Triage

| Symptom | Cause | Fix |
| --- | --- | --- |
| `401` everywhere, `/healthz` fine | `PROGNOS_API_TOKEN` set, caller is not sending it | Add the `Authorization` header |
| `401` everywhere and the token is unset | Trailing space or quote in `.env` | `PROGNOS_API_TOKEN` must be a bare value |
| `429` under normal load | One client, 120 requests in 60s | Raise `--rate-limit`, or find the caller |
| `429` from a proxy you control | Everything arrives from one peer address | Set `trust_forwarded_for` |

## Reproducing a number

Every figure in the README comes from a committed script. If one does not match:

```bash
uv run pytest tests/test_results.py -v   # names the drift
uv run prognos evaluate --config configs/base.yaml --out results/phase3_sequence.json --sequence --epochs 30
```

`tests/test_results.py` fails CI when a README figure drifts from a committed result
file, so this should never get far.

## Calibration set is too small

```bash
uv run prognos stream --config configs/base.yaml --calibration-fraction 0.3 --out /tmp/wide.json
```

Raising the fraction widens the interval and raises coverage. There is a cost — the
interval gets less useful for scheduling — and that trade is the point of the sweep.
Measured on this dataset: 0.05 → 0.986, 0.15 → 0.995, 0.30 → 0.991.

## When the model is genuinely wrong

There is no shortcut and the runbook should not pretend otherwise.

1. Open the final test group. It has been closed since Phase 1 and opening it is a
   one-way decision — it is no longer a test set afterwards.
2. Re-run the whole evaluation, not just the failing slice.
3. Re-freeze and re-record. The result file's `versions` and `generated_at` fields must
   change to match what actually produced the numbers.

Do not tune on the final test group. Every number this project reports came from
folds the model never saw, and the first version of the Phase 5 replay driver broke
that by accident.

## Escalation

Nothing here pages anyone, deliberately: this is a portfolio project with no on-call
rotation and a single contributor. Every alert condition is a thing you look at when
you look, not a thing that interrupts you.