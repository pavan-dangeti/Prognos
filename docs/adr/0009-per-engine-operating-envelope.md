# ADR-0009: Judge operating shifts against the fleet's envelope and the engine's own

Date: 2026-11-01
Status: Accepted

## Context

A prediction outside the flight conditions the model was fitted on should be marked
low confidence. The first check compared each reading against the fleet's envelope:
the range of altitude, Mach, throttle and inlet temperature in the serving model's
training windows. Measured, it had three faults:

- It could not see a short-haul engine suddenly flying long-haul altitudes, because
  long-haul engines are in the fleet.
- It fired on healthy engines whose ordinary flights leave the fit engines' range:
  4 of 9 held-out engines.
- A shift opened and closed with each flight that crossed the line, raising up to 14
  alerts for one event.

## Decision

`ShiftTracker` judges each completed flight, on its lowest and highest flight
conditions, against two envelopes: the fleet's, and the engine's own, learned from its
first flights (excluding any the fleet already calls abnormal) and widened by a margin
of its range. A shift opens after a number of flights outside and closes after a number
fully inside, so one event raises one alert. The settings (learn 10 flights, margin
10%, open after 2, release after 5) were chosen by the protocol in ADR-0007.

## Rationale

A per-engine envelope is what makes "new for this engine" visible, and engines in
N-CMAPSS fly stable flight profiles, so an engine's own first flights bound its later
ones well. Judging per flight rather than per reading matches what a shift is, a
property of a flight, and lets the study and the live pipeline run the same tracker.
Open and release counts turn flights into events, which is what an operator acts on.

## Consequences

- On held-out engines: false shift alarms 4/9 to 0/9, within-fleet shifts named 9/9,
  one alert per event (`results/phase6_envelope.json`).
- A shift is named at the end of the flight it occurs on, not mid-flight.
- Two flights outside in a row are needed to open, so a shift on strictly alternating
  flights is never named. A test pins this.
- An engine's own envelope needs its first 10 flights; until then only the fleet's
  applies.
