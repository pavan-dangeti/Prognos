# ADR-0006: Share one feature builder between offline and online paths

Date: 2026-10-09
Status: Accepted

## Context

Phase 2 trains models on features computed from Parquet. Phase 5 serves those
models on features computed from a live streaming buffer. If those are two
implementations of the same arithmetic, they will agree on the day they are
written and diverge over the following weeks as each is fixed in response to its
own bugs. The failure is silent: predictions get slightly worse, with nothing in
the logs to say why, and the usual debugging path is to suspect the model.

## Decision

One feature builder. It takes a window of readings and returns a feature vector.
Offline it is fed windows from Parquet; online it is fed the streaming buffer. A
test asserts both paths produce identical output for identical input.

The streaming window buffer must be able to materialise exactly the window shape
the offline path produces, including handling partial windows and late arrivals,
or the shared function cannot be shared.

## Rationale

The whole point of this project is that the online system answers the same
question the offline evaluation answered. If the feature arithmetic differs, the
offline metrics describe a system that does not exist. This is the most expensive
class of silent bug available here and the cheapest to prevent.

It also removes a whole class of "works offline, broken online" debugging, which is
where most of the time in a streaming project goes.

## Consequences

- The feature builder has to be free of side effects and free of Parquet
  dependencies, so it can be called from the inference container. It takes plain
  arrays.
- Window construction logic becomes a shared, tested unit: partial windows,
  windows that span a late arrival, windows that span a rejected reading. That is
  more logic than the naive version, and it is the logic that was going to be
  written badly twice.
- The identity test is the contract. When it fails, something changed on one side
  only, and that is the whole diagnosis.

## Alternatives

- Two implementations, cross-checked in an integration test: rejected. The test
  catches a mismatch once; it does not stop either side drifting afterwards.
- Share only the column naming, not the arithmetic: rejected, this is the part
  that drifts.