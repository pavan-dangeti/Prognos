# ADR-0003: Split by engine, and open the test group once

Date: 2026-10-09
Status: Accepted (revised 2026-10-09 after grouping leaked across sub-datasets)

## Context

Sliding windows overlap heavily. Two windows from the same engine share most of
their rows. A random split over windows therefore places near-duplicates on both
sides of the boundary, and reported error collapses to near zero for reasons that
have nothing to do with the model.

Engine-level splitting is therefore necessary. It is also not sufficient, on its
own: N-CMAPSS restarts unit numbers at 1 in every sub-dataset, so DS01 unit 2 and
DS02 unit 2 are different engines. Grouping on the bare unit number merges them and
puts two different engines in one fold. This project used unit numbers alone until
Phase 2, where it reported 16 engines where there are 29, and every reported number
was affected.

## Decision

An engine is the pair (sub-dataset, unit), carried as `GroupKey`. `GroupKFold` on
those keys for cross-validation, with one group held out entirely as the final test
set. Hyperparameters are selected on the validation folds only. The final test group
is opened once, after the model and its configuration are frozen, and never used to
select anything.

Confidence intervals on every reported comparison come from resampling at the engine
level, not the window level, because the engine is the independent unit of
observation.

## Rationale

The engine is the correct resampling unit. Two windows from one engine are not
independent observations of how a model generalises to a new engine, which is the
question that matters. And it is the correct identity: two engines that share a unit
number are different engines, and pooling them makes the reported error describe
neither one.

With 29 engines across three sub-datasets the resulting intervals are wide, and that
width is the honest state of the benchmark. Reporting a narrow interval computed
over windows would be reporting a number that means something else.

## Consequences

- Confidence intervals on model comparisons are wide. Many plausible models will
  be statistically indistinguishable on this evidence, and we will say so
  instead of ranking them on a decimal place.
- Every grouping key in the codebase is a `GroupKey`. Anything that takes a bare
  integer "unit" is a bug, because it means the wrong engine.
- Life-stage fractions are normalised by each engine's own length, not the
  dataset's longest. A 63-cycle engine and a 100-cycle engine are at different
  points in life at the same cycle number.
- Training data is small. Sequence models will overfit unless regularisation and
  early stopping are handled carefully.

## Alternatives

- Random window split: rejected, leaks through overlap.
- Group on the bare unit number: rejected, merges engines across sub-datasets. This
  was the original implementation and it produced plausible, wrong numbers.
- Leave-one-engine-out with a fixed fold order: rejected for reporting, kept only as
  a sensitivity check.