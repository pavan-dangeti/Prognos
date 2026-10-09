# ADR-0003: Split by engine unit, and open the test group once

Date: 2026-10-09
Status: Accepted

## Context

Sliding windows overlap heavily. Two windows from the same engine unit share most
of their rows. A random split over windows therefore places near-duplicates on
both sides of the boundary, and reported error collapses to near zero for reasons
that have nothing to do with the model.

N-CMAPSS has 9 to 10 units per sub-dataset, so unit-level splitting is coarse.

## Decision

Split on engine unit ID. `GroupKFold` for cross-validation, with one group of
units held out entirely as the final test set. Hyperparameters are selected on the
validation folds only. The final test group is opened once, after the model and
its configuration are frozen, and never used to select anything.

Confidence intervals on every reported comparison come from resampling at the
unit level, not the window level, because the unit is the independent unit of
observation.

## Rationale

Engine unit is the correct resampling unit. Two windows from one engine are not
independent observations of how a model generalises to a new engine, which is the
question that matters. With 9 groups the resulting intervals are wide, and that
width is the honest state of the benchmark. Reporting a narrow interval computed
over windows would be reporting a number that means something else.

## Consequences

- Confidence intervals on model comparisons are wide. Many plausible models will
  be statistically indistinguishable on this evidence, and we will say so
  instead of ranking them on a decimal place.
- Nine-fold CV on 9 units leaves single-unit folds, so results are sensitive to
  fold assignment. Grouped CV with repeated seeds and the spread across seeds is
  reported alongside the mean.
- Training data is small. Sequence models will overfit unless regularisation and
  early stopping are handled carefully.

## Alternatives

- Random window split: rejected, leaks through overlap.
- Leave-one-unit-out with a fixed fold order: rejected for reporting, kept only as
  a sensitivity check.