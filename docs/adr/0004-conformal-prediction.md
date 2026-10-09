# ADR-0004: Conformal prediction for intervals, over probabilistic models

Date: 2026-10-09
Status: Accepted

## Context

A maintenance decision is "replace or do not replace", and the two error
directions are not symmetric: replacing an engine with life left is a bounded,
known cost; running an engine past its life is a larger and unbounded cost. A
point estimate alone does not support that decision, because it hides how much the
model actually knows.

Candidate approaches: split conformal on held-out residuals, conformalised
quantile regression, Bayesian neural networks, deep ensembles.

## Decision

Split conformal as the default interval method, with CQR reported alongside it
for local adaptivity. Target coverage 90%, configurable.

## Rationale

Split conformal gives a finite-sample marginal coverage guarantee under no
distributional assumption, at the cost of one calibration pass over held-out
residuals. It works with any point model, so the interval method does not constrain
the architecture choice. It is also easy to state and defend in an interview,
which matters for a project whose owner has to explain every part live.

CQR is included because error is not constant across the life cycle. Early-life
windows are far more uncertain than late-life ones, and a single additive
quantile offset from split conformal cannot express that. CQR adapts the width
locally at modest extra cost. Reporting both makes the width difference visible
rather than assumed.

Coverage is monitored live, not just offline, because a marginal guarantee is
exact at calibration time and drifts afterwards. A model with 90% offline coverage
and 71% live coverage is degraded, and that is an SLO breach.

## Consequences

- The guarantee is marginal, not conditional. Under distribution shift the
  realised coverage can differ from nominal. Stated in the README, and covered by
  the live coverage SLO.
- Calibration requires a held-out set of units distinct from the ones used for
  point estimation. On 9 units this is expensive and must be accounted for in the
  split, not taken from data already used.
- Intervals are not natively probabilistic model outputs, so they do not compose
  into a full predictive distribution the way a Bayesian model would. Adequate for
  a threshold decision; not adequate if we later need quantiles of the RUL
  distribution itself.

## Alternatives

- Bayesian neural network: rejected, triples inference cost and complicates the
  model registry for no coverage advantage.
- Deep ensemble: rejected for the same reason.
- Heteroscedastic regression only: rejected, uncalibrated under shift, which is
  the regime that matters.