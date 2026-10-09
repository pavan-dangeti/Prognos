# ADR-0005: Act on the lower bound of the interval

Date: 2026-10-09
Status: Accepted

## Context

A calibrated 90% interval is the output of the uncertainty layer. Something has to
turn it into an action, and the choice of which part of the interval to threshold
is a risk decision, not a modelling decision.

## Decision

The maintenance policy replaces an engine when `rul_lower <= threshold`, where
`rul_lower` is the lower bound of the prediction interval. The threshold is
swept, and the resulting cost curve is reported rather than a single chosen point.
A point-estimate policy is implemented alongside it as the control case.

## Rationale

The two error directions have very different costs. Replacing an engine with life
remaining costs the planned maintenance visit and returns a partly used engine.
Running an engine past its life costs the failure, which is larger and has a tail
that a fixed number does not bound well. Thresholding the lower bound is the
conservative choice: it acts when the evidence that the engine is nearly done is
strong, not merely when the estimate says so.

It also has a property that makes it defensible without a cost model: at 90%
coverage, the lower bound is below the truth about 95% of the time. So the policy
fires late only rarely, while firing early is cheap.

Reporting the whole threshold sweep matters because the chosen threshold is where
the cost model enters the result. A single number hides the fact that the optimal
point depends entirely on the cost assumptions, which the sensitivity sweep makes
visible.

## Consequences

- Conservative bias: the policy over-replaces relative to the point estimate.
  Under a cost model where replacement is very expensive, that bias costs money,
  and the sweep will show where it stops being worth it.
- Interval width feeds directly into policy aggressiveness. A model with tight
  but wrong intervals produces a policy that is confidently wrong. This is the
  strongest argument for monitoring coverage live rather than trusting the offline
  number.
- The cost result is only as good as the cost parameters. They are stated with
  every result and swept.

## Alternatives

- Point estimate: rejected as the primary policy, kept as the control. It is the
  comparison that shows the interval is doing work.
- Optimise the threshold directly on the cost objective: deferred. It optimises
  the same thing the sweep finds, with more machinery and less transparency.