# ADR-0001: Use N-CMAPSS (data set 17) rather than C-MAPSS

Date: 2026-10-09
Status: Accepted

## Context

Two NASA benchmark datasets exist for turbofan RUL estimation. Data set 6
(C-MAPSS, 2008) records one snapshot per cycle from a single synthetic operating
profile, four sub-datasets over two fault modes. Data set 17 (N-CMAPSS, 2021)
records complete flight traces under realistic flight profiles, nine readable
sub-datasets covering up to seven failure modes across fan, LPC, HPC, HPT and LPT.

## Decision

Use N-CMAPSS. Restrict to DS02 as primary, DS01 secondary, DS06 tertiary.

## Rationale

N-CMAPSS keeps the full time series within a flight rather than collapsing each
cycle to a single snapshot. That is what makes the streaming phase honest:
operating conditions genuinely vary within a window, so drift monitoring and
regime-shift detection have something real to detect. On C-MAPSS the operating
columns are effectively constant per cycle and drift detection would be
answering a question the data does not pose.

N-CMAPSS is also the current reference benchmark, so published numbers exist for
comparison.

DS02 leads because it is the most-cited sub-dataset in the literature, which is
what makes the published comparison meaningful. DS01 is a clean single-condition
isolation of degradation. DS06 has five failure modes, which is the only
evidence available here that a model learned degradation rather than one
failure signature. The rest add units but not questions, and the full archive is
14.7 GiB against a 16 GB laptop.

## Consequences

- We accept that a published RMSE only compares when window length, RUL capping
  and split protocol all match. Where they do not, the mismatch is stated.
- Claims about failure-mode generality rest on one sub-dataset.
- Three sub-datasets keep the working set small enough to iterate quickly.

## Alternatives

- C-MAPSS (data set 6): rejected, hides operating-regime variation.
- All nine N-CMAPSS sub-datasets: rejected, ~14.7 GiB and largely redundant.
- PHM 2021 challenge subset: rejected, gated distribution with its own rules.