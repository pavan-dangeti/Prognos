# ADR-0007: Tune detectors on development engines, report on held-out ones

Date: 2026-10-31
Status: Accepted

## Context

The drift detector fired on healthy engines, and the operating-shift check later
proved to as well. Both have settings (thresholds, windows, persistence, release) that
could be adjusted until a chosen healthy engine came out clean. Doing that on the
engines the result is then reported on produces a false-alarm rate of zero that means
nothing: it measures how well the settings were fitted to those engines, not how the
detector behaves on an engine it has not seen.

## Decision

Every detector setting is chosen by a fixed protocol:

1. The 19 cross-validation engines are split by a seeded draw within each sub-dataset
   into 10 development and 9 held-out engines. The final test group is not read.
2. A grid of settings is evaluated on development engines only, for false alarms on
   the clean engines and detection of faults injected into them.
3. The setting is chosen by a rule written into the code before the run
   (`selection_key` in `drift_study.py` and `envelope_study.py`): fewest engines with
   any false alarm first, then detection, then the most sensitive setting.
4. Before and after are reported on the held-out engines, which played no part in the
   choice, and the chosen values become the defaults, with a test tying the two
   together.

## Rationale

The selection rule is fixed before the results exist, so it cannot be bent towards a
result. The held-out half gives an honest estimate of the false-alarm rate on unseen
engines. Splitting within each sub-dataset keeps DS06's different failure modes on
both sides. The same split serves every detector, so their results are comparable and
no engine is development for one and held-out for another.

## Consequences

- With 9 held-out engines, a rate of 0/9 is weak evidence of a low rate, not proof of
  zero. The README quotes it as a count.
- A control run on a development engine is no longer evidence. The fault and failure
  scenarios run on DS01-005/unit001, a development engine, and their artifacts say so.
- Anything the grid did not search is reported, not tuned around afterwards: drift on
  six low-magnitude sensors is never detected, and a shift on strictly alternating
  flights is never named.
