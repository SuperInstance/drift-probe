# RESULTS — drift-probe on real history (132 commits, 2026-10-17)

Detector: dual-mode (EWMA-z winsorized + Holt forecast error, |z| > 2.5),
36h episode grouping, health-weighted daily timeline, recency-weighted
current score (2-day halflife). Extraction: 0.7s for all 132 commits.

## Top drift episodes

**1. 2026-10-16 03:24 → 10-17 04:37 UTC — day score 155.1 (largest in
history).** Attribution: "test count rising (0→44, z=+10) while days
since real work falling (6.9→0, z=−10) while MEMORY.md size falling."
This is exactly the v0.3/v0.4/Sprint-0 maintenance push: 87810e6/819fc2f
landed in that window; tests/test_agent.py first appears in v0.3 (37
tests) growing to 52 by Sprint 0; the repo had gone 6.9 days with zero
non-bookkeeping commits before it. Dates, directions, and values all
verify against the git log. **REAL.**

**2. 2026-10-15 → 10-16.** MEMORY.md churn collapse: the echo-loop era
grew it to ~4.4 kB; Sprint 0 D6 purged it to 886 B. **REAL.**

**3. 2026-10-06 → 10-08** (build-era → heartbeat transition). Right
regime, trivial magnitude — borderline noise.

## Honest assessment: real or noise?

The Oct-16 detection is real and precisely attributed — but the headline
weakness is the inverse of the lead-time promise: the **idle era itself
(10-09 → 10-15) scores near zero**. Holt absorbs the steady climb of
days_since_substantive, so the detector flags *regime changes* loudly
and *steady slides* weakly. Early warnings of the idle era exist but are
faint (authors 3→1 on 10-12 at z=4; cadence change 10-06 at z=2.6).

It also has **no valence**: healthy changes (adding 52 tests) flag as
hard as degradations — judging which is which is the future LLM gate's
job, not a z-score's.

Current score: **37.9 recent / 1.4 raw-today** (post-push calm).

**Verdict: worth keeping as a "something changed, look" trigger** —
which is all the scout report asked of level 1. Steady-state rot
detection needs a different instrument: the duty-cycle probe reads
`days_since_substantive` directly and does not forget.

## Test suite

15 tests green: hand-computed EWMA math, z-capping, Holt ramp-tracking,
stable-series silence, injected drift (up/down, with persistence
assertion), step change flagged by both modes, episode gap-splitting,
aggregate scoring, recency decay, TASKS checkbox parsing, and full
extraction over a synthetic 3-commit repo with null-metric handling
(zero-poc has no dependency manifest; `dep_age_days` is null throughout
by design).
