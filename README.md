# drift-probe — is this repo drifting from its healthy manifold?

The world-model lane's honest first step (per
`vision/research/world-models.md`): NOT a neural world model — a
classical trajectory scorer over git history. System-1 judges score
snapshots; this scores *trajectories*: ~15 metrics extracted per commit,
forecast-error and z-score anomaly detection over the resulting series,
one drift score with attribution.

## What it does

- `metrics.py --repo <path> > series.jsonl` — per-commit snapshots:
  commits_7d, test_count, tests_touched_7d, doc_files, docs_bytes,
  memory_bytes, tasks_open, authors_7d, dep_age_days, ci_age_days,
  scratch_files, msg_len_avg10, days_since_substantive, repo_files,
  py_loc. Missing files → nulls (handled), not fake zeros.
- `drift.py --repo <path> [--report report.json] [--markdown]
  [--print-score]` — EWMA-z (winsorized variance) + Holt linear forecast
  error, |z| > 2.5 both modes; 36h episode grouping; health-weighted
  daily timeline; recency-weighted current score; value attribution
  ("test count rising (0→44, z=+10) while days since real work falling").
- `nightly.sh` — the future integration: fetch, extract, detect, and on
  threshold append one row to a DRIVE.md-style ledger
  (`ts | expected | observed | drift_score`). This is where the
  world-model lane meets the probe lane.

## Results on real history (see RESULTS.md)

Top episode = the v0.3/v0.4 push, precisely attributed and verified
against the git log. Honest limits, measured: steady slides (the idle
era) score near zero — Holt absorbs them; this detector flags regime
*changes*, not rot (the duty-cycle probe is the rot instrument). No
valence: healthy and degrading changes flag identically — judging which
is the future LLM gate's job. Use as a "something changed, look"
trigger. 15 tests green.

## Kill criterion (inherited from the scout)

The lane dies if the nightly proposer experiment reaches 10 PRs opened,
0 merged. Written down before it starts.

## Next

GitHub-API metrics (CI duration, issue backlog shape — currently
invisible to git), then the appetite-gated nightly proposer: drift
anomaly arms qwen2.5-coder:7b to draft exactly one maintenance PR with
evidence in the body.
