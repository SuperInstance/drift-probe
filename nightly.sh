#!/usr/bin/env bash
# drift-probe nightly — the future integration sketch.
# Fetch, extract, detect; on threshold, append a DRIVE.md-style row.
# The ledger row is where the world-model lane meets the probe lane.
set -euo pipefail
cd "$(dirname "$0")"

REPO="${1:-/home/proart/projects/zero-poc}"
THRESHOLD="${DRIFT_THRESHOLD:-50}"

git -C "$REPO" fetch -q origin 2>/dev/null || true
python3 metrics.py --repo "$REPO" > series.jsonl
python3 drift.py --series series.jsonl --report report.json

SCORE=$(python3 drift.py --series series.jsonl --print-score | tail -1)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ATTR=$(python3 - <<'EOF'
import json
try:
    r = json.load(open("report.json"))
    eps = r.get("episodes") or []
    print((eps[-1].get("attribution", "stable") if eps else "stable")[:120])
except Exception:
    print("unknown")
EOF
)

echo "$TS | drift_score=$SCORE | observed: $ATTR" >> drift-log.md
if awk -v s="$SCORE" 'BEGIN{exit !(s > THRESHOLD)}'; then
    echo "$TS | probe | expected: metrics within baseline | observed: $ATTR | action:proposed | drift_score $SCORE > $THRESHOLD" >> DRIVE.md
    echo "drift-probe: score $SCORE exceeds $THRESHOLD — DRIVE.md row appended"
else
    echo "drift-probe: score $SCORE within threshold"
fi
