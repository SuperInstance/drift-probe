#!/usr/bin/env python3
"""drift.py — classical drift detector over git metric time series.

Two detection modes per metric, per the scout report (item 2a):
  1. EWMA-z:    one-step residual vs EWMA baseline, normalized by an EWMA of
                squared residuals. Flags surprises against the recent level.
  2. Forecast:  Holt linear (level+trend) one-step forecast error vs an EWMA
                of past errors. A steady trend does NOT trip Holt — only
                breaks in the trend do. EWMA-z catches level shifts Holt
                has absorbed; the two modes disagree usefully.

Flags: |z| > threshold (default 2.5), with direction. Flags are grouped into
episodes, weighted per metric, aggregated into a per-day drift timeline and a
recency-weighted "current" score, with human-readable attribution.

Usage:
    python3 drift.py --repo /path/to/repo --report report.json [--markdown]
    python3 drift.py --series series.jsonl --report report.json
    python3 drift.py --repo ... --print-score     # just the current score

Python 3 stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import metrics as mx

Z_THRESHOLD = 2.5
BURN_IN = 10          # observations of warmup before flagging
ALPHA_LEVEL = 0.30    # EWMA level
BETA_VAR = 0.10       # EWMA residual variance
ALPHA_HOLT = 0.40     # Holt level
BETA_HOLT = 0.15      # Holt trend
FLOOR_REL = 0.05      # sd floor: 5% of |level| (guards constant series)
FLOOR_ABS = 0.25      # absolute sd floor (count metrics sitting at 0)
EPISODE_MAX_GAP_S = 36 * 3600.0
RECENCY_HALFLIFE_D = 2.0
RECENCY_WINDOW_D = 7.0
Z_CAP = 10.0          # beyond 10 sigma we only say "very large"

# Health-weighted importance of each metric for the aggregate score.
WEIGHTS = {
    "test_count": 1.5,
    "days_since_substantive": 1.4,
    "tests_touched_7d": 1.2,
    "dep_age_days": 1.2,
    "memory_bytes": 1.1,
    "ci_age_days": 1.0,
    "commits_7d": 0.9,
    "authors_7d": 0.9,
    "tasks_open": 0.8,
    "docs_bytes": 0.7,
    "doc_files": 0.7,
    "repo_files": 0.6,
    "py_loc": 0.6,
    "msg_len_avg10": 0.5,
    "scratch_files": 0.5,
}

FRIENDLY = {
    "commits_7d": "commit rate",
    "test_count": "test count",
    "tests_touched_7d": "test activity",
    "doc_files": "doc files",
    "docs_bytes": "docs volume",
    "memory_bytes": "MEMORY.md size",
    "tasks_open": "open tasks",
    "authors_7d": "distinct authors",
    "dep_age_days": "dependency age",
    "ci_age_days": "CI config age",
    "scratch_files": "scratch files",
    "msg_len_avg10": "commit-message length",
    "days_since_substantive": "days since real work",
    "repo_files": "repo file count",
    "py_loc": "python LOC",
}


# --------------------------------------------------------------------------
# pure detection math (unit-tested)
# --------------------------------------------------------------------------

def _floor_sd(level: float) -> float:
    return max(FLOOR_REL * abs(level), FLOOR_ABS, 1e-9)


def ewma_zscores(values: list[float], alpha: float = ALPHA_LEVEL,
                 beta: float = BETA_VAR) -> tuple[list[float], list[float]]:
    """Returns (levels, zscores). zscores[t] is the surprise of values[t]
    relative to the level/variance known at t-1 (zscores[0] == 0)."""
    if not values:
        return [], []
    levels = [values[0]]
    sd = _floor_sd(values[0])
    var = sd * sd
    zs = [0.0]
    for t in range(1, len(values)):
        pred = levels[-1]
        r = values[t] - pred
        sd = max(math.sqrt(var), _floor_sd(pred))
        zs.append(math.copysign(min(abs(r) / sd, Z_CAP), r) if r else 0.0)
        levels.append(alpha * values[t] + (1 - alpha) * pred)
        # winsorized variance update: a single surprise must not inflate the
        # scale and mask the persistent shift that follows it
        r_upd = max(-3.0 * sd, min(3.0 * sd, r))
        var = beta * r_upd * r_upd + (1 - beta) * var
    return levels, zs


def holt_zscores(values: list[float], alpha: float = ALPHA_HOLT,
                 beta: float = BETA_HOLT) -> tuple[list[float], list[float]]:
    """Holt linear one-step forecast errors as z-scores.
    Returns (predictions, zscores); a pure linear ramp yields ~0 z."""
    if len(values) < 2:
        return list(values), [0.0] * len(values)
    level, trend = values[0], values[1] - values[0]
    preds, zs = [values[0]], [0.0]
    var = _floor_sd(values[0]) ** 2
    for t in range(1, len(values)):
        pred = level + trend
        e = values[t] - pred
        sd = max(math.sqrt(var), _floor_sd(pred))
        zs.append(math.copysign(min(abs(e) / sd, Z_CAP), e) if e else 0.0)
        new_level = alpha * values[t] + (1 - alpha) * pred
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        e_upd = max(-3.0 * sd, min(3.0 * sd, e))
        var = beta * e_upd * e_upd + (1 - beta) * var
        preds.append(pred)
    return preds, zs


@dataclass
class Flag:
    ts: float
    value: float
    z: float
    direction: str            # "up" | "down"
    modes: set = field(default_factory=set)

    @property
    def excess(self) -> float:
        return abs(self.z) - Z_THRESHOLD


@dataclass
class Episode:
    metric: str
    flags: list  # list[Flag]
    baseline: Optional[float] = None
    during: Optional[float] = None

    @property
    def start(self) -> float:
        return self.flags[0].ts

    @property
    def end(self) -> float:
        return self.flags[-1].ts

    @property
    def peak_z(self) -> float:
        return max(abs(f.z) for f in self.flags)

    @property
    def direction(self) -> str:
        """Narrative direction from values (baseline -> during); z-sign can
        disagree when the EWMA level has already run past the values."""
        if self.baseline is not None and self.during is not None:
            return "up" if self.during > self.baseline else "down"
        return statistics.mode(f.direction for f in self.flags)

    @property
    def weight(self) -> float:
        return WEIGHTS.get(self.metric, 0.5)

    @property
    def magnitude(self) -> float:
        return self.peak_z * self.weight


def detect_metric_flags(points: list[tuple[float, float]],
                        threshold: float = Z_THRESHOLD,
                        burn: int = BURN_IN) -> list[Flag]:
    """points: [(ts, value)] sorted, nulls removed. Combines both modes;
    an observation flagged by either mode yields one Flag (max |z|)."""
    ts = [p[0] for p in points]
    values = [p[1] for p in points]
    if len(values) < burn + 2 or len(set(values)) < 2:
        return []
    _, z_ewma = ewma_zscores(values)
    _, z_holt = holt_zscores(values)
    flags: list[Flag] = []
    for t in range(len(values)):
        if t < burn:
            continue
        cands = [(z, mode) for z, mode in
                 ((z_ewma[t], "ewma_z"), (z_holt[t], "holt"))
                 if abs(z) > threshold]
        if not cands:
            continue
        z, mode = max(cands, key=lambda zm: abs(zm[0]))
        flags.append(Flag(ts=ts[t], value=values[t], z=z,
                          direction="up" if z > 0 else "down",
                          modes={m for _, m in cands}))
    return flags


def group_episodes(flags: list[Flag], metric: str,
                    values_by_ts: dict[float, float],
                    max_gap_s: float = EPISODE_MAX_GAP_S) -> list[Episode]:
    """Consecutive flags of one metric within max_gap_s form one episode."""
    if not flags:
        return []
    eps, cur = [], [flags[0]]
    for f in flags[1:]:
        if f.ts - cur[-1].ts <= max_gap_s:
            cur.append(f)
        else:
            eps.append(cur)
            cur = [f]
    eps.append(cur)
    episodes = [Episode(metric=metric, flags=g) for g in eps]
    for ep in episodes:
        ep.baseline = _window_median(values_by_ts, ep.start,
                                     before_s=7 * 86400, at_least=3)
        ep.during = statistics.median([f.value for f in ep.flags])
    return episodes


def _window_median(values_by_ts: dict[float, float], end_ts: float,
                   before_s: float, at_least: int) -> Optional[float]:
    xs = [v for ts, v in values_by_ts.items() if end_ts - before_s <= ts < end_ts]
    all_ts = sorted(values_by_ts)
    if len(xs) < at_least and all_ts:
        # fall back to the earliest at_least points (young repos)
        idx = min(max(bisect_ts(all_ts, end_ts), at_least), len(all_ts))
        xs = [values_by_ts[all_ts[k]] for k in range(idx)]
    return statistics.median(xs) if xs else None


def bisect_ts(sorted_ts: list[float], x: float) -> int:
    lo, hi = 0, len(sorted_ts)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# aggregation + attribution
# --------------------------------------------------------------------------

def day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def aggregate_timeline(all_flags: list[tuple[str, Flag]]) -> dict[str, dict]:
    """day -> {'score': sum of weighted excess, 'metrics': {name: max|z|}}"""
    tl: dict[str, dict] = {}
    for name, f in all_flags:
        k = day_key(f.ts)
        d = tl.setdefault(k, {"score": 0.0, "metrics": {}})
        d["score"] += max(f.excess, 0.0) * WEIGHTS.get(name, 0.5)
        prev = d["metrics"].get(name, 0.0)
        d["metrics"][name] = max(prev, abs(f.z))
    return tl


def recent_score(timeline: dict[str, dict], now: float,
                 halflife_d: float = RECENCY_HALFLIFE_D,
                 window_d: float = RECENCY_WINDOW_D) -> float:
    """Recency-weighted mean daily drift score over the trailing window.
    Quiet days count as zero, so an old burst decays out of the score."""
    num = den = 0.0
    day0 = datetime.fromtimestamp(now - window_d * 86400, tz=timezone.utc)
    last = datetime.fromtimestamp(now, tz=timezone.utc)
    d = day0
    while d <= last:
        k = d.strftime("%Y-%m-%d")
        t = d.replace(hour=12).timestamp()
        w = 0.5 ** (max((now - t) / 86400.0, 0.0) / halflife_d)
        num += w * timeline.get(k, {}).get("score", 0.0)
        den += w
        d += timedelta(days=1)
    return num / den if den else 0.0


DIRWORD = {"up": "rising", "down": "falling"}


def fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return f"{x:.4g}"


def attribute(ep: Episode, co_eps: list[Episode]) -> str:
    """'test count falling (12->7, z=-3.4) while MEMORY.md size rises (...)'"""
    name = FRIENDLY.get(ep.metric, ep.metric)
    main = (f"{name} {DIRWORD[ep.direction]} "
            f"({fmt_num(ep.baseline)}->{fmt_num(ep.during)}, "
            f"z={ep.peak_z:+.1f})")
    parts = []
    for co in co_eps[:2]:
        cname = FRIENDLY.get(co.metric, co.metric)
        parts.append(f"while {cname} {DIRWORD[co.direction]} "
                     f"({fmt_num(co.baseline)}->{fmt_num(co.during)}, "
                     f"z={co.peak_z:+.1f})")
    return "; ".join([main] + parts)


def build_report(repo: str, rows: list[dict]) -> dict:
    metric_names = [m for m in mx.METRIC_ORDER]
    episodes: list[Episode] = []
    all_flags: list[tuple[str, Flag]] = []
    metrics_summary: dict[str, dict] = {}
    for m in metric_names:
        pts = [(r["ts"], r[m]) for r in rows
               if r.get(m) is not None and isinstance(r[m], (int, float))]
        flags = detect_metric_flags(pts)
        values_by_ts = dict(pts)
        eps = group_episodes(flags, m, values_by_ts)
        episodes += eps
        all_flags += [(m, f) for f in flags]
        metrics_summary[m] = {
            "n_points": len(pts),
            "n_flags": len(flags),
            "n_episodes": len(eps),
            "last": pts[-1][1] if pts else None,
        }
        if flags:
            metrics_summary[m]["last_z"] = round(flags[-1].z, 2)

    episodes.sort(key=lambda e: e.magnitude, reverse=True)
    timeline = aggregate_timeline(all_flags)
    now = rows[-1]["ts"] if rows else 0

    def overlaps(a: Episode, b: Episode, pad: float = 43200.0) -> bool:
        return a.start - pad <= b.end and b.start - pad <= a.end

    top = []
    for ep in episodes[:5]:
        co = [e for e in episodes
              if e is not ep and e.metric != ep.metric and overlaps(ep, e)]
        co.sort(key=lambda e: e.magnitude, reverse=True)
        # prefer opposite-direction co-movers for the story, then magnitude
        co.sort(key=lambda e: (e.direction == ep.direction), reverse=False)
        top.append({
            "metric": ep.metric,
            "start": iso(ep.start), "end": iso(ep.end),
            "direction": ep.direction,
            "peak_z": round(ep.peak_z, 2),
            "baseline": ep.baseline, "during": ep.during,
            "n_flags": len(ep.flags),
            "attribution": attribute(ep, co),
            "co_movers": [e.metric for e in co[:3]],
        })

    current_score = recent_score(timeline, now)
    # dedupe per metric: keep each metric's strongest recent divergence
    now_div = {}
    for m, f in sorted(all_flags, key=lambda mf: abs(mf[1].z), reverse=True):
        if now - f.ts <= 3 * 86400 and m not in now_div:
            now_div[m] = {"metric": m, "z": round(f.z, 2),
                          "direction": f.direction, "value": f.value}
    now_div = sorted(now_div.values(), key=lambda d: -abs(d["z"]))
    report = {
        "repo": repo,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_commits": len(rows),
        "config": {
            "z_threshold": Z_THRESHOLD, "burn_in": BURN_IN,
            "episode_max_gap_h": EPISODE_MAX_GAP_S / 3600,
            "recency_halflife_d": RECENCY_HALFLIFE_D,
        },
        "current": {
            "as_of": iso(now) if rows else None,
            "recent_drift_score": round(current_score, 3),
            "raw_today_score": round(
                timeline.get(day_key(now), {}).get("score", 0.0), 3),
            "top_divergences_now": now_div[:6],
        },
        "episodes_top": top,
        "metrics": metrics_summary,
        "timeline": {k: {"score": round(v["score"], 3),
                         "metrics": v["metrics"]}
                     for k, v in sorted(timeline.items())},
        "total_episodes": len(episodes),
    }
    return report


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")


def render_markdown(rep: dict) -> str:
    lines = [f"# Drift report — {rep['repo']}",
             f"generated {rep['generated_utc']} · "
             f"{rep['n_commits']} commits · "
             f"z>{rep['config']['z_threshold']} "
             f"(burn-in {rep['config']['burn_in']})", ""]
    cur = rep["current"]
    lines += [f"## Current drift score: {cur['recent_drift_score']} "
              f"(raw today: {cur['raw_today_score']})",
              f"as of {cur['as_of']}"]
    if cur["top_divergences_now"]:
        lines += ["", "Active divergences (last 3 days):"]
        lines += [f"- {FRIENDLY.get(d['metric'], d['metric'])} "
                  f"{d['direction']} z={d['z']:+} value={fmt_num(d['value'])}"
                  for d in cur["top_divergences_now"]]
    lines += ["", "## Top drift episodes"]
    for i, ep in enumerate(rep["episodes_top"], 1):
        lines += [f"{i}. **{ep['start']} → {ep['end']}** — {ep['attribution']}"]
    lines += ["", "## Daily drift timeline",
              "| day | score | top metrics |", "|---|---|---|"]
    for k, v in rep["timeline"].items():
        tops = ", ".join(f"{FRIENDLY.get(m, m)} z={z:.1f}"
                         for m, z in sorted(v["metrics"].items(),
                                            key=lambda kv: -kv[1])[:3])
        lines.append(f"| {k} | {v['score']:.2f} | {tops} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=mx.DEFAULT_REPO)
    ap.add_argument("--series", default=None,
                    help="read JSONL from metrics.py instead of extracting")
    ap.add_argument("--report", default="report.json",
                    help="output JSON report path")
    ap.add_argument("--markdown", action="store_true",
                    help="also print a human-readable report to stdout")
    ap.add_argument("--print-score", action="store_true",
                    help="print only the current recent drift score")
    args = ap.parse_args()

    if args.series:
        with open(args.series) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        repo = args.repo
    else:
        if not os.path.isdir(args.repo):
            print(f"drift: repo not found: {args.repo}", file=sys.stderr)
            return 2
        repo = os.path.abspath(args.repo)
        rows = mx.extract_series(repo)
    if not rows:
        print("drift: no commits found", file=sys.stderr)
        return 2

    rep = build_report(repo, rows)
    with open(args.report, "w") as f:
        json.dump(rep, f, indent=2)
    if args.print_score:
        print(rep["current"]["recent_drift_score"])
    if args.markdown:
        sys.stdout.write(render_markdown(rep))
    else:
        print(f"[drift] report -> {args.report} "
              f"(current score {rep['current']['recent_drift_score']}, "
              f"{rep['total_episodes']} episodes)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
