#!/usr/bin/env python3
"""metrics.py — extract per-commit metric time series from a git repo.

The honest first step of the world-model lane (see
/home/proart/projects/vision/research/world-models.md, item 2a): classical
drift detection over git history, NOT a neural world model.

For every commit (chronological order) it reconstructs a cumulative repo
snapshot (file list + sizes + selected blob contents via ls-tree/cat-file)
plus trailing-window activity, and emits one JSONL row:

    {"ts": 1760000000, "date": "2026-10-09T...", "sha": "abcd1234",
     "commits_7d": 25, "test_count": 0, ...}

Missing files (repo may lack MEMORY.md, dependency manifests, ...) yield
null values; the detector skips null-only metrics.

Python 3 stdlib only. No network, no GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

DEFAULT_REPO = "/home/proart/projects/zero-poc"

WINDOW_DAYS = 7.0
MSG_WINDOW = 10  # trailing commits averaged for commit-message length

# Files that count as pure bookkeeping (heartbeat state, not real work).
BOOKKEEPING_PATHS = {"MEMORY.md", "TASKS.md", ".agent-state.json"}

# Dependency manifests we look for (basename prefixes / exact names).
DEP_EXACT = {"pyproject.toml", "setup.py", "package.json", "Pipfile",
             "poetry.lock", "package-lock.json", "Cargo.toml", "go.mod",
             "go.sum", "Gemfile", "pom.xml"}
DEP_PREFIX = ("requirements",)

METRIC_ORDER = [
    "commits_7d", "test_count", "tests_touched_7d", "doc_files",
    "docs_bytes", "memory_bytes", "tasks_open", "authors_7d",
    "dep_age_days", "ci_age_days", "scratch_files", "msg_len_avg10",
    "days_since_substantive", "repo_files", "py_loc",
]


def run_git(repo: str, *args: str) -> str:
    p = subprocess.run(["git", "-C", repo, *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


@dataclass
class Commit:
    sha: str
    ts: int
    author: str
    subject: str
    files: list[str] = field(default_factory=list)

    @property
    def substantive(self) -> bool:
        return any(f not in BOOKKEEPING_PATHS for f in self.files)

    @property
    def touches_tests(self) -> bool:
        return any(f == "tests" or f.startswith("tests/") for f in self.files)

    @property
    def touches_dep(self) -> bool:
        for f in self.files:
            base = f.rsplit("/", 1)[-1]
            if base in DEP_EXACT or base.startswith(DEP_PREFIX):
                return True
        return False

    @property
    def touches_ci(self) -> bool:
        return any(f.startswith(".github/workflows/") for f in self.files)


def load_commits(repo: str) -> list[Commit]:
    """All commits (all refs) with changed-file lists, sorted by commit time.
    Same-timestamp batch pushes are tie-broken in topological order so the
    walk order matches the real build order."""
    sep_h, sep_r = "\x1f", "\x1e"
    # record = "\x1e<sha>\x1f<ct>\x1f<author>\x1f<subject>\n[blank]\n<files>"
    raw = run_git(repo, "log", "--all",
                  f"--format={sep_r}%H{sep_h}%ct{sep_h}%ae{sep_h}%s",
                  "--name-only")
    commits: list[Commit] = []
    seen: set[str] = set()
    for chunk in raw.split(sep_r):
        if not chunk.strip():
            continue
        lines = chunk.split("\n")
        sha, ts, ae, subject = lines[0].split(sep_h, 3)
        if sha in seen:  # reachable from multiple refs
            continue
        seen.add(sha)
        commits.append(Commit(sha=sha, ts=int(ts), author=ae, subject=subject,
                              files=[l.strip() for l in lines[1:] if l.strip()]))
    # rev-list --topo-order lists newest first; older => higher index.
    rev_pos = {sha: i for i, sha in enumerate(
        run_git(repo, "rev-list", "--all", "--topo-order").split())}
    commits.sort(key=lambda c: (c.ts, -rev_pos.get(c.sha, 0)))
    return commits


def snapshot_files(repo: str, sha: str) -> dict[str, tuple[int, str]]:
    """path -> (size_bytes, blob_sha) for the full tree at `sha`."""
    out = run_git(repo, "ls-tree", "-r", "--full-tree", "-l", sha)
    files: dict[str, tuple[int, str]] = {}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        files[path.strip('"')] = (int(parts[3]), parts[2])
    return files


class BlobCache:
    """Content cache keyed by blob sha (unchanged files cost nothing)."""
    def __init__(self, repo: str):
        self.repo = repo
        self.cache: dict[str, bytes] = {}

    def fetch(self, shas: list[str]) -> None:
        todo = [s for s in dict.fromkeys(shas) if s not in self.cache]
        if not todo:
            return
        proc = subprocess.run(
            ["git", "-C", self.repo, "cat-file", "--batch"],
            input=("\n".join(todo) + "\n").encode(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("git cat-file --batch failed")
        data, i = proc.stdout, 0
        for _ in todo:
            j = data.index(b"\n", i)
            sha, _typ, size = data[i:j].decode().split()
            size = int(size)
            self.cache[sha] = data[j + 1:j + 1 + size]
            i = j + 1 + size + 1  # skip trailing newline

    def get(self, sha: str) -> bytes:
        return self.cache[sha]


def count_open_tasks(text: str) -> int:
    """Unchecked checkbox items ('- [ ] x') in TASKS.md."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if s == "- [ ]" or s.startswith("- [ ] "):
            n += 1
    return n


def _py_loc(content: bytes) -> int:
    return content.decode("utf-8", errors="replace").count("\n") + (
        1 if content and not content.endswith(b"\n") else 0)


def extract_series(repo: str) -> list[dict]:
    """One JSONL-able row per commit, in chronological order."""
    commits = load_commits(repo)
    if not commits:
        return []
    blobs = BlobCache(repo)
    ts_list = [c.ts for c in commits]
    rows: list[dict] = []

    last_substantive: Optional[int] = None
    last_dep: Optional[int] = None
    last_ci: Optional[int] = None

    for i, c in enumerate(commits):
        t = c.ts
        # ---- trailing-window activity metrics ------------------------------
        w0 = t - WINDOW_DAYS * 86400
        j0 = bisect_left(ts_list, w0)
        window = commits[j0:i + 1]
        commits_7d = len(window)
        authors_7d = len({w.author for w in window})
        tests_touched_7d = sum(1 for w in window if w.touches_tests)
        msg_len_avg10 = (sum(len(w.subject) for w in commits[max(0, i - MSG_WINDOW + 1):i + 1])
                         / min(i + 1, MSG_WINDOW))
        # ages are "as of this commit" — a commit that does the work resets
        # its own age counter to 0
        if c.substantive:
            last_substantive = t
        if c.touches_dep:
            last_dep = t
        if c.touches_ci:
            last_ci = t
        days_since_substantive = ((t - last_substantive) / 86400.0
                                  if last_substantive is not None else 0.0)
        dep_age_days = ((t - last_dep) / 86400.0 if last_dep is not None else None)
        ci_age_days = ((t - last_ci) / 86400.0 if last_ci is not None else None)

        # ---- cumulative snapshot metrics ------------------------------------
        files = snapshot_files(repo, c.sha)
        test_paths = [p for p in files
                      if p == "tests" or p.startswith("tests/")]
        test_paths = [p for p in test_paths if p.endswith(".py")]
        doc_paths = [p for p in files if p.startswith("docs/")]
        py_paths = [p for p in files if p.endswith(".py")]
        scratch_files = sum(1 for p in files if p.startswith("scratch/"))

        need = []
        if "TASKS.md" in files:
            need.append(files["TASKS.md"][1])
        need += [files[p][1] for p in test_paths]
        need += [files[p][1] for p in py_paths]
        blobs.fetch(need)

        test_count = sum(blobs.get(files[p][1]).count(b"def test_")
                         for p in test_paths)
        tasks_open = (count_open_tasks(blobs.get(files["TASKS.md"][1])
                                       .decode("utf-8", errors="replace"))
                      if "TASKS.md" in files else None)
        py_loc = sum(_py_loc(blobs.get(files[p][1])) for p in py_paths)
        memory_bytes = files["MEMORY.md"][0] if "MEMORY.md" in files else None

        row = {
            "ts": t,
            "date": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
            "sha": c.sha[:10],
            "commits_7d": commits_7d,
            "test_count": test_count,
            "tests_touched_7d": tests_touched_7d,
            "doc_files": len(doc_paths),
            "docs_bytes": sum(files[p][0] for p in doc_paths),
            "memory_bytes": memory_bytes,
            "tasks_open": tasks_open,
            "authors_7d": authors_7d,
            "dep_age_days": round(dep_age_days, 3) if dep_age_days is not None else None,
            "ci_age_days": round(ci_age_days, 3) if ci_age_days is not None else None,
            "scratch_files": scratch_files,
            "msg_len_avg10": round(msg_len_avg10, 2),
            "days_since_substantive": round(days_since_substantive, 3),
            "repo_files": len(files),
            "py_loc": py_loc,
        }
        rows.append(row)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"target repo (default: {DEFAULT_REPO})")
    ap.add_argument("--out", default="-", help="output JSONL path ('-' = stdout)")
    args = ap.parse_args()

    rows = extract_series(args.repo)
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    if args.out == "-":
        sys.stdout.write(payload)
    else:
        with open(args.out, "w") as f:
            f.write(payload)
    print(f"[metrics] {len(rows)} commits -> {args.out} "
          f"({len(METRIC_ORDER)} metrics/commit)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
