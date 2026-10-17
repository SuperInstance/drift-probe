#!/usr/bin/env python3
"""Tests for drift-probe: EWMA/Holt math, flag detection, episode grouping,
aggregation, and metrics extraction over a synthetic git repo.

Run:  python3 -m unittest discover -s tests -v   (from repo root)
      python3 tests/test_drift.py                (also works)
"""
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import drift  # noqa: E402
import metrics as mx  # noqa: E402

H = 3600.0


def _stable(n=40, seed=7, level=100.0, spread=2.0):
    rng = random.Random(seed)
    return [level + rng.uniform(-spread, spread) for _ in range(n)]


class TestEwmaMath(unittest.TestCase):
    def test_ewma_level_and_z_hand_computed(self):
        # alpha=0.5: levels [10,10,10,11]; final residual 2 against sd floor
        # max(sqrt(0.2025), 0.05*10) = 0.5 -> z = 4.0
        levels, zs = drift.ewma_zscores([10, 10, 10, 12], alpha=0.5, beta=0.1)
        self.assertEqual(levels, [10.0, 10.0, 10.0, 11.0])
        self.assertEqual(zs[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(zs[3], 4.0)

    def test_z_is_capped(self):
        # 0 -> 1000 against a tiny floor must not produce an absurd z
        _, zs = drift.ewma_zscores([1.0] * 20 + [1000.0])
        self.assertLessEqual(abs(zs[-1]), drift.Z_CAP)

    def test_holt_tracks_pure_linear_ramp(self):
        values = [3.0 * t for t in range(20)]
        preds, zs = drift.holt_zscores(values)
        self.assertTrue(all(abs(z) < 0.5 for z in zs[2:]),
                        f"ramp should track, got {zs}")


class TestDetection(unittest.TestCase):
    def test_constant_series_not_flagged(self):
        pts = [(i * H, 5.0) for i in range(30)]
        self.assertEqual(drift.detect_metric_flags(pts), [])

    def test_stable_noisy_series_not_flagged(self):
        vals = _stable()
        pts = [(i * H, v) for i, v in enumerate(vals)]
        self.assertEqual(drift.detect_metric_flags(pts), [])

    def test_injected_upward_drift_flagged(self):
        vals = _stable() + [v + 25.0 for v in _stable(seed=8)]
        pts = [(i * H, v) for i, v in enumerate(vals)]
        flags = drift.detect_metric_flags(pts)
        self.assertTrue(flags, "upward shift must be flagged")
        after = [f for f in flags if f.ts >= 40 * H]
        before = [f for f in flags if f.ts < 40 * H]
        self.assertFalse(before, "stable region must stay quiet")
        self.assertTrue(all(f.direction == "up" for f in after))
        self.assertGreater(max(abs(f.z) for f in after), 2.5)

    def test_injected_downward_drift_flagged(self):
        vals = _stable() + [max(v - 30.0, 0.5) for v in _stable(seed=9)]
        pts = [(i * H, v) for i, v in enumerate(vals)]
        flags = drift.detect_metric_flags(pts)
        after = [f for f in flags if f.ts >= 40 * H]
        self.assertTrue(after)
        self.assertTrue(all(f.direction == "down" for f in after))

    def test_step_flagged_by_both_modes(self):
        vals = [10.0] * 15 + [20.0] * 15
        pts = [(i * H, v) for i, v in enumerate(vals)]
        flags = drift.detect_metric_flags(pts)
        self.assertTrue(flags)
        self.assertIn("ewma_z", flags[0].modes)
        self.assertIn("holt", flags[0].modes)

    def test_short_series_skipped(self):
        self.assertEqual(drift.detect_metric_flags([(i * H, i) for i in range(5)]), [])


class TestEpisodesAndScore(unittest.TestCase):
    def test_episode_grouping_by_gap(self):
        vmap = {float(t): 1.0 for t in range(0, 200 * int(H), int(H))}
        mk = lambda t: drift.Flag(ts=t, value=1.0, z=3.0, direction="up")
        flags = [mk(0), mk(12 * H), mk(24 * H),
                 mk(124 * H), mk(136 * H)]  # 100h gap splits the run
        eps = drift.group_episodes(flags, "test_count", vmap)
        self.assertEqual(len(eps), 2)
        self.assertEqual([len(e.flags) for e in eps], [3, 2])

    def test_score_higher_during_injected_drift_than_stability(self):
        stable = self._pts(_stable(seed=11))
        drifted = self._pts(_stable(seed=12) + [v + 40 for v in _stable(seed=13)])
        flags_stable = drift.detect_metric_flags(stable)
        flags_drift = drift.detect_metric_flags(drifted)
        # a persistent shift must fire more than once (winsorized variance
        # keeps flagging until the baseline accepts the new level)
        self.assertGreaterEqual(len(flags_drift), 2)
        tl_stable = drift.aggregate_timeline([("m", f) for f in flags_stable])
        tl_drift = drift.aggregate_timeline([("m", f) for f in flags_drift])
        s_stable = sum(d["score"] for d in tl_stable.values())
        s_drift = sum(d["score"] for d in tl_drift.values())
        self.assertGreater(s_drift, s_stable + 2.0)

    @staticmethod
    def _pts(vals):
        return [(i * H, v) for i, v in enumerate(vals)]

    def test_recent_score_decays_with_age(self):
        tl = {"2026-10-01": {"score": 10.0, "metrics": {"m": 3.0}}}
        now = drift.datetime.strptime("2026-10-02", "%Y-%m-%d").replace(
            tzinfo=drift.timezone.utc).timestamp()
        later = now + 5 * 86400
        self.assertGreater(drift.recent_score(tl, now),
                           drift.recent_score(tl, later))
        self.assertEqual(drift.recent_score({}, now), 0.0)

    def test_episode_direction_follows_values(self):
        vmap = {float(i * H): 10.0 + i for i in range(40)}
        # flags claim "up" but values during episode are below baseline
        flags = [drift.Flag(ts=30 * H, value=20.0, z=-3.0, direction="down")]
        ep = drift.group_episodes(flags, "m", vmap)[0]
        self.assertEqual(ep.direction, "down")


class TestMetricsExtraction(unittest.TestCase):
    def test_count_open_tasks(self):
        text = "# T\n\n- [x] done\n- [ ] one\n  - [ ] two\n- [ ]three\n"
        self.assertEqual(mx.count_open_tasks(text), 2)

    def test_extract_series_on_synthetic_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "r")
            os.makedirs(os.path.join(repo, "tests"))
            self._git(repo, "init", "-b", "main")
            # c1: template + 2 tests + task board
            self._write(repo, "MEMORY.md", "m" * 100)
            self._write(repo, "TASKS.md", "# T\n- [ ] a\n- [ ] b\n- [x] c\n")
            self._write(repo, "tests/test_a.py",
                        "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n")
            self._commit(repo, "2026-01-01T12:00:00 +0000", "seed template")
            # c2: bookkeeping-only (MEMORY.md append)
            self._write(repo, "MEMORY.md", "m" * 150)
            self._commit(repo, "2026-01-03T12:00:00 +0000", "chore: memory")
            # c3: real work — more tests + a dependency manifest
            self._write(repo, "tests/test_a.py",
                        "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n"
                        "\n\ndef test_three():\n    pass\n\n\ndef test_four():\n    pass\n")
            self._write(repo, "requirements.txt", "flask\n")
            self._commit(repo, "2026-01-05T12:00:00 +0000", "feat: more tests")

            rows = mx.extract_series(repo)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["test_count"] for r in rows], [2, 2, 4])
            self.assertEqual([r["tasks_open"] for r in rows], [2, 2, 2])
            self.assertTrue(rows[1]["memory_bytes"] > rows[0]["memory_bytes"])
            # c2 touches only MEMORY.md -> 2 idle days; c3 is substantive -> 0
            self.assertAlmostEqual(rows[1]["days_since_substantive"], 2.0, places=3)
            self.assertEqual(rows[2]["days_since_substantive"], 0.0)
            # no dep manifest until c3: null, null, 0.0
            self.assertEqual(rows[0]["dep_age_days"], None)
            self.assertEqual(rows[2]["dep_age_days"], 0.0)
            # missing docs/ handled as zeros, not errors
            self.assertEqual(rows[0]["doc_files"], 0)
            self.assertEqual(rows[0]["docs_bytes"], 0)
            self.assertEqual([r["repo_files"] for r in rows], [3, 3, 4])

    @staticmethod
    def _write(repo, rel, content):
        path = os.path.join(repo, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    @staticmethod
    def _git(repo, *args):
        subprocess.run(["git", "-C", repo, *args], check=True,
                       capture_output=True)

    def _commit(self, repo, date, msg):
        env = dict(os.environ,
                   GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date,
                   GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@x")
        self._git(repo, "add", "-A")
        subprocess.run(["git", "-C", repo, "commit", "-m", msg],
                       check=True, capture_output=True, env=env)


if __name__ == "__main__":
    unittest.main(verbosity=2)
