"""P3 批次 3 验收：对手配比预置（mirror 0.5→0.3 / expert 0.1→0.2）与正式晋级协议。

背景（2026-09-14 两次 n=200 形式化门控）：同源模型之间 72%-91.5% 的对局以
no_capture 判和，门控只剩 17-56 局胜负样本、区分度枯竭。mirror 自对弈对抗梯度
接近零，而 expert 对局既提供真实对抗压力、又是客观终局 Value 样本的最廉价来源。
故本批次降低 mirror 占比、提高 expert 占比（一次只改这一个变量）。

同时验收批次 1b 的"轮内门控降级为只记录"：
    n=16 时 Wilson 判据需得分率 ≥0.75（约 +191 Elo）才显著，对每轮 +10~30 Elo 的
    真实进步无功效；默认不得据此改动发布模型，晋升只能经正式 SPRT 门控
    （`--promote-to-best`）。
"""
from __future__ import annotations

import os
import random
import tempfile
import unittest

from junqi.eval_gate import promote_candidate
from junqi.train_rl import (LR_FLOOR_RATIO, OPP_MIX, OPP_MIX_PRESETS,
                            inloop_gate_decision, lr_for_epoch, opponent_type_for)


def _perfect_record(n: int = 16) -> dict:
    """构造一个"远超晋级判据"的轮内门控记录（全胜）。"""
    stages = {name: {"wins": 4, "draws": 0, "losses": 0, "games": 4, "reasons": {}}
              for name in ("opening", "midgame", "endgame", "random")}
    stages["overall"] = {"wins": n, "draws": 0, "losses": 0, "games": n, "reasons": {}}
    return stages


class TestOppMixPresets(unittest.TestCase):

    def test_presets_are_valid_distributions(self):
        for name, mix in OPP_MIX_PRESETS.items():
            self.assertEqual(set(mix), set(OPP_MIX), f"{name} 键不一致")
            self.assertAlmostEqual(sum(mix.values()), 1.0, places=9, msg=name)
            for k, v in mix.items():
                self.assertGreaterEqual(v, 0.0, f"{name}.{k}")
                self.assertLessEqual(v, 1.0, f"{name}.{k}")

    def test_baseline_preset_equals_current_constant(self):
        self.assertEqual(OPP_MIX_PRESETS["baseline"], OPP_MIX)

    def test_diverse_preset_matches_plan(self):
        d = OPP_MIX_PRESETS["diverse"]
        self.assertAlmostEqual(d["mirror"], 0.30)
        self.assertAlmostEqual(d["expert"], 0.20)
        self.assertLess(d["mirror"], OPP_MIX["mirror"])
        self.assertGreater(d["expert"], OPP_MIX["expert"])

    def test_sampling_distribution_follows_preset(self):
        mix = OPP_MIX_PRESETS["diverse"]
        rng = random.Random(7)
        n = 20000
        counts = {}
        for _ in range(n):
            t = opponent_type_for(rng.random(), net1_available=True, mix=mix)
            counts[t] = counts.get(t, 0) + 1
        for k, w in mix.items():
            self.assertAlmostEqual(counts.get(k, 0) / n, w, delta=0.02, msg=k)

    def test_best_degrades_to_expert_without_opponent_net(self):
        mix = OPP_MIX_PRESETS["diverse"]
        seen = {opponent_type_for(r / 1000.0, net1_available=False, mix=mix)
                for r in range(1000)}
        self.assertNotIn("best", seen)
        self.assertIn("expert", seen)

    def test_diverse_mix_raises_mirror_opponent_count_of_expert_games(self):
        """同一批随机数下，diverse 预置的 expert 局数应显著多于 baseline。"""
        rng_seed, n = 11, 5000
        def expert_share(mix):
            rng = random.Random(rng_seed)
            hits = sum(1 for _ in range(n)
                       if opponent_type_for(rng.random(), True, mix) == "expert")
            return hits / n
        self.assertAlmostEqual(expert_share(OPP_MIX_PRESETS["baseline"]), 0.10, delta=0.02)
        self.assertAlmostEqual(expert_share(OPP_MIX_PRESETS["diverse"]), 0.20, delta=0.02)


class TestInloopGateDemotion(unittest.TestCase):

    def test_default_blocks_promotion_even_with_perfect_record(self):
        promote, reason = inloop_gate_decision(_perfect_record(16))
        self.assertFalse(promote, "轮内门控默认必须只记录、不判定")
        self.assertIn("无统计功效", reason)
        self.assertIn("--promote-to-best", reason)

    def test_rollback_flag_restores_old_behaviour(self):
        promote, reason = inloop_gate_decision(_perfect_record(16),
                                              inloop_gate_promote=True)
        self.assertTrue(promote, f"回滚开关应恢复晋升判定，实际: {reason}")

    def test_rollback_flag_still_rejects_mediocre_record(self):
        stages = _perfect_record(16)
        stages["overall"] = {"wins": 2, "draws": 12, "losses": 2, "games": 16,
                             "reasons": {}}
        promote, _ = inloop_gate_decision(stages, inloop_gate_promote=True)
        self.assertFalse(promote)


class TestPromotionProtocol(unittest.TestCase):

    def _mk(self, tmp, promote: bool):
        cand = os.path.join(tmp, "candidate.pt")
        best = os.path.join(tmp, "best.pt")
        open(cand, "w").write("NEW")
        open(best, "w").write("OLD")
        report = {"promote": promote, "score_rate": 0.62,
                  "sprt": {"decision": "accept_h1"}}
        return cand, best, report

    def test_promotes_only_when_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand, best, report = self._mk(tmp, promote=True)
            changed, msg = promote_candidate(report, cand, best_path=best, backup_dir=tmp)
            self.assertTrue(changed)
            self.assertEqual(open(best).read(), "NEW")
            backups = [f for f in os.listdir(tmp) if f.startswith("best_legacy_")]
            self.assertEqual(len(backups), 1, "旧发布模型必须带时间戳备份")
            self.assertEqual(open(os.path.join(tmp, backups[0])).read(), "OLD")
            self.assertIn("已更新", msg)

    def test_rejected_gate_leaves_best_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cand, best, report = self._mk(tmp, promote=False)
            changed, msg = promote_candidate(report, cand, best_path=best, backup_dir=tmp)
            self.assertFalse(changed)
            self.assertEqual(open(best).read(), "OLD")
            self.assertEqual([f for f in os.listdir(tmp) if f.startswith("best_legacy_")], [])
            self.assertIn("未晋级", msg)

    def test_missing_candidate_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            best = os.path.join(tmp, "best.pt")
            open(best, "w").write("OLD")
            changed, msg = promote_candidate(
                {"promote": True, "score_rate": 0.7}, os.path.join(tmp, "nope.pt"),
                best_path=best, backup_dir=tmp)
            self.assertFalse(changed)
            self.assertEqual(open(best).read(), "OLD")


class TestLrSchedule(unittest.TestCase):

    def test_constant_returns_base_for_all_epochs(self):
        for ep in range(1, 7):
            self.assertAlmostEqual(
                lr_for_epoch(1e-4, ep, 1, 6, schedule="constant"), 1e-4, places=12)

    def test_cosine_endpoints_and_monotonicity(self):
        start, end, base = 1, 6, 1e-4
        self.assertAlmostEqual(lr_for_epoch(base, start, start, end, "cosine"), base)
        self.assertAlmostEqual(lr_for_epoch(base, end, start, end, "cosine"),
                               base * LR_FLOOR_RATIO, places=12)
        vals = [lr_for_epoch(base, ep, start, end, "cosine") for ep in range(start, end + 1)]
        self.assertEqual(vals, sorted(vals, reverse=True), "cosine 必须单调不增")
        self.assertTrue(all(base * LR_FLOOR_RATIO - 1e-12 <= v <= base + 1e-12 for v in vals))

    def test_never_below_floor(self):
        v = lr_for_epoch(1e-4, 99, 1, 6, "cosine")     # 超出范围的轮次被 clamp
        self.assertAlmostEqual(v, 1e-4 * LR_FLOOR_RATIO, places=12)

    def test_floor_ratio_is_configurable(self):
        v = lr_for_epoch(1e-4, 6, 1, 6, "cosine", floor_ratio=0.5)
        self.assertAlmostEqual(v, 5e-5, places=12)


if __name__ == "__main__":
    unittest.main()
