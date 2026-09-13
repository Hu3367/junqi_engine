"""S0/S1 整改专项单测（依据 docs/TRAINING_ROOT_CAUSE_REVIEW.md）。

覆盖：
1. S0 靶场标签一致性（true_value 与 true_val_class 不再矛盾）与新增校准指标；
2. S1 晋升判据新语义（整体 Wilson 下界 + ref_vs_search2 硬条件 + 正式协议开关）；
3. S1 对手配比纯函数（边界、总和、经验分布、无 net1 降级）；
4. S1 对手走子不再写入 one-hot 策略目标（策略样本数 == NN 搜索次数）。
"""
from __future__ import annotations

import hashlib
import random
import unittest
from collections import Counter
from unittest import mock

from junqi.benchmark import create_benchmark_suite, evaluate_net_benchmark
from junqi.net import JunqiNet
from junqi.selfplay import RandomStrategy
from junqi.train_rl import (OPP_MIX, decide_promotion, opponent_type_for,
                            play_selfplay_game)

CLASS_TO_VALUE = {0: 1.0, 1: 0.0, 2: -1.0}


class _StubNet:
    """无 torch 前向的桩网络：均匀先验 + 零估值。"""

    def eval(self):
        return self

    def predict_state(self, state, seat=None, world=None, history_counts=None, device="cpu"):
        acts = state.legal_actions()
        return ({a: 1.0 / len(acts) for a in acts}, 0.0)

    def predict_batch(self, items, device="cpu"):
        return [self.predict_state(it[0], it[1], it[2], it[3] if len(it) > 3 else None)
                for it in items]


class _CountingRandom(RandomStrategy):
    """计数版随机策略：记录对手走子手数。"""

    def __init__(self):
        self.calls = 0

    def choose(self, state, rng, avoid=None, history_counts=None):
        self.calls += 1
        return super().choose(state, rng, avoid=avoid, history_counts=history_counts)


# ---------------------------------------------------------------- S0 靶场修复

class TestS0BenchmarkFixes(unittest.TestCase):

    def test_suite_label_consistency(self):
        """S0：所有题目的 true_value 必须与 true_val_class 一致（±1/0 极端值）。"""
        suite = create_benchmark_suite()
        self.assertEqual(len(suite), 50)
        for item in suite:
            tv, tc = item["true_value"], item["true_val_class"]
            self.assertIn(tv, (-1.0, 0.0, 1.0),
                          f"{item['id']} 真值 {tv} 不是极端类值")
            self.assertEqual(tv, CLASS_TO_VALUE[tc],
                             f"{item['id']} 真值 {tv} 与类别 {tc} 矛盾")

    def test_new_calibration_metrics(self):
        """S0：靶场结果必须包含 Brier 分数与预测熵，且取值范围合法。"""
        res = evaluate_net_benchmark(JunqiNet(in_channels=38), device="cpu")
        self.assertIn("value_brier", res)
        self.assertIn("value_pred_entropy", res)
        self.assertTrue(0.0 <= res["value_brier"] <= 2.0)
        import math
        self.assertTrue(0.0 <= res["value_pred_entropy"] <= math.log(3.0) + 1e-6)

    def test_collapse_threshold_tightened(self):
        """S0：单类预测占比 ≥70% 必须触发塌缩告警。"""
        # 题库真值 Win=17/Draw=25/Loss=8；构造一个“恒预测 Draw”的桩网络
        class _DrawNet(_StubNet):
            def predict_state(self, state, seat=None, world=None,
                              history_counts=None, device="cpu"):
                policy, _ = super().predict_state(state, seat, world,
                                                  history_counts, device)
                return policy, 0.0

            def predict_probabilities(self, state, seat=None, world=None,
                                      history_counts=None, device="cpu"):
                policy, _ = super().predict_state(state, seat, world,
                                                  history_counts, device)
                return policy, {"win": 0.05, "draw": 0.90, "loss": 0.05}

        res = evaluate_net_benchmark(_DrawNet(), device="cpu")
        self.assertTrue(res["collapse_warning"], "90% Draw 预测必须触发塌缩告警")


# ---------------------------------------------------------------- S1 晋升判据

class TestS1PromotionGate(unittest.TestCase):

    def s(self, w, d, l):
        return {"wins": w, "draws": d, "losses": l,
                "games": w + d + l, "reasons": {}}

    def _balanced(self):
        """各阶段得分均 ≥0.3、整体 0.65（n=100）的可晋升统计。"""
        return {
            "overall": self.s(35, 60, 5), "opening": self.s(25, 15, 0),
            "midgame": self.s(5, 25, 5), "endgame": self.s(5, 20, 5),
            "random": self.s(0, 0, 0),
        }

    def test_overall_wilson_passes_at_moderate_n(self):
        """S1：整体 0.65（n=100）应可晋升——旧判据在此样本量下需单阶段 ≥0.75。"""
        ok, _ = decide_promotion(self._balanced())
        self.assertTrue(ok)
        # 更小样本（n=32，得分 0.719）同样可达：功效修复的直接证据
        ok, _ = decide_promotion({"overall": self.s(14, 18, 0),
                                  "opening": self.s(8, 8, 0),
                                  "midgame": self.s(6, 10, 0)})
        self.assertTrue(ok)

    def test_all_draw_mirror_still_rejected(self):
        ok, _ = decide_promotion({
            "overall": self.s(0, 80, 0), "opening": self.s(0, 20, 0),
            "midgame": self.s(0, 20, 0), "endgame": self.s(0, 20, 0)})
        self.assertFalse(ok)

    def test_ref_hard_conditions(self):
        """S1：ref_vs_search2 <0.5 或相对上轮退化时一票否决；未跑参考对抗时跳过。"""
        base = self._balanced()
        ok, _ = decide_promotion(base, ref_score=0.4)
        self.assertFalse(ok)
        ok, _ = decide_promotion(base, ref_score=0.55)
        self.assertTrue(ok)
        ok, _ = decide_promotion(base, ref_score=0.55, prev_ref_score=0.6)
        self.assertFalse(ok)
        ok, _ = decide_promotion(base, ref_score=None)   # 未跑参考对抗
        self.assertTrue(ok)

    def test_strict_stages_for_formal_protocol(self):
        """S1：strict_stages=True（正式 200 局协议）恢复子阶段显著性要求。"""
        stats = {
            "overall": self.s(35, 60, 5),
            "opening": self.s(8, 24, 0),    # 各子阶段得分下界均 ≤0.5
            "midgame": self.s(9, 22, 4),
            "endgame": self.s(8, 24, 3),
            "random": self.s(0, 0, 0),
        }
        ok, _ = decide_promotion(stats)
        self.assertTrue(ok, "快速门控：整体显著即可晋升")
        ok, _ = decide_promotion(stats, strict_stages=True)
        self.assertFalse(ok, "正式协议：还要求至少一个子阶段显著改善")


# ---------------------------------------------------------------- S1 对手配比

class TestS1OpponentMix(unittest.TestCase):

    def test_mix_sums_to_one(self):
        self.assertAlmostEqual(sum(OPP_MIX.values()), 1.0)

    def test_boundary_mapping(self):
        self.assertEqual(opponent_type_for(0.0), "mirror")
        self.assertEqual(opponent_type_for(0.499), "mirror")
        self.assertEqual(opponent_type_for(0.5), "best")
        self.assertEqual(opponent_type_for(0.749), "best")
        self.assertEqual(opponent_type_for(0.75), "expert")
        self.assertEqual(opponent_type_for(0.85), "greedy")
        self.assertEqual(opponent_type_for(0.949), "greedy")
        self.assertEqual(opponent_type_for(0.95), "random")
        self.assertEqual(opponent_type_for(0.999), "random")

    def test_no_net1_degrades_to_expert(self):
        self.assertEqual(opponent_type_for(0.6, net1_available=False), "expert")

    def test_empirical_ratio(self):
        """2000 个确定性均匀随机数下，经验占比与配置偏差 <3 个百分点。"""
        counts = Counter(opponent_type_for(random.Random(i).random())
                         for i in range(2000))
        total = sum(counts.values())
        for k, want in OPP_MIX.items():
            self.assertAlmostEqual(counts.get(k, 0) / total, want, delta=0.03,
                                   msg=f"对手 {k} 实际占比偏离配置过大")


# ---------------------------------------------------------------- S1 对手样本隔离

class TestS1OpponentSamples(unittest.TestCase):

    def _hash(self, p, v):
        h = hashlib.md5()
        for arr, _, pi, _ in p:
            h.update(arr.tobytes())
            h.update(pi.tobytes())
        for item in v:
            # S2 起 Value 样本为 4 元组 (arr, z, phase, is_world)，兼容旧 3 元组
            h.update(item[0].tobytes())
            h.update(bytes([item[1]]))
        return h.hexdigest()

    def test_opponent_moves_excluded_from_policy_targets(self):
        """S1：策略样本数必须等于 NN 搜索次数（对手手不再写 one-hot 目标）。"""
        from junqi import train_rl as trl
        opp = _CountingRandom()
        search_calls = {"n": 0}
        orig_search = trl.MCTS.search

        def wrapped(self, *a, **k):
            search_calls["n"] += 1
            return orig_search(self, *a, **k)

        with mock.patch.object(trl.MCTS, "search", wrapped):
            p, v, pv, _rec = play_selfplay_game(_StubNet(), opp_strategy=opp, sims=4,
                                          device="cpu", seed=77, curriculum_prob=0.0,
                                          quiet_tail_cutoff=10**9)
        self.assertGreater(opp.calls, 0, "对手策略未被调用（测试前提不成立）")
        self.assertGreater(search_calls["n"], 0)
        self.assertEqual(len(p), search_calls["n"],
                         "策略样本数 != NN 搜索次数：对手手仍被记入策略目标")

    def test_opponent_game_determinism(self):
        """含外部对手的自对弈同种子必须完全可复现。"""
        def run_once():
            p, v, pv, _rec = play_selfplay_game(_StubNet(), opp_strategy=RandomStrategy(),
                                          sims=4, device="cpu", seed=991,
                                          curriculum_prob=0.0)
            return len(p), len(v), len(pv), self._hash(p, v)

        self.assertEqual(run_once(), run_once())


if __name__ == "__main__":
    unittest.main()
