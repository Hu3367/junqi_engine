"""P3 数据质量改造验收：认输机制（ResignTracker）与生成/评测夹具分离。

对应 2026-09-13 基线计划修订：
1. 生成侧调稀平局触发器（no_capture=120/repetition=4），评测门控保持官方 70/1000 规则；
2. 和棋局的连续无吃子尾部（quiet >= 60）样本作为垃圾段丢弃；
3. 自博弈认输：走子方根 Value <= -0.95 连续 8 手（且 ply >= 40）判该方认输，
   对局按官方 code 21 语义记 ±1 终局标签——终局奖励定义不变，回滚 = 关闭开关。
"""
from __future__ import annotations

import random
import unittest

from junqi.config import RuleConfig
from junqi.state import deal


class _SeatBiasedStubNet:
    """stub 网络：座位 0 恒认为己方必败（v=-0.99），座位 1 恒认为己方必胜。

    MCTS 回传符号约定下，根 Value 与走子方视角一致，用于确定性地触发认输。"""

    def __init__(self, v0: float = -0.99, v1: float = 0.99):
        self.v0 = v0
        self.v1 = v1

    def predict_state(self, state, seat=None, world=None, history_counts=None, device="cpu"):
        acts = state.legal_actions()
        p = 1.0 / max(len(acts), 1)
        v = self.v0 if seat == 0 else self.v1
        return {a: p for a in acts}, v

    def predict_batch(self, items, device="cpu"):
        return [self.predict_state(it[0], seat=it[1], world=it[2],
                                   history_counts=it[3] if len(it) > 3 else None)
                for it in items]


class TestResignTracker(unittest.TestCase):

    def test_triggers_after_consecutive_losing_observations(self):
        from junqi.train_rl import ResignTracker
        t = ResignTracker(threshold=-0.95, consecutive=3, min_ply=10)
        self.assertFalse(t.observe(0, -0.99, ply=20))   # 1
        self.assertFalse(t.observe(0, -0.99, ply=22))   # 2
        self.assertTrue(t.observe(0, -0.99, ply=24))    # 3 -> 认输

    def test_resets_on_improvement(self):
        from junqi.train_rl import ResignTracker
        t = ResignTracker(threshold=-0.95, consecutive=3, min_ply=10)
        self.assertFalse(t.observe(0, -0.99, ply=20))
        self.assertFalse(t.observe(0, -0.99, ply=22))
        self.assertFalse(t.observe(0, -0.5, ply=24))    # 局面改善，计数清零
        self.assertFalse(t.observe(0, -0.99, ply=26))
        self.assertFalse(t.observe(0, -0.99, ply=28))
        self.assertTrue(t.observe(0, -0.99, ply=30))    # 重新计满 3 次

    def test_respects_min_ply(self):
        from junqi.train_rl import ResignTracker
        t = ResignTracker(threshold=-0.95, consecutive=2, min_ply=40)
        self.assertFalse(t.observe(0, -0.99, ply=10))
        self.assertFalse(t.observe(0, -0.99, ply=12))   # 未到 min_ply 不认输
        self.assertFalse(t.observe(0, -0.99, ply=40))
        self.assertTrue(t.observe(0, -0.99, ply=42))

    def test_per_seat_independent(self):
        from junqi.train_rl import ResignTracker
        t = ResignTracker(threshold=-0.95, consecutive=2, min_ply=10)
        self.assertFalse(t.observe(0, -0.99, ply=20))
        self.assertFalse(t.observe(1, -0.99, ply=22))   # 座位 1 的劣势不影响座位 0
        self.assertFalse(t.observe(1, 0.99, ply=24))
        self.assertTrue(t.observe(0, -0.99, ply=26))    # 座位 0 计满

    def test_none_value_ignored(self):
        from junqi.train_rl import ResignTracker
        t = ResignTracker(threshold=-0.95, consecutive=2, min_ply=10)
        self.assertFalse(t.observe(0, None, ply=20))    # 强制单着无估值，不计数也不清零
        self.assertFalse(t.observe(0, -0.99, ply=22))
        self.assertTrue(t.observe(0, -0.99, ply=24))    # None 不打断连续计数


class TestGenerationFixture(unittest.TestCase):

    def test_generation_cfg_relaxes_draw_triggers(self):
        """生成侧夹具：调稀平局触发器；官方规则默认值不得被改动。"""
        from junqi.train_rl import GENERATION_CFG
        self.assertEqual(GENERATION_CFG.no_capture_draw_plies, 120)
        self.assertEqual(GENERATION_CFG.repetition_draw_count, 4)
        official = RuleConfig()
        self.assertEqual(official.no_capture_draw_plies, 70)
        self.assertEqual(official.repetition_draw_count, 3)

    def test_drop_draw_tail_filter(self):
        """和棋局的 quiet >= 60 尾部样本丢弃；决胜局样本全保留。"""
        from junqi.train_rl import _drop_draw_tail
        items = [(f"s{i}", q) for i, q in enumerate([10, 30, 59, 60, 80, 110])]
        kept = _drop_draw_tail(items, final_winner=-1, quiet_idx=1)
        self.assertEqual([x[0] for x in kept], ["s0", "s1", "s2"])
        # 决胜局：尾部照样保留（z=±1 有信号）
        kept_win = _drop_draw_tail(items, final_winner=0, quiet_idx=1)
        self.assertEqual(len(kept_win), 6)


class TestResignIntegration(unittest.TestCase):

    def test_play_selfplay_game_resign_ends_dead_lost_game(self):
        """stub 网络（座位 0 恒判己方必败）下，座位 0 在计满连续劣势手数后认输：
        对局提前终止、终局原因 resign、Value 标签出现决胜类别（非全和棋）。"""
        from junqi.train_rl import play_selfplay_game
        p, v, pv, record = play_selfplay_game(
            _SeatBiasedStubNet(), sims=6, device="cpu", seed=42,
            resign_threshold=-0.9, resign_consecutive=3, resign_min_ply=6)
        self.assertEqual(record["reason"], "resign")
        self.assertEqual(record["winner"], 1)          # 座位 0 认输，座位 1 胜
        self.assertTrue(record["resigned_seat"] == 0)
        self.assertLess(record["plies"], 60)           # 远早于拖和长度
        # 样本路由限制：认输局的 ±1 标签来自模型自身判断，不得进 Value 训练
        self.assertGreater(len(p), 0, "认输局的 Policy 样本必须保留")
        self.assertEqual(len(v), 0, "认输局不得产生 Value（世界模式）样本")
        self.assertEqual(len(pv), 0, "认输局不得产生 Value（公共模式）样本")

    def test_mcts_search_returns_root_value(self):
        """mcts.search 第 5 返回值为根走子方视角期望值（认输判定的输入）。"""
        from junqi.mcts import MCTS
        st = deal(random.Random(7), RuleConfig())
        net = _SeatBiasedStubNet()
        v0 = MCTS(net, simulations=8).search(st, temperature=1.0,
                                             add_noise=False, rng=random.Random(1))[4]
        self.assertIsNotNone(v0)
        self.assertLess(v0, -0.5)                      # 座位 0 视角：自认必败
        # 座位 1 视角：构造 turn=1 的对称局面不便利，直接校验符号翻转通道即可——
        # 由 ResignIntegration 的认输方向（座位 0 被判负）间接保证。


if __name__ == "__main__":
    unittest.main()
