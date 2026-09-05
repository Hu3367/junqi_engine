"""P4.1 贝叶斯暗子信念跟踪与 P4.2 多世界混合决策引擎专项单元测试。"""
from __future__ import annotations

import random
import unittest

from junqi.belief import BeliefTracker
from junqi.config import RuleConfig
from junqi.hybrid_engine import HybridDecisionEngine
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece, deal


class TestP4HybridAndBelief(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()
        self.rng = random.Random(2026)

    def test_belief_tracker_initialization(self):
        """测试暗子信念跟踪器初始化与概率归一化。"""
        st = deal(self.rng, self.cfg)
        tracker = BeliefTracker(st, seed=42)

        self.assertEqual(len(tracker.hidden_positions), 50)
        self.assertEqual(sum(tracker.remaining_pool.values()), 50)

        # 检查各暗子格点概率归一化
        for pos, dist in tracker.beliefs.items():
            tot_p = sum(dist.values())
            self.assertAlmostEqual(tot_p, 1.0, places=5)

    def test_belief_tracker_flip_update(self):
        """测试翻棋动作后的后验崩缩与剩余池扣减。"""
        st = deal(self.rng, self.cfg)
        tracker = BeliefTracker(st, seed=42)

        flip_act = Action("flip", (5, 0))
        st_after = st.apply(flip_act)

        tracker.on_action(flip_act, st, st_after)

        self.assertEqual(len(tracker.hidden_positions), 49)
        self.assertEqual(sum(tracker.remaining_pool.values()), 49)
        self.assertNotIn((5, 0), tracker.beliefs)

        # 验证其余格点重归一化
        for pos, dist in tracker.beliefs.items():
            tot_p = sum(dist.values())
            self.assertAlmostEqual(tot_p, 1.0, places=5)

    def test_belief_tracker_determinization_sampling(self):
        """测试多世界确定化采样的一致性。"""
        st = deal(self.rng, self.cfg)
        tracker = BeliefTracker(st, seed=42)

        world = tracker.sample_world(st)
        self.assertEqual(len(world), 50)
        for pos, pc in world.items():
            self.assertTrue(pc.revealed)
            self.assertIn(pos, tracker.hidden_positions)

        k_worlds = tracker.sample_k_worlds(st, k=4)
        self.assertEqual(len(k_worlds), 4)
        for w in k_worlds:
            self.assertEqual(len(w), 50)

    def test_hybrid_engine_instant_flag_capture(self):
        """测试混合决策引擎在有一步吃旗机会时绝对优先吃旗。"""
        st = GameState(board={
            (1, 1): Piece("r", Rank.GONG, True),
            (0, 1): Piece("b", Rank.QI, True),
            (11, 1): Piece("r", Rank.QI, True),
        }, dead=[Piece("b", Rank.LEI, True)] * 3, turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        engine = HybridDecisionEngine(k_worlds=2, device="cpu", seed=42)
        best_act = engine.select_action(st)

        self.assertEqual(best_act, Action("move", (1, 1), (0, 1)))

    def test_hybrid_engine_evaluate_position(self):
        """测试态势评估接口输出格式与数值有效性。"""
        st = deal(self.rng, self.cfg)
        engine = HybridDecisionEngine(k_worlds=2, device="cpu", seed=42)

        info = engine.evaluate_position(st)

        self.assertIn("win", info)
        self.assertIn("draw", info)
        self.assertIn("loss", info)
        self.assertIn("value", info)
        self.assertIn("best_action", info)

        p_sum = info["win"] + info["draw"] + info["loss"]
        self.assertAlmostEqual(p_sum, 1.0, places=3)
        self.assertTrue(-1.0 <= info["value"] <= 1.0)
        self.assertIsNotNone(info["best_action"])

    def test_hybrid_engine_repetition_avoidance(self):
        """测试即将导致循环判和的动作被战术规则有效惩罚。"""
        st = GameState(board={
            (5, 2): Piece("r", Rank.SI, True),
            (0, 1): Piece("b", Rank.QI, True),
            (11, 1): Piece("r", Rank.QI, True),
        }, turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        engine = HybridDecisionEngine(k_worlds=2, device="cpu", seed=42)

        # 构造若走到 (6, 2) 则命中已出现 2 次的局面
        st_next = st.apply(Action("move", (5, 2), (6, 2)))
        from junqi.state import position_key
        hist = {position_key(st_next): 2}

        scored = engine.choose_actions(st, topn=5, history_counts=hist)
        # 验证走到 (6,2) 的分数显著低于其他合法移动
        score_dict = {a: s for a, s in scored}
        rep_act = Action("move", (5, 2), (6, 2))
        if rep_act in score_dict:
            self.assertLess(score_dict[rep_act], 0.0)


if __name__ == "__main__":
    unittest.main()
