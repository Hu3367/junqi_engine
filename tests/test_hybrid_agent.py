"""P2 阶段混合引擎 (HybridAgent) 单元测试与战术验证。"""
from __future__ import annotations

import random
import unittest

from junqi.ai import HybridAgent
from junqi.config import RuleConfig
from junqi.rules import Rank
from junqi.selfplay import make_strategy
from junqi.state import Action, GameState, Piece, deal


class TestHybridAgent(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def test_hybrid_agent_initialization_and_inference(self):
        """测试 HybridAgent 初始化与正常局面决策。"""
        agent = HybridAgent(model_path="models/bc_best.pt", search_depth=2, top_k=4)
        st = deal(rng=random.Random(42))

        scored = agent.choose_actions(st, topn=3)
        self.assertGreater(len(scored), 0)
        self.assertLessEqual(len(scored), 3)
        best_act, best_score = scored[0]
        self.assertIn(best_act, st.legal_actions())

    def test_hybrid_agent_tactical_kill_override(self):
        """测试战术覆盖：当存在一步吃旗制胜走法时，HybridAgent 必须选择吃旗。"""
        agent = HybridAgent(model_path="models/bc_best.pt", search_depth=2)

        # 构造局面：地雷已清空，红工兵在 (1,1)，蓝军旗在 (0,1)
        dead = (Piece("b", Rank.LEI), Piece("b", Rank.LEI), Piece("b", Rank.LEI))
        board = {
            (1, 1): Piece("r", Rank.GONG, revealed=True),
            (0, 1): Piece("b", Rank.QI, revealed=True),
        }
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0, dead=dead, cfg=self.cfg)

        scored = agent.choose_actions(st, topn=1)
        self.assertEqual(len(scored), 1)
        best_act, _ = scored[0]

        # 必须选择一步吃旗 (1,1) -> (0,1)
        self.assertEqual(best_act, Action("move", (1, 1), (0, 1)))

    def test_hybrid_strategy_factory(self):
        """测试 make_strategy 支持 hybrid 策略。"""
        strat = make_strategy("hybrid2", seed=123)
        self.assertEqual(strat.name, "hybrid_d2")
        st = deal(rng=random.Random(10))
        act = strat.choose(st, random.Random(10))
        self.assertIn(act, st.legal_actions())


if __name__ == "__main__":
    unittest.main()
