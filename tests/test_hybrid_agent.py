"""P2 阶段混合引擎 (HybridAgent) 单元测试与战术验证。"""
from __future__ import annotations

import random
import unittest

from junqi.ai import HybridAgent
from junqi.config import RuleConfig
from junqi.rules import Rank
from junqi.selfplay import make_strategy
from junqi.state import Action, GameState, Piece, deal, position_key


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

    def test_hybrid_agent_penalizes_repetition_position(self):
        """规避集合命中时，该走法必须被重罚到末位（防重复判和）。

        回归守卫（2026-09-17 定证）：`HybridAgent.choose_actions` 曾把 avoid 判定
        写成 `a in avoid`——`a` 是 `Action` dataclass，而 `avoid` 装的是
        `position_key()` 产出的 4 元局面键 ⇒ 判定恒为 False，混合引擎在实战中
        **从不规避重复局面**。P2 验收第 4 条据此暴露出重复判和率 40%（对照 expert2 6.7%）。
        同仓其余 5 处消费者（`Agent`/`ExpertAgent`/`MCTS`/`HybridDecisionEngine`/
        `search` 根节点）都写作 `position_key(state.apply(a)) in avoid`。
        """
        agent = HybridAgent(model_path="models/bc_best.pt", search_depth=1, top_k=64)

        # 一个明子（可走）+ 一个暗子（可翻），保证合法动作数 > 1。
        # (3,1) 不在铁路、不是行营，其公路邻格 (3,0)/(3,2)/(2,1)/(4,1) 全部为空。
        board = {
            (3, 1): Piece("r", Rank.SHI, revealed=True),
            (3, 3): Piece("b", Rank.LIAN, revealed=False),
        }
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0, cfg=self.cfg)
        acts = st.legal_actions()
        self.assertGreater(len(acts), 1, "本用例需要多个合法动作")

        # 基准：不传 avoid 时的完整打分，用于挑出"本来最优的那步走子"
        plain = dict(agent.choose_actions(st, topn=len(acts)))
        self.assertEqual(set(plain), set(acts), "top_k 需覆盖全部合法动作")
        best_move = max((a for a in acts if a.kind == "move"), key=lambda a: plain[a])

        # 把该走法的后继局面放进规避集合，再问一次
        avoid = {position_key(st.apply(best_move))}
        scored = agent.choose_actions(st, topn=len(acts), avoid=avoid)
        d = dict(scored)

        self.assertIn(best_move, d, "被规避的走法仍应出现在打分表里（只是垫底）")
        self.assertEqual(scored[-1][0], best_move,
                         "命中 avoid 的走法必须被罚到末位")
        self.assertLess(d[best_move], -1e5, "应按 WIN_SCORE 量级重罚")


if __name__ == "__main__":
    unittest.main()
