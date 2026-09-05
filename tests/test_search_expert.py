"""专家级搜索核心 (QSearch, Star1, IDS, ExpertAgent) 单元与战术测试。"""
from __future__ import annotations

import unittest

from junqi.ai import Agent, ExpertAgent
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.rules import COMPOSITION, Rank
from junqi.search import ExpertSearchEngine
from junqi.state import Action, GameState, Piece


def mk(board_spec, dead=(), turn=0, seat_color=None, quiet=0, cfg=None):
    board = {pos: Piece(color, Rank[rk], revealed)
             for pos, (color, rk, revealed) in board_spec.items()}
    return GameState(board=board, dead=dead,
                     seat_color=seat_color or {0: "r", 1: "b"},
                     turn=turn, first_flip_done=True, quiet=quiet,
                     cfg=cfg or RuleConfig())


class TestSearchExpert(unittest.TestCase):

    def setUp(self):
        self.w = EvalWeights()

    def test_one_step_flag_capture(self):
        """一步可吃军旗时，搜索必须将吃旗动作作为最高优先级绝对必选。"""
        # 红工兵在 (1, 1)，蓝军旗在 (0, 1)，蓝 3 颗地雷已全死
        dead = [Piece("b", Rank.LEI, True) for _ in range(3)]
        st = mk({(1, 1): ("r", "GONG", True), (0, 1): ("b", "QI", True),
                 (5, 4): ("b", "SI", True)}, dead=dead, turn=0)

        agent = ExpertAgent(search=SearchConfig(depth=2), weights=self.w, seed=42)
        acts = agent.choose_actions(st, topn=1)

        self.assertTrue(len(acts) > 0)
        self.assertEqual(acts[0][0], Action("move", (1, 1), (0, 1)), "必须一步吃旗终局")

    def test_qsearch_avoids_poisoned_piece_trap(self):
        """静态搜索 (QSearch) 必须识破'贪吃小子随即被炸弹/大子反杀'的地平线陷阱。"""
        # 局面:
        # 红司令在 (5, 2)，红连长在 (10, 0)
        # 蓝排长在 (4, 2) (诱饵)
        # 蓝炸弹在 (3, 2) (守卫排长)
        # 蓝军长在 (0, 0) (后方大子)
        # 如果红司令吃 (4, 2)，在 QSearch 中下一步蓝炸弹直接吃 (4, 2) 炸死司令，
        # 随后蓝军长将碾压红连长；
        # 因此静态搜索必须识破该陷阱，避免走 (5, 2) -> (4, 2)。
        st = mk({
            (5, 2): ("r", "SI", True),
            (4, 2): ("b", "PAI", True),
            (3, 2): ("b", "ZHA", True),
            (10, 0): ("r", "LIAN", True),
            (0, 0): ("b", "JUN", True),
        }, turn=0)

        engine = ExpertSearchEngine(weights=self.w, seed=42)
        best_act, score, _ = engine.search(st, max_depth=1)

        # 在没有 QSearch 时，depth=1 贪吃排长 (+18 分)；
        # 启用 QSearch 后，吃排长会导致司令阵亡且蓝军长控场 (-35+ 分)，因此绝不走 (5, 2)->(4, 2)
        self.assertNotEqual(best_act, Action("move", (5, 2), (4, 2)),
                            "静态搜索必须防止司令贪吃受保护诱饵后被炸")

    def test_iterative_deepening_timeout_safety(self):
        """迭代加深在给定时间预算内能平稳返回，且不出现非法动作。"""
        st = mk({
            (5, 2): ("r", "SI", True),
            (5, 0): ("r", "JUN", True),
            (7, 2): ("b", "SI", True),
            (7, 4): ("b", "JUN", True),
        }, turn=0)

        engine = ExpertSearchEngine(weights=self.w, seed=42)
        # 限时 50ms 搜索高深度 (如 depth=20)
        best_act, score, stats = engine.search(st, max_depth=20, time_limit_ms=50)

        self.assertIsNotNone(best_act)
        self.assertTrue(stats.time_elapsed_ms <= 300.0)  # 允许短时间调度误差，但不会无限卡死

    def test_expert_agent_topn_structure(self):
        """ExpertAgent.choose_actions 能正确返回 topn 排序走法列表。"""
        st = mk({
            (5, 2): ("r", "SI", True),
            (5, 0): ("r", "JUN", True),
            (7, 2): ("b", "SI", True),
        }, turn=0)

        agent = ExpertAgent(search=SearchConfig(depth=2), weights=self.w, seed=42)
        results = agent.choose_actions(st, topn=3)

        self.assertEqual(len(results), 3)
        # 验证降序排列
        self.assertGreaterEqual(results[0][1], results[1][1])
        self.assertGreaterEqual(results[1][1], results[2][1])


if __name__ == "__main__":
    unittest.main()
