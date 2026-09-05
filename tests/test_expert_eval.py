"""专家级评估函数 (Dynamic Dominance, Mobility, Flip Safety) 单元测试。"""
from __future__ import annotations

import unittest

from junqi.config import EvalWeights, RuleConfig
from junqi.eval_expert import evaluate_expert
from junqi.rules import Rank
from junqi.state import GameState, Piece


def mk(board_spec, dead=(), turn=0, seat_color=None, quiet=0, cfg=None):
    board = {pos: Piece(color, Rank[rk], revealed)
             for pos, (color, rk, revealed) in board_spec.items()}
    return GameState(board=board, dead=dead,
                     seat_color=seat_color or {0: "r", 1: "b"},
                     turn=turn, first_flip_done=True, quiet=quiet,
                     cfg=cfg or RuleConfig())


class TestExpertEvaluate(unittest.TestCase):

    def setUp(self):
        self.w = EvalWeights()

    def test_commander_hegemony_bonus(self):
        """当敌方司令阵亡且我方司令在位时，我方估值应获得额外的制霸加成。"""
        # 局面 A: 双方司令均在
        dead_a = []
        st_a = mk({(5, 2): ("r", "SI", True), (7, 2): ("b", "SI", True)}, dead=dead_a)
        score_a = evaluate_expert(st_a, 0, self.w)

        # 局面 B: 敌方司令阵亡，我方司令称霸 (其他物质对称)
        dead_b = [Piece("b", Rank.SI, True)]
        st_b = mk({(5, 2): ("r", "SI", True)}, dead=dead_b)
        score_b = evaluate_expert(st_b, 0, self.w)

        # 局面 B 我方不仅有物质净胜，还有制霸额外加成
        self.assertGreater(score_b, score_a + 90.0)

    def test_trapped_piece_mobility_penalty(self):
        """被完全卡死、机动力为 0 的大子应受到死子惩罚。"""
        # 红司令在 (0, 0)，周围被己方地雷完全卡死在角落
        st_trapped = mk({
            (0, 0): ("r", "SI", True),
            (0, 1): ("r", "LEI", True),
            (1, 0): ("r", "LEI", True),
            (10, 2): ("b", "SI", True),  # 蓝司令在开阔地
        })

        # 红司令在开阔地 (5, 2)
        st_free = mk({
            (5, 2): ("r", "SI", True),
            (0, 1): ("r", "LEI", True),
            (1, 0): ("r", "LEI", True),
            (10, 2): ("b", "SI", True),
        })

        v_trapped = evaluate_expert(st_trapped, 0, self.w)
        v_free = evaluate_expert(st_free, 0, self.w)

        self.assertGreater(v_free, v_trapped, "开阔司令估值应显著高于受困死子")

    def test_gongbing_depletion_mine_security(self):
        """当敌方工兵全灭时，己方地雷和军旗价值应获得安全加成。"""
        # 局面 1: 敌方工兵均存活
        st1 = mk({(11, 1): ("r", "QI", True), (11, 0): ("r", "LEI", True)})
        v1 = evaluate_expert(st1, 0, self.w)

        # 局面 2: 敌方工兵 3 个全灭
        dead_all_gong = [Piece("b", Rank.GONG, True) for _ in range(3)]
        st2 = mk({(11, 1): ("r", "QI", True), (11, 0): ("r", "LEI", True)}, dead=dead_all_gong)
        v2 = evaluate_expert(st2, 0, self.w)

        # 扣除阵亡工兵本身的物质差 (3 * 42 = 126)，v2 的安全增益应当带来额外超额加分
        self.assertGreater(v2 - v1, 126.0 + 20.0, "工兵全灭后地雷/军旗应获得安全额外加成")


if __name__ == "__main__":
    unittest.main()
