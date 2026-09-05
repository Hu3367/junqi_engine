"""Zobrist 哈希与置换表 (Transposition Table) 单元测试。"""
from __future__ import annotations

import unittest

from junqi.config import RuleConfig
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece
from junqi.tt import FLAG_EXACT, FLAG_LOWER_BOUND, FLAG_UPPER_BOUND, TranspositionTable
from junqi.zobrist import compute_zobrist


def mk(board_spec, dead=(), turn=0, seat_color=None, quiet=0, cfg=None):
    board = {pos: Piece(color, Rank[rk], revealed)
             for pos, (color, rk, revealed) in board_spec.items()}
    return GameState(board=board, dead=dead,
                     seat_color=seat_color or {0: "r", 1: "b"},
                     turn=turn, first_flip_done=True, quiet=quiet,
                     cfg=cfg or RuleConfig())


class TestZobristAndTT(unittest.TestCase):

    def test_zobrist_determinism(self):
        """相同局面必须生成完全相同的 Zobrist 哈希。"""
        st1 = mk({(5, 2): ("r", "LIAN", True), (10, 0): ("b", "LIAN", True)})
        st2 = mk({(5, 2): ("r", "LIAN", True), (10, 0): ("b", "LIAN", True)})
        self.assertEqual(compute_zobrist(st1), compute_zobrist(st2))

    def test_zobrist_distinction(self):
        """走子后状态哈希必须发生变化。"""
        st = mk({(5, 2): ("r", "LIAN", True), (10, 0): ("b", "LIAN", True)})
        h1 = compute_zobrist(st)
        child = st.apply(Action("move", (5, 2), (4, 2)))
        h2 = compute_zobrist(child)
        self.assertNotEqual(h1, h2)

    def test_zobrist_turn_distinction(self):
        """相同棋盘但轮次不同，哈希必须不同。"""
        st1 = mk({(5, 2): ("r", "LIAN", True)}, turn=0)
        st2 = mk({(5, 2): ("r", "LIAN", True)}, turn=1)
        self.assertNotEqual(compute_zobrist(st1), compute_zobrist(st2))

    def test_tt_exact_store_and_lookup(self):
        """置换表存入与精确命中测试。"""
        tt = TranspositionTable(size_power=10)
        key = 0x123456789ABCDEF0
        act = Action("move", (5, 2), (4, 2))

        tt.store(key, depth=3, score=150.0, flag=FLAG_EXACT, best_action=act)

        # 深度足够时应命中精确分
        score, best_act = tt.lookup(key, depth=2, alpha=-1000.0, beta=1000.0)
        self.assertEqual(score, 150.0)
        self.assertEqual(best_act, act)

        # 深度不足时只返回 best_action，不剪枝
        score, best_act = tt.lookup(key, depth=4, alpha=-1000.0, beta=1000.0)
        self.assertIsNone(score)
        self.assertEqual(best_act, act)

    def test_tt_bounds_cutoffs(self):
        """置换表边界剪枝测试 (Lower bound & Upper bound)。"""
        tt = TranspositionTable(size_power=10)
        key = 0xABCDEF0123456789
        act = Action("move", (1, 1), (1, 2))

        # 下界 (Cut-node)
        tt.store(key, depth=3, score=200.0, flag=FLAG_LOWER_BOUND, best_action=act)
        # 当 beta <= score 时触发剪枝
        score, _ = tt.lookup(key, depth=3, alpha=0.0, beta=150.0)
        self.assertEqual(score, 200.0)
        # 当 beta > score 时不剪枝
        score, _ = tt.lookup(key, depth=3, alpha=0.0, beta=250.0)
        self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()
