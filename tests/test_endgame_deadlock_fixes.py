import os
import random
import unittest
from collections import Counter

from junqi.state import GameState, Piece, Action
from junqi.config import RuleConfig, EvalWeights
from junqi.rules import Rank, COMPOSITION, COLORS
from junqi.analysis import is_dead_draw
from junqi.ai import evaluate_expert
from junqi.hybrid_engine import HybridDecisionEngine


class TestEndgameDeadlockFixes(unittest.TestCase):
    def setUp(self):
        self.cfg = RuleConfig()

    def _build_full_dead(self, board):
        counts = Counter({(c, r): n for c in COLORS for r, n in COMPOSITION.items()})
        for pc in board.values():
            counts[(pc.color, pc.rank)] -= 1
        dead = []
        for (c, r), cnt in counts.items():
            for _ in range(cnt):
                dead.append(Piece(c, r, revealed=True))
        return dead

    def test_user_screenshot_endgame_not_dead_draw(self):
        """测试 2026-09-06 用户截图局面：蓝方双师长压制红方营连排残局，绝非必和。"""
        board = {
            # 红方
            (3, 0): Piece("r", Rank.QI, revealed=True),
            (4, 0): Piece("r", Rank.LEI, revealed=True),
            (1, 4): Piece("r", Rank.LEI, revealed=True),
            (11, 2): Piece("r", Rank.LEI, revealed=True),
            (3, 3): Piece("r", Rank.PAI, revealed=True),
            (7, 1): Piece("r", Rank.LIAN, revealed=True),
            (11, 3): Piece("r", Rank.YING, revealed=True),
            # 蓝方
            (7, 3): Piece("b", Rank.QI, revealed=True),
            (5, 4): Piece("b", Rank.LEI, revealed=True),
            (2, 2): Piece("b", Rank.PAI, revealed=True),
            (7, 4): Piece("b", Rank.PAI, revealed=True),
            (8, 2): Piece("b", Rank.SHI, revealed=True),
            (9, 3): Piece("b", Rank.YING, revealed=True),
            (10, 2): Piece("b", Rank.SHI, revealed=True),
        }
        dead = self._build_full_dead(board)
        st = GameState(board=board, dead=dead, seat_color={0: "r", 1: "b"},
                       turn=0, ply=30, quiet=2, first_flip_done=True, cfg=self.cfg)

        # 1. 绝不能被判定为必和死局
        is_draw, reason = is_dead_draw(st)
        self.assertFalse(is_draw, f"Should not be draw, got reason: {reason}")

        # 2. 专家估值必须正确反映蓝方胜势与红方劣势
        score_red = evaluate_expert(st, seat=0)
        score_blue = evaluate_expert(st, seat=1)
        self.assertLess(score_red, -50.0, f"Red should be heavily losing, got {score_red}")
        self.assertGreater(score_blue, 50.0, f"Blue should be heavily winning, got {score_blue}")

        # 3. HybridDecisionEngine 必须成功生成走法，绝不能返回空列表卡死
        mp = "models/best.pt" if os.path.exists("models/best.pt") else "models/bc_best.pt"
        engine = HybridDecisionEngine(model_path=mp, k_worlds=4)
        scored = engine.choose_actions(st, topn=3)
        self.assertTrue(len(scored) > 0, "choose_actions must return non-empty list")
        best_act = scored[0][0]
        self.assertIn(best_act, st.legal_actions(), "Chosen action must be legal")

        # 4. evaluate_position 结构必须完备且包含 action_scores
        info = engine.evaluate_position(st)
        self.assertIn("action_scores", info)
        self.assertTrue(len(info["action_scores"]) > 0)

    def test_1v1_equal_rank_dead_draw(self):
        """测试 1v1 同级子力（如连长对连长，双无工兵）属于理论必和。"""
        board = {
            (5, 2): Piece("r", Rank.LIAN, revealed=True),
            (6, 2): Piece("b", Rank.LIAN, revealed=True),
            (0, 1): Piece("r", Rank.QI, revealed=True),
            (0, 2): Piece("r", Rank.LEI, revealed=True),
            (11, 1): Piece("b", Rank.QI, revealed=True),
            (11, 2): Piece("b", Rank.LEI, revealed=True),
        }
        dead = self._build_full_dead(board)
        st = GameState(board=board, dead=dead, seat_color={0: "r", 1: "b"},
                       turn=0, ply=60, quiet=10, first_flip_done=True, cfg=self.cfg)

        is_draw, reason = is_dead_draw(st)
        self.assertTrue(is_draw)
        self.assertEqual(reason, "1v1_equal_rank_deadlock")

        # HybridDecisionEngine 在必和时也必须正常返回合法走法，不能卡死
        engine = HybridDecisionEngine(model_path=None, k_worlds=2)
        scored = engine.choose_actions(st, topn=3)
        self.assertTrue(len(scored) > 0)
        self.assertIn(scored[0][0], st.legal_actions())

    def test_mines_physical_partition_dead_draw(self):
        """测试全盘地雷将双方多子物理彻底隔绝时的断连必和。"""
        # 在第 5 行布置一整排地雷，彻底阻断上下半盘
        board = {
            (4, 2): Piece("r", Rank.SI, revealed=True),
            (3, 2): Piece("r", Rank.LIAN, revealed=True),
            (7, 2): Piece("b", Rank.SHI, revealed=True),
            (8, 2): Piece("b", Rank.PAI, revealed=True),
            (5, 0): Piece("r", Rank.LEI, revealed=True),
            (5, 1): Piece("r", Rank.LEI, revealed=True),
            (5, 2): Piece("r", Rank.LEI, revealed=True),
            (5, 3): Piece("r", Rank.LEI, revealed=True),
            (5, 4): Piece("r", Rank.LEI, revealed=True),
            (0, 1): Piece("r", Rank.QI, revealed=True),
            (11, 1): Piece("b", Rank.QI, revealed=True),
        }
        dead = self._build_full_dead(board)
        st = GameState(board=board, dead=dead, seat_color={0: "r", 1: "b"},
                       turn=0, ply=40, quiet=5, first_flip_done=True, cfg=self.cfg)

        is_draw, reason = is_dead_draw(st)
        self.assertTrue(is_draw)
        self.assertEqual(reason, "mines_partition_board_disconnected")

    def test_quiet_moves_limit_dead_draw(self):
        """测试达到无吃子限步（70手）时直接判和。"""
        board = {
            (5, 2): Piece("r", Rank.SI, revealed=True),
            (6, 2): Piece("b", Rank.JUN, revealed=True),
        }
        dead = self._build_full_dead(board)
        st = GameState(board=board, dead=dead, seat_color={0: "r", 1: "b"},
                       turn=0, ply=100, quiet=70, first_flip_done=True, cfg=self.cfg)

        is_draw, reason = is_dead_draw(st)
        self.assertTrue(is_draw)
        self.assertEqual(reason, "quiet_moves_limit_reached")


if __name__ == "__main__":
    unittest.main()
