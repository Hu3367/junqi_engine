import unittest
from junqi.state import GameState, Piece, Action
from junqi.rules import Rank, COLORS
from junqi.tactical_sampler import (
    state_to_dict, dict_to_state, action_to_dict, dict_to_action,
    is_active_tactical_contact, score_to_class, WIN_SCORE
)

RED = COLORS[0]
BLUE = COLORS[1]


class TestTacticalSampler(unittest.TestCase):
    def test_state_serialization_roundtrip(self):
        # 构造典型中局盘面
        board = {
            (0, 0): Piece(color=BLUE, rank=Rank.SI, revealed=True),
            (5, 2): Piece(color=RED, rank=Rank.JUN, revealed=True),
            (6, 2): Piece(color=RED, rank=Rank.SHI, revealed=False),
        }
        st = GameState(
            board=board,
            seat_color={0: RED, 1: BLUE},
            turn=0,
            ply=42,
            quiet=5,
        )

        st_dict = state_to_dict(st)
        self.assertEqual(st_dict["turn"], 0)
        self.assertEqual(st_dict["ply"], 42)
        self.assertEqual(st_dict["quiet"], 5)
        self.assertEqual(len(st_dict["board"]), 3)

        restored_st = dict_to_state(st_dict)
        self.assertEqual(restored_st.turn, 0)
        self.assertEqual(restored_st.ply, 42)
        self.assertEqual(restored_st.quiet, 5)
        self.assertEqual(len(restored_st.board), 3)

        # 检查棋子属性
        pc = restored_st.board[(0, 0)]
        self.assertEqual(pc.color, BLUE)
        self.assertEqual(pc.rank, Rank.SI)
        self.assertTrue(pc.revealed)

        pc_dark = restored_st.board[(6, 2)]
        self.assertEqual(pc_dark.color, RED)
        self.assertEqual(pc_dark.rank, Rank.SHI)
        self.assertFalse(pc_dark.revealed)

    def test_action_serialization_roundtrip(self):
        act_move = Action(kind="move", frm=(5, 2), to=(4, 2))
        act_dict = action_to_dict(act_move)
        self.assertEqual(act_dict["kind"], "move")
        self.assertEqual(tuple(act_dict["frm"]), (5, 2))
        self.assertEqual(tuple(act_dict["to"]), (4, 2))

        restored_move = dict_to_action(act_dict)
        self.assertEqual(restored_move, act_move)

        act_flip = Action(kind="flip", frm=(6, 2), to=None)
        act_dict_flip = action_to_dict(act_flip)
        self.assertEqual(act_dict_flip["kind"], "flip")
        restored_flip = dict_to_action(act_dict_flip)
        self.assertEqual(restored_flip, act_flip)

    def test_score_to_class(self):
        self.assertEqual(score_to_class(WIN_SCORE - 100), 0)  # 胜
        self.assertEqual(score_to_class(500.0), 0)            # 优势胜 (>= 300)
        self.assertEqual(score_to_class(50.0), 1)             # 僵持和
        self.assertEqual(score_to_class(-50.0), 1)            # 僵持和
        self.assertEqual(score_to_class(-400.0), 2)           # 劣势负 (<= -300)
        self.assertEqual(score_to_class(-WIN_SCORE), 2)       # 负

    def test_active_tactical_contact(self):
        # 1. 双方皆有明子，且有移动走法
        board = {
            (5, 2): Piece(color=RED, rank=Rank.JUN, revealed=True),   # 红军长
            (4, 2): Piece(color=BLUE, rank=Rank.SHI, revealed=True),  # 蓝师长
        }
        st = GameState(board=board, seat_color={0: RED, 1: BLUE}, turn=0, ply=30, quiet=2)
        self.assertTrue(is_active_tactical_contact(st))

        # 2. 只有暗子，无明子，不是交火状态
        board_dark = {
            (5, 2): Piece(color=RED, rank=Rank.JUN, revealed=False),
            (4, 2): Piece(color=BLUE, rank=Rank.SHI, revealed=False),
        }
        st_dark = GameState(board=board_dark, seat_color={0: RED, 1: BLUE}, turn=0, ply=1, quiet=0)
        self.assertFalse(is_active_tactical_contact(st_dark))


if __name__ == "__main__":
    unittest.main()
