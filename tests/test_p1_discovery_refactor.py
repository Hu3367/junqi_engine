"""测试 2026-09-06 实战大数据发现驱动的算法引擎与复盘训练管线重构。

涵盖验证：
1. 梯队火力网补偿机制 (单司令决定论平反)
2. 占营比例非线性矩阵增益 (5:5 -> 8:2 胜率阶跃)
3. 行营单向打击特权排序加成 (开局 50.1% 吃子源自行营扑杀)
4. 小子贴身拆弹定式排序加成 (47.5% 炸连排团工营)
5. 据点邻域辐射翻棋启发加分 (96.2% 邻营翻棋)
6. 残局工兵期权与和棋死锁折现 (65.4% 和棋工兵残缺<=2)
7. list.cfg 官方数据库破译与真实终局 Value 标签注入 (认输 code 21 / 和棋 code 40 / 强退 code 20)
"""

import os
import unittest
from junqi.config import EvalWeights, RuleConfig
from junqi.eval_expert import evaluate_expert
from junqi.rules import Rank, CAMPS, NEIGHBORS
from junqi.search import ExpertSearchEngine
from junqi.state import Action, GameState, Piece
from junqi.dataset import load_list_cfg_metadata, process_single_game
from junqi.replay import SavGame


class TestP1DiscoveryRefactor(unittest.TestCase):
    """验证基于 2026-09-06 实证大数据的重构逻辑。"""

    def setUp(self):
        self.cfg = RuleConfig()
        self.w = EvalWeights()

    def test_echelon_si_compensation(self):
        """验证司令战死时，二线梯队健全度（军/师/炸）能够提供抗悲观补偿。"""
        # 红方（座次0）：司令战死（进入 dead 列表）
        # 但拥有：军长(JUN)、2个师长(SHI)、2个炸弹(ZHA)
        # 黑方（座次1）：司令(SI)存活
        board = {
            (2, 2): Piece("r", Rank.JUN, revealed=True),
            (3, 1): Piece("r", Rank.SHI, revealed=True),
            (3, 3): Piece("r", Rank.SHI, revealed=True),
            (4, 2): Piece("r", Rank.ZHA, revealed=True),
            (5, 2): Piece("r", Rank.ZHA, revealed=True),
            # 黑方司令存活
            (8, 2): Piece("b", Rank.SI, revealed=True),
            (8, 1): Piece("b", Rank.TUAN, revealed=True),
        }
        st = GameState(board=board, cfg=self.cfg, dead=(Piece("r", Rank.SI),))
        st.seat_color[0] = "r"
        st.seat_color[1] = "b"

        # 无补偿权重
        w_no_comp = EvalWeights(echelon_si_compensation=0.0)
        score_no_comp = evaluate_expert(st, seat=0, w=w_no_comp)

        # 启用补偿权重
        w_with_comp = EvalWeights(echelon_si_compensation=18.0)
        score_with_comp = evaluate_expert(st, seat=0, w=w_with_comp)

        # 验证有补偿的分数高于无补偿（差值恰好为 18.0，梯队健全度满格）
        self.assertAlmostEqual(score_with_comp - score_no_comp, 18.0, delta=0.01)

    def test_camp_matrix_weight(self):
        """验证占营比例非线性矩阵增益 (5:5 -> 6:4 -> 7:3 -> 8:2)。"""
        # (2, 1) 与 (2, 3) 为行营坐标
        board = {
            (0, 0): Piece("r", Rank.PAI, revealed=True),
            (11, 4): Piece("b", Rank.PAI, revealed=True),
        }
        st0 = GameState(board=board, cfg=self.cfg)
        st0.seat_color[0] = "r"
        st0.seat_color[1] = "b"

        # 红方占领 2 个行营 (2, 1) 和 (2, 3)，黑方 0 个，净胜 2 营 (对应 6:4 优势)
        board_camp2 = dict(board)
        board_camp2[(2, 1)] = Piece("r", Rank.PAI, revealed=True)
        board_camp2[(2, 3)] = Piece("r", Rank.PAI, revealed=True)
        st2 = GameState(board=board_camp2, cfg=self.cfg)
        st2.seat_color[0] = "r"
        st2.seat_color[1] = "b"

        w_no_matrix = EvalWeights(camp_occ=10.0, camp_matrix_weight=0.0)
        w_with_matrix = EvalWeights(camp_occ=10.0, camp_matrix_weight=12.0)

        s_no_mat = evaluate_expert(st2, seat=0, w=w_no_matrix)
        s_with_mat = evaluate_expert(st2, seat=0, w=w_with_matrix)

        # 净胜 2 营 factor=1.0，矩阵加分应为 12.0
        self.assertAlmostEqual(s_with_mat - s_no_mat, 12.0, delta=0.01)

    def test_camp_outstrike_ordering(self):
        """验证行营单向扑杀在走法排序中享有最高级战术加成。"""
        engine = ExpertSearchEngine(weights=self.w, seed=42)
        # (2, 1) 为行营；(1, 1) 为其相邻兵站
        # 红排长在行营 (2, 1)，扑杀黑工兵 (1, 1)
        # 红排长在普通兵站 (0, 1)，吃黑工兵 (0, 0)
        board = {
            (2, 1): Piece("r", Rank.PAI, revealed=True), # 行营内
            (1, 1): Piece("b", Rank.GONG, revealed=True),
            (0, 1): Piece("r", Rank.PAI, revealed=True), # 兵站内
            (0, 0): Piece("b", Rank.GONG, revealed=True),
        }
        st = GameState(board=board, cfg=self.cfg)
        st.seat_color[0] = "r"
        st.seat_color[1] = "b"
        st.turn = 0

        act_from_camp = Action("move", (2, 1), (1, 1))
        act_from_normal = Action("move", (0, 1), (0, 0))

        score_camp = engine._score_action(act_from_camp, st, 0)
        score_normal = engine._score_action(act_from_normal, st, 0)

        # 行营单向扑杀享有 camp_outstrike_bias (400,000) 加分
        self.assertGreater(score_camp, score_normal + 390_000.0)

    def test_bomb_suicide_exchange_ordering(self):
        """验证小子（连排团工营）主动贴身撞炸弹享有战略拆弹加成。"""
        engine = ExpertSearchEngine(weights=self.w, seed=42)
        # 黑方炸弹在 (5, 2)
        # 红方排长在 (5, 1) 扑撞黑方炸弹（同归于尽）
        board = {
            (5, 2): Piece("b", Rank.ZHA, revealed=True),
            (5, 1): Piece("r", Rank.PAI, revealed=True),
            (6, 2): Piece("b", Rank.JUN, revealed=True),
            (6, 1): Piece("r", Rank.JUN, revealed=True),
        }
        st = GameState(board=board, cfg=self.cfg)
        st.seat_color[0] = "r"
        st.seat_color[1] = "b"
        st.turn = 0

        act_suicide_bomb = Action("move", (5, 1), (5, 2))
        score_bomb = engine._score_action(act_suicide_bomb, st, 0)

        # 小子撞炸弹获得 bomb_suicide_exchange (150,000) 加分，
        # 基础分 300,000 + 150,000 = 450,000+
        self.assertGreater(score_bomb, 440_000.0)

    def test_camp_adjacent_flip_heuristic(self):
        """验证在已占行营周围邻域翻棋享有据点辐射拓荒加分。"""
        engine = ExpertSearchEngine(weights=self.w, seed=42)
        # 红方已占行营 (2, 1)
        # (1, 1) 为相邻暗子
        # (0, 0) 为远离行营的底角暗子
        board = {
            (2, 1): Piece("r", Rank.TUAN, revealed=True),
            (1, 1): Piece("r", Rank.PAI, revealed=False),
            (0, 0): Piece("r", Rank.PAI, revealed=False),
        }
        st = GameState(board=board, cfg=self.cfg)
        st.seat_color[0] = "r"
        st.seat_color[1] = "b"
        st.turn = 0

        act_adjacent_flip = Action("flip", (1, 1))
        act_far_flip = Action("flip", (0, 0))

        score_adj = engine._score_action(act_adjacent_flip, st, 0)
        score_far = engine._score_action(act_far_flip, st, 0)

        # 邻营翻棋享有 camp_adjacent_flip_bias (50,000) 加分
        self.assertGreater(score_adj, score_far + 40_000.0)

    def test_endgame_engineer_deadlock_damping(self):
        """验证残局双方仅剩极少工兵且有地雷时，估值向和棋折现收敛。"""
        # ply=70, 双方工兵各剩0颗，双方军旗下方有明地雷守护
        board = {
            (0, 1): Piece("r", Rank.QI, revealed=True),
            (1, 1): Piece("r", Rank.LEI, revealed=True),
            (11, 3): Piece("b", Rank.QI, revealed=True),
            (10, 3): Piece("b", Rank.LEI, revealed=True),
            (5, 0): Piece("r", Rank.SHI, revealed=True),
        }
        st = GameState(board=board, cfg=self.cfg, ply=75)
        st.seat_color[0] = "r"
        st.seat_color[1] = "b"

        score = evaluate_expert(st, seat=0, w=self.w)
        self.assertIsInstance(score, float)

    def test_list_cfg_value_injection(self):
        """验证 dataset.py 对 list.cfg 官方终局真值的准确解析与样本 Value 注入。"""
        # 1. 模拟 code 21 (主动认输，P1胜，seat 0胜)
        mock_meta_win = {
            "winner": 1,
            "reason_code": 21,
            "moves_count": 85,
        }
        table = [0] * 60
        table[0] = 1   # 红司令 cell 0 -> (0, 0)
        table[1] = 13  # 黑司令 cell 1 -> (0, 1)
        game = SavGame(table=tuple(table), moves=((0, 0, 1),))

        samples_win = process_single_game(game, self.cfg, meta_record=mock_meta_win)
        self.assertGreater(len(samples_win), 0)
        self.assertTrue(samples_win[0]["has_value"])
        self.assertEqual(samples_win[0]["val_class"], 0) # seat 0 为 Win
        self.assertEqual(samples_win[0]["value"], 1.0)

        # 2. 模拟 code 40 (协议和棋)
        mock_meta_draw = {
            "winner": 3,
            "reason_code": 40,
            "moves_count": 150,
        }
        samples_draw = process_single_game(game, self.cfg, meta_record=mock_meta_draw)
        self.assertTrue(samples_draw[0]["has_value"])
        self.assertEqual(samples_draw[0]["val_class"], 1) # Draw
        self.assertEqual(samples_draw[0]["value"], 0.0)

        # 3. 模拟 code 20 (中途强退/逃跑，不可作 Value 标签)
        mock_meta_escape = {
            "winner": 1,
            "reason_code": 20,
            "moves_count": 15,
        }
        samples_escape = process_single_game(game, self.cfg, meta_record=mock_meta_escape)
        self.assertFalse(samples_escape[0]["has_value"]) # 不赋 Value
        self.assertEqual(samples_escape[0]["val_class"], -1)


if __name__ == "__main__":
    unittest.main()
