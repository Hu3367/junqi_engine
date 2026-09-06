"""P1 传统估值与搜索增强单元测试 (动态炸弹、等比子力梯度、地雷护旗与 PVS 剪枝)。"""
from __future__ import annotations

import unittest

from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.eval_expert import evaluate_expert
from junqi.rules import Rank
from junqi.search import ExpertSearchEngine
from junqi.state import Action, GameState, Piece


class TestP1TraditionalEnhancement(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def test_dynamic_bomb_valuation_scaling(self):
        """测试动态炸弹估值随敌方最高军衔动态缩放。"""
        w = EvalWeights.apk_weights()
        self.assertTrue(w.use_dynamic_bomb)
        self.assertAlmostEqual(w.bomb_ratio, 1.0 / 3.0)

        # 局面 1: 红方有炸弹，蓝方有司令 (2560)
        st_with_si = GameState(board={
            (5, 2): Piece("r", Rank.ZHA, True),
            (11, 1): Piece("r", Rank.QI, True),
            (6, 2): Piece("b", Rank.SI, True),
            (0, 1): Piece("b", Rank.QI, True),
        }, dead=[
            # 蓝方司令在场，其他大官全部阵亡
            Piece("b", Rank.JUN, True), Piece("b", Rank.SHI, True), Piece("b", Rank.SHI, True),
        ], turn=0, cfg=self.cfg)
        st_with_si.seat_color[0], st_with_si.seat_color[1] = "r", "b"

        # 局面 2: 红方有炸弹，蓝方无司令/军长，最大只剩师长 (640)
        st_with_shi = GameState(board={
            (5, 2): Piece("r", Rank.ZHA, True),
            (11, 1): Piece("r", Rank.QI, True),
            (6, 2): Piece("b", Rank.SHI, True),
            (0, 1): Piece("b", Rank.QI, True),
        }, dead=[
            Piece("b", Rank.SI, True), Piece("b", Rank.JUN, True), Piece("b", Rank.SHI, True),
        ], turn=0, cfg=self.cfg)
        st_with_shi.seat_color[0], st_with_shi.seat_color[1] = "r", "b"

        score_si = evaluate_expert(st_with_si, seat=0, w=w)
        score_shi = evaluate_expert(st_with_shi, seat=0, w=w)

        # 敌方有司令时，红方炸弹价值(2560/3=853)远高于敌方仅剩师长时的炸弹价值(640/3=213)
        # 故红炸弹单体的估值分差应当显著体现
        # 验证在 APK 权重下炸弹在面对司令时的估值基准
        self.assertGreater(score_si, score_shi - (2560 - 640))

    def test_landmine_flag_guard_bonus(self):
        """测试地雷护旗关键阵地加分。"""
        w_with_bonus = EvalWeights(mine_flag_guard_bonus=80.0)
        w_without_bonus = EvalWeights(mine_flag_guard_bonus=0.0)

        st_guarded = GameState(board={
            (11, 1): Piece("r", Rank.QI, True),
            (10, 1): Piece("r", Rank.LEI, True),
            (5, 2): Piece("r", Rank.TUAN, True),
            (0, 1): Piece("b", Rank.QI, True),
            (6, 2): Piece("b", Rank.TUAN, True),
        }, turn=0, cfg=self.cfg)
        st_guarded.seat_color[0], st_guarded.seat_color[1] = "r", "b"

        score_with = evaluate_expert(st_guarded, seat=0, w=w_with_bonus)
        score_without = evaluate_expert(st_guarded, seat=0, w=w_without_bonus)

        # 净增益严格为 80.0
        self.assertAlmostEqual(score_with - score_without, 80.0, places=1)

    def test_pvs_search_execution(self):
        """测试 PVS 搜索正确执行并产生合理决策。"""
        engine = ExpertSearchEngine(weights=EvalWeights.apk_weights(), seed=42)

        # 构造一步吃旗胜势局面: 工兵在 (1, 1)，军旗在 (0, 1)，正交一步直达
        st = GameState(board={
            (1, 1): Piece("r", Rank.GONG, True),
            (11, 1): Piece("r", Rank.QI, True),
            (0, 1): Piece("b", Rank.QI, True),
        }, dead=[
            Piece("b", Rank.LEI, True), Piece("b", Rank.LEI, True), Piece("b", Rank.LEI, True)
        ], turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        best_act, score, stats = engine.search(st, max_depth=2, time_limit_ms=500)
        self.assertIsNotNone(best_act)
        self.assertEqual(best_act.kind, "move")
        self.assertEqual(best_act.to, (0, 1))  # 必须选择一步扛旗
        self.assertGreater(stats.nodes, 0)
        self.assertTrue(hasattr(stats, "pvs_researches"))


if __name__ == "__main__":
    unittest.main()
