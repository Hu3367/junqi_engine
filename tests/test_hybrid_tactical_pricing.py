"""P2 战术定价验收测试：QSearch 深前向计算取代 1-ply 检测与加性硬打分。

对应 docs/REPLAY_ANALYSIS_AND_SYSTEMIC_IMPROVEMENT_PLAN_20260913.md §1.3 的
两大实战翻车场景，作为重构后的确定性回归验收：

1. 军长沿铁路开向明炸弹火力圈：1-ply 检测看不见"下一手炸弹扑杀"，
   QSearch 定价必须把该走法压到所有安全走法之下（地平线效应修复）；
2. 行营小子面对"炸弹炸雷 -> 工兵拔旗"的强制拆弹局面：加性硬打分时代
   "出营扣 150"导致拒不出营等死；QSearch 定价必须让"出营拆弹"自然涌现。

测试隔离说明：prior_weight=0 以隔离被测的战术定价通道（先验融合另有
HybridAgent 既有测试覆盖）；双方保留远处机动子避免困毙终局污染分差。
"""
from __future__ import annotations

import unittest

from junqi.config import RuleConfig
from junqi.hybrid_engine import HybridDecisionEngine
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece


class TestTacticalPricing(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def _engine(self, tactical_depth: int = 2) -> HybridDecisionEngine:
        """候选全覆盖定价、零先验干扰的测试引擎。"""
        return HybridDecisionEngine(model_path="models/best.pt", k_worlds=2,
                                    device="cpu", seed=42, candidate_k=64,
                                    tactical_depth=tactical_depth,
                                    prior_weight=0.0)

    def _score_map(self, st: GameState, tactical_depth: int = 2):
        eng = self._engine(tactical_depth)
        info = eng.evaluate_position(st)
        return {a: s for a, s in info["action_scores"]}, info["best_action"]

    def test_jun_toward_bomb_suppressed(self):
        """军长走向明炸弹火力圈的走法必须被定价压制（军长撞炸弹回归验收）。"""
        board = {
            (6, 2): Piece("r", Rank.JUN, True),   # 军长在铁路上
            (5, 1): Piece("b", Rank.ZHA, True),   # 敌明炸弹在铁路线上，可扑杀 (5,2)
            (10, 1): Piece("r", Rank.PAI, True),  # 红方远处机动子（防困毙终局污染）
            (1, 1): Piece("b", Rank.PAI, True),   # 蓝方远处机动子
            (0, 1): Piece("b", Rank.QI, True),
            (11, 1): Piece("r", Rank.QI, True),
        }
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0, cfg=self.cfg)

        scores, best = self._score_map(st)
        deadly = Action("move", (6, 2), (5, 2))  # 落点在炸弹一步扑杀圈内
        self.assertIn(deadly, scores)
        # 所有远离炸弹火力圈的铁路走法都必须优于送死走法
        for safe_to in [(6, 0), (6, 1), (6, 3), (6, 4)]:
            safe = Action("move", (6, 2), safe_to)
            self.assertIn(safe, scores)
            self.assertGreater(scores[safe], scores[deadly],
                               f"安全走法 {safe_to} 应优于撞弹火力圈 "
                               f"({scores[safe]=:.1f} vs {scores[deadly]=:.1f})")
        # 送死走法应为显著负值（军长换炸弹的子力期望亏损被定价捕获）
        self.assertLess(scores[deadly], -40.0)
        self.assertNotEqual(best, deadly)

    def test_camp_outstrike_bomb_sweep_emerges(self):
        """行营小子出营同尽敌明炸弹必须涌现为最优动作。

        场景还原（报告 §1.3-P4 缺陷）：蓝方炸弹威胁红方最后一颗地雷 (5,2)，
        地雷一旦被炸除，蓝方工兵 (1,1) 下一线即可长驱直入拔红旗 (0,1)。
        行营 (4,1) 内的排长是唯一能同尽炸弹的防守子——不出营 = 输棋。
        （旧代码对出营吃子一律扣 150 分，本测试即为该缺陷的回归验收。）
        """
        board = {
            (4, 1): Piece("r", Rank.PAI, True),   # 排长驻行营 (4,1)，唯一拆弹子
            (5, 1): Piece("b", Rank.ZHA, True),   # 敌明炸弹威胁地雷 (5,2)
            (5, 2): Piece("r", Rank.LEI, True),   # 红方最后一颗地雷（另两颗已阵亡）
            (1, 1): Piece("b", Rank.GONG, True),  # 蓝方工兵已就位，清雷后直取军旗
            (0, 1): Piece("r", Rank.QI, True),    # 红方军旗暴露在工兵攻击线上
            (10, 1): Piece("r", Rank.PAI, True),  # 红方远处机动子（鞭长莫及）
        }
        # 红方已阵亡两颗地雷 => 场上 (5,2) 是最后一颗，炸除即解锁吃旗
        dead = [Piece("r", Rank.LEI), Piece("r", Rank.LEI)]
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                       dead=dead, cfg=self.cfg)

        # 拆弹线深 4 ply（出营 -> 炸雷 -> 缓着 -> 拔旗），需 tactical_depth=4 才可见
        scores, best = self._score_map(st, tactical_depth=4)
        sweep = Action("move", (4, 1), (5, 1))  # 出营同尽炸弹
        self.assertIn(sweep, scores)
        # 所有不拆弹的消极走法都导致"炸雷 -> 拔旗"的强制败线，必须被深度定价压制
        for passive_to in [(3, 0), (3, 1), (3, 2), (4, 0), (4, 2), (5, 0)]:
            passive = Action("move", (4, 1), passive_to)
            self.assertIn(passive, scores)
            self.assertGreater(scores[sweep], scores[passive],
                               f"出营拆弹应优于消极走法 {passive_to} "
                               f"({scores[sweep]=:.1f} vs {scores[passive]=:.1f})")
        self.assertEqual(best, sweep)


if __name__ == "__main__":
    unittest.main()
