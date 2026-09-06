"""全盘 10 个行营拓扑参数化模糊测试 (Camp Topology Fuzz Testing)。

严格验证军棋棋盘对称几何下，行营“据点庇护”、“占营优于吃小子”与“反诱杀”规则的不变式：
1. 全盘 10 个行营任意坐标，驻守小子出营吃诱饵必受严惩；
2. 存在反杀或丢营威胁时，AI 决策绝对拦截弃营吃小子；
3. 营间机动（camp_to_camp）与进营（enters_camp）受到正向战术激励，不受弃营惩罚。
"""
from __future__ import annotations

import unittest
import pytest

from junqi.config import RuleConfig
from junqi.hybrid_engine import HybridDecisionEngine
from junqi.rules import CAMPS, HQS, NEIGHBORS, Rank, is_camp, is_hq
from junqi.state import Action, GameState, Piece


class TestCampTopologyFuzz(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()
        self.engine = HybridDecisionEngine(k_worlds=2, seed=42)

    def test_all_10_camps_exist(self):
        self.assertEqual(len(CAMPS), 10, "军棋标准盘面必须恰好有 10 个行营")

    def test_all_camps_abandon_bait_penalized(self):
        """遍历全盘 10 个行营，验证驻守工兵/排长/营长出营吃诱饵并面临反杀时，
        在所有行营坐标下均必须受到严厉扣分，且决策引擎绝不出营。"""
        all_camps = sorted(list(CAMPS))

        for camp_pos in all_camps:
            # 找到行营的一个非行营、非大本营邻居作为诱饵兵站
            bait_candidates = [n for n in NEIGHBORS[camp_pos] if not is_camp(n) and not is_hq(n)]
            self.assertTrue(len(bait_candidates) > 0, f"行营 {camp_pos} 必须拥有邻接兵站")
            bait_pos = bait_candidates[0]

            # 找到诱饵兵站的一个邻居作为敌方反杀伏兵位置 (不能是行营、大本营或诱饵自身)
            killer_candidates = [
                n for n in NEIGHBORS[bait_pos]
                if n != camp_pos and not is_camp(n) and not is_hq(n)
            ]
            self.assertTrue(len(killer_candidates) > 0, f"诱饵格 {bait_pos} 必须有外部邻接格放置敌军")
            killer_pos = killer_candidates[0]

            # 另外放置一枚可翻暗子，确保 AI 拥有合理的替代动作
            dark_pos = (0, 0) if (0, 0) not in (camp_pos, bait_pos, killer_pos) else (11, 4)

            # 测试不同驻防兵种: 工兵 (Rank.GONG)、排长 (Rank.PAI)、营长 (Rank.YING)
            test_ranks = [
                (Rank.GONG, Rank.LEI, Rank.LIAN),  # 工兵弃营挖雷遭连长反杀
                (Rank.PAI, Rank.PAI, Rank.JUN),    # 排长弃营吃排长遭军长反杀
                (Rank.YING, Rank.PAI, Rank.SI),    # 营长弃营吃排长遭司令反杀
            ]

            for mover_rank, bait_rank, killer_rank in test_ranks:
                board = {
                    camp_pos: Piece("r", mover_rank, revealed=True),
                    bait_pos: Piece("b", bait_rank, revealed=True),
                    killer_pos: Piece("b", killer_rank, revealed=True),
                    dark_pos: Piece("b", Rank.PAI, revealed=False),
                }

                st = GameState(board=board, dead=[], seat_color={0: "r", 1: "b"}, turn=0,
                               first_flip_done=True, ply=10, winner=None, win_reason=None, cfg=self.cfg)

                abandon_act = Action("move", camp_pos, bait_pos)
                self.assertIn(abandon_act, st.legal_actions(),
                              f"动作 {abandon_act} 在行营 {camp_pos} 必须合法")

                scored = self.engine.choose_actions(st, topn=len(st.legal_actions()))
                score_dict = {a: s for a, s in scored}

                self.assertIn(abandon_act, score_dict)
                act_score = score_dict[abandon_act]

                # 断言 1: 弃营吃诱饵并面临反杀的动作必须大幅负分 (< -150)
                self.assertLess(
                    act_score, -150.0,
                    f"行营 {camp_pos} 驻军 {mover_rank} 出营吃 {bait_rank} 遭 {killer_rank} 反杀未受严惩！得分: {act_score}"
                )

                # 断言 2: 最佳决策决不能是弃营自杀动作
                best_act = scored[0][0]
                self.assertNotEqual(
                    best_act, abandon_act,
                    f"行营 {camp_pos} 最佳动作误选弃营吃小子 {abandon_act}，得分: {act_score}"
                )

    def test_camp_to_camp_not_penalized(self):
        """验证行营之间的战术调动 (如 (7,3) -> (8,2)) 享有 camp_to_camp 战术奖励，绝不触发弃营惩罚。"""
        # (7, 3) 与 (8, 2) 互为行营邻居
        board = {
            (7, 3): Piece("r", Rank.GONG, revealed=True),
            (0, 0): Piece("b", Rank.PAI, revealed=False),
        }
        st = GameState(board=board, dead=[], seat_color={0: "r", 1: "b"}, turn=0,
                       first_flip_done=True, ply=10, winner=None, win_reason=None, cfg=self.cfg)

        c2c_act = Action("move", (7, 3), (8, 2))
        self.assertIn(c2c_act, st.legal_actions())

        scored = self.engine.choose_actions(st, topn=len(st.legal_actions()))
        score_dict = {a: s for a, s in scored}

        self.assertIn(c2c_act, score_dict)
        # 验证营间转移为正向加分 (>= 15.0)
        self.assertGreaterEqual(score_dict[c2c_act], 15.0, "营间安全转移必须获得正向战术加分")


if __name__ == "__main__":
    unittest.main()
