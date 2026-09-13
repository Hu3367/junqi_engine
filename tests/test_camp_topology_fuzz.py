"""全盘 10 个行营拓扑参数化模糊测试 (Camp Topology Fuzz Testing)。

2026-09-13 P2 重构后契约（加性硬打分常数已废除，见
docs/REPLAY_ANALYSIS_AND_SYSTEMIC_IMPROVEMENT_PLAN_20260913.md §4.1）：

1. 【地平线修复不变式】同一"出营吃诱饵"动作，在存在真实可执行反杀伏兵时
   的 QSearch 定价，必须显著低于无伏兵对照局面——直接验证深度搜索确实
   看见了下一手的反杀线（旧 1-ply 检测的地平线盲区正是本缺陷根源）；
2. 【反杀真实性】伏兵必须满足 battle(killer, mover) ∈ {attacker_wins,
   both_die}（旧版用例存在"连长反杀工兵"的前提错误——defender_wins
   不构成反杀），且伏兵不放在 row0/row11 冻结位；
3. 【营间机动】camp_to_camp 由搜索按局面定价，不受弃营式硬惩罚压制
   （旧契约的 ±15 固定加减分已废除）。

测试设计说明：诱饵紧邻行营时，任何非吃子走法都会丢营（诱饵下手进营），
"吃诱饵被反杀"与"弃营"的真实代价相近、孰优取决于具体子力与先验，
故"绝不吃诱饵"不构成普遍不变式——旧版靠 -150 硬扣制造了该假象。
"""
from __future__ import annotations

import unittest

from junqi.config import RuleConfig
from junqi.hybrid_engine import HybridDecisionEngine
from junqi.rules import (ATTACKER_WINS, BOTH_DIE, CAMPS, HQS, NEIGHBORS,
                         Rank, battle, is_camp, is_hq)
from junqi.state import Action, GameState, Piece

# 暗子必须放在可动格：row0/row11 为度数 0 冻结位，翻完最后一枚暗子会
# 直接触发困毙终局判定，把翻棋价值抬到 ±WIN_SCORE 量级（棋盘病理）
DARK_CANDIDATES = [(5, 2), (6, 2), (5, 3), (6, 1), (6, 3)]

# 红方远处机动子候选：与诱饵/伏兵/行营保持距离，防止交换触发
# "一方无子可动"的困毙终局，掩盖地平线定价差（须非行营、非冻结位）
RESERVE_CANDIDATES = [(5, 2), (6, 2), (5, 3), (6, 1), (6, 3), (1, 2), (10, 2)]


def _pick_pos(candidates, used) -> tuple:
    for pos in candidates:
        if pos not in used:
            return pos
    raise AssertionError("候选位耗尽")


def _far_from(pos, others) -> bool:
    """曼哈顿距离均 > 1（不与其他关键格相邻）。"""
    return all(abs(pos[0] - o[0]) + abs(pos[1] - o[1]) > 1 for o in others)


class TestCampTopologyFuzz(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()
        self.engine = HybridDecisionEngine(k_worlds=2, seed=42, candidate_k=64)

    def test_all_10_camps_exist(self):
        self.assertEqual(len(CAMPS), 10, "军棋标准盘面必须恰好有 10 个行营")

    def test_all_camps_abandon_bait_horizon_fixed(self):
        """遍历全盘 10 个行营 x 三组真实反杀组合：出营吃诱饵动作在有伏兵时
        的定价必须显著低于无伏兵对照（QSearch 看见反杀线 = 地平线修复）。"""
        for camp_pos in sorted(CAMPS):
            bait_candidates = [n for n in NEIGHBORS[camp_pos]
                               if not is_camp(n) and not is_hq(n)]
            self.assertTrue(len(bait_candidates) > 0, f"行营 {camp_pos} 必须拥有邻接兵站")
            bait_pos = bait_candidates[0]

            # 反杀伏兵：诱饵邻居，排除行营/大本营/冻结位，且与诱饵不同格
            killer_candidates = [
                n for n in NEIGHBORS[bait_pos]
                if n != camp_pos and not is_camp(n) and not is_hq(n)
                and n[0] not in (0, 11)
            ]
            self.assertTrue(len(killer_candidates) > 0,
                            f"诱饵格 {bait_pos} 必须有可机动的邻接伏兵格")
            killer_pos = killer_candidates[0]

            test_ranks = [
                (Rank.GONG, Rank.LEI, Rank.TUAN),  # 工兵挖雷（存活）遭团长反杀
                (Rank.LIAN, Rank.PAI, Rank.SI),    # 连长吃排长（存活）遭司令反杀
            ]
            # 注：同归于尽组合（如排长×排长）不纳入——兑子后移动子已阵亡，
            # 不存在"反杀"环节；且若红方仅此一子，交换直接触发困毙终局。

            for mover_rank, bait_rank, killer_rank in test_ranks:
                self.assertEqual(battle(mover_rank, bait_rank), ATTACKER_WINS,
                                 f"{mover_rank} 必须能吃 {bait_rank} 且存活才构成诱饵")
                self.assertIn(battle(killer_rank, mover_rank),
                              (ATTACKER_WINS, BOTH_DIE),
                              f"{killer_rank} 必须能反杀 {mover_rank} 才构成伏杀")

                base_board = {
                    camp_pos: Piece("r", mover_rank, revealed=True),
                    bait_pos: Piece("b", bait_rank, revealed=True),
                    killer_pos: Piece("b", killer_rank, revealed=True),
                }
                # 红方远处机动子：远离诱饵/伏兵/行营，防止困毙终局污染定价差
                used = set(base_board)
                reserve_pos = next(p for p in RESERVE_CANDIDATES
                                   if p not in used and _far_from(p, used))
                base_board[reserve_pos] = Piece("r", Rank.PAI, revealed=True)
                used.add(reserve_pos)
                dark_pos = _pick_pos(DARK_CANDIDATES, used)
                base_board[dark_pos] = Piece("b", Rank.PAI, revealed=False)

                st = GameState(board=dict(base_board), dead=[],
                               seat_color={0: "r", 1: "b"}, turn=0,
                               first_flip_done=True, ply=10, winner=None,
                               win_reason=None, cfg=self.cfg)
                abandon_act = Action("move", camp_pos, bait_pos)
                self.assertIn(abandon_act, st.legal_actions(),
                              f"动作 {abandon_act} 在行营 {camp_pos} 必须合法")

                scored = self.engine.choose_actions(st, topn=64)
                score_with = {a: s for a, s in scored}[abandon_act]

                # 对照组：移除反杀伏兵，同一吃子动作重新定价
                ctrl_board = dict(base_board)
                del ctrl_board[killer_pos]
                st_ctrl = GameState(board=ctrl_board, dead=[],
                                    seat_color={0: "r", 1: "b"}, turn=0,
                                    first_flip_done=True, ply=10, winner=None,
                                    win_reason=None, cfg=self.cfg)
                scored_ctrl = self.engine.choose_actions(st_ctrl, topn=64)
                score_without = {a: s for a, s in scored_ctrl}[abandon_act]

                self.assertLess(
                    score_with, score_without - 5.0,
                    f"行营 {camp_pos} {mover_rank.name} 吃 {bait_rank.name}："
                    f"有 {killer_rank.name} 伏兵定价 {score_with:.1f} 应显著低于"
                    f"无伏兵对照 {score_without:.1f}（QSearch 未看见反杀线？）")

    def test_camp_to_camp_not_penalized(self):
        """验证行营之间的战术调动 (如 (7,3) -> (8,2)) 由搜索按局面定价，
        不受弃营式硬惩罚压制（旧契约的 ±15 固定加减分已废除）。"""
        board = {
            (7, 3): Piece("r", Rank.GONG, revealed=True),
            (5, 2): Piece("b", Rank.PAI, revealed=False),
        }
        st = GameState(board=board, dead=[], seat_color={0: "r", 1: "b"}, turn=0,
                       first_flip_done=True, ply=10, winner=None, win_reason=None, cfg=self.cfg)

        c2c_act = Action("move", (7, 3), (8, 2))
        self.assertIn(c2c_act, st.legal_actions())

        scored = self.engine.choose_actions(st, topn=64)
        score_dict = {a: s for a, s in scored}

        self.assertIn(c2c_act, score_dict)
        ranked = [a for a, _ in scored]
        # 营间机动不得是唯一垫底动作（弃营式硬惩罚已废除）
        self.assertLess(ranked.index(c2c_act), len(ranked) - 1,
                        "营间转移不应被系统性压制到候选末位")
