"""P1 阶段静态搜索 (QSearch) 深度延伸、长程反杀识别、分支因子受控与 IDS 动态早停专项测试。"""
from __future__ import annotations

import unittest

from junqi.ai import ApkNativeAgent, ExpertAgent
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.rules import Rank
from junqi.search import ExpertSearchEngine
from junqi.state import Action, GameState, Piece


def mk_state(board_spec, dead=(), turn=0, seat_color=None, cfg=None):
    board = {pos: Piece(color, Rank[rk], revealed)
             for pos, (color, rk, revealed) in board_spec.items()}
    return GameState(board=board, dead=dead,
                     seat_color=seat_color or {0: "r", 1: "b"},
                     turn=turn, first_flip_done=True,
                     cfg=cfg or RuleConfig())


class TestP1QuiescenceDepth(unittest.TestCase):

    def setUp(self):
        self.w = EvalWeights.apk_weights()

    def test_long_range_counterattack_avoidance(self):
        """测试静态搜索深度延伸：能识别 5~6 步后的长程连环反杀陷阱，杜绝地平线盲区丢大子。

        交火链设计：
        - 红旅长在 (5, 2)
        - 蓝排长在 (4, 2) (诱饵)
        - 蓝师长在 (3, 2) (伏击：反吃红旅长)
        - 红军长在 (5, 0) (支援：沿铁路吃蓝师长)
        - 蓝炸弹在 (3, 0) (核武伏击：沿铁路炸红军长)
        - 红连长在 (10, 0) (后方弱子)
        - 蓝军长在 (0, 0) (后方压制)

        推演：
        1. 红旅长吃蓝排长 (5,2)->(4,2)
        2. 蓝师长吃红旅长 (3,2)->(4,2)
        3. 红军长吃蓝师长 (5,0)->(3,2) (经铁路/交火)
        4. 蓝炸弹炸红军长 (3,0)->(3,2)
        5. 终局蓝方军长掌控全盘，红方仅剩连长，红方大崩盘！
        在旧实现 (depth=4) 时，由于深度截断看不见第 4-5 步红军长被炸，红方会贪吃排长；
        在对齐原生 16 ply QSearch 后，红方能够识破陷阱，严禁走 (5, 2) -> (4, 2)。
        """
        st = mk_state({
            (5, 2): ("r", "LV", True),
            (4, 2): ("b", "PAI", True),
            (3, 2): ("b", "SHI", True),
            (5, 0): ("r", "JUN", True),
            (3, 0): ("b", "ZHA", True),
            (10, 0): ("r", "LIAN", True),
            (0, 0): ("b", "JUN", True),
        }, turn=0)

        engine = ExpertSearchEngine(weights=self.w, seed=42, qsearch_depth=16)
        best_act, score, stats = engine.search(st, max_depth=1, time_limit_ms=0)

        self.assertNotEqual(best_act, Action("move", (5, 2), (4, 2)),
                            "静态搜索必须看清 4~5 步后军长被炸的长程反杀陷阱，拒绝贪吃诱饵")
        self.assertGreater(stats.qnodes, 0)

    def test_qsearch_pure_captures_no_camp_explosion(self):
        """测试 QSearch 纯吃子原则：空行营移动不得进入 QSearch，分支因子受控。"""
        # 构造局面：有 1 个吃子走法，同时周围有 4 个空行营
        st = mk_state({
            (2, 2): ("r", "SHI", True),
            (2, 3): ("b", "LIAN", True),  # 可吃目标
            (6, 2): ("b", "SI", True),
        }, turn=0)
        # (2, 2) 邻接空行营 (3, 2), (1, 2) 等

        engine = ExpertSearchEngine(weights=self.w, seed=42)
        acts = st.legal_actions()
        # 确认局面中既有吃子动作也有进营动作
        camp_moves = [a for a in acts if a.kind == "move" and a.to in [(3, 2), (1, 2)]]
        self.assertTrue(len(camp_moves) > 0, "局面应包含进营候选动作")

        # 针对该局面执行 QSearch，验证节点数极低（不会因为空营爆炸）
        engine.stats.qnodes = 0
        val = engine._qsearch(st, alpha=-10000.0, beta=10000.0, depth_left=16)

        # 纯吃子局面下，QSearch 仅展开吃连长的单条交火线，qnodes 应在 10 个以内
        self.assertLessEqual(engine.stats.qnodes, 10,
                             f"纯吃子 QSearch 不应展开进营动作，节点数过高: {engine.stats.qnodes}")

    def test_ids_dynamic_cutoff_safety(self):
        """测试 IDS 动态时间控制 (对齐 0x5ac3a 耗时超 25% 早停)：能够平稳安全返回最优解。"""
        st = mk_state({
            (5, 2): ("r", "SI", True),
            (5, 0): ("r", "JUN", True),
            (7, 2): ("b", "SI", True),
            (7, 4): ("b", "JUN", True),
        }, turn=0)

        engine = ExpertSearchEngine(weights=self.w, seed=42)
        # 设置限时 100ms，请求搜索 20 层
        best_act, score, stats = engine.search(st, max_depth=20, time_limit_ms=100)

        self.assertIsNotNone(best_act)
        # 验证耗时不会失控超时，并且成功返回最高深度的迭代加深解
        self.assertGreaterEqual(stats.max_depth, 1)
        self.assertLess(stats.time_elapsed_ms, 300.0)

    def test_apk_native_agent_levels_and_qdepth(self):
        """测试 ApkNativeAgent 三级难度档位与 QSearch 深度配置对齐。"""
        bot_b = ApkNativeAgent(level="beginner", seed=1)
        self.assertEqual(bot_b.depth, 2)
        self.assertEqual(bot_b.qsearch_depth, 8)

        bot_i = ApkNativeAgent(level="intermediate", seed=2)
        self.assertEqual(bot_i.depth, 3)
        self.assertEqual(bot_i.qsearch_depth, 12)

        bot_a = ApkNativeAgent(level="advanced", seed=3)
        self.assertEqual(bot_a.depth, 4)
        self.assertEqual(bot_a.qsearch_depth, 16)

        # 验证单步决策
        st = mk_state({
            (5, 2): ("r", "SI", True),
            (7, 2): ("b", "SI", True),
        }, turn=0)
        act = bot_a.select_action(st)
        self.assertIn(act, st.legal_actions())


if __name__ == "__main__":
    unittest.main()
