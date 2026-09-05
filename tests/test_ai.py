"""传统估值增强测试（A2：行营势力 / 死区势能 / 暗子时差）。

依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §4.2 第一条线。
红测流程：本文件先于实现存在，实现后全部转绿。
"""
from __future__ import annotations

import unittest

from junqi.ai import Agent, evaluate
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece, deal


def mk(board_spec, dead=(), turn=0, seat_color=None, quiet=0, cfg=None):
    board = {pos: Piece(color, Rank[rk], revealed)
             for pos, (color, rk, revealed) in board_spec.items()}
    return GameState(board=board, dead=dead,
                     seat_color=seat_color or {0: "r", 1: "b"},
                     turn=turn, first_flip_done=True, quiet=quiet,
                     cfg=cfg or RuleConfig())


class TestEvaluateEnhancements(unittest.TestCase):
    """三类显式判断的估值方向性测试（其余盘面保持对称，仅目标项产生差异）。"""

    def setUp(self):
        self.w = EvalWeights()

    # ------------------------------------------------- 1. 行营势力

    def test_camp_zone_prefers_approach(self):
        """贴近空行营的走法估值应高于走离行营的走法（行营势力范围）。"""
        # 行营 (3,2) 为空；红连长在 (5,2)：走 (4,2) 邻接行营，走 (6,2) 远离
        st = mk({(5, 2): ("r", "LIAN", True), (10, 0): ("b", "LIAN", True)})
        near = evaluate(st.apply(Action("move", (5, 2), (4, 2))), 0, self.w)
        away = evaluate(st.apply(Action("move", (5, 2), (6, 2))), 0, self.w)
        self.assertGreater(near, away, "贴近空行营应更优（行营势力项）")

    def test_camp_zone_disabled(self):
        """camp_zone=0 时两走法估值差中不含行营势力贡献。"""
        w = EvalWeights()
        w.camp_zone = 0.0
        st = mk({(5, 2): ("r", "LIAN", True), (10, 0): ("b", "LIAN", True)})
        near = evaluate(st.apply(Action("move", (5, 2), (4, 2))), 0, w)
        away = evaluate(st.apply(Action("move", (5, 2), (6, 2))), 0, w)
        self.assertAlmostEqual(near, away, places=6)

    # ------------------------------------------------- 2. 死区势能

    def test_fortress_term_boosts_sealed_side(self):
        """死区势能项应按 fortress_score 差拉开双方视角估值差：
        封死死区的一方获得和棋势能加分，对方视角对称减分。
        局面：红旗被对方明雷+己方明雷封死（fortress=1.0），蓝无工兵且无旗。"""
        board = {
            (11, 1): ("r", "QI", True),    # 红旗
            (10, 1): ("b", "LEI", True),   # 蓝明雷做永久墙
            (11, 0): ("r", "LEI", True),   # 红明雷
            (11, 2): ("r", "LEI", True),   # 红明雷
            (0, 0): ("b", "SI", True),     # 蓝仅存司令（无工兵）
        }
        dead = [Piece("b", Rank.GONG, True) for _ in range(3)]  # 蓝工兵全灭
        st = mk(board, dead=dead)
        # 启用死区项：视角差 = 物质差 + 2*fortress(=1.0) 加成；
        # 关闭时仅剩物质差（此局面蓝物质领先，视角差为负）
        gap_on = evaluate(st, 0, self.w) - evaluate(st, 1, self.w)
        w0 = EvalWeights()
        w0.fortress = 0.0
        gap_off = evaluate(st, 0, w0) - evaluate(st, 1, w0)
        self.assertAlmostEqual(gap_on - gap_off, 2 * 25.0, places=5,
                               msg="死区项应贡献 ±fortress 权重差")
        self.assertGreater(gap_on, gap_off, "死区势能应利好封死死区的一方视角")

    # ------------------------------------------------- 3. 暗子时差

    def test_hidden_tempo_prefers_active_pieces(self):
        """同等明子物质下，可行动明子多的一方估值更高（暗子需两回合激活）。"""
        # 红两枚活动子；蓝一明子 + 一暗子（暗子暂不可动）——总物质近似对称
        st = mk({(5, 0): ("r", "LIAN", True), (5, 4): ("r", "PAI", True),
                 (7, 0): ("b", "LIAN", True), (7, 4): ("b", "PAI", False)})
        v_on = evaluate(st, 0, self.w)
        w0 = EvalWeights()
        w0.hidden_tempo = 0.0
        v_off = evaluate(st, 0, w0)
        self.assertGreater(v_on, v_off, "活动子优势应通过时差项体现")

    # ------------------------------------------------- 兼容性

    def test_eval_weights_compat_and_roundtrip(self):
        """新增权重键序列化往返；旧字典（缺新键）加载取默认值。"""
        w = EvalWeights()
        d = w.to_dict()
        for key in ("camp_zone", "fortress", "hidden_tempo"):
            self.assertIn(key, d)
        w2 = EvalWeights.from_dict(d)
        self.assertEqual(w2.camp_zone, w.camp_zone)
        self.assertEqual(w2.fortress, w.fortress)
        self.assertEqual(w2.hidden_tempo, w.hidden_tempo)
        # 旧字典兼容
        old = {k: v for k, v in d.items()
               if k not in ("camp_zone", "fortress", "hidden_tempo")}
        w3 = EvalWeights.from_dict(old)
        self.assertEqual(w3.camp_zone, EvalWeights().camp_zone)

    def test_agent_smoke_with_enhanced_eval(self):
        """增强估值接入 PIMC 搜索后，开局与中盘决策不崩溃、动作合法。"""
        st = deal(__import__("random").Random(88))
        agent = Agent(SearchConfig(depth=1, samples=2), seed=1)
        scored = agent.choose_actions(st, topn=3)
        self.assertTrue(scored)
        self.assertIn(scored[0][0], st.legal_actions())
        st2 = st.apply(scored[0][0]).apply(st.apply(scored[0][0]).legal_actions()[0])
        scored2 = agent.choose_actions(st2, topn=3)
        self.assertTrue(scored2)
        self.assertIn(scored2[0][0], st2.legal_actions())


if __name__ == "__main__":
    unittest.main()
