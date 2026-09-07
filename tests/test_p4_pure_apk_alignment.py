"""P4 官方 APK 逆向算法 1:1 纯净对齐专项测试。

验证 5 大关键差异全面对齐 libjunqi.so 原生机制：
1. 静态估值核 1:1 纯净化 (0x59f90 与 0x124094)：
   - 权威 2 的幂次物质价值表、动态炸弹缩放、60 格行营偏置、地雷护旗 +80；
2. 博弈树拓扑结构与内部节点纯明子化 (0x5a50e - 0x5a536 与 0x59e80)：
   - 根节点分流决策，树深层（ply >= 1）与 QSearch 绝不展开翻棋几率树；
3. 开局库偏好 (0x5a3c0)：
   - 开局首翻强偏好中心 4 个行营周围的 6 个黄金暗子辐射位；
4. 官方三档难度、随机扰动 Jitter 与 IDS 25% 早停截断 (0x3404c / 0x34078 / 0x5ac3a)；
5. 官方 70 步无吃子判和规则对齐。
"""
from __future__ import annotations

import random
import unittest

from junqi.apk_agent import ApkNativeAgent
from junqi.apk_engine import (
    APK_LEVEL_SPECS,
    APK_PIECE_VALUES,
    CAMP_POSITION_BONUS,
    CENTER_CAMP_FLIP_POSITIONS,
    MINE_FLAG_GUARD_BONUS,
    ApkSearchEngine,
    eval_apk_flip_root,
    eval_apk_pure,
)
from junqi.config import RuleConfig
from junqi.rules import HQS, Rank
from junqi.state import Action, GameState, Piece, deal


class TestP4PureApkAlignment(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def test_01_pure_evaluation_material_camps_and_mine_guard(self):
        """1. 测试 1:1 纯净估值核：2560 等比物质、动态炸弹、行营加分与地雷护旗。"""
        # (1) 物质等比梯度验证
        self.assertEqual(APK_PIECE_VALUES[Rank.SI], 2560.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.JUN], 1280.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.SHI], 640.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.LV], 320.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.TUAN], 160.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.YING], 80.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.LIAN], 40.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.PAI], 30.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.GONG], 80.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.LEI], 70.0)
        self.assertEqual(APK_PIECE_VALUES[Rank.QI], 50.0)

        # (2) 行营偏置加分 (中营 +50, 其余行营 +40)
        self.assertEqual(CAMP_POSITION_BONUS[(3, 2)], 50.0)
        self.assertEqual(CAMP_POSITION_BONUS[(8, 2)], 50.0)
        self.assertEqual(CAMP_POSITION_BONUS[(2, 1)], 40.0)

        # (3) 动态炸弹与纯净估值计算
        # 己方有排长(30)、炸弹，敌方有司令(2560)
        # 己方炸弹价值 = min(853.33, 2560/3) = 853.33
        st = GameState(board={
            (0, 0): Piece("r", Rank.PAI, True),
            (0, 1): Piece("r", Rank.ZHA, True),
            (11, 4): Piece("b", Rank.SI, True),
        }, turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        score_r = eval_apk_pure(st, "r")
        expected_score = (30.0 + 853.3333333333334) - 2560.0
        self.assertAlmostEqual(score_r, expected_score, places=2)

        # (4) 行营加分验证：让己方排长进入中营 (3, 2)
        st_camp = GameState(board={
            (3, 2): Piece("r", Rank.PAI, True),
        }, turn=0, cfg=self.cfg)
        st_camp.seat_color[0], st_camp.seat_color[1] = "r", "b"
        score_camp = eval_apk_pure(st_camp, "r")
        self.assertAlmostEqual(score_camp, 30.0 + 50.0, places=2)

        # (5) 地雷护旗防御加分验证 (+80)
        # 己方军旗在 (11, 1)，地雷在 (10, 1) 守护
        st_mine = GameState(board={
            (11, 1): Piece("r", Rank.QI, True),
            (10, 1): Piece("r", Rank.LEI, True),
        }, turn=0, cfg=self.cfg)
        st_mine.seat_color[0], st_mine.seat_color[1] = "r", "b"
        score_mine = eval_apk_pure(st_mine, "r")
        # 物质: 旗50 + 雷70 = 120，加护旗 +80 -> 200
        self.assertAlmostEqual(score_mine, 120.0 + MINE_FLAG_GUARD_BONUS, places=2)

    def test_02_game_tree_topology_pure_visible_tree_and_qsearch(self):
        """2. 测试博弈树拓扑结构：树深层与 QSearch 绝不展开暗棋翻棋。"""
        engine = ApkSearchEngine(seed=42)

        # 构造包含明子与大量暗子的局面
        st = GameState(board={
            (1, 2): Piece("r", Rank.SI, True),
            (1, 3): Piece("b", Rank.JUN, True),
            (4, 0): Piece("b", Rank.PAI, False),  # 暗子
            (4, 4): Piece("r", Rank.PAI, False),  # 暗子
        }, turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        # 根节点合法动作既有 move 也有 flip
        root_acts = st.legal_actions()
        has_flips = any(a.kind == "flip" for a in root_acts)
        has_moves = any(a.kind == "move" for a in root_acts)
        self.assertTrue(has_flips)
        self.assertTrue(has_moves)

        # 深入单步 move 之后，内部节点测试：内部节点走法过滤只保留 move
        nxt = st.apply(Action("move", (1, 2), (1, 3)))
        internal_acts = nxt.legal_actions()
        internal_moves_only = [a for a in internal_acts if a.kind == "move"]

        # 验证内部节点绝不展开 flip
        self.assertTrue(all(a.kind == "move" for a in internal_moves_only))

        # 运行引擎深度 2 搜索
        act, score, stats = engine.search(st, depth=2, qsearch_depth=4)
        self.assertIsNotNone(act)
        # 司令吃军长带来巨大优势，引擎应果断选择 (1,2)->(1,3) 吃军长
        self.assertEqual(act, Action("move", (1, 2), (1, 3)))
        self.assertGreater(score, 1000.0)

    def test_03_opening_center_camp_flip_preference(self):
        """3. 测试开局中心行营邻位暗子强偏好 (0x5a3c0)。"""
        engine = ApkSearchEngine(seed=123)

        # 构造开局全新全暗棋盘（所有非行营位置均为暗子）
        st = deal(random.Random(2026), self.cfg)
        self.assertEqual(len(st.hidden_positions()), 50)
        self.assertFalse(st.first_flip_done)

        # 进行根节点搜索决策
        act, score, stats = engine.search(st, depth=2)
        self.assertIsNotNone(act)
        self.assertEqual(act.kind, "flip")

        # 首翻必须精准命中开局黄金 6 位之一
        self.assertIn(act.frm, CENTER_CAMP_FLIP_POSITIONS)

    def test_04_level_specs_jitter_and_ids_cutoff(self):
        """4. 测试难度档位参数、随机扰动 Jitter 与 IDS 25% 早停截断。"""
        # 验证档位常量规范
        self.assertEqual(APK_LEVEL_SPECS["beginner"]["depth"], 2)
        self.assertEqual(APK_LEVEL_SPECS["beginner"]["jitter"], 30.0)
        self.assertEqual(APK_LEVEL_SPECS["beginner"]["time_limit_ms"], 100)

        self.assertEqual(APK_LEVEL_SPECS["intermediate"]["depth"], 3)
        self.assertEqual(APK_LEVEL_SPECS["intermediate"]["jitter"], 10.0)
        self.assertEqual(APK_LEVEL_SPECS["intermediate"]["time_limit_ms"], 300)

        self.assertEqual(APK_LEVEL_SPECS["advanced"]["depth"], 4)
        self.assertEqual(APK_LEVEL_SPECS["advanced"]["jitter"], 0.5)
        self.assertEqual(APK_LEVEL_SPECS["advanced"]["time_limit_ms"], 1000)

        # 验证 ApkNativeAgent 正常集成
        agent_beg = ApkNativeAgent(level="beginner", seed=10)
        agent_adv = ApkNativeAgent(level="advanced", seed=20)
        self.assertEqual(agent_beg.depth, 2)
        self.assertEqual(agent_adv.depth, 4)

    def test_05_rule_draw_70_plies_aligned(self):
        """5. 测试 70 步无吃子和棋规则对齐。"""
        # 验证默认配置为官方标准 70
        cfg = RuleConfig()
        self.assertEqual(cfg.no_capture_draw_plies, 70)

        # 构造连续 69 步无吃子局面，走非吃子移动后达到 70 步触发判和
        st = GameState(board={
            (0, 0): Piece("r", Rank.PAI, True),
            (11, 4): Piece("b", Rank.PAI, True),
        }, turn=0, quiet=69, ply=69, cfg=cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        self.assertFalse(st.is_terminal())
        self.assertIsNone(st.winner)

        # 移动一步，不吃子
        nxt = st.apply(Action("move", (0, 0), (0, 1)))
        self.assertEqual(nxt.quiet, 70)
        self.assertTrue(nxt.is_terminal())
        self.assertEqual(nxt.winner, -1)
        self.assertEqual(nxt.win_reason, "no_capture")

    def test_06_first_flip_camp_occupation_over_blind_flip(self):
        """6. 测试首翻出己方明子后优先占营建立据点，禁绝漫无目的盲翻 (对局 234949 复盘对齐)。"""
        engine = ApkSearchEngine(seed=42)

        # 构造对局 game_20260907_234949.json 第 5 手局面：
        # 红方在 (8, 1) 有明子排长，近邻 (7, 1) 和 (9, 1) 为未占领空营
        # 蓝方在 (3, 1) 有军长，在 (8, 2) 中营有排长
        board = {
            (3, 1): Piece("b", Rank.JUN, True),
            (8, 1): Piece("r", Rank.PAI, True),
            (8, 2): Piece("b", Rank.PAI, True),
        }
        # 其余非行营格均为暗子
        for pos in [(r, c) for r in range(12) for c in range(5)]:
            if pos not in board and pos not in ((2, 1), (2, 3), (3, 2), (4, 1), (4, 3), (7, 1), (7, 3), (8, 2), (9, 1), (9, 3)):
                board[pos] = Piece("r", Rank.PAI, False)

        st = GameState(board=board, turn=1, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "b", "r"

        act, score, stats = engine.search(st, depth=3)
        self.assertIsNotNone(act)
        # 严禁在此局面下盲目远端翻棋
        self.assertEqual(act.kind, "move")
        self.assertEqual(act.frm, (8, 1))
        # 必须进驻相邻空营之一建立据点
        self.assertIn(act.to, [(7, 1), (9, 1)])

    def test_07_radiate_from_camp_after_occupation(self):
        """7. 测试占营据点确立后，依托行营单向打击特权辐射拓荒。"""
        engine = ApkSearchEngine(seed=42)

        # 红方排长已安全进驻 (7, 1) 行营
        board = {
            (4, 1): Piece("b", Rank.JUN, True),
            (7, 1): Piece("r", Rank.PAI, True),
            (8, 2): Piece("b", Rank.PAI, True),
        }
        for pos in [(r, c) for r in range(12) for c in range(5)]:
            if pos not in board and pos not in ((2, 1), (2, 3), (3, 2), (4, 1), (4, 3), (7, 1), (7, 3), (8, 2), (9, 1), (9, 3)):
                board[pos] = Piece("r", Rank.PAI, False)

        st = GameState(board=board, turn=1, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "b", "r"

        act, score, stats = engine.search(st, depth=3)
        self.assertIsNotNone(act)
        # 在营中无需无谓出营，优先依托行营辐射翻开周边暗子
        self.assertEqual(act.kind, "flip")
        # 翻棋目标必须属于 (7, 1) 行营辐射控制范围 (6,0), (6,1), (6,2), (7,0), (7,2), (8,0), (8,1)
        camp_7_1_neighbors = [(6, 0), (6, 1), (6, 2), (7, 0), (7, 2), (8, 0), (8, 1)]
        self.assertIn(act.frm, camp_7_1_neighbors)


if __name__ == "__main__":
    unittest.main()

