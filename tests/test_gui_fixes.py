"""测试 GUI 界面修复：防范 KeyError 回调异常、last_action_desc 正确传递、APK 与专家模型正确接入。"""
import tkinter as tk
import unittest

from junqi.ai import ApkNativeAgent, ExpertAgent
from junqi.config import RuleConfig, SearchConfig
from junqi.gui import GuiApp
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece


class TestGuiFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = GuiApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_describe_defensive(self):
        """测试 describe() 在棋子已被移走（落地后）时不会引发 KeyError。"""
        st = GameState(board={
            (3, 2): Piece("b", Rank.PAI, True),
            (3, 3): Piece("r", Rank.GONG, True),
        }, turn=1, first_flip_done=True, seat_color={0: "r", 1: "b"}, cfg=RuleConfig())
        self.app.state = st

        act = Action("move", (3, 2), (3, 3))
        # 移动前生成描述
        desc_before = self.app.describe(act)
        self.assertIn("排长吃", desc_before)

        # 模拟走子后
        st_after = st.apply(act)
        self.app.state = st_after

        # 以前在 st_after 上调用 describe(act) 会报 KeyError: (3, 2)
        # 现在必须防御式安全降级，绝不抛出异常
        desc_after = self.app.describe(act)
        self.assertIsInstance(desc_after, str)
        self.assertIn("(3,2)->(3,3)", desc_after)

    def test_update_status_after_ai_move(self):
        """测试 AI 走子后 update_status() 正常呈现 [AI刚走] 提示，无异常。"""
        st = GameState(board={
            (3, 2): Piece("b", Rank.PAI, True),
            (3, 3): Piece("r", Rank.GONG, True),
            (4, 4): Piece("r", Rank.SHI, True),
            (4, 2): Piece("b", Rank.SHI, True),
        }, turn=1, first_flip_done=True, seat_color={0: "r", 1: "b"}, cfg=RuleConfig())
        self.app.state = st
        self.app.human_seat = 0

        act = Action("move", (3, 2), (3, 3))
        # 执行动作
        self.app.do_action(act, ai_meta={"engine": "apk"})

        # 检查是否轮到人类，并且 last_action_desc 完整记录且呈现在状态栏中
        self.assertEqual(self.app.state.turn, 0)
        self.assertIn("排长吃", self.app.last_action_desc)
        status_text = self.app.status.cget("text")
        self.assertIn("[AI刚走]", status_text)
        self.assertIn("排长吃", status_text)

    def test_gui_engines_integration(self):
        """测试 GUI 界面中的原生 APK 引擎与专家搜索引擎能够正常初始化并做出合法决策。"""
        self.app.new_game(human_seat=0)
        # 测试 APK 引擎对接
        self.app.ai_engine.set("apk")
        self.assertEqual(self.app.ai_engine.get(), "apk")
        apk_agent = ApkNativeAgent(level="advanced", seed=42)
        act = apk_agent.select_action(self.app.state)
        self.assertIn(act, self.app.state.legal_actions())

        # 测试专家引擎对接与 QSearch 深度
        self.app.ai_engine.set("expert")
        self.assertEqual(self.app.ai_engine.get(), "expert")
        expert_agent = ExpertAgent(SearchConfig(depth=2, time_limit_ms=500, qsearch_depth=12), seed=42)
        act2 = expert_agent.engine.search(self.app.state, max_depth=1)[0]
        self.assertIn(act2, self.app.state.legal_actions())
