"""P4 假想敌 ApkNativeAgent 及其基准对弈测试。"""
from __future__ import annotations

import random
import unittest

from junqi.ai import Agent, ApkNativeAgent, ExpertAgent
from junqi.benchmark import run_apk_challenge, run_tournament_match
from junqi.config import RuleConfig
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece, deal


class TestP4ApkAgent(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def test_apk_agent_initialization_and_levels(self):
        """测试 ApkNativeAgent 各难度档位的初始化。"""
        bot_b = ApkNativeAgent(level="beginner", seed=101)
        self.assertEqual(bot_b.depth, 2)
        self.assertEqual(bot_b.time_limit_ms, 100)

        bot_i = ApkNativeAgent(level="intermediate", seed=102)
        self.assertEqual(bot_i.depth, 3)
        self.assertEqual(bot_i.time_limit_ms, 300)

        bot_a = ApkNativeAgent(level="advanced", seed=103)
        self.assertEqual(bot_a.depth, 4)
        self.assertEqual(bot_a.time_limit_ms, 1000)

        with self.assertRaises(ValueError):
            ApkNativeAgent(level="godlike")

    def test_apk_agent_non_peeking_and_move_selection(self):
        """测试 ApkNativeAgent 遵守公共信息屏障，且正常输出合法动作。"""
        bot = ApkNativeAgent(level="beginner", seed=2026)

        # 随机开局洗牌
        st = deal(random.Random(12345), self.cfg)
        hidden_before = [p for p, pc in st.board.items() if not pc.revealed]
        self.assertGreater(len(hidden_before), 0)

        # 进行单步决策
        act = bot.select_action(st)
        self.assertIn(act, st.legal_actions())

        # 验证暗子未被内部窥探并修改为 revealed
        hidden_after = [p for p, pc in st.board.items() if not pc.revealed]
        self.assertEqual(len(hidden_before), len(hidden_after))

    def test_apk_agent_tactical_win(self):
        """测试 ApkNativeAgent 在面临一步吃旗时能果断取胜。"""
        bot = ApkNativeAgent(level="beginner", seed=42)

        # 构造工兵一步吃旗胜势局面 (已挖光地雷，工兵在 (1, 1)，敌旗在 (0, 1))
        st = GameState(board={
            (1, 1): Piece("r", Rank.GONG, True),
            (11, 1): Piece("r", Rank.QI, True),
            (0, 1): Piece("b", Rank.QI, True),
        }, dead=[
            Piece("b", Rank.LEI, True), Piece("b", Rank.LEI, True), Piece("b", Rank.LEI, True)
        ], turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        act = bot.select_action(st)
        self.assertEqual(act.kind, "move")
        self.assertEqual(act.to, (0, 1))

    def test_run_apk_challenge_match(self):
        """测试对战调度器与 ApkNativeAgent 的自动化基准对弈。"""
        candidate = ExpertAgent(seed=42)
        # 设置短限步快速验证集成链路
        fast_cfg = RuleConfig(max_plies=4)
        res = run_apk_challenge(candidate, n_games=2, level="beginner", base_seed=777, cfg=fast_cfg)

        self.assertIn("games", res)
        self.assertEqual(res["games"], 2)
        self.assertIn("score_rate_a", res)
        self.assertIn("apk_level", res)
        self.assertEqual(res["apk_level"], "beginner")
        self.assertIn("elo_diff", res)


if __name__ == "__main__":
    unittest.main()
