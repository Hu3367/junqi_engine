"""单元测试：复盘管理器、单步点评存储与全量汇总改进数据管线。"""
from __future__ import annotations

import glob
import json
import os
import tempfile
import unittest

from junqi.config import RuleConfig
from junqi.replay_manager import ReplayManager, describe_action
from junqi.review_storage import ReviewStorage
from junqi.review_summary import ReviewSummaryPipeline
from junqi.state import Action, GameState, Piece


class TestReplayReview(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, "test_annotations.json")
        self.storage = ReviewStorage(self.storage_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_json_replay(self):
        """测试加载 games/*.json 对战复盘文件并验证逐步状态还原。"""
        json_files = glob.glob("games/game_*.json")
        if not json_files:
            self.skipTest("未找到本地 games/ 对战记录")
        test_file = json_files[0]

        session = ReplayManager.load_game(test_file)
        self.assertEqual(session.game_type, "json")
        self.assertGreater(session.total_plies, 0)
        self.assertIn("seed", session.metadata)

        # 步进验证
        step0 = session.get_step(0)
        self.assertIsNotNone(step0)
        self.assertEqual(step0.ply, 0)
        self.assertTrue(len(step0.action_desc) > 0)
        self.assertIsNotNone(step0.state_after)

        # 终局状态验证
        final_st = session.final_state
        self.assertIsNotNone(final_st)

    def test_load_sav_replay(self):
        """测试加载军旗原生 .sav 复盘文件并验证状态还原。"""
        sav_files = glob.glob("军旗复盘/*.sav")
        if not sav_files:
            self.skipTest("未找到本地 军旗复盘/*.sav 文件")
        test_file = sav_files[0]

        session = ReplayManager.load_game(test_file)
        self.assertEqual(session.game_type, "sav")
        self.assertGreater(session.total_plies, 0)
        self.assertIn("names", session.metadata)

        step0 = session.get_step(0)
        self.assertIsNotNone(step0)
        self.assertTrue(len(step0.action_desc) > 0)

    def test_review_storage_crud(self):
        """测试点评记录增删改查。"""
        st = GameState(board={}, seat_color={0: "r", 1: "b"}, turn=0, ply=10, quiet=2)
        act = Action("move", (2, 3), (2, 2))
        rec_act = Action("move", (7, 2), (7, 3))

        # 保存点评
        saved = self.storage.save_review(
            source_file="test_game.json",
            ply=10,
            state=st,
            action_played=act,
            label="camp_abandon",
            comment="测试弃营批评",
            action_recommended=rec_act,
            action_played_desc="排长弃营",
            action_recommended_desc="挺进空营",
        )

        self.assertEqual(saved["ply"], 10)
        self.assertEqual(saved["label"], "camp_abandon")
        self.assertEqual(self.storage.count("test_game.json"), 1)

        # 查询点评
        fetched = self.storage.get_review("test_game.json", 10)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["comment"], "测试弃营批评")
        self.assertEqual(fetched["action_recommended"]["frm"], [7, 2])

        # 更新点评
        self.storage.save_review(
            source_file="test_game.json",
            ply=10,
            state=st,
            action_played=act,
            label="expert_blunder",
            comment="更新后的点评说明",
        )
        updated = self.storage.get_review("test_game.json", 10)
        self.assertEqual(updated["label"], "expert_blunder")
        self.assertEqual(updated["comment"], "更新后的点评说明")

        # 删除点评
        deleted = self.storage.delete_review("test_game.json", 10)
        self.assertTrue(deleted)
        self.assertEqual(self.storage.count("test_game.json"), 0)

    def test_review_summary_pipeline(self):
        """测试全量点评汇总管线：生成报告、导出蒸馏数据集与自动化回归测试。"""
        from junqi.rules import Rank
        # 构造 2 条点评
        st1 = GameState(board={(2, 3): Piece("r", Rank.PAI, True)}, seat_color={0: "r", 1: "b"}, turn=0, ply=5)
        st2 = GameState(board={(5, 2): Piece("r", Rank.SI, True)}, seat_color={0: "r", 1: "b"}, turn=0, ply=15)

        self.storage.save_review(
            source_file="game_A.json",
            ply=5,
            state=st1,
            action_played=Action("move", (2, 3), (2, 2)),
            label="camp_abandon",
            comment="排长不可弃营",
            action_recommended=Action("move", (2, 3), (2, 4)),
        )
        self.storage.save_review(
            source_file="game_B.sav",
            ply=15,
            state=st2,
            action_played=Action("move", (5, 2), (5, 3)),
            label="good_move",
            comment="司令占领中路要塞",
        )

        pipeline = ReviewSummaryPipeline(self.storage)
        out_report = os.path.join(self.temp_dir.name, "report.md")
        out_dataset = os.path.join(self.temp_dir.name, "labeled.json")
        out_test = os.path.join(self.temp_dir.name, "test_cases.py")

        res = pipeline.run_all(out_report, out_dataset, out_test)

        self.assertEqual(res["total_reviews"], 2)
        self.assertTrue(os.path.exists(out_report))
        self.assertTrue(os.path.exists(out_dataset))
        self.assertTrue(os.path.exists(out_test))

        # 检查报告内容
        with open(out_report, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("排长不可弃营", content)
            self.assertIn("camp_abandon", content)

        # 检查导出的训练数据集格式
        with open(out_dataset, "r", encoding="utf-8") as f:
            ds = json.load(f)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]["action_judgment"], "expert_blunder")
            self.assertEqual(ds[1]["action_judgment"], "both_fine")
            self.assertTrue(ds[0]["verified"])

        # 检查生成的回归测试代码
        with open(out_test, "r", encoding="utf-8") as f:
            test_code = f.read()
            self.assertIn("def test_review_case_1_", test_code)
            self.assertIn("assert chosen != blunder", test_code)


if __name__ == "__main__":
    unittest.main()
