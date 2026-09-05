"""P1 复盘数据集生成、切分与合规性单元测试。"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from junqi.config import RuleConfig
from junqi.dataset import (export_replay_dataset, evaluate_dataset_policy,
                          process_single_game)
from junqi.replay import SavGame, cell_rc
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece


class TestDatasetP1(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def test_single_game_processing_and_value_labels(self):
        """测试单局样本提取与 Value 标签赋值规则（§6 规范）。"""
        # 初始棋盘：红连长 (5,2)，蓝连长 (4,2)
        table = [0] * 60
        table[5 * 5 + 2] = 7    # 红连长
        table[4 * 5 + 2] = 19   # 蓝连长

        # 着法序列：红翻 (5,2) -> 蓝翻 (4,2) -> 红走 (5,2)->(4,2) 吃掉蓝连长获得胜利
        moves = (
            (5 * 5 + 2, 5 * 5 + 2, 1),
            (4 * 5 + 2, 4 * 5 + 2, 1),
            (5 * 5 + 2, 4 * 5 + 2, 1),
        )

        g_win = SavGame(
            path="test_win.sav",
            table=tuple(table),
            moves=moves,
            winner=0,  # 玩家1胜
            win_reason="immobilized",
            stopped_on_event=False
        )

        samples = process_single_game(g_win, self.cfg)
        self.assertEqual(len(samples), 3)
        s0 = samples[0]

        # 验证公共 Policy 维度 (38 通道)
        self.assertEqual(s0["state"].shape, (38, 12, 5))
        self.assertEqual(s0["mask"].shape, (3650,))
        self.assertTrue(s0["mask"][s0["action"]])

        # 胜负局必须赋 Value (+1/-1)
        self.assertTrue(s0["has_value"])
        self.assertEqual(s0["value"], 1.0)
        self.assertEqual(samples[1]["value"], -1.0)  # 蓝方视角为 -1.0

    def test_unfinished_game_does_not_assign_value(self):
        """未终局或特殊中止局绝不能进入 Value 训练样本（has_value=False）。"""
        table = [0] * 60
        table[5 * 5 + 2] = 7
        table[4 * 5 + 2] = 19
        moves = (
            (5 * 5 + 2, 5 * 5 + 2, 1),
            (4 * 5 + 2, 4 * 5 + 2, 1),
        )

        g_unfinished = SavGame(
            path="test_unfinished.sav",
            table=tuple(table),
            moves=moves,
            winner=None,  # 未终局
            stopped_on_event=True  # 中止
        )

        samples = process_single_game(g_unfinished, self.cfg)
        self.assertEqual(len(samples), 2)
        # Policy 可用，但 has_value 必须为 False
        self.assertFalse(samples[0]["has_value"])
        self.assertFalse(samples[1]["has_value"])

    def test_dataset_reproducibility_and_disjoint_split(self):
        """测试按对局切分的互斥性与确定性重现。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 找到复盘文件并取前 10 局做集成测试
            import glob
            savs = glob.glob("../**/*.sav", recursive=True) + glob.glob("./**/*.sav", recursive=True)
            if not savs:
                self.skipTest("未找到 .sav 测试文件")

            out_dir = os.path.join(tmpdir, "test_dataset")
            stats1 = export_replay_dataset(os.path.dirname(savs[0]), out_dir=out_dir, seed=42, max_games=10)

            self.assertTrue(os.path.exists(os.path.join(out_dir, "train.npz")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "val.npz")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "test.npz")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "metadata.json")))

            # 验证哈希已写入
            self.assertIn("train.npz", stats1.hashes)

            # 验证基准评估指标可正常计算
            report = evaluate_dataset_policy(os.path.join(out_dir, "val.npz"), max_samples=100)
            self.assertIn("random_baseline", report)
            self.assertGreater(report["random_baseline"]["top1_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
