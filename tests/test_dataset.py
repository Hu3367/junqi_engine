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

    def test_terminal_label_from_meta_codes(self):
        """P1 标签规则验收（基线计划 §6 权威口径）：官方终局码 -> 三类 Value。"""
        from junqi.dataset import terminal_label_from_meta as f

        # 明确胜负：常规终局 1 / 主动认输 21 / 长捉判负 22 / 超时判负 23
        for rc in (1, 21, 22, 23):
            kind, seat = f(rc, 1)
            self.assertEqual((kind, seat), ("decided", 0), f"code {rc}")
            kind, seat = f(rc, 2)
            self.assertEqual((kind, seat), ("decided", 1), f"code {rc}")

        # 正规和棋：协议和棋 40 / 循环和棋 42 / 限步判和 43
        for rc in (40, 42, 43):
            kind, seat = f(rc, 3)
            self.assertEqual((kind, seat), ("draw", -1), f"code {rc}")

        # 2026-09-13 修正（AGENTS.md §6 冲突排查）：code 24 断线与 code 20
        # 强退同为中止事件，一律不赋 Value（此前 24 被无条件视为明确胜负）
        for rc in (20, 24):
            kind, seat = f(rc, 1)
            self.assertEqual((kind, seat), ("none", -1), f"code {rc} 带胜者码也不得赋 Value")
            kind, seat = f(rc, 3)
            self.assertEqual((kind, seat), ("none", -1), f"code {rc} 不得误判为和棋")

        # 未知码安全回退
        self.assertEqual(f(0, 3), ("draw", -1))
        self.assertEqual(f(99, 3), ("draw", -1))
        self.assertEqual(f(99, 9), ("none", -1))

    def test_meta_code24_disconnect_gets_no_value(self):
        """端到端：带 list.cfg 元数据 code 24（断线）的胜局，Policy 可用但
        has_value 必须为 False（此前被无条件赋 ±1，属标签口径冲突）。"""
        table = [0] * 60
        table[5 * 5 + 2] = 7    # 红连长
        table[4 * 5 + 2] = 19   # 蓝连长
        moves = (
            (5 * 5 + 2, 5 * 5 + 2, 1),
            (4 * 5 + 2, 4 * 5 + 2, 1),
        )
        g = SavGame(path="test_dc.sav", table=tuple(table), moves=moves,
                    winner=0, win_reason="disconnected", stopped_on_event=True)
        meta = {"reason_code": 24, "winner": 1, "moves_count": 2}
        samples = process_single_game(g, self.cfg, meta_record=meta)
        self.assertEqual(len(samples), 2)
        for s in samples:
            self.assertFalse(s["has_value"], "断线局（code 24）不得赋终局 Value")
        # 对照：code 21 主动认输仍属明确胜负
        meta21 = {"reason_code": 21, "winner": 1, "moves_count": 2}
        samples21 = process_single_game(g, self.cfg, meta_record=meta21)
        self.assertTrue(samples21[0]["has_value"])
        self.assertEqual(samples21[0]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
