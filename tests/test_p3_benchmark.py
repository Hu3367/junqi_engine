"""P3.1 与 P3.2 架构升级与实验靶场单元测试。"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from junqi.benchmark import (create_benchmark_suite, evaluate_net_benchmark,
                             run_tournament_match, save_metrics_report)
from junqi.config import RuleConfig
from junqi.encoder import NUM_CHANNELS, encode_state, encode_state_np
from junqi.net import JunqiNet
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece, position_key


class TestP3ArchitectureAndBenchmark(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()

    def test_encoder_38_channels_and_repetition(self):
        """测试 38 通道编码及重复计数平面。"""
        st = GameState(board={
            (5, 2): Piece("r", Rank.SI, True),
            (6, 2): Piece("b", Rank.SI, True),
        }, turn=0, cfg=self.cfg)

        # 默认无 history_counts -> 36 通道为 1.0, 37 通道为 0.0
        t1 = encode_state_np(st)
        self.assertEqual(t1.shape, (38, 12, 5))
        self.assertEqual(t1[36, 0, 0], 1.0)
        self.assertEqual(t1[37, 0, 0], 0.0)

        # 传入历史计数
        pk = position_key(st)
        hist = {pk: 2}  # 出现过 2 次
        t2 = encode_state_np(st, history_counts=hist)
        self.assertEqual(t2[36, 0, 0], 1.0)
        self.assertEqual(t2[37, 0, 0], 1.0)

    def test_net_3_class_value_head_and_backward_compatibility(self):
        """测试 Value 三分类头输出与标量期望换算。"""
        net = JunqiNet(in_channels=38)
        st = GameState(board={
            (5, 2): Piece("r", Rank.SI, True),
            (6, 2): Piece("b", Rank.SI, True),
        }, turn=0, cfg=self.cfg)

        # 前向推理
        policy_map, scalar_v = net.predict_state(st)
        self.assertIsInstance(scalar_v, float)
        self.assertTrue(-1.0 <= scalar_v <= 1.0)

        # 概率字典推理
        _, v_probs = net.predict_probabilities(st)
        self.assertIn("win", v_probs)
        self.assertIn("draw", v_probs)
        self.assertIn("loss", v_probs)
        prob_sum = v_probs["win"] + v_probs["draw"] + v_probs["loss"]
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_benchmark_suite_evaluation(self):
        """测试基准靶场评估指标计算。"""
        suite = create_benchmark_suite()
        self.assertEqual(len(suite), 50)

        net = JunqiNet(in_channels=38)
        res = evaluate_net_benchmark(net, suite=suite)

        self.assertIn("value_mae_overall", res)
        self.assertIn("value_class_acc", res)
        self.assertIn("value_mae_endgame", res)
        self.assertTrue(0.0 <= res["value_mae_overall"] <= 2.0)

    def test_metrics_saving_and_dashboard(self):
        """测试 metrics/ 目录看板输出生成。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_data = {
                "value_mae": {
                    "value_mae_overall": 0.12,
                    "value_class_acc": 0.88,
                },
                "tournament": {
                    "score_rate_a": 0.65,
                    "draw_rate": 0.20,
                    "repetition_rate": 0.02,
                    "avg_plies": 105.4,
                    "elo_diff": 108.2,
                }
            }
            save_metrics_report(metrics_data, output_dir=tmpdir)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "value_mae.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tournament.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "benchmark_dashboard.md")))


if __name__ == "__main__":
    unittest.main()
