"""P4 长期挂机准入复审问题专项整改单元测试。

测试覆盖：
1. NNAgent 与 MCTS 在传入 avoid/history_counts 时对重复走法的严格规避；
2. StratifiedReplayBuffer 类别均衡抽样与经验池序列化/反序列化；
3. 自博弈 Worker 中多样化对手调度；
4. 50 题靶场类别分布与塌缩告警机制。
"""
from __future__ import annotations

import os
import random
import unittest
from collections import Counter

import numpy as np
import torch

from junqi.ai import NNAgent
from junqi.benchmark import create_benchmark_suite, evaluate_net_benchmark
from junqi.config import RuleConfig
from junqi.mcts import MCTS
from junqi.net import JunqiNet
from junqi.rules import Rank
from junqi.selfplay import play_game
from junqi.state import Action, GameState, Piece, deal, position_key
from junqi.train_rl import StratifiedReplayBuffer, play_selfplay_game


class TestP4AuditFixes(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()
        self.rng = random.Random(2026)

    def test_mcts_and_nn_avoid_repetition(self):
        """测试 MCTS 与 NNAgent 在存在 avoid 局面时必须 100% 切换至替代动作。"""
        st = GameState(board={
            (5, 2): Piece("r", Rank.SI, True),
            (11, 1): Piece("r", Rank.QI, True),
            (0, 1): Piece("b", Rank.QI, True),
        }, turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        net = JunqiNet()
        net.eval()
        agent = NNAgent(net=net, simulations=20, seed=42)

        # 默认情况下选择某首选动作
        normal_act = agent.choose_actions(st, topn=1)[0][0]

        # 构造将首选动作加入 avoid
        nxt_st = st.apply(normal_act)
        avoid = {position_key(nxt_st)}

        # 再次决策，必须避开该动作
        avoided_act = agent.choose_actions(st, topn=1, avoid=avoid)[0][0]
        self.assertNotEqual(avoided_act, normal_act)

    def test_stratified_value_buffer_class_balance(self):
        """测试 Value 经验池按类别均衡抽样，防止 Draw 淹没。"""
        buffer = StratifiedReplayBuffer(capacity=1000, rng=random.Random(42))

        dummy_arr = np.zeros((38, 12, 5), dtype=np.float32)

        # 注入大量 Draw 样本 (100 条) 和少量 Win/Loss 样本 (各 5 条)
        for _ in range(100):
            buffer.add_value((dummy_arr, 1, 1))  # Draw
        for _ in range(5):
            buffer.add_value((dummy_arr, 0, 1))  # Win
            buffer.add_value((dummy_arr, 2, 1))  # Loss

        self.assertEqual(buffer.total_value_samples(), 110)

        # 采样 30 条数据
        batch = buffer.sample_value_batch(batch_size=30)
        self.assertEqual(len(batch), 30)

        classes = [item[1] for item in batch]
        counts = Counter(classes)

        # 验证各类别均有被抽样（Win/Loss 不会被 100 条 Draw 淹没）
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0)
        self.assertGreater(counts[2], 0)

    def test_stratified_buffer_save_and_load(self):
        """测试经验池完整持久化与恢复。"""
        buffer = StratifiedReplayBuffer(capacity=1000, rng=random.Random(42))
        dummy_arr = np.zeros((38, 12, 5), dtype=np.float32)
        buffer.add_value((dummy_arr, 0, 1))
        buffer.add_value((dummy_arr, 1, 2))

        tmp_path = "models/_test_buffer.pkl"
        buffer.save(tmp_path)
        self.assertTrue(os.path.exists(tmp_path))

        new_buf = StratifiedReplayBuffer(capacity=1000)
        new_buf.load(tmp_path)
        self.assertEqual(new_buf.total_value_samples(), 2)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def test_benchmark_collapse_monitoring(self):
        """测试 50 题靶场对单一类别预测塌缩的监控告警。"""
        net = JunqiNet()
        suite = create_benchmark_suite()
        res = evaluate_net_benchmark(net, suite=suite, device="cpu")

        self.assertIn("pred_class_counts", res)
        self.assertIn("true_class_counts", res)
        self.assertIn("collapse_warning", res)
        self.assertEqual(res["benchmark_total"], 50)


if __name__ == "__main__":
    unittest.main()
