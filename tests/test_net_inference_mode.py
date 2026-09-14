"""批次 4 验收：神经网络推理热路径的模式切换短路（性能优化，语义不变）。

背景（cProfile，1 局 301 手 / sims=20）：
    self.eval() 在每次单状态推理都被调用，而 nn.Module.train/eval 会递归遍历全部
    子模块（实测 369,812 次遍历、约 8.7s 累计，占单局墙钟 123.7s 约 7%）。
    优化为「模式未变则 O(1) 返回」，语义与 nn.Module.train 一致。

验证口径（对齐 AGENTS.md「每项实现必须有对应测试或固定评测证据」）：
    1. 本文件：短路确实生效 + train/eval 语义不变 + 直接切换子模块后仍可恢复；
    2. scratch/nn_predict_baseline.py：24 单状态 + 6 批量组逐位一致（capture/verify）。
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from junqi.net import JunqiNet


class TestModeShortCircuit(unittest.TestCase):

    def setUp(self):
        self.net = JunqiNet(in_channels=38)
        self._orig_base_train = nn.Module.train
        self.base_train_calls = []

    def tearDown(self):
        nn.Module.train = self._orig_base_train

    def _spy_base_train(self):
        """统计基类 nn.Module.train 被真正调用的次数（= 是否发生整树遍历）。"""
        calls = self.base_train_calls
        orig = self._orig_base_train

        def spy(self, mode=True):
            calls.append(mode)
            return orig(self, mode)

        nn.Module.train = spy

    def test_repeated_eval_only_walks_tree_once(self):
        self._spy_base_train()
        # 一次整树遍历 = 网络自身 + 递归到每个子模块，故用"增量"断言而非绝对次数
        for _ in range(5):
            self.net.eval()
        after_first_walks = len(self.base_train_calls)
        self.assertGreater(after_first_walks, 1, "首次 eval 必须真正遍历子模块")
        for _ in range(5):
            self.net.eval()
        self.assertEqual(len(self.base_train_calls), after_first_walks,
                         "已是 eval 模式时 eval() 必须 O(1) 短路（不得再遍历子树）")
        self.net.train()
        self.assertGreater(len(self.base_train_calls), after_first_walks,
                           "模式发生变化时必须触发整树遍历")

    def test_train_eval_semantics_preserved(self):
        self.net.train()
        self.assertTrue(self.net.training)
        self.assertTrue(all(m.training for m in self.net.modules()))
        self.net.eval()
        self.assertFalse(self.net.training)
        self.assertFalse(any(m.training for m in self.net.modules()))

    def test_recovers_after_direct_submodule_switch(self):
        """直接切换子模块后，net 级别的模式切换仍能恢复整树一致（子模块切换点）。"""
        self.net.eval()
        self.net.value_head.train()                       # 模拟 train_value_distill 的用法
        self.assertTrue(self.net.value_head.training)
        self.net.train()                                  # 模式变化 → 整树遍历
        self.assertTrue(all(m.training for m in self.net.modules()))
        self.net.eval()
        self.assertFalse(any(m.training for m in self.net.modules()))

    def test_predict_paths_force_eval_mode(self):
        """predict_* 仍会把网络切到 eval（保留原语义：避免 BN 用批内统计推理）。"""
        import random as _random

        from junqi.state import deal
        st = deal(_random.Random(0), None)
        self.net.train()
        self.assertTrue(self.net.training)
        self.net.predict_probabilities(st, seat=st.turn, device="cpu")
        self.assertFalse(self.net.training, "predict_* 必须把网络切到 eval")
        self.assertFalse(any(m.training for m in self.net.modules()))

    def test_eval_returns_self(self):
        self.assertIs(self.net.eval(), self.net)
        self.assertIs(self.net.train(), self.net)


if __name__ == "__main__":
    unittest.main()
