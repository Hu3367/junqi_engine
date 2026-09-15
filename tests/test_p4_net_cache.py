"""P4：权重加载缓存（审查 P4）。

门控/评测链路默认每阶段 32 局、双向共 256 局/轮，而 `selfplay.make_strategy`
对**每一局**都新建策略 → 原先每局从磁盘反序列化一份 ~34MB 权重。
本测试固化：同路径+同 mtime 只加载一次；文件被改写（mtime 变化）后必须重新加载；
缓存有容量上界；缓存对象只用于推理。
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

import torch

from junqi import ai as ai_mod
from junqi.ai import NNAgent, load_net_cached
from junqi.net import JunqiNet


class TestNetCache(unittest.TestCase):

    def setUp(self):
        ai_mod._NET_CACHE.clear()
        self._tmp = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        self.path = os.path.join(self._tmp.name, "m.pt")
        JunqiNet(num_blocks=1, channels=8).save(self.path)

    def tearDown(self):
        ai_mod._NET_CACHE.clear()
        self._tmp.cleanup()

    def test_same_file_loads_once(self):
        a = load_net_cached(self.path, "cpu")
        b = load_net_cached(self.path, "cpu")
        self.assertIs(a, b, "同路径同 mtime 必须命中缓存")
        self.assertEqual(len(ai_mod._NET_CACHE), 1)

    def test_mtime_change_invalidates(self):
        load_net_cached(self.path, "cpu")
        time.sleep(0.01)
        JunqiNet(num_blocks=1, channels=8).save(self.path)   # 触发新 mtime
        os.utime(self.path, (time.time() + 5, time.time() + 5))
        c = load_net_cached(self.path, "cpu")
        self.assertEqual(len(ai_mod._NET_CACHE), 2,
                         "权重文件被改写后必须重新加载（不能继续用旧权重）")

    def test_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(ai_mod._NET_CACHE_MAX + 6):
                p = os.path.join(d, f"m{i}.pt")
                JunqiNet(num_blocks=1, channels=8).save(p)
                load_net_cached(p, "cpu")
        self.assertLessEqual(len(ai_mod._NET_CACHE), ai_mod._NET_CACHE_MAX)

    def test_nn_agent_uses_cache(self):
        a1 = NNAgent(model_path=self.path, simulations=1, device="cpu", seed=1)
        a2 = NNAgent(model_path=self.path, simulations=1, device="cpu", seed=2)
        self.assertIs(a1.net, a2.net,
                      "同权重的两个 NNAgent 应共享同一份网络，而非各自反序列化")

    def test_cached_net_is_in_eval_mode(self):
        net = load_net_cached(self.path, "cpu")
        self.assertFalse(net.training, "缓存副本必须处于 eval 模式（仅推理）")

    def test_missing_file_falls_back(self):
        p = os.path.join(self._tmp.name, "nope.pt")
        with self.assertRaises(Exception):
            load_net_cached(p, "cpu")


if __name__ == "__main__":
    unittest.main()
