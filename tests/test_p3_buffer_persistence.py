"""P3：经验池持久化瘦身（审查 P1）。

现状缺陷：`save_checkpoint(..., buffer=buffer)` **每轮**把整个回放池 pickle 落盘。
实测 models/candidate_latest_buffer.pkl 单文件 3.8 GB（三个实验目录合计约 11.4 GB），
单轮磁盘与耗时主要开销都来自这里，且写一半被中断会留下损坏的巨文件。

修复：
  1. 新增 `should_save_buffer(epoch, end_epoch, every)` 节流判据；
  2. `save_checkpoint(save_buffer=False)` 时完全不落池；
  3. 原子写（先写 .tmp 再 replace），杜绝半截文件。
"""
from __future__ import annotations

import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from junqi.net import JunqiNet
from junqi.train_rl import (StratifiedReplayBuffer, load_checkpoint,
                            save_checkpoint, should_save_buffer)

CH = 38
PHASE_MID = 1


def _policy_sample(phase: int = 1):
    """与 add_policy 契约一致：(state, mask, target, phase)。"""
    return (np.zeros((CH, 12, 5), dtype=np.float32),
            np.zeros(3650, dtype=bool),
            np.zeros(3650, dtype=np.float32),
            phase)


def _value_sample(z_cls: int = 1, is_world: int = 0):
    """与 add_value 契约一致：(state, z_cls, is_world)。"""
    return (np.zeros((CH, 12, 5), dtype=np.float32), z_cls, is_world)


class TestShouldSaveBuffer(unittest.TestCase):

    def test_every_one_always_saves(self):
        self.assertTrue(should_save_buffer(1, 10, 1))
        self.assertTrue(should_save_buffer(7, 10, 1))

    def test_every_zero_disabled_except_last_epoch(self):
        """every=0 表示不落盘，但最后一轮必须保存（否则白训）。"""
        self.assertFalse(should_save_buffer(1, 10, 0))
        self.assertFalse(should_save_buffer(5, 10, 0))
        self.assertTrue(should_save_buffer(10, 10, 0))

    def test_throttles_between_intervals(self):
        """every=5：第 5/10/... 轮与最后一轮保存，中间轮跳过。"""
        self.assertFalse(should_save_buffer(1, 12, 5))
        self.assertFalse(should_save_buffer(4, 12, 5))
        self.assertTrue(should_save_buffer(5, 12, 5))
        self.assertFalse(should_save_buffer(6, 12, 5))
        self.assertTrue(should_save_buffer(10, 12, 5))
        self.assertTrue(should_save_buffer(12, 12, 5), "最后一轮必须保存")

    def test_invalid_every_falls_back_to_always(self):
        self.assertTrue(should_save_buffer(3, 10, -1))


class _CkptCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        torch.manual_seed(0)
        self.net = JunqiNet(num_blocks=1, channels=8)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        import random as _r
        self.rng = _r.Random(0)
        self.buffer = StratifiedReplayBuffer(capacity=300, rng=_r.Random(1))
        self.buffer.add_policy(_policy_sample(PHASE_MID))
        self.buffer.add_value(_value_sample(1, 0))
        self.buffer.add_value(_value_sample(2, 1))

    def tearDown(self):
        self._tmp.cleanup()

    def _buf_path(self, ckpt):
        return ckpt.replace(".pt", "_buffer.pkl")


class TestSaveCheckpoint(_CkptCase):

    def test_save_buffer_false_writes_no_pool(self):
        ckpt = os.path.join(self.dir, "c.pt")
        save_checkpoint(ckpt, self.net, self.opt, 3, 1500.0, self.rng,
                        buffer=self.buffer, save_buffer=False)
        self.assertTrue(os.path.exists(ckpt))
        self.assertFalse(os.path.exists(self._buf_path(ckpt)),
                         "save_buffer=False 时不得写经验池文件")

    def test_save_buffer_true_roundtrip(self):
        ckpt = os.path.join(self.dir, "c2.pt")
        save_checkpoint(ckpt, self.net, self.opt, 3, 1500.0, self.rng,
                        buffer=self.buffer, save_buffer=True)
        self.assertTrue(os.path.exists(self._buf_path(ckpt)))

        restored = StratifiedReplayBuffer(capacity=300, rng=None)
        load_checkpoint(ckpt, JunqiNet(num_blocks=1, channels=8), None,
                        device="cpu", buffer=restored)
        self.assertEqual(restored.total_policy_samples(),
                         self.buffer.total_policy_samples())

    def test_no_temp_file_left_behind(self):
        ckpt = os.path.join(self.dir, "c3.pt")
        save_checkpoint(ckpt, self.net, self.opt, 1, 1500.0, self.rng,
                        buffer=self.buffer, save_buffer=True)
        leftovers = [f for f in os.listdir(self.dir) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [], "原子写后不得残留 .tmp 文件")

    def test_checkpoint_without_buffer_still_loads(self):
        ckpt = os.path.join(self.dir, "c4.pt")
        save_checkpoint(ckpt, self.net, self.opt, 2, 1501.0, self.rng)
        ck = load_checkpoint(ckpt, JunqiNet(num_blocks=1, channels=8), None,
                             device="cpu")
        self.assertEqual(ck["epoch"], 2)
        self.assertAlmostEqual(float(ck["elo"]), 1501.0)

    def test_saved_pool_is_readable_pickle(self):
        """回归保护：文件格式必须与 load 端一致（含空池场景）。"""
        ckpt = os.path.join(self.dir, "c5.pt")
        empty = StratifiedReplayBuffer(capacity=30, rng=None)
        save_checkpoint(ckpt, self.net, self.opt, 1, 1500.0, self.rng,
                        buffer=empty, save_buffer=True)
        with open(self._buf_path(ckpt), "rb") as f:
            data = pickle.load(f)
        self.assertIn("policy", data)
        self.assertIn("value", data)


if __name__ == "__main__":
    unittest.main()
