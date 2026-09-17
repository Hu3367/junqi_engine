"""P3 修订（2026-09-14）验收：Value 重锚、健康探针与 p1 训练器崩溃回归。

背景（docs/SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md 及其验证）：
1. 联合 RL 训练漂移共享主干 → Value 头失准；受控实验证据：lr=1e-3 微调 1 轮即把
   平衡准确率 0.763 → 0.567，lr=1e-4 保持 0.713 且策略学得更好 → 默认学习率降至 1e-4；
2. 每轮联合训练后追加"冻结主干、仅训 Value 头"的重锚（池客观终局样本 + p1_v3 官方标签）；
3. Value 验收改用 p1_v3/test 独立留出集，指标 = 平衡准确率（MAE 单项会被塌缩模型通过：
   该集和棋占 54%，"恒定预测单一类别"的 MAE 仅约 0.50）；
4. train_value_from_p1_dataset 此前引用未定义变量 mae 必然崩溃（NameError），且无测试。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from junqi.train_rl import StratifiedReplayBuffer, build_anchor_dataset, value_acceptance
from junqi.train_value_distill import (evaluate_value_health, load_p1_arrays,
                                       train_value_from_p1_dataset,
                                       train_value_head_only, value_health_metrics)


# ------------------------------------------------------------------ 桩网络

class _ChannelTrunk(nn.Module):
    """无参数主干：把编码器通道 25（material_diff）取值暴露给 Value 头。

    固定特征使"仅训 Value 头"可确定性收敛，避免依赖随机主干的偶然可学性。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x[:, 25, 0, 0].unsqueeze(-1)
        return v.repeat(1, 8)


class _StubValueNet(nn.Module):
    """测试桩：value_head 之外还有一个冻结参数 dummy，用于验证"只改 Value 头"。"""

    def __init__(self):
        super().__init__()
        self.trunk = _ChannelTrunk()
        self.dummy = nn.Parameter(torch.randn(4))
        self.policy_head = nn.Linear(8, 3650)
        self.value_head = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor):
        f = self.trunk(x)
        return self.policy_head(f), self.value_head(f)


def _separable_dataset(n: int = 512, seed: int = 0) -> dict:
    """按通道 25 取值构造可分的 W/D/L 数据（+1→Win, 0→Draw, -1→Loss）。"""
    rs = np.random.RandomState(seed)
    vals = rs.choice([1.0, 0.0, -1.0], size=n).astype(np.float32)
    x = rs.randn(n, 38, 12, 5).astype(np.float32)
    x[:, 25, 0, 0] = vals
    y = np.where(vals > 0, 0, np.where(vals < 0, 2, 1)).astype(np.int64)
    return {"x": x, "y": y, "v": vals}


def _write_p1_npz(dir_path: str, n_train: int = 384, n_val: int = 256, seed: int = 3):
    """写出最小可用的 p1 npz 划分（load_p1_arrays 只读 4 个键）。"""
    rs = np.random.RandomState(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        d = _separable_dataset(n, seed=seed + len(split))
        np.savez(os.path.join(dir_path, f"{split}.npz"),
                 states=d["x"], val_classes=d["y"], values=d["v"],
                 has_values=np.ones(n, dtype=bool))


# ------------------------------------------------------------------ 健康度指标

class TestValueHealthMetrics(unittest.TestCase):

    def test_balanced_accuracy_and_healthy_case(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        pred = labels.copy()
        m = value_health_metrics(pred, np.array([1.0, 1, 0, 0, -1, -1]), labels,
                                 np.array([1.0, 1, 0, 0, -1, -1]))
        self.assertAlmostEqual(m["balanced_acc"], 1.0)
        self.assertAlmostEqual(m["class_acc"], 1.0)
        self.assertAlmostEqual(m["mae"], 0.0)
        self.assertFalse(m["collapse_warning"])

    def test_collapse_detected_even_with_low_mae(self):
        """塌缩模型（恒定预测 Loss）必须被识别——单看 MAE 会漏判。

        忠实复现实测形态（models/candidate_latest.pt 在 p1_v3 val 上）：argmax 全为
        Loss，但 softmax 接近均匀 → 标量 p(Win)-p(Loss) 接近 0 → MAE 仅约 0.49，
        而该集和棋占 54%、真实值以 0 为中心，故 MAE 低并不代表校准良好。
        """
        labels = np.array([0] * 23 + [1] * 54 + [2] * 23)
        pred = np.full(len(labels), 2)
        m = value_health_metrics(pred, np.full(len(labels), -0.05), labels,
                                 np.array([1.0] * 23 + [0.0] * 54 + [-1.0] * 23))
        self.assertTrue(m["collapse_warning"])
        self.assertLess(m["mae"], 0.55)                    # MAE 单项判据会误放行
        self.assertLess(m["balanced_acc"], 0.4)            # 平衡准确率识破
        self.assertEqual(m["per_class_recall"][0], 0.0)

    def test_random_level_is_one_third(self):
        labels = np.array([0, 1, 2] * 10)
        pred = np.zeros(len(labels), dtype=np.int64)
        m = value_health_metrics(pred, np.zeros(len(labels)), labels, np.zeros(len(labels)))
        self.assertAlmostEqual(m["balanced_acc"], 1.0 / 3.0, places=6)


# ------------------------------------------------------------------ 仅训 Value 头

class TestTrainValueHeadOnly(unittest.TestCase):

    def test_only_value_head_parameters_change(self):
        net = _StubValueNet()
        before_dummy = net.dummy.detach().clone()
        before_policy = net.policy_head.weight.detach().clone()
        before_value = net.value_head[0].weight.detach().clone()
        tr, va = _separable_dataset(256, 1), _separable_dataset(128, 2)
        res = train_value_head_only(net, tr, va, epochs=3, batch_size=64, lr=1e-2,
                                    seed=1, device="cpu", verbose=False)
        self.assertTrue(torch.equal(before_dummy, net.dummy.detach()))
        self.assertTrue(torch.equal(before_policy, net.policy_head.weight.detach()))
        self.assertFalse(torch.equal(before_value, net.value_head[0].weight.detach()))
        self.assertEqual(res["train_samples"], 256)

    def test_recovers_scrambled_head_and_restores_best(self):
        net = _StubValueNet()
        with torch.no_grad():                              # 打乱 Value 头 = 模拟塌缩起点
            for p in net.value_head.parameters():
                p.copy_(torch.randn_like(p) * 3.0)
        tr, va = _separable_dataset(512, 5), _separable_dataset(256, 6)
        cold = evaluate_value_health(net, va["x"], va["y"], va["v"], device="cpu")
        res = train_value_head_only(net, tr, va, epochs=12, batch_size=64, lr=1e-2,
                                    seed=5, device="cpu", verbose=False)
        warm = evaluate_value_health(net, va["x"], va["y"], va["v"], device="cpu")
        self.assertGreater(warm["balanced_acc"], cold["balanced_acc"])
        self.assertGreater(warm["balanced_acc"], 0.8)
        # 结束时权重必须是"最优 epoch"的那份（模型选择依据 = 验证集平衡准确率）
        self.assertAlmostEqual(res["val_balanced_acc"], warm["balanced_acc"], places=6)
        self.assertFalse(warm["collapse_warning"])

    def test_requires_grad_restored_for_joint_training(self):
        """重锚后必须恢复全部参数的 requires_grad，否则后续联合训练只更新 Value 头。"""
        net = _StubValueNet()
        tr, va = _separable_dataset(128, 7), _separable_dataset(64, 8)
        train_value_head_only(net, tr, va, epochs=1, batch_size=64, lr=1e-3,
                              seed=7, device="cpu", verbose=False)
        for name, p in net.named_parameters():
            self.assertTrue(p.requires_grad, f"{name} 的 requires_grad 未恢复")

    def test_train_value_head_only_freezes_batchnorm_running_stats(self):
        """A1: 验证 train_value_head_only 在训练前后，冻结模块的 BatchNorm running 统计严格不变。"""
        from junqi.net import JunqiNet
        net = JunqiNet(in_channels=38, num_blocks=1, channels=16)
        before_bn = {
            name: buf.detach().clone()
            for name, buf in net.named_buffers()
            if "value_head" not in name and ("running_mean" in name or "running_var" in name)
        }
        self.assertGreater(len(before_bn), 0, "必须存在待验证的 BatchNorm 统计量")
        tr, va = _separable_dataset(128, 9), _separable_dataset(64, 10)
        train_value_head_only(net, tr, va, epochs=2, batch_size=32, lr=1e-2,
                              seed=9, device="cpu", verbose=False)
        for name, before_buf in before_bn.items():
            current_buf = dict(net.named_buffers())[name]
            self.assertTrue(
                torch.equal(current_buf, before_buf),
                f"冻结模块 BatchNorm 统计量被改写: {name}")

    def test_train_value_head_only_runs_when_samples_fewer_than_batch_size(self):
        """新增 2: 样本数少于 batch_size 时不得 0 batch 空跑，必须完整执行全部样本。"""
        net = _StubValueNet()
        before_value = net.value_head[0].weight.detach().clone()
        # 30 条样本，batch_size=64（旧代码 range(0, 30-64+1, 64) 产生空区间直接空跑）
        tr, va = _separable_dataset(30, 11), _separable_dataset(20, 12)
        res = train_value_head_only(net, tr, va, epochs=2, batch_size=64, lr=1e-2,
                                    seed=11, device="cpu", verbose=False)
        self.assertEqual(res["train_samples"], 30)
        self.assertFalse(torch.equal(before_value, net.value_head[0].weight.detach()),
                         "样本数小于 batch_size 时应正常执行训练，而非空跑丢弃")

    def test_value_target_scale_and_domain(self):
        """B3: 终局合成局面分值经 tanh 映射后值域在 [-1, 1]，且明确胜负映射为 ±1.0。"""
        import math
        from junqi.state import WIN_SCORE
        from junqi.train_value_distill import SCORE_SCALE
        win_target = math.tanh(float(WIN_SCORE) / SCORE_SCALE)
        loss_target = math.tanh(-float(WIN_SCORE) / SCORE_SCALE)
        draw_target = math.tanh(0.0 / SCORE_SCALE)
        self.assertAlmostEqual(win_target, 1.0, places=4)
        self.assertAlmostEqual(loss_target, -1.0, places=4)
        self.assertAlmostEqual(draw_target, 0.0, places=4)

    def test_train_value_head_only_rejects_insufficient_samples(self):
        """边界用例: 训练样本数少于 2 时抛出 ValueError。"""
        net = _StubValueNet()
        tr_empty = {"x": np.zeros((0, 38, 12, 5), dtype=np.float32), "y": np.zeros(0, dtype=np.int64), "v": np.zeros(0, dtype=np.float32)}
        tr_one = {"x": np.zeros((1, 38, 12, 5), dtype=np.float32), "y": np.zeros(1, dtype=np.int64), "v": np.zeros(1, dtype=np.float32)}
        va = _separable_dataset(10, 1)

        with self.assertRaises(ValueError):
            train_value_head_only(net, tr_empty, va, epochs=1, batch_size=32)
        with self.assertRaises(ValueError):
            train_value_head_only(net, tr_one, va, epochs=1, batch_size=32)

    def test_label_with_expert_serial_vs_multiprocessing(self):
        """A3: 验证 label_with_expert 在 workers=0 (单进程) 与 workers=2 (多进程) 下打标结果完全一致。"""
        from junqi.state import deal, RuleConfig
        from junqi.train_value_distill import label_with_expert
        import random

        cfg = RuleConfig()
        states = [deal(random.Random(100 + i), cfg) for i in range(12)]
        labels_serial = label_with_expert(states, depth=1, time_limit_ms=50, seed=42, workers=0)
        labels_mp = label_with_expert(states, depth=1, time_limit_ms=50, seed=42, workers=2)
        self.assertEqual(labels_serial, labels_mp, "单进程与多进程冷 TT 打标结果必须完全一致")


# ------------------------------------------------------------------ 重锚数据集组装

def _fake_p1(dir_key: str, split: str) -> dict:
    n = 100 if split == "train" else 20
    d = _separable_dataset(n, seed=11 if split == "train" else 12)
    return {"x": d["x"], "y": d["y"], "v": d["v"], "n_total": n, "n_labeled": n}


class TestBuildAnchorDataset(unittest.TestCase):

    def test_mixes_pool_objective_samples_with_p1_labels(self):
        buf = StratifiedReplayBuffer(capacity=900, rng=__import__("random").Random(0))
        for cls in (0, 1, 2):                              # 每类 60 条客观终局样本
            for _ in range(60):
                arr = np.random.randn(38, 12, 5).astype(np.float32)
                buf.add_value((arr, cls, 1, 1))
        with mock.patch("junqi.train_rl._load_anchor_p1", _fake_p1):
            data = build_anchor_dataset(buf, p1_dir="<fake>", per_class=50, p1_ratio=0.3,
                                        seed=0)
        src = data["sources"]
        self.assertEqual(src["pool_total"], 150)           # 每类 50 条上限
        self.assertEqual(src["p1_total"], round(150 * 0.3 / 0.7))
        self.assertEqual(len(data["train"]["y"]), 150 + src["p1_total"])
        self.assertEqual(sorted(np.unique(data["train"]["y"]).tolist()), [0, 1, 2])
        self.assertEqual(set(data["train"]["v"].tolist()), {1.0, 0.0, -1.0})
        self.assertEqual(len(data["val"]["y"]), 20)

    def test_empty_buffer_raises(self):
        buf = StratifiedReplayBuffer(capacity=90, rng=__import__("random").Random(0))
        with mock.patch("junqi.train_rl._load_anchor_p1", _fake_p1):
            with self.assertRaises(RuntimeError):
                build_anchor_dataset(buf, p1_dir="<fake>", seed=0)


class TestValueAcceptance(unittest.TestCase):

    def test_acceptance_requires_balanced_acc_and_no_collapse(self):
        ok, _ = value_acceptance({"balanced_acc": 0.50, "mae": 0.40,
                                  "collapse_warning": False})
        self.assertTrue(ok)
        ok, detail = value_acceptance({"balanced_acc": 0.60, "mae": 0.40,
                                       "collapse_warning": True})
        self.assertFalse(ok)
        self.assertIn("塌缩=True", detail)
        ok, _ = value_acceptance({"balanced_acc": 0.40, "mae": 0.40,
                                  "collapse_warning": False})
        self.assertFalse(ok)


# ------------------------------------------------------------------ 崩溃回归 + 端到端

class TestP1TrainerRegression(unittest.TestCase):

    def test_train_value_from_p1_dataset_runs_end_to_end(self):
        """回归：该函数曾因引用未定义变量 mae 在首轮打印处必然 NameError。"""
        from junqi.net import JunqiNet
        with tempfile.TemporaryDirectory() as tmp:
            _write_p1_npz(tmp)
            base = os.path.join(tmp, "base.pt")
            network = JunqiNet(in_channels=38)
            network.save(base)
            out = os.path.join(tmp, "out.pt")
            res = train_value_from_p1_dataset(p1_dir=tmp, base_model=base, out_path=out,
                                              epochs=1, batch_size=64, device="cpu")
            self.assertTrue(os.path.exists(out))
            self.assertIn("val_acc", res)
            self.assertGreaterEqual(res["val_acc"], 0.0)

    def test_load_p1_arrays_filters_has_values_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs = np.random.RandomState(0)
            n = 10
            d = _separable_dataset(n, seed=1)
            hv = np.array([True] * 6 + [False] * 4)        # code 20/24 不赋 Value
            np.savez(os.path.join(tmp, "train.npz"), states=d["x"], val_classes=d["y"],
                     values=d["v"], has_values=hv)
            got = load_p1_arrays(tmp, "train")
            self.assertEqual(got["n_total"], 10)
            self.assertEqual(got["n_labeled"], 6)
            self.assertEqual(len(got["y"]), 6)


if __name__ == "__main__":
    unittest.main()
