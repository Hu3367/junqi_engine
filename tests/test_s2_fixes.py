"""S2 整改专项单测（依据 docs/TRAINING_ROOT_CAUSE_REVIEW.md §S2）。

覆盖：
1. 温度表上调（探索恢复）；
2. 公共模式 Value 样本（模式标志、标签合法性）与世界模式叶子并存；
3. StratifiedReplayBuffer 公共/世界混合采样与旧 3 元组兼容；
4. 决胜课程与中盘注入的确定性与可运行性；
5. 辅助回归头（forward_with_aux 形状/值域、forward 2 元组兼容、train_epoch 四元组）；
6. 专家价值蒸馏：分数→类别映射、局面采样与打标冒烟、冻结策略头。
"""
from __future__ import annotations

import os
import random
import tempfile
import unittest

import numpy as np
import torch

from junqi.analysis import PHASE_ENDGAME, PHASE_MIDGAME, PHASE_OPENING
from junqi.config import RuleConfig
from junqi.net import JunqiNet
from junqi.state import WIN_SCORE
from junqi.train_rl import (PUBLIC_VALUE_RATIO, TEMP_BY_PHASE,
                            StratifiedReplayBuffer, play_selfplay_game,
                            train_epoch)
from junqi.train_value_distill import (gen_positions, label_with_expert,
                                       score_to_class, train_value_distill)


class _StubNet:
    """无 torch 前向的桩网络：均匀先验 + 零估值。"""

    def eval(self):
        return self

    def predict_state(self, state, seat=None, world=None, history_counts=None, device="cpu"):
        acts = state.legal_actions()
        return ({a: 1.0 / len(acts) for a in acts}, 0.0)

    def predict_batch(self, items, device="cpu"):
        return [self.predict_state(it[0], it[1], it[2], it[3] if len(it) > 3 else None)
                for it in items]


# ---------------------------------------------------------------- 温度与课程

class TestS2TemperaturesAndCurriculum(unittest.TestCase):

    def test_temperature_table_raised(self):
        self.assertAlmostEqual(TEMP_BY_PHASE[PHASE_OPENING], 1.2)
        self.assertAlmostEqual(TEMP_BY_PHASE[PHASE_MIDGAME], 1.0)
        self.assertAlmostEqual(TEMP_BY_PHASE[PHASE_ENDGAME], 0.5)

    def test_curriculum_game_determinism(self):
        """决胜课程（curriculum_prob=1.0）同种子两次运行样本流必须一致。"""
        def run_once():
            p, v, pv, _rec = play_selfplay_game(_StubNet(), sims=4, device="cpu",
                                                seed=555, curriculum_prob=1.0)
            return len(p), len(v), len(pv)

        self.assertEqual(run_once(), run_once())

    def test_midgame_injection_runs(self):
        """中盘注入（midgame_prob=1.0）必须能完整跑完一局且可复现。"""
        def run_once():
            p, v, pv, _rec = play_selfplay_game(_StubNet(), sims=4, device="cpu",
                                          seed=777, curriculum_prob=0.0,
                                          midgame_prob=1.0)
            return len(p), len(v), len(pv)

        res = run_once()
        self.assertGreater(res[0], 0)
        self.assertEqual(res, run_once())


# ---------------------------------------------------------------- Value 双流样本

class TestS2ValueSamples(unittest.TestCase):

    def test_public_and_world_samples_structure(self):
        p, v, pv, _rec = play_selfplay_game(_StubNet(), sims=4, device="cpu",
                                      seed=123, curriculum_prob=0.0)
        self.assertGreater(len(pv), 0, "必须产生公共模式 Value 样本")
        for arr, z, phase, is_world in pv:
            self.assertEqual(is_world, 0)
            self.assertEqual(arr[35, 0, 0], 0.0, "公共模式样本的模式通道必须为 0")
            self.assertIn(z, (0, 1, 2))
            self.assertIn(phase, (0, 1, 2))
        for arr, z, phase, is_world in v:
            self.assertEqual(is_world, 1)
            self.assertIn(z, (0, 1, 2))

    def test_buffer_mixed_sampling_ratio(self):
        """公共/世界样本各半时，批内公共占比应接近 PUBLIC_VALUE_RATIO。"""
        buf = StratifiedReplayBuffer(capacity=4000, rng=random.Random(7))
        arr = np.zeros((38, 12, 5), dtype=np.float32)
        for i in range(300):
            buf.add_value((arr, 1, 1, 0))   # 公共模式 Draw
            buf.add_value((arr, 1, 1, 1))   # 世界模式 Draw
        n_pub = n_tot = 0
        for _ in range(30):
            batch = buf.sample_value_batch(64)
            n_tot += len(batch)
            n_pub += sum(1 for s in batch if (s[3] if len(s) > 3 else 1) == 0)
        ratio = n_pub / n_tot
        self.assertLess(abs(ratio - PUBLIC_VALUE_RATIO), 0.18,
                        f"公共模式占比 {ratio:.3f} 偏离目标 {PUBLIC_VALUE_RATIO} 过大")

    def test_buffer_backward_compat_3_tuple(self):
        """旧版 3 元组样本（无模式标志）必须仍可采样（按世界模式处理）。"""
        buf = StratifiedReplayBuffer(capacity=100, rng=random.Random(3))
        arr = np.zeros((38, 12, 5), dtype=np.float32)
        for z in (0, 1, 2):
            buf.add_value((arr, z, 1))
        batch = buf.sample_value_batch(8)
        self.assertEqual(len(batch), 8)


# ---------------------------------------------------------------- 辅助回归头

class TestS2AuxHead(unittest.TestCase):

    def _small_net(self):
        return JunqiNet(in_channels=38, num_blocks=1, channels=16)

    def test_forward_with_aux_shapes_and_range(self):
        net = self._small_net()
        x = torch.randn(2, 38, 12, 5)
        out = net.forward_with_aux(x)
        self.assertEqual(len(out), 3)
        logits, v_logits, aux = out
        self.assertEqual(logits.shape, (2, 3650))
        self.assertEqual(v_logits.shape, (2, 3))
        self.assertEqual(aux.shape, (2,))
        self.assertTrue(torch.all(aux >= -1.0) and torch.all(aux <= 1.0))

    def test_forward_still_returns_2_tuple(self):
        net = self._small_net()
        out = net(torch.randn(1, 38, 12, 5))
        self.assertEqual(len(out), 2, "推理接口必须保持 2 元组（向后兼容）")

    def test_train_epoch_returns_aux_loss(self):
        net = self._small_net()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        buf = StratifiedReplayBuffer(capacity=100, rng=random.Random(5))
        arr = np.random.RandomState(1).rand(38, 12, 5).astype(np.float32)
        mask = np.zeros(3650, dtype=bool)
        mask[0] = True
        pi = np.zeros(3650, dtype=np.float32)
        pi[0] = 1.0
        for _ in range(6):
            buf.add_policy((arr, mask, pi, 1))
            buf.add_value((arr, 1, 1, 0))
            buf.add_value((arr, 0, 1, 1))
        res = train_epoch(net, buf, opt, batch_size=4, steps_per_epoch=2,
                          device="cpu")
        self.assertEqual(len(res), 4, "train_epoch 必须返回 (loss, p, v, aux)")
        self.assertTrue(all(x >= 0.0 for x in res))


# ---------------------------------------------------------------- 专家蒸馏

class TestS2Distillation(unittest.TestCase):

    def test_score_to_class_mapping(self):
        self.assertEqual(score_to_class(WIN_SCORE), 0)
        self.assertEqual(score_to_class(-WIN_SCORE), 2)
        self.assertEqual(score_to_class(0.0), 1)
        self.assertEqual(score_to_class(400.0), 0)
        self.assertEqual(score_to_class(-400.0), 2)
        self.assertEqual(score_to_class(60.0), 1)

    def test_gen_positions_and_label_smoke(self):
        rng = random.Random(5)
        cfg = RuleConfig()
        positions = gen_positions(rng, cfg, n_opening=2, n_midgame=2,
                                  n_endgame=2)
        self.assertGreaterEqual(len(positions), 1)
        for st, phase in positions:
            self.assertFalse(st.is_terminal())
            self.assertIsNotNone(st.my_color())
            self.assertIn(phase, (0, 1, 2))
        labels = label_with_expert([st for st, _ in positions][:3],
                                   depth=1, time_limit_ms=100, seed=5)
        self.assertEqual(len(labels), min(3, len(positions)))
        for z in labels:
            self.assertIn(z, (0, 1, 2))

    def test_distill_smoke_freezes_policy(self):
        """蒸馏冒烟：产出文件、返回指标，且策略头/主干权重冻结不变。"""
        with tempfile.TemporaryDirectory() as td:
            base_path = os.path.join(td, "base.pt")
            out_path = os.path.join(td, "distilled.pt")
            base = JunqiNet(in_channels=38, num_blocks=1, channels=16)
            base.save(base_path)

            metrics = train_value_distill(base_model=base_path, out_path=out_path,
                                          n_samples=8, epochs=1, batch_size=4,
                                          depth=1, time_limit_ms=50, seed=3,
                                          device="cpu")
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("best_val_acc", metrics)
            self.assertGreaterEqual(metrics["samples"], 1)

            distilled = JunqiNet.load_from_file(out_path)
            for k in ("in_conv.0.weight", "policy_head.4.weight"):
                self.assertTrue(
                    torch.equal(distilled.state_dict()[k], base.state_dict()[k]),
                    f"蒸馏不得改动冻结参数 {k}")


if __name__ == "__main__":
    unittest.main()
