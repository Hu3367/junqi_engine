"""P3 批次 3 验收：Policy 分桶采样权重失衡修复（adaptive 模式）。

背景（2026-09-14 实测，checkpoint buffer_stats）：
    policy 桶 opening 8,589 / midgame 3,208 / endgame 66,409
固定权重 (0.2, 0.5, 0.3) 下，50% 的 batch 从仅 3,208 条 midgame 样本中抽取，
单样本每轮被重复曝光约 12 次，而占池 85% 的 endgame 只过 0.35 遍——既过拟合
midgame 又浪费多数数据。adaptive 模式按 √桶容量 归一化，把"单样本曝光率"的
最大/最小比从约 35 压到约 4.5，且不饿死小桶。

默认模式仍为 fixed（已通过 5 轮验证的基线），本改动以 --pool-weights adaptive 显式启用，
遵循"训练中途不动参数、变量逐个加"的纪律。
"""
from __future__ import annotations

import random
import unittest

import numpy as np

from junqi.train_rl import (StratifiedReplayBuffer, effective_policy_weights)

OBSERVED_COUNTS = (8589, 3208, 66409)      # 2026-09-14 实战池实测
MIDGAME, ENDGAME = 1, 2


def _exposure_ratio(counts, weights):
    """单样本曝光率 w_p / n_p 的最大/最小比（衡量桶失衡程度）。"""
    counts = np.asarray(counts, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    nz = (counts > 0) & (w > 0)
    per_sample = w[nz] / counts[nz]
    return float(per_sample.max() / per_sample.min())


class TestEffectivePolicyWeights(unittest.TestCase):

    def test_fixed_mode_is_legacy_weights(self):
        w = effective_policy_weights(OBSERVED_COUNTS, mode="fixed")
        np.testing.assert_allclose(w, [0.2, 0.5, 0.3], atol=1e-9)

    def test_adaptive_mode_is_sqrt_normalized(self):
        w = effective_policy_weights(OBSERVED_COUNTS, mode="adaptive")
        expected = np.sqrt(np.asarray(OBSERVED_COUNTS, dtype=float))
        np.testing.assert_allclose(w, expected / expected.sum(), atol=1e-9)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)

    def test_adaptive_fixes_exposure_imbalance(self):
        fixed = effective_policy_weights(OBSERVED_COUNTS, mode="fixed")
        adaptive = effective_policy_weights(OBSERVED_COUNTS, mode="adaptive")
        ratio_fixed = _exposure_ratio(OBSERVED_COUNTS, fixed)
        ratio_adaptive = _exposure_ratio(OBSERVED_COUNTS, adaptive)
        self.assertGreater(ratio_fixed, 30.0)          # 基线：约 34.5 倍失衡
        self.assertLess(ratio_adaptive, 6.0)           # 修复后：约 4.5 倍
        self.assertLess(ratio_adaptive, ratio_fixed / 5.0)

    def test_adaptive_does_not_starve_small_bucket(self):
        w = effective_policy_weights(OBSERVED_COUNTS, mode="adaptive")
        for i, n in enumerate(OBSERVED_COUNTS):
            quota = int(round(128 * w[i]))
            self.assertGreaterEqual(quota, 1, f"桶 {i} (n={n}) 配额被饿死")

    def test_empty_buckets_get_zero_weight(self):
        for mode in ("fixed", "adaptive"):
            w = effective_policy_weights((1000, 0, 0), mode=mode)
            np.testing.assert_allclose(w, [1.0, 0.0, 0.0], atol=1e-9)

    def test_all_empty_returns_zeros(self):
        np.testing.assert_allclose(effective_policy_weights((0, 0, 0), mode="adaptive"),
                                   [0.0, 0.0, 0.0], atol=1e-12)


class TestBufferWeightMode(unittest.TestCase):

    def _fill(self, buf, n_open, n_mid, n_end):
        for phase, n in ((0, n_open), (1, n_mid), (2, n_end)):
            for _ in range(n):
                buf.add_policy((np.zeros((38, 12, 5), dtype=np.float32),
                                np.zeros(3650, dtype=bool),
                                np.zeros(3650, dtype=np.float32), phase))

    def test_adaptive_mode_batches_from_all_buckets(self):
        buf = StratifiedReplayBuffer(capacity=900, rng=random.Random(0))
        buf.policy_weight_mode = "adaptive"
        self._fill(buf, 300, 60, 300)                  # 模拟中盘桶偏小
        batch = buf.sample_policy_batch(128)
        self.assertEqual(len(batch), 128)
        phases = {s[3] for s in batch}
        self.assertEqual(phases, {0, 1, 2}, "三个阶段都应出现在 batch 中")

    def test_default_mode_is_fixed(self):
        buf = StratifiedReplayBuffer(capacity=900, rng=random.Random(0))
        self.assertEqual(buf.policy_weight_mode, "fixed")

    def test_adaptive_shifts_batch_toward_endgame(self):
        """端到端：中盘桶偏小时，adaptive 的 batch 中盘占比显著低于 fixed。"""
        shares = {}
        for mode in ("fixed", "adaptive"):
            buf = StratifiedReplayBuffer(capacity=2000, rng=random.Random(1))
            buf.policy_weight_mode = mode
            self._fill(buf, 200, 60, 600)
            batches = [buf.sample_policy_batch(128) for _ in range(30)]
            mid = sum(1 for b in batches for s in b if s[3] == MIDGAME)
            end = sum(1 for b in batches for s in b if s[3] == ENDGAME)
            shares[mode] = mid / max(mid + end, 1)
        self.assertGreater(shares["fixed"], 0.5)       # 基线：中盘被过度采样
        self.assertLess(shares["adaptive"], 0.35)      # 自适应：向数据量大的桶倾斜


if __name__ == "__main__":
    unittest.main()
