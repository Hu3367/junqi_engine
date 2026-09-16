"""P2 搜索蒸馏「人类策略锚点」的回归测试（2026-09-16 新增）。

背景：无锚点的蒸馏会把 BC 策略**整体覆盖**成搜索偏好 —— 实测人类测试集
top-1 从 0.529 掉到 0.204/0.353，门控裁决判分仅 0.267/0.325 且显著为负。
故给损失加第二项 `β · CE(student, base_soft)`（常数意义下 = KL(base ‖ student)）。

本文件只守卫**接口与默认行为**（不重跑完整蒸馏）：
  1. `anchor_weight` 默认 0.0 ⇒ 旧行为完全不变（向后兼容）；
  2. 该值会被写入产物元信息，便于事后追溯；
  3. 端到端能跑通小规模蒸馏并产出可加载的检查点。
"""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest

from junqi.state import Action, Rank
from junqi.train_search_distill import teacher_confidence, train_search_distill


class TestAnchorInterface(unittest.TestCase):

    def test_default_is_zero_backward_compatible(self):
        """默认 anchor_weight 必须是 0.0 —— 不改变任何历史行为。"""
        sig = inspect.signature(train_search_distill)
        self.assertIn("anchor_weight", sig.parameters)
        self.assertEqual(sig.parameters["anchor_weight"].default, 0.0)

    def test_cli_exposes_flag(self):
        """`python -m junqi distill_search` 必须暴露 --anchor-weight。

        注意：该参数的 argparse 定义在 junqi/__main__.py（`ds.add_argument`），
        不是模块自身的 main()——只改后者命令行会报 unrecognized arguments。
        """
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "junqi" / "__main__.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn('ds.add_argument("--anchor-weight"', text)
        self.assertIn("anchor_weight=args.anchor_weight", text)

    def test_loss_anchors_to_base_policy(self):
        """源码层面确认锚点项确实进入了损失（防被误删）。"""
        src = inspect.getsource(train_search_distill)
        self.assertIn("ref_net", src)
        self.assertIn("anchor_weight * _soft_ce(", src)
        self.assertIn('"anchor_weight": anchor_weight', src)


class TestTeacherConfidence(unittest.TestCase):
    """教师置信度过滤：只有在教师**有明确偏好**的局面上才学习。

    病因：随机推进采样的局面里教师根分值 max−median 中位仅 27.6，
    配 T=120 时软分布接近均匀（94% 最大熵）。强行拟合会把策略推向均匀分布。
    """

    @staticmethod
    def _scored(values):
        return [(Action("flip", (r, c)), v)
                for (r, c), v in zip([(0, 0), (0, 2), (1, 1), (2, 4)], values)]

    def test_disabled_keeps_everything(self):
        self.assertEqual(teacher_confidence(self._scored([10, 5, 3, 0]), 0.0), 1.0)

    def test_flat_teacher_is_dropped(self):
        """首选只比中位高一点 ⇒ 教师没意见 ⇒ 权重 0。"""
        self.assertEqual(teacher_confidence(self._scored([11, 10, 10, 10]), 30.0), 0.0)

    def test_sharp_teacher_is_kept(self):
        """首选显著优于中位 ⇒ 保留。"""
        self.assertEqual(teacher_confidence(self._scored([100, 10, 10, 10]), 30.0), 1.0)

    def test_uses_max_minus_median_not_max_minus_min(self):
        """判据是 max−median（对单个离群差着法稳健），不是 max−min。

        此例 max−min = 100（很大），但 max−median = 5（教师其实没意见）。
        """
        self.assertEqual(teacher_confidence(self._scored([15, 14, 14, -85]), 30.0), 0.0)

    def test_degenerate_inputs(self):
        self.assertEqual(teacher_confidence([], 30.0), 0.0)
        self.assertEqual(teacher_confidence(self._scored([5]), 30.0), 0.0)
        # 关闭过滤时，单动作局面视为有效
        self.assertEqual(teacher_confidence(self._scored([5]), 0.0), 1.0)

    def test_win_score_is_clipped(self):
        """终局 ±WIN_SCORE 不应把展布算出天文数字（裁剪后再算）。"""
        big = 9_999_999.0
        self.assertEqual(teacher_confidence(self._scored([big, 0, 0, 0]), 30.0), 1.0)


class TestAnchorEndToEnd(unittest.TestCase):
    """小规模端到端：跑通并核对元信息。"""

    @unittest.skipUnless(os.path.exists("models/best.pt"), "缺少基座模型")
    def test_tiny_run_records_anchor_weight(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "tiny.pt")
            res = train_search_distill(
                base_model="models/best.pt", out_path=out,
                n_states=8, epochs=1, batch_size=4,
                depth=1, time_limit_ms=50, temperature=20.0,
                seed=7, workers=0, device="cpu",
                anchor_weight=0.5,
                val_ratio=0.2,
            )
            self.assertTrue(os.path.exists(out))
            self.assertGreaterEqual(res["states"], 1)
            ck = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(ck["anchor_weight"], 0.5)
            self.assertEqual(ck["teacher_temperature"], 20.0)


if __name__ == "__main__":
    unittest.main()
