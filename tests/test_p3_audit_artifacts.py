"""P3：训练产物核查工具（`scripts/audit_artifacts.py`）的纯函数测试。

背景（2026-09-15）：重训前需要判断哪些产物已失效。踩过的两个坑：
  1. `models/best.pt` 与 `models/pool/bc_best.pt` **逐位相同** —— 发布模型其实
     只是 BC 基线副本，从未被训练/晋升更新。只看文件名完全看不出来。
  2. `candidate_latest.pt` 是完整 checkpoint（`{"net","optimizer",...}`），
     直接当作裸 state_dict 比对会得出错误结论（曾据此误判"7 个键不同但差值为 0"）。
本测试覆盖取权重与比对这两步纯逻辑，不需要加载真实模型。
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE, "scripts", "audit_artifacts.py")


def _load():
    spec = importlib.util.spec_from_file_location("audit_artifacts_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLoadStateDictAny(unittest.TestCase):

    def setUp(self):
        self.mod = _load()
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _p(self, name):
        return os.path.join(self._tmp.name, name)

    def test_wrapped_save_format(self):
        p = self._p("wrapped.pt")
        torch.save({"model_state": {"a": torch.zeros(2)},
                    "in_channels": 38, "num_blocks": 6, "channels": 128}, p)
        sd, kind = self.mod.load_state_dict_any(p)
        self.assertEqual(list(sd), ["a"])
        self.assertIn("包装", kind)

    def test_checkpoint_format(self):
        p = self._p("ckpt.pt")
        torch.save({"net": {"a": torch.zeros(2)}, "optimizer": {}, "epoch": 5,
                    "elo": 1500.0}, p)
        sd, kind = self.mod.load_state_dict_any(p)
        self.assertEqual(list(sd), ["a"])
        self.assertIn("checkpoint", kind)
        self.assertIn("5", kind)

    def test_bare_state_dict(self):
        p = self._p("bare.pt")
        torch.save({"a": torch.zeros(2)}, p)
        sd, kind = self.mod.load_state_dict_any(p)
        self.assertEqual(list(sd), ["a"])
        self.assertIn("裸", kind)

    def test_raw_module(self):
        p = self._p("mod.pt")
        torch.save(torch.nn.Linear(2, 3), p)
        sd, kind = self.mod.load_state_dict_any(p)
        self.assertIn("weight", sd)
        self.assertIn("pickle", kind)


class TestCompareStateDicts(unittest.TestCase):

    def setUp(self):
        self.mod = _load()

    def test_identical_detected(self):
        a = {"w": torch.ones(2), "b": torch.zeros(1)}
        r = self.mod.compare_state_dicts(a, dict(a))
        self.assertTrue(r["identical"])
        self.assertEqual(r["diff"], 0)
        self.assertEqual(r["same"], 2)

    def test_diff_and_max_delta(self):
        a = {"w": torch.zeros(3)}
        b = {"w": torch.tensor([0.0, 0.5, -2.0])}
        r = self.mod.compare_state_dicts(a, b)
        self.assertFalse(r["identical"])
        self.assertEqual(r["diff"], 1)
        self.assertAlmostEqual(r["max_delta"], 2.0)

    def test_missing_key_breaks_identity(self):
        """只在一边出现的键（如旧模型缺 aux_head）必须使 identical=False。"""
        a = {"w": torch.ones(2), "aux_head.6.weight": torch.ones(1)}
        b = {"w": torch.ones(2)}
        r = self.mod.compare_state_dicts(a, b)
        self.assertFalse(r["identical"])
        self.assertEqual(r["missing"], ["aux_head.6.weight"])

    def test_shape_mismatch_is_reported_separately(self):
        a = {"w": torch.ones(2)}
        b = {"w": torch.ones(3)}
        r = self.mod.compare_state_dicts(a, b)
        self.assertFalse(r["identical"])
        self.assertEqual(r["shape_mismatch"], ["w"])
        self.assertEqual(r["diff"], 0, "形状不符不应计入 diff")

    def test_int64_keys_compared_exactly(self):
        """BN 的 num_batches_tracked 是 int64，浮点化比较会漏判。"""
        a = {"n": torch.tensor(7, dtype=torch.int64)}
        b = {"n": torch.tensor(8, dtype=torch.int64)}
        r = self.mod.compare_state_dicts(a, b)
        self.assertFalse(r["identical"])


class TestCliContract(unittest.TestCase):

    def test_module_importable_and_has_main(self):
        mod = _load()
        self.assertTrue(callable(mod.main))

    def test_missing_dir_returns_error_code(self):
        mod = _load()
        self.assertEqual(mod.main(["--models-dir", os.path.join(BASE, "不存在")]), 1)

    def test_defaults_point_to_shipped_paths(self):
        mod = _load()
        self.assertEqual(mod.DEFAULT_MODELS, "models")
        self.assertEqual(mod.DEFAULT_P1.replace("\\", "/"), "datasets/p1_v3")


if __name__ == "__main__":
    unittest.main()
