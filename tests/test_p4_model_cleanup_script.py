"""P4：models/ 清理脚本必须保护热启动链（审查附带发现）。

原 `scripts/cleanup_models.py` 的删除模式含 `*_distilled.pt`，会把
`models/value_distilled_v2.pt`（P3 修订后 train_rl 热启动链的**首选**健康 Value
头，p1_v3/test 平衡准确率 0.735）一并删除；删掉后训练静默退回 `bc_best.pt`
（价值头未校准），候选初期反而更弱。且脚本无 dry-run，直接交互式删除。
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE, "scripts", "cleanup_models.py")


def _load():
    spec = importlib.util.spec_from_file_location("cleanup_models_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestProtectedArtifacts(unittest.TestCase):

    def setUp(self):
        self.mod = _load()

    def test_warmstart_chain_is_protected(self):
        for name in ("best.pt", "bc_best.pt", "value_distilled_v2.pt",
                     "value_distilled.pt"):
            self.assertFalse(self.mod.should_delete(name),
                             f"{name} 不能出现在可删除集合里（热启动/发布链依赖）")

    def test_pool_snapshots_are_deletable(self):
        for name in ("candidate_latest_buffer.pkl", "_candidate_gate.pt",
                     "best_legacy_20260901.pt"):
            self.assertTrue(self.mod.should_delete(name), f"{name} 应可清理")

    def test_resume_entrypoint_and_distill_outputs_kept(self):
        for name in ("candidate_latest.pt", "search_distilled_20k.pt"):
            self.assertFalse(self.mod.should_delete(name),
                             f"{name} 保留（断点续训入口 / P2 蒸馏产物）")

    def test_protected_set_covers_train_rl_hotstart_chain(self):
        """与 train_rl 实际读取的热启动文件保持一致。"""
        src = open(os.path.join(BASE, "junqi", "train_rl.py"),
                   encoding="utf-8").read()
        for name in ("value_distilled_v2.pt", "value_distilled.pt", "bc_best.pt",
                     "best.pt"):
            self.assertIn(name, src, f"train_rl 未引用 {name}，保护名单需复核")
            self.assertIn(name, self.mod.PROTECTED_EXACT,
                          f"train_rl 会用 {name}，清理脚本必须保护它")


class TestDryRunByDefault(unittest.TestCase):

    def test_default_does_not_delete(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            victim = p / "candidate_latest_buffer.pkl"
            victim.write_bytes(b"x" * 16)
            rc = mod.main(["--models-dir", d])
            self.assertEqual(rc, 0)
            self.assertTrue(victim.exists(), "默认必须是 dry-run，不得删除文件")

    def test_yes_actually_deletes(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            victim = p / "_candidate_gate.pt"
            victim.write_bytes(b"x" * 16)
            keeper = p / "value_distilled_v2.pt"
            keeper.write_bytes(b"x" * 16)
            rc = mod.main(["--models-dir", d, "--yes"])
            self.assertEqual(rc, 0)
            self.assertFalse(victim.exists(), "--yes 时应删除中间产物")
            self.assertTrue(keeper.exists(), "--yes 时也必须保护热启动权重")

    def test_missing_dir_is_not_fatal(self):
        mod = _load()
        self.assertEqual(mod.main(["--models-dir", os.path.join(BASE, "nope")]), 1)


if __name__ == "__main__":
    unittest.main()
