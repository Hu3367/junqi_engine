"""P1：数据集标签口径一致性、版本守卫与高危脚本冒烟（审查 C7 / R6 / C12）。

  C7 `dataset.py` 的 outcome 统计把 code 24（断线）计入 `decided_win`，而标签函数
     `terminal_label_from_meta` 把 24 判为 `none`（不赋 Value）。两处硬编码各写一遍，
    导致 metadata.json 的 `decided_win` 虚高（p1_v3 记 488，实际含断线局）。
  R6 CLI/函数的默认数据集路径仍指向 p1_v1 / p1_v2，而训练锚点用 p1_v3；
    p1_v2 生成于 2026-09-06，早于 2026-09-13 的 code 24 标签修正，口径不同。
  C12 `train_bc` / `eval_bc` / `fit_weights` 此前零测试覆盖。
"""
from __future__ import annotations

import json
import os
import unittest

import numpy as np

from junqi.dataset import (DEFAULT_P1_DIR, MIN_P1_VERSION, outcome_bucket_from_meta,
                           terminal_label_from_meta)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestTerminalLabelTable(unittest.TestCase):
    """官方终局码 -> 标签（基线 §6 权威口径）。"""

    def test_decided_codes(self):
        for rc in (1, 21, 22, 23):
            kind, seat = terminal_label_from_meta(rc, 1)
            self.assertEqual(kind, "decided", f"code {rc} 应为明确胜负")
            self.assertEqual(seat, 0)
            self.assertEqual(terminal_label_from_meta(rc, 2)[1], 1)

    def test_draw_codes(self):
        for rc in (40, 42, 43):
            self.assertEqual(terminal_label_from_meta(rc, 3)[0], "draw",
                             f"code {rc} 应为正规和棋")

    def test_abandoned_codes_have_no_value(self):
        for rc in (20, 24):
            self.assertEqual(terminal_label_from_meta(rc, 1)[0], "none",
                             f"code {rc} 不得赋 Value（审查 §6 硬约束）")

    def test_decided_requires_valid_winner_code(self):
        self.assertEqual(terminal_label_from_meta(21, 3)[0], "draw")


class TestOutcomeBucketConsistency(unittest.TestCase):
    """C7：统计桶必须与标签同源，不得各自硬编码。"""

    def test_code_24_is_not_decided_win(self):
        bucket = outcome_bucket_from_meta({"reason_code": 24, "winner": 1})
        self.assertEqual(bucket, "special_or_unfinished",
                         "code 24（断线）不得计入 decided_win（C7 回归）")

    def test_code_20_is_not_decided_win(self):
        self.assertEqual(
            outcome_bucket_from_meta({"reason_code": 20, "winner": 2}),
            "special_or_unfinished")

    def test_decided_and_draw_buckets(self):
        self.assertEqual(outcome_bucket_from_meta({"reason_code": 21, "winner": 1}),
                         "decided_win")
        self.assertEqual(outcome_bucket_from_meta({"reason_code": 40, "winner": 3}),
                         "rule_draw")

    def test_bucket_never_contradicts_label(self):
        """穷举 code x winner：桶名与标签类别必须严格对应。"""
        mapping = {"decided": "decided_win", "draw": "rule_draw",
                   "none": "special_or_unfinished"}
        for rc in (1, 20, 21, 22, 23, 24, 40, 42, 43, 0):
            for w in (0, 1, 2, 3):
                kind, _ = terminal_label_from_meta(rc, w)
                self.assertEqual(
                    outcome_bucket_from_meta({"reason_code": rc, "winner": w}),
                    mapping[kind],
                    f"rc={rc} w={w} 统计桶与标签口径冲突")

    def test_missing_meta_fields_follow_label_table(self):
        """缺字段时取官方默认 winner=3（和棋），与标签表保持一致而非另立规则。"""
        self.assertEqual(outcome_bucket_from_meta({}),
                         outcome_bucket_from_meta({"reason_code": 0, "winner": 3}))
        self.assertEqual(outcome_bucket_from_meta({}), "rule_draw")

    def test_none_winner_value_does_not_crash(self):
        self.assertEqual(
            outcome_bucket_from_meta({"reason_code": 1, "winner": None}),
            "rule_draw")


class TestDefaultDatasetVersion(unittest.TestCase):
    """R6：默认数据集必须指向修正后的 p1_v3。"""

    def test_default_dir_is_p1_v3(self):
        self.assertEqual(DEFAULT_P1_DIR.replace("\\", "/"), "datasets/p1_v3")

    def test_min_version_is_3(self):
        self.assertEqual(tuple(MIN_P1_VERSION), (3, 0, 0))

    def test_callers_use_the_constant(self):
        """train_bc / eval_bc / train_value_distill / __main__ 不得再写死 p1_v1/v2。"""
        offenders = []
        for rel in ("junqi/train_bc.py", "junqi/eval_bc.py",
                    "junqi/train_value_distill.py", "junqi/__main__.py",
                    "junqi/dataset.py"):
            src = open(os.path.join(BASE, rel), encoding="utf-8").read()
            for bad in ('"datasets/p1_v1', "'datasets/p1_v1",
                        '"datasets/p1_v2', "'datasets/p1_v2"):
                if bad in src:
                    offenders.append(f"{rel}: {bad}")
        self.assertEqual(offenders, [], f"仍有写死的旧数据集路径: {offenders}")

    def test_shipped_metadata_version_acceptance(self):
        """现有 p1_v3 必须通过版本守卫（否则训练会直接失败）。"""
        from junqi.train_value_distill import check_p1_version

        meta_path = os.path.join(BASE, "datasets", "p1_v3", "metadata.json")
        if not os.path.exists(meta_path):
            self.skipTest("p1_v3 数据集不存在")
        meta = json.load(open(meta_path, encoding="utf-8"))
        ok, msg = check_p1_version(meta)
        self.assertTrue(ok, msg)

    def test_old_version_is_rejected(self):
        from junqi.train_value_distill import check_p1_version

        ok, msg = check_p1_version({"version": "2.0.0"})
        self.assertFalse(ok, "p1_v2 口径早于 code 24 修正，必须拒绝")
        self.assertIn("2.0.0", msg)


class TestPlanCodeConsistencyR5(unittest.TestCase):
    """R5：基地方案与代码对"认输局 Value 口径"的表述必须一致。

    审查发现方案 §6 写"双方按 ±1 计入 Value"，而 §5 P3 与 train_rl.py 相反
    （认输局只进 Policy）。代码是对的（防 Value 自证回路），方案已回写；
    本测试防止两边再次漂移。
    """

    def _plan(self) -> str:
        return open(os.path.join(BASE, "AI_TRAINING_AND_HUMAN_PLAY_PLAN.md"),
                    encoding="utf-8").read()

    def _train_rl(self) -> str:
        return open(os.path.join(BASE, "junqi", "train_rl.py"),
                    encoding="utf-8").read()

    def test_code_filters_resigned_value_samples(self):
        src = self._train_rl()
        self.assertIn("resigned_seat is not None", src)
        self.assertIn("value_samples = []", src)

    def test_plan_no_longer_claims_resign_value_is_used(self):
        plan = self._plan()
        self.assertNotIn("双方按 ±1 计入 Value", plan,
                         "方案仍在宣称认输局 ±1 进 Value，与代码相反")

    def test_plan_documents_reason_impact_rollback(self):
        plan = self._plan()
        idx = plan.index("自博弈认输局")
        section = plan[idx:idx + 1600]
        for kw in ("修订原因", "影响范围", "验证方法", "回滚"):
            self.assertIn(kw, section,
                          f"基线文档改动必须记录「{kw}」（AGENTS.md 第 10 条）")


class TestHighRiskScriptSmoke(unittest.TestCase):
    """C12：此前零覆盖的高危脚本至少要能导入、关键纯函数语义正确。"""

    def test_modules_importable(self):
        import junqi.eval_bc          # noqa: F401
        import junqi.fit_weights      # noqa: F401
        import junqi.train_bc         # noqa: F401
        import junqi.train_value_distill as tvd

        self.assertTrue(hasattr(tvd, "load_p1_arrays"))
        self.assertTrue(hasattr(tvd, "value_health_metrics"))

    def test_load_p1_arrays_filters_unlabeled(self):
        import tempfile

        from junqi.train_value_distill import load_p1_arrays

        with tempfile.TemporaryDirectory() as d:
            np.savez(os.path.join(d, "train.npz"),
                     states=np.zeros((3, 38, 12, 5), dtype=np.float32),
                     masks=np.zeros((3, 3650), dtype=bool),
                     actions=np.zeros(3, dtype=np.int64),
                     phases=np.zeros(3, dtype=np.int64),
                     values=np.array([1.0, 0.0, -1.0], dtype=np.float32),
                     val_classes=np.array([0, 1, 2], dtype=np.int64),
                     has_values=np.array([True, False, True]))
            out = load_p1_arrays(d, "train")
        self.assertEqual(out["n_total"], 3)
        self.assertEqual(out["n_labeled"], 2)
        self.assertEqual(list(out["y"]), [0, 2])

    def test_fit_weights_import_surface(self):
        import junqi.fit_weights as fw

        self.assertTrue(any(not n.startswith("_") for n in dir(fw)))


if __name__ == "__main__":
    unittest.main()
