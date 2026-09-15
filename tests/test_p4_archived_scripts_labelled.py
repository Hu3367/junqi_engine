"""守卫：历史脚本目录必须保持"明确标注为归档"，且不冒充测试。

审查 C10/C12 附带发现：`scripts/archive/` 与 `tests/utils/` 里放的是**一次性验证
脚本**，但既没有测试覆盖、也不在 pytest 收集范围内。尤其 `tests/utils/` 位于
`tests/` 之下，极易被误读为"被执行过的测试"——实际上它们文件名不匹配
`pytest.ini` 的 `python_files = test_*.py`，从未被收集执行过。

本测试守住三条线：
  1. 两个目录都有 README 标注，说明它们不是测试/不是 API；
  2. 这些脚本确实**不会**被 pytest 收集（防止有人改名成 test_*.py 后
     把一次性脚本塞进质量门）；
  3. 脚本依赖的 `train_rl` 内部原语当前仍然存在（若将来删除，这里会先失败，
     提醒同步处理，而不是静默 ImportError）。
"""
from __future__ import annotations

import os
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(BASE, "scripts", "archive")
TEST_UTILS = os.path.join(BASE, "tests", "utils")


class TestArchivedScriptsLabelled(unittest.TestCase):

    def _read(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_archive_readme_exists(self):
        p = os.path.join(ARCHIVE, "README.md")
        self.assertTrue(os.path.exists(p), "scripts/archive/README.md 缺失")
        text = self._read(p)
        for kw in ("归档", "一次性", "不要", "pytest"):
            self.assertIn(kw, text, f"archive README 应说明 {kw}")

    def test_tests_utils_readme_exists(self):
        p = os.path.join(TEST_UTILS, "README.md")
        self.assertTrue(os.path.exists(p), "tests/utils/README.md 缺失")
        text = self._read(p)
        for kw in ("不是测试", "test_*.py", "评测", "pytest"):
            self.assertIn(kw, text, f"tests/utils README 应说明 {kw}")

    def test_archived_scripts_are_not_pytest_collectable(self):
        """归档脚本文件名不得匹配 python_files，否则会被当成正式测试。"""
        offenders = []
        for d in (ARCHIVE, TEST_UTILS):
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if name.endswith(".py") and name.startswith("test_"):
                    offenders.append(os.path.join(os.path.basename(d), name))
        self.assertEqual(offenders, [],
                         f"归档脚本不得命名为 test_*.py（会被 pytest 收集）: {offenders}")

    def test_pytest_ini_scopes_collection_to_tests(self):
        ini = self._read(os.path.join(BASE, "pytest.ini"))
        self.assertIn("testpaths = tests", ini)
        self.assertIn("python_files = test_*.py", ini)

    def test_archive_scripts_reference_surviving_primitives(self):
        """这些脚本 import 的 train_rl 原语若被删除，应在此处先失败而非静默爆掉。"""
        import junqi.train_rl as tr

        used = set()
        for d in (ARCHIVE, TEST_UTILS):
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if not name.endswith(".py"):
                    continue
                src = self._read(os.path.join(d, name))
                for prim in ("wilson_lower_bound", "decide_promotion"):
                    if f"import {prim}" in src or f", {prim}" in src:
                        used.add(prim)
        for prim in sorted(used):
            self.assertTrue(hasattr(tr, prim),
                            f"归档脚本仍在 import train_rl.{prim}，它被删了会导致 ImportError")

    def test_script_expectations_are_documented(self):
        """归档 README 必须给出'需要长期守门就写正式用例'的替代路径。"""
        text = self._read(os.path.join(ARCHIVE, "README.md"))
        for kw in ("tests/", "eval_gate", "benchmark"):
            self.assertIn(kw, text, f"archive README 应给出 {kw} 相关替代路径")


if __name__ == "__main__":
    unittest.main()
