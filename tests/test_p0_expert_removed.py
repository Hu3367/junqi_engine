"""守卫：`junqi/expert/` 已彻底删除，不得复活为"死代码"。

背景（审查 R4）：该包 11 个模块引用了一批在 `GameState`/`RuleConfig` 上
**不存在**的成员（`Move`、`get_piece_at`、`current_turn`、`turn_count`、
`get_pieces`、`PIECE_RANKS`、`is_my_base`），`import junqi.expert` 直接 NameError；
AST 依赖图显示其 fan-in 为 0。2026-09-15 按用户决策 `git rm -r junqi/expert/`。

本测试守住三条线：
  1. 目录不再存在（也防止 `__pycache__` 残留造成"似乎还在"的假象）；
  2. 全仓没有任何模块/脚本再 import 它（否则会立刻 ImportError）；
  3. 删除记录文档仍在，且含可回滚的 git 恢复命令。
"""
from __future__ import annotations

import ast
import os
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(BASE, "docs", "06-References", "DEPRECATED_EXPERT_PACKAGE.md")

SCAN_DIRS = ("junqi", "scripts", "tests")


def _py_files():
    for sub in SCAN_DIRS:
        root_dir = os.path.join(BASE, sub)
        for root, _d, fs in os.walk(root_dir):
            if "__pycache__" in root:
                continue
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    yield os.path.relpath(p, BASE).replace("\\", "/")


class TestExpertPackageRemoved(unittest.TestCase):

    def test_package_directory_is_gone(self):
        self.assertFalse(os.path.exists(os.path.join(BASE, "junqi", "expert")),
                         "junqi/expert/ 应已删除")

    def test_dependent_script_also_removed(self):
        self.assertFalse(
            os.path.exists(os.path.join(BASE, "scripts", "test_expert_core.py")),
            "scripts/test_expert_core.py 是唯一引用该包的脚本，应一并删除")

    def test_no_module_imports_expert_anywhere(self):
        offenders = []
        for rel in _py_files():
            src = open(os.path.join(BASE, rel), encoding="utf-8", errors="ignore").read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                if any(n == "junqi.expert" or n.startswith("junqi.expert.")
                       for n in names):
                    offenders.append(rel)
                    break
        self.assertEqual(sorted(set(offenders)), [],
                         "不得再 import 已删除的 junqi.expert（会立即 ImportError）")

    def test_removal_doc_exists_with_recovery_command(self):
        self.assertTrue(os.path.exists(DOC),
                        "删除记录文档缺失（docs/06-References/DEPRECATED_EXPERT_PACKAGE.md）")
        text = open(DOC, encoding="utf-8").read()
        for kw in ("git rm -r junqi/expert", "git checkout", "ExpertSearchEngine",
                   "Move", "get_piece_at"):
            self.assertIn(kw, text, f"删除记录应包含 {kw}")

    def test_online_engine_still_is_search_py(self):
        src = open(os.path.join(BASE, "junqi", "ai.py"), encoding="utf-8").read()
        self.assertIn("ExpertSearchEngine", src,
                      "在线传统搜索引擎应为 junqi/search.py::ExpertSearchEngine")
        self.assertTrue(hasattr(
            __import__("junqi.search", fromlist=["ExpertSearchEngine"]),
            "ExpertSearchEngine"))


if __name__ == "__main__":
    unittest.main()
