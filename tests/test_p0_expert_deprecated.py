"""P0/Zombie code 守卫：`junqi/expert/` 不得重新接回主流程。

背景（审查 R4，详见 reviews/CODE_REVIEW_2026-09-15.md §R4）：
`junqi/expert/` 11 个模块引用了一批在 `GameState`/`RuleConfig` 上**不存在**的
成员（`Move`、`get_piece_at`、`current_turn`、`turn_count`、`get_pieces`、
`PIECE_RANKS`、`is_my_base`），`import junqi.expert` 直接 NameError；依赖图显示
其 fan-in 为 0，主流程使用的是 `junqi/search.py` 的 `ExpertSearchEngine`。

本测试不修复该包（那是一次重写），只把两件事钉死：
  1. 包处于"废弃"状态并自陈现状；
  2. **没有任何主流程模块依赖它**——将来若有人把它接回主流程，这里会立刻失败，
     迫使对方先去读 DEPRECATED.md。
"""
from __future__ import annotations

import ast
import os
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根
PKG = os.path.join(BASE, "junqi")


def _module_source_map() -> dict:
    out = {}
    for root, _d, fs in os.walk(PKG):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, BASE).replace("\\", "/")
                out[rel] = open(p, encoding="utf-8", errors="ignore").read()
    return out


class TestExpertPackageDeprecated(unittest.TestCase):

    def test_deprecation_notice_exists(self):
        doc = os.path.join(PKG, "expert", "DEPRECATED.md")
        self.assertTrue(os.path.exists(doc),
                        "废弃说明缺失：junqi/expert/DEPRECATED.md")
        text = open(doc, encoding="utf-8").read()
        for kw in ("Move", "get_piece_at", "PIECE_RANKS", "junqi/search.py"):
            self.assertIn(kw, text, f"DEPRECATED.md 应说明 {kw} 相关现状")

    def test_no_mainline_module_imports_expert(self):
        """除 expert 自身外，任何 junqi 模块都不得 import junqi.expert。"""
        offenders = []
        for rel, src in _module_source_map().items():
            if rel.startswith("junqi/expert/") or rel == "junqi/expert.py":
                continue
            for node in ast.walk(ast.parse(src)):
                targets = []
                if isinstance(node, ast.ImportFrom):
                    targets = [node.module or ""]
                    if node.level:
                        targets.append("junqi")
                elif isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                if any(t.startswith("junqi.expert") or t == "expert"
                       for t in targets):
                    offenders.append(rel)
                    break
        self.assertEqual(sorted(set(offenders)), [],
                         "主流程模块不得依赖已废弃的 junqi.expert")

    def test_online_engine_is_search_py(self):
        """在线传统搜索引擎必须是 junqi/search.py 的 ExpertSearchEngine。"""
        self.assertTrue(os.path.exists(os.path.join(PKG, "search.py")))
        src = open(os.path.join(PKG, "ai.py"), encoding="utf-8").read()
        self.assertIn("ExpertSearchEngine", src,
                      "ai.py 应继续复用 junqi/search.py 的 ExpertSearchEngine")


if __name__ == "__main__":
    unittest.main()
