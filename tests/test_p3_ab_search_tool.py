"""P3：传统搜索 A/B 对照工具（`scripts/ab_search_compare.py`）的守卫。

工具本身靠两个子进程跑真实搜索，不适合在单测里端到端执行；这里守住
"能导入、探针代码可编译、参数校验合理"这三条，避免脚本腐化后才发现。

它解决的是一次真实教训：C1 想修"限时搜索浪费预算"，第一版用比值外推，
实测反而在 endgame 局面少搜一层（1000ms 预算只用 171ms）。结论必须靠
A/B 实测得出，不能靠读代码推断。
"""
from __future__ import annotations

import ast
import importlib.util
import os
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE, "scripts", "ab_search_compare.py")


def _load():
    spec = importlib.util.spec_from_file_location("ab_search_compare_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAbTool(unittest.TestCase):

    def setUp(self):
        self.mod = _load()

    def test_importable_with_expected_entrypoints(self):
        self.assertTrue(callable(self.mod.main))
        self.assertTrue(callable(self.mod.export_commit_pkg))
        self.assertTrue(callable(self.mod.run_probe))

    def test_embedded_probes_compile(self):
        for name in ("PROBE", "COMPARE"):
            src = getattr(self.mod, name)
            self.assertGreater(len(src), 200, f"{name} 内容异常")
            ast.parse(src)          # 语法必须正确，否则子进程会直接崩

    def test_probe_reports_all_three_axes(self):
        """探针必须同时覆盖：默认深度搜索 / 限时搜索 / APK 引擎。"""
        probe = self.mod.PROBE
        for token in ("depth2", "timed", "apk", "ExpertSearchEngine",
                      "ApkSearchEngine", "time_limit_ms=0", "time_limit_ms=1000"):
            self.assertIn(token, probe, f"探针缺少 {token}")

    def test_compare_flags_shallower_as_regression(self):
        """比对逻辑必须把"真的少搜了"识别为回归（唯一真正的坏信号）。"""
        comp = self.mod.COMPARE
        self.assertIn("更浅(回归!)", comp)
        self.assertIn("更深(改善)", comp)

    def test_compare_distinguishes_bookkeeping_from_real_regression(self):
        """只看 max_depth 会误判——本项目真实踩过一次。

        修复"根循环超时中断"时把降级结果退回浅一层，工具报 5 处"更浅(回归)"；
        逐层追踪后发现决策真的退化了。故比对必须同时看**节点数与决策**，
        并把"节点/决策都没变、只有 max_depth 记法不同"单列一档。
        """
        comp = self.mod.COMPARE
        self.assertIn("更浅(仅记录口径)", comp)
        self.assertIn('a.get("nodes") == b.get("nodes")', comp)
        self.assertIn('a.get("act") == b.get("act")', comp)
        self.assertIn("degraded", comp)

    def test_probe_records_decision_and_degraded_for_timed_run(self):
        """探针必须为限时搜索记录 act/score/degraded，否则比对无从谈起。"""
        probe = self.mod.PROBE
        for token in ('r["timed"] = {"act"', '"degraded"', '"score"', '"nodes"'):
            self.assertIn(token, probe, f"探针缺少 {token}")

    def test_no_repo_mutating_git_commands(self):
        """工具只允许 git archive 到临时目录，不得 checkout/worktree/reset。"""
        src = open(SCRIPT, encoding="utf-8").read()
        for bad in ("checkout", "worktree", "reset", "git rm"):
            self.assertNotIn(f'"{bad}"', src, f"工具不得使用 git {bad}")

    def test_bad_commit_raises_systemexit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                self.mod.export_commit_pkg("this-commit-does-not-exist", d)


if __name__ == "__main__":
    unittest.main()
