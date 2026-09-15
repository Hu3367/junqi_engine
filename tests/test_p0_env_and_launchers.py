"""守卫：虚拟环境移出工程后，启动链路必须给出可操作的提示。

背景（2026-09-15 用户实测）：
    $ python -m junqi gui
      ...
      File "junqi/hybrid_engine.py", line 28, in <module>
        import torch
    ModuleNotFoundError: No module named 'torch'

`junqi/__init__.py` 会连带 `import torch`（经 ai/hybrid_engine），而虚拟环境已从
`junqi_engine/venv/` 移到同级 `../venv_junqi_engine/`。此时若用 PATH 上的系统
python（本机为 Python 3.13，无 torch）运行，就会得到上面这条**看不出该换哪个
解释器**的报错。

本测试固化三件事：
  1. 用"没有 torch 的解释器"跑 `python -m junqi --help`，报错必须是可操作提示
     （提到 venv 路径 + 当前解释器），而不是裸的 ModuleNotFoundError；
  2. 工程根目录提供 `run.bat` 统一启动器，且同时识别"工程内 venv"与"同级 venv"
     两种布局；
  3. `scripts/run_test.ps1` 不再裸调 `python`。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR_NAMES = ("venv_junqi_engine",)


def _find_torchless_interpreter():
    """找一个没有 torch 的解释器，用来复现"用错解释器"的场景。"""
    cands = []
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p:
            cands.append(p)
    cands.append(sys.executable)
    for p in cands:
        try:
            r = subprocess.run([p, "-c", "import torch"], capture_output=True,
                               timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0:
            return p
    return None


class TestFriendlyEnvDiagnostic(unittest.TestCase):

    def test_wrong_interpreter_gets_actionable_message(self):
        py = _find_torchless_interpreter()
        if py is None:
            self.skipTest("本机找不到缺 torch 的解释器，无法复现场景")
        r = subprocess.run([py, "-m", "junqi", "--help"], cwd=BASE,
                           capture_output=True, timeout=180)
        out = (r.stdout + r.stderr).decode("utf-8", "replace")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("缺少第三方依赖", out, "应给出中文可操作提示")
        self.assertIn("venv_junqi_engine", out, "提示里必须写明正确的虚拟环境路径")
        self.assertIn("run.bat", out, "提示里应给出封装脚本入口")
        self.assertIn("当前解释器", out, "提示里应回显实际使用的解释器")

    def test_init_keeps_original_exception_chain(self):
        src = open(os.path.join(BASE, "junqi", "__init__.py"), encoding="utf-8").read()
        self.assertIn("from _exc", src, "必须保留原始异常链（raise ... from exc）")
        self.assertIn("_THIRD_PARTY_DEPS", src,
                      "只应拦截第三方依赖缺失，不得吞掉项目自身的 ImportError")

    def test_non_dependency_import_errors_still_propagate(self):
        """项目自身的 ImportError 必须原样抛出，不能被包装成环境提示。"""
        src = open(os.path.join(BASE, "junqi", "__init__.py"), encoding="utf-8").read()
        self.assertIn("if _exc.name not in _THIRD_PARTY_DEPS:", src)
        self.assertIn("raise\n", src)


class TestLaunchers(unittest.TestCase):

    def test_run_bat_exists_and_locates_both_layouts(self):
        p = os.path.join(BASE, "run.bat")
        self.assertTrue(os.path.exists(p), "工程根目录应提供 run.bat 统一启动器")
        text = open(p, encoding="utf-8", errors="ignore").read()
        self.assertIn("venv\\Scripts\\python.exe", text,
                      "必须支持工程内 venv（旧布局/自建环境）")
        self.assertIn("..\\venv_junqi_engine\\Scripts\\python.exe", text,
                      "必须支持同级 venv（当前布局）")
        self.assertIn("-m junqi", text)
        self.assertIn("test", text, "run.bat test 应转发到 pytest")

    def test_run_tests_bat_locates_both_layouts(self):
        text = open(os.path.join(BASE, "run_tests.bat"),
                    encoding="utf-8", errors="ignore").read()
        self.assertIn("venv\\Scripts\\python.exe", text)
        self.assertIn("..\\venv_junqi_engine\\Scripts\\python.exe", text)

    def test_powershell_runner_does_not_call_bare_python(self):
        text = open(os.path.join(BASE, "scripts", "run_test.ps1"),
                    encoding="utf-8", errors="ignore").read()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("REM"):
                continue
            self.assertFalse(stripped.startswith("python "),
                             f"run_test.ps1 不得裸调 python：{stripped[:70]}")
        self.assertIn("venv_junqi_engine", text)

    def test_readme_documents_new_venv_location(self):
        text = open(os.path.join(BASE, "README.md"), encoding="utf-8").read()
        self.assertIn("venv_junqi_engine", text)
        self.assertIn("run.bat", text)


class TestPowerShellScripts(unittest.TestCase):
    """PowerShell 5.1 把「无 BOM 的 UTF-8」按本地编码（GBK）读取。

    含中文的 `.ps1` 若不带 BOM，中文注释会被解析成乱码，进而出现
    `ParseException: 语句块或类型定义中缺少右"}"` 这类看起来毫无道理的语法错误
    （2026-09-15 实测：`.\\activate_env.ps1` 直接无法运行）。
    所以：**含非 ASCII 的 .ps1 必须带 UTF-8 BOM**。
    """

    BOM = b"\xef\xbb\xbf"

    def _ps1_files(self):
        for root, dirs, files in os.walk(BASE):
            dirs[:] = [d for d in dirs if d not in (
                ".git", "__pycache__", "apk_extracted", "军旗复盘", "models",
                "datasets", "build", ".pytest_cache", "scratch")]
            for f in files:
                if f.endswith(".ps1"):
                    yield os.path.join(root, f)

    def test_non_ascii_ps1_has_utf8_bom(self):
        offenders = []
        for p in self._ps1_files():
            raw = open(p, "rb").read()
            if any(b > 127 for b in raw) and raw[:3] != self.BOM:
                offenders.append(os.path.relpath(p, BASE))
        self.assertEqual(offenders, [],
                         f"含中文的 .ps1 必须带 UTF-8 BOM（否则 PS 5.1 解析失败）: {offenders}")

    def test_ps1_scripts_parse_cleanly(self):
        """用 PowerShell 自身的解析器验证语法（BOM 问题会在这一步暴露）。"""
        pwsh = shutil.which("powershell") or shutil.which("pwsh")
        if not pwsh:
            self.skipTest("本机没有 PowerShell，跳过语法校验")
        targets = [os.path.join(BASE, "activate_env.ps1"),
                   os.path.join(BASE, "scripts", "run_test.ps1")]
        for p in targets:
            if not os.path.exists(p):
                continue
            script = (
                "$e=$null;"
                f"[System.Management.Automation.Language.Parser]::ParseFile('{p}',"
                "[ref]$null,[ref]$e) | Out-Null; $e.Count"
            )
            r = subprocess.run([pwsh, "-NoProfile", "-NonInteractive",
                                "-Command", script],
                               capture_output=True, timeout=120)
            out = (r.stdout + r.stderr).decode("utf-8", "replace").strip()
            self.assertTrue(out.endswith("0"),
                            f"{os.path.basename(p)} 存在 PowerShell 语法错误：{out[:400]}")


class TestActivateEnvScripts(unittest.TestCase):
    """新增的根目录激活脚本（解决'手敲相对路径少一个点'的问题）。"""

    def test_files_exist(self):
        for name in ("activate_env.ps1", "activate_env.cmd"):
            self.assertTrue(os.path.exists(os.path.join(BASE, name)),
                            f"工程根目录应提供 {name}")

    def test_ps1_locates_both_layouts_and_checks_torch(self):
        text = open(os.path.join(BASE, "activate_env.ps1"),
                    encoding="utf-8-sig").read()
        self.assertIn("venv\\Scripts\\Activate.ps1", text)
        self.assertIn("..", text)
        self.assertIn("venv_junqi_engine\\Scripts\\Activate.ps1", text)
        self.assertIn("import torch", text, "激活后应做依赖自检")
        self.assertIn("残留", text, "应清理残留的旧激活状态（PATH 指向已删除目录）")

    def test_cmd_locates_both_layouts(self):
        text = open(os.path.join(BASE, "activate_env.cmd"),
                    encoding="utf-8", errors="ignore").read()
        self.assertIn("venv\\Scripts\\activate.bat", text)
        self.assertIn("..\\venv_junqi_engine\\Scripts\\activate.bat", text)

    def test_ps1_uses_scripts_subdir_for_interpreter(self):
        """回归：解释器路径必须拼到 <venv>\\Scripts\\python.exe，而不是 <venv>\\python.exe。"""
        text = open(os.path.join(BASE, "activate_env.ps1"),
                    encoding="utf-8-sig").read()
        self.assertIn("'Scripts\\python.exe'", text)

    def test_readme_mentions_activate_env(self):
        text = open(os.path.join(BASE, "README.md"), encoding="utf-8").read()
        self.assertIn("activate_env.ps1", text)


class TestNoStaleVenvPaths(unittest.TestCase):
    """文档/脚本中不应再残留指向工程内 venv 的**可执行**路径。"""

    ALLOW = {
        # 这两处是在描述"移动前的位置"，属历史记录，非可执行示例
        "docs/CHANGELOG.md",
        "reviews/CODE_REVIEW_2026-09-15.md",
        # 本测试自身用该字符串做检查
        "tests/test_p0_env_and_launchers.py",
    }

    def test_no_runnable_stale_paths(self):
        offenders = []
        old_fs = "junqi_engine/venv"
        for root, dirs, files in os.walk(BASE):
            dirs[:] = [d for d in dirs if d not in (
                ".git", "__pycache__", "apk_extracted", "军旗复盘", "models",
                "datasets", "build", ".pytest_cache", "scratch", "metrics",
                ".workbuddy")]
            for f in files:
                if not f.endswith((".md", ".py", ".bat", ".ps1", ".json")):
                    continue
                p = os.path.join(root, f)
                rel = os.path.relpath(p, BASE).replace("\\", "/")
                if rel in self.ALLOW:
                    continue
                try:
                    s = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if old_fs in s:
                    offenders.append(rel)
        self.assertEqual(sorted(set(offenders)), [],
                         f"仍指向工程内 venv 的旧路径: {sorted(set(offenders))}")


if __name__ == "__main__":
    unittest.main()
