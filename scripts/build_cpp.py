"""重建 junqi_core C++ 扩展（Windows / MSVC）。

为什么需要这个脚本（而不是直接 `pip install -e .`）：
本机构建环境里 `reg.exe` 被安全策略拦截，而 setuptools/distutils 的
`MSVCCompiler` **依赖注册表**定位 Windows SDK，于是它会：
  1. 找不到 ucrt/um/shared 头文件 → `fatal error C1083: 无法打开包括文件 'io.h'`
  2. 找不到 rc.exe → `LINK : fatal error LNK1158: cannot run 'rc.exe'`
本脚本绕过注册表，直接把 SDK 的 include/lib 写进 `INCLUDE`/`LIB`，
并把 rc.exe 所在目录前置到 `PATH`（distutils 两者都会读取）。

用法:
    python scripts/build_cpp.py            # 增量编译，产出根目录 junqi_core.*.pyd
    python scripts/build_cpp.py --clean    # 先删 build/ 与 .pyd，全量重编
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _detect_sdk() -> tuple[str, str, str]:
    """返回 (sdk_include_root, sdk_lib_root, rc_dir)，自动挑选最新的 SDK 版本。"""
    kits = Path(r"C:\Program Files (x86)\Windows Kits\10")
    if not kits.is_dir():
        raise SystemExit(f"未找到 Windows SDK: {kits}")

    def newest(sub: str) -> Path:
        d = kits / sub
        vers = sorted(p for p in d.iterdir() if p.is_dir() and p.name[0].isdigit())
        if not vers:
            raise SystemExit(f"SDK {sub} 下没有版本目录: {d}")
        return vers[-1]

    inc = newest("Include")
    lib = newest("Lib")
    rc = kits / "bin" / inc.name / "x64"
    if not (rc / "rc.exe").is_file():
        raise SystemExit(f"未找到 rc.exe: {rc}")
    return str(inc), str(lib), str(rc)


def _detect_msvc_lib() -> str:
    base = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
                r"\VC\Tools\MSVC")
    if not base.is_dir():
        return ""
    vers = sorted(p for p in base.iterdir() if p.is_dir())
    if not vers:
        return ""
    lib = vers[-1] / "lib" / "x64"
    return str(lib) if lib.is_dir() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="删除 build/ 与 .pyd 后全量重编")
    args = ap.parse_args()

    inc, lib, rc_dir = _detect_sdk()
    msvc_lib = _detect_msvc_lib()

    if args.clean:
        shutil.rmtree(ROOT / "build", ignore_errors=True)
        for p in glob.glob(str(ROOT / "junqi_core*.pyd")):
            os.remove(p)
            print("removed", p)

    os.environ["INCLUDE"] = os.pathsep.join([
        os.path.join(inc, s) for s in ("ucrt", "um", "shared", "winrt")
    ])
    os.environ["LIB"] = os.pathsep.join(
        [msvc_lib, os.path.join(lib, "ucrt", "x64"), os.path.join(lib, "um", "x64")]
        if msvc_lib else
        [os.path.join(lib, "ucrt", "x64"), os.path.join(lib, "um", "x64")]
    )
    os.environ["PATH"] = os.pathsep.join([rc_dir, os.environ.get("PATH", "")])

    print(f"SDK include : {inc}")
    print(f"SDK lib     : {lib}")
    print(f"rc.exe dir  : {rc_dir}")

    rc = subprocess.call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(ROOT),
    )
    if rc != 0:
        raise SystemExit(f"build_ext failed with exit code {rc}")

    built = sorted(glob.glob(str(ROOT / "junqi_core*.pyd")))
    print("OK ->", built)


if __name__ == "__main__":
    main()
