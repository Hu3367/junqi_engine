import os
import sys
from setuptools import setup, find_packages

# 动态检测 pybind11 支持
ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    ext_modules = [
        Pybind11Extension(
            "junqi_core",
            [
                "src_cpp/bindings/python_bindings.cpp",
                "src_cpp/src/board.cpp",
                "src_cpp/src/rules.cpp",
                "src_cpp/src/zobrist.cpp",
                "src_cpp/src/eval_apk.cpp",
                "src_cpp/src/eval_expert.cpp",
                "src_cpp/src/eval_expert_tables.cpp",
                "src_cpp/src/expert_qsearch.cpp",
                "src_cpp/src/expert_search.cpp",
                "src_cpp/src/apk_engine.cpp",
            ],
            include_dirs=["src_cpp/include"],
            cxx_std=17,
            extra_compile_args=["/utf-8"] if sys.platform == "win32" else [],
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    pass

setup(
    name="junqi_engine",
    version="2.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
