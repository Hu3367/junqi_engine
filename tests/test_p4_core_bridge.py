"""P4 C++ 引擎网桥与双轨自动降级机制测试。"""
from __future__ import annotations

import unittest
from pathlib import Path

from junqi.core_bridge import (
    HAS_CPP_CORE,
    is_cpp_available,
    search_apk_auto,
    state_to_cpp,
)
from junqi.config import RuleConfig
from junqi.rules import Rank
from junqi.state import Action, GameState, Piece


class TestCoreBridge(unittest.TestCase):

    def setUp(self):
        self.cfg = RuleConfig()
        self.src_cpp_dir = Path(__file__).resolve().parent.parent / "src_cpp"

    def test_cpp_source_files_exist(self):
        """测试 src_cpp 核心头文件、源文件与 CMake 配置文件完整存在。"""
        self.assertTrue(self.src_cpp_dir.exists())
        self.assertTrue((self.src_cpp_dir / "CMakeLists.txt").exists())
        self.assertTrue((self.src_cpp_dir / "include" / "types.h").exists())
        self.assertTrue((self.src_cpp_dir / "include" / "constants.h").exists())
        self.assertTrue((self.src_cpp_dir / "include" / "board.h").exists())
        self.assertTrue((self.src_cpp_dir / "include" / "rules.h").exists())
        self.assertTrue((self.src_cpp_dir / "include" / "eval_apk.h").exists())
        self.assertTrue((self.src_cpp_dir / "include" / "apk_engine.h").exists())
        self.assertTrue((self.src_cpp_dir / "src" / "board.cpp").exists())
        self.assertTrue((self.src_cpp_dir / "src" / "rules.cpp").exists())
        self.assertTrue((self.src_cpp_dir / "src" / "eval_apk.cpp").exists())
        self.assertTrue((self.src_cpp_dir / "src" / "apk_engine.cpp").exists())
        self.assertTrue((self.src_cpp_dir / "bindings" / "python_bindings.cpp").exists())
        self.assertTrue((self.src_cpp_dir / "tests" / "bench_main.cpp").exists())

    def test_bridge_availability_check(self):
        """测试网桥可用性状态检测。"""
        avail = is_cpp_available()
        self.assertIsInstance(avail, bool)
        self.assertEqual(avail, HAS_CPP_CORE)

    def test_search_apk_auto_fallback_or_native(self):
        """测试 search_apk_auto 统一搜索引擎在无论是否编译的情况下均能正常输出合法走法。"""
        st = GameState(board={
            (1, 2): Piece("r", Rank.SI, True),
            (1, 3): Piece("b", Rank.JUN, True),
            (4, 0): Piece("b", Rank.PAI, False),
        }, turn=0, cfg=self.cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"

        act, score, stats = search_apk_auto(st, level="beginner", depth=2)
        self.assertIsNotNone(act)
        self.assertIn(act, st.legal_actions())
        self.assertIn("is_cpp", stats)
        self.assertIn("nodes", stats)

        # 司令吃军长或进营占据高价值点
        self.assertIn(act, [Action("move", (1, 2), (1, 3)), Action("move", (1, 2), (2, 3))])
        self.assertGreater(score, 1000.0)


if __name__ == "__main__":
    unittest.main()
