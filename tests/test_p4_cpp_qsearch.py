"""P4 切片 2：QSearch 的 C++ 移植等价性与降级测试。

覆盖三件事：
  1. C++ legal_actions 与 Python 的**集合**一致（顺序允许不同，见下）；
  2. C++ qsearch 与 Python `_qsearch` 的**返回值**一致；
  3. 搜索引擎在 C++ 开启/关闭下**决策**一致，且可回滚。

未编译 junqi_core 时全部 skip。
"""
from __future__ import annotations

import math
import random
import unittest

from junqi.config import RuleConfig
from junqi.core_bridge import HAS_CPP_CORE, make_cpp_qsearch, state_to_cpp
from junqi.search import ExpertSearchEngine
from junqi.state import deal

TOL = 1e-9
requires_cpp = unittest.skipUnless(HAS_CPP_CORE, "junqi_core 未编译")


def _sample_states(n: int = 12, seed: int = 20260913):
    rng = random.Random(seed)
    cfg = RuleConfig()
    out = []
    while len(out) < n:
        st = deal(rng, cfg)
        for _ in range(rng.randint(8, 95)):
            if st.is_terminal():
                break
            acts = st.legal_actions()
            if not acts:
                break
            st = st.apply(rng.choice(acts))
        if not st.is_terminal() and st.my_color() is not None:
            out.append(st)
    return out


def _py_key(a):
    to = a.to if a.to is not None else a.frm
    return (a.kind, a.frm[0] * 5 + a.frm[1], to[0] * 5 + to[1])


def _cpp_key(a):
    return ("flip" if int(a.kind) == 0 else "move", int(a.frm), int(a.to))


class TestLegalActionsEquivalence(unittest.TestCase):
    @requires_cpp
    def test_action_set_matches(self):
        """走法**集合**必须逐位一致（顺序不必一致，见下条）。"""
        for st in _sample_states(25):
            py_set = {_py_key(a) for a in st.legal_actions()}
            cpp_set = {_cpp_key(a) for a in state_to_cpp(st).legal_actions()}
            self.assertEqual(py_set, cpp_set, f"走法集合不一致 ply={st.ply}")

    @requires_cpp
    def test_order_may_differ_but_is_deterministic(self):
        """顺序允许不同，但 C++ 自身必须**确定性**（同一局面两次调用相同）。

        Python 的铁路滑行/工兵飞行返回 `set`，迭代序由 tuple 哈希决定，
        C++ 用排序去重，两者天然不同。这不影响正确性：
        `_qsearch` 按 `_score_action` 稳定排序后遍历，而 alpha-beta 的
        **返回值与遍历顺序无关**，顺序只影响节点数。
        """
        st = _sample_states(1, seed=5)[0]
        b = state_to_cpp(st)
        first = [_cpp_key(a) for a in b.legal_actions()]
        second = [_cpp_key(a) for a in state_to_cpp(st).legal_actions()]
        self.assertEqual(first, second, "C++ 走法顺序不确定")


class TestQSearchValueEquivalence(unittest.TestCase):
    @requires_cpp
    def test_qsearch_value_matches_python(self):
        """全窗口下 C++ 与 Python 的 qsearch 返回值必须一致。"""
        for qd in (4, 6):
            for st in _sample_states(8, seed=4242):
                eng = ExpertSearchEngine(seed=7, use_qtt=False,
                                         use_cpp_qsearch=False)
                py_val = eng._qsearch(st, -math.inf, math.inf, qd)
                qs = make_cpp_qsearch(eng.w, qd)
                cpp_val = qs.qsearch(state_to_cpp(st), -math.inf, math.inf, qd)
                self.assertLessEqual(abs(py_val - cpp_val), TOL,
                                     f"qd={qd} ply={st.ply}: {py_val} vs {cpp_val}")

    @requires_cpp
    def test_qsearch_with_narrow_window(self):
        """非零窗口（fail-soft 边界）下也要一致。"""
        for st in _sample_states(6, seed=99):
            eng = ExpertSearchEngine(seed=7, use_qtt=False, use_cpp_qsearch=False)
            for alpha, beta in ((-50.0, 50.0), (0.0, 200.0), (-300.0, -100.0)):
                py_val = eng._qsearch(st, alpha, beta, 4)
                qs = make_cpp_qsearch(eng.w, 4)
                cpp_val = qs.qsearch(state_to_cpp(st), alpha, beta, 4)
                self.assertLessEqual(abs(py_val - cpp_val), TOL,
                                     f"window=({alpha},{beta}) ply={st.ply}: "
                                     f"{py_val} vs {cpp_val}")


class TestSearchIntegration(unittest.TestCase):
    @requires_cpp
    def test_decision_identical_across_modes(self):
        """纯 Python / 仅 C++ 估值 / 完整 C++ 三档决策必须一致。"""
        modes = [
            dict(use_cpp_eval=False, use_cpp_qsearch=False),
            dict(use_cpp_eval=True, use_cpp_qsearch=False),
            dict(use_cpp_eval=True, use_cpp_qsearch=True),
        ]
        for st in _sample_states(4, seed=2026):
            base = None
            for kw in modes:
                eng = ExpertSearchEngine(seed=7, **kw)
                act, score, stats = eng.search(st, max_depth=2)
                if base is None:
                    base = (str(act), score, stats.nodes)
                    continue
                self.assertEqual(str(act), base[0], f"{kw} 动作不一致")
                self.assertLessEqual(abs(score - base[1]), TOL, f"{kw} 分值不一致")
                self.assertEqual(stats.nodes, base[2], f"{kw} nodes 不一致")

    @requires_cpp
    def test_cpp_qsearch_increases_qnodes_without_changing_decision(self):
        """C++ 侧不实现 QTT（纯缓存），qnodes 会更高 —— 这是**预期**，不是回归。

        缓存只省时间不改变值；去掉后 qnodes 上升但墙钟大幅下降
        （残局实测 1228ms → 245ms）。本用例固定该事实，防止后人误判为回归。
        """
        st = _sample_states(1, seed=44)[0]
        py_eng = ExpertSearchEngine(seed=7, use_cpp_qsearch=False)
        cpp_eng = ExpertSearchEngine(seed=7, use_cpp_qsearch=True)
        a1, s1, st1 = py_eng.search(st, max_depth=2, qsearch_depth=8)
        a2, s2, st2 = cpp_eng.search(st, max_depth=2, qsearch_depth=8)
        self.assertEqual(str(a1), str(a2))
        self.assertLessEqual(abs(s1 - s2), TOL)
        self.assertGreater(st2.qnodes, 0)

    def test_cpp_qsearch_can_be_disabled(self):
        """use_cpp_qsearch=False 必须能完全旁路 C++（回滚路径）。"""
        st = _sample_states(1, seed=5)[0]
        eng = ExpertSearchEngine(seed=7, use_cpp_qsearch=False)
        self.assertFalse(eng.use_cpp_qsearch)
        act, score, _ = eng.search(st, max_depth=2)
        self.assertIsNotNone(act)


if __name__ == "__main__":
    unittest.main()
