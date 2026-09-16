"""P4 切片 3：完整搜索子树（`_negamax` + Star1 机会节点）的 C++ 等价性测试。

边界说明（重要）：C++ 只拥有 `_negamax` 与机会节点子树；**迭代加深的根循环
留在 Python**。因此这里比对的是"同一根循环 + 不同子树实现"的最终决策。

未编译 junqi_core 时全部 skip。
"""
from __future__ import annotations

import random
import unittest

from junqi.config import EvalWeights, RuleConfig
from junqi.core_bridge import HAS_CPP_CORE, encode_state_blob, make_cpp_search
from junqi.rules import Rank
from junqi.search import ExpertSearchEngine
from junqi.state import GameState, Piece, deal

TOL = 1e-9
requires_cpp = unittest.skipUnless(HAS_CPP_CORE, "junqi_core 未编译")

PY_ONLY = dict(use_cpp_eval=False, use_cpp_qsearch=False, use_cpp_search=False)
FULL_CPP = dict(use_cpp_eval=True, use_cpp_qsearch=True, use_cpp_search=True)


def _sample_states(n: int = 6, seed: int = 20260913):
    rng = random.Random(seed)
    cfg = RuleConfig()
    out = []
    while len(out) < n:
        st = deal(rng, cfg)
        for _ in range(rng.randint(8, 70)):
            if st.is_terminal():
                break
            acts = st.legal_actions()
            if not acts:
                break
            st = st.apply(rng.choice(acts))
        if not st.is_terminal() and st.my_color() is not None:
            out.append(st)
    return out


class TestBlobCarriesWinner(unittest.TestCase):
    """blob 必须携带 winner —— 踩过一次，固定住。

    背景：`encode_state_blob` 最初没有编码 `winner`，于是 Python 根循环交给 C++ 的
    **已终局**子状态（如"一步扛旗"）被当成未终局，C++ 返回静态估值而非胜负分。
    实测症状：一步扛旗的动作在 depth=1 被估成 350 分而不是 999999，导致决策改变。
    """

    def test_terminal_state_is_seen_as_terminal_by_cpp(self):
        cfg = RuleConfig()
        st = GameState(board={(1, 1): Piece("r", Rank.GONG, True),
                              (0, 1): Piece("b", Rank.QI, True)},
                       dead=[Piece("b", Rank.LEI, True)] * 3, turn=0, cfg=cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"
        child = st.apply(st.legal_actions()[0])
        blob = encode_state_blob(child)
        # 头部最后一个 int16 之后是 max_plies；winner 位于头部第 8 个 int16
        import struct
        nd = len(child.dead)
        winner = struct.unpack_from("<h", blob, 60 + nd + 2 * 7)[0]
        self.assertEqual(winner, child.winner)
        self.assertIsNotNone(winner if winner != -2 else None)

        if HAS_CPP_CORE:
            import junqi_core
            qs = junqi_core.ExpertQSearch()
            qs.weights = junqi_core.ExpertWeights()
            val = qs.qsearch_blob(blob, float("-inf"), float("inf"), 4)
            # 终局子状态必须返回胜负分（WIN_SCORE 量级），而不是静态估值
            self.assertGreater(abs(val), 900_000.0,
                               f"C++ 未把终局状态判为终局：val={val}")


class TestDecisionEquivalence(unittest.TestCase):
    @requires_cpp
    def test_decision_identical_depths_1_to_3(self):
        for d in (1, 2, 3):
            for i, st in enumerate(_sample_states(6)):
                py = ExpertSearchEngine(seed=7, **PY_ONLY)
                cp = ExpertSearchEngine(seed=7, **FULL_CPP)
                a1, s1, _ = py.search(st, max_depth=d)
                a2, s2, _ = cp.search(st, max_depth=d)
                self.assertEqual(str(a1), str(a2),
                                 f"depth={d} state={i} 动作不一致: {a1} vs {a2}")
                self.assertLessEqual(abs(s1 - s2), TOL,
                                     f"depth={d} state={i} 分值不一致: {s1} vs {s2}")

    @requires_cpp
    def test_instant_flag_win_at_depth_1(self):
        """一步扛旗在 depth=1 必须被选中（winner 编码缺失时的直接回归用例）。"""
        cfg = RuleConfig()
        st = GameState(board={(1, 1): Piece("r", Rank.GONG, True),
                              (11, 1): Piece("r", Rank.QI, True),
                              (0, 1): Piece("b", Rank.QI, True)},
                       dead=[Piece("b", Rank.LEI, True)] * 3, turn=0, cfg=cfg)
        st.seat_color[0], st.seat_color[1] = "r", "b"
        for d in (1, 2):
            eng = ExpertSearchEngine(weights=EvalWeights.apk_weights(), seed=42,
                                     **FULL_CPP)
            act, score, _ = eng.search(st, max_depth=d)
            self.assertEqual(act.kind, "move")
            self.assertEqual(act.to, (0, 1), f"depth={d} 未选择一步扛旗")
            self.assertGreaterEqual(score, 900_000.0)

    @requires_cpp
    def test_apk_weights_equivalence(self):
        for st in _sample_states(4, seed=777):
            py = ExpertSearchEngine(weights=EvalWeights.apk_weights(), seed=7, **PY_ONLY)
            cp = ExpertSearchEngine(weights=EvalWeights.apk_weights(), seed=7, **FULL_CPP)
            a1, s1, _ = py.search(st, max_depth=2)
            a2, s2, _ = cp.search(st, max_depth=2)
            self.assertEqual(str(a1), str(a2))
            self.assertLessEqual(abs(s1 - s2), TOL)


class TestSubtreeDirectEquivalent(unittest.TestCase):
    """直接比对子树入口的返回值（绕过根循环，隔离 C++ 与 Python 实现）。"""

    @requires_cpp
    def test_negamax_top_level_value(self):
        for st in _sample_states(4, seed=4242):
            eng = ExpertSearchEngine(seed=7, use_qtt=False, **PY_ONLY)
            for d in (1, 2):
                py = eng._negamax(st, d, 1, float("-inf"), float("inf"), set())
                cs = make_cpp_search(eng.w, eng.qsearch_depth)
                cpp = cs.negamax_blob(encode_state_blob(st), d, 1,
                                      float("-inf"), float("inf"))
                self.assertLessEqual(abs(py - cpp), TOL,
                                     f"depth={d} ply={st.ply}: {py} vs {cpp}")

    @requires_cpp
    def test_chance_flip_value(self):
        """机会节点的解析期望（depth<=1 分支）必须一致。"""
        for st in _sample_states(3, seed=99):
            hidden = st.hidden_positions()
            if not hidden:
                continue
            pos = hidden[0]
            from junqi.state import Action as _A
            flip = _A("flip", pos)
            eng = ExpertSearchEngine(seed=7, use_qtt=False, **PY_ONLY)
            py = eng._evaluate_chance_flip(st, flip, 1, 0,
                                           float("-inf"), float("inf"), set())
            cs = make_cpp_search(eng.w, eng.qsearch_depth)
            cpp = cs.chance_flip_blob(encode_state_blob(st), pos[0] * 5 + pos[1],
                                      1, 0, float("-inf"), float("inf"))
            self.assertLessEqual(abs(py - cpp), TOL,
                                 f"py={py} cpp={cpp} ply={st.ply}")


class TestRollback(unittest.TestCase):
    def test_cpp_search_can_be_disabled(self):
        st = _sample_states(1, seed=5)[0]
        eng = ExpertSearchEngine(seed=7, use_cpp_search=False)
        self.assertFalse(eng.use_cpp_search)
        act, score, _ = eng.search(st, max_depth=2)
        self.assertIsNotNone(act)

    def test_stale_pyd_falls_back(self):
        """暂存一个坏的 C++ 引擎，搜索必须优雅退回 Python 而非崩溃。"""
        import junqi.core_bridge as cb
        import junqi.search as se

        class Boom:
            nodes = qnodes = chance_nodes = star1_cutoffs = 0
            pvs_researches = tt_hits = 0
            stopped = False
            qsearch_depth = 16

            def negamax_blob(self, *a, **k):
                raise RuntimeError("boom")

        st = _sample_states(1, seed=3)[0]
        eng = ExpertSearchEngine(seed=7, use_cpp_search=True)
        real_make = cb.make_cpp_search
        try:
            cb.make_cpp_search = lambda *a, **k: Boom()
            se.make_cpp_search = cb.make_cpp_search
            eng._cpp_search = None
            act, score, _ = eng.search(st, max_depth=2)
            self.assertIsNotNone(act)
            self.assertFalse(eng.use_cpp_search, "失败后应永久关闭 C++ 子树")
        finally:
            cb.make_cpp_search = real_make
            se.make_cpp_search = real_make


if __name__ == "__main__":
    unittest.main()
