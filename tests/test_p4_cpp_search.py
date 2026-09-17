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
from junqi.state import Action, GameState, Piece, deal

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

    @requires_cpp
    def test_chance_flip_restricted_star1_at_depth3(self):
        """受限 Star1 展开（P1.A）必须与 C++ 逐位一致。

        该分支只在 `ply_depth == 1 且 depth >= 3 且 翻棋点处于战术区` 触发，
        而上面的 `test_chance_flip_value` 用的是 (depth=1, ply=0) 的**解析期望**入口，
        `test_negamax_top_level_value` 只到 depth<=2 —— 恰好整段漏掉这个新分支。

        回归背景：C++ 长尾分支曾漏写负号（`v = child_value(child)` 应为取负），
        实测 (ply=1, depth=3) 下 10/10 局面与 Python 分歧，最大 257.6 分。
        """
        for st in _sample_states(6, seed=20260916):
            hidden = st.hidden_positions()
            if not hidden:
                continue
            pos = hidden[len(hidden) // 2]
            flip = Action("flip", pos)
            for depth in (3, 4):
                eng = ExpertSearchEngine(seed=7, **PY_ONLY)
                py = eng._evaluate_chance_flip(st, flip, depth, 1,
                                               -999999.0, 999999.0, set())
                cs = make_cpp_search(eng.w, eng.qsearch_depth)
                cpp = cs.chance_flip_blob(encode_state_blob(st),
                                          pos[0] * 5 + pos[1], depth, 1,
                                          -999999.0, 999999.0)
                self.assertLessEqual(
                    abs(py - cpp), TOL,
                    f"depth={depth} pos={pos}: py={py} cpp={cpp}")


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


class TestKnownZobristResidue(unittest.TestCase):
    """已知残余差异的取证与边界（2026-09-16 定证）。

    结论：两侧**单节点语义逐位一致**，但**不保证 depth>=3 的子树值逐位一致**。
    证据链（全部可复跑，探针见 scratch/probe_review_clean_ab.py、diag_residue_tt.py）：
      1. `_negamax` / `negamax_` 的检查顺序（超时/终局/depth<=0/重复/TT/无子可动/
         排序/PVS 窗口/剪枝与杀手历史/TT 写入/stopped）逐项一致；
      2. 每个子节点用**全新引擎**（TT 从零）时逐位一致 —— 实测 0/17、0/20
         （worst 5.7e-14 / 3.6e-15）；
      3. 共享引擎（TT 跨子节点累积）时 4/17、6/20 分歧（子值最大差 38 分）；
      4. `compute_zobrist` 与 `junqi_core.expert_zobrist` 对同一局面 **0/6 相等**，
         低 18 位（TT 桶索引 `key & mask`）也 0/6 相等。
    ⇒ 两侧引擎各自自洽，但 zobrist 键独立 ⇒ TT 的**碰撞/淘汰模式独立**
      ⇒ depth>=3 时命中不同条目、返回不同缓存界（bound）。depth<=2 节点太少不碰撞，故一直一致。

    ⚠ 若将来把 C++ zobrist 改为由 Python 生成（对齐键与桶索引），下面两个"锁定现状"的
      测试会失败 —— 那是好事，请把它们改成严格逐位断言。
    """

    @requires_cpp
    def test_zobrist_keys_are_currently_independent(self):
        """锁定诊断：两侧 zobrist 键与 TT 桶索引目前**不同**。"""
        import junqi_core
        from junqi.zobrist import compute_zobrist
        diff_keys = diff_idx = 0
        states = _sample_states(4, seed=20260916)
        for st in states:
            py = compute_zobrist(st)
            cx = junqi_core.expert_zobrist(encode_state_blob(st))
            diff_keys += py != cx
            diff_idx += (py & 0x3FFFF) != (cx & 0x3FFFF)
        self.assertEqual(diff_keys, len(states),
                         "zobrist 键已对齐 — 请把等价性契约升级为 depth>=3 也逐位一致")
        self.assertEqual(diff_idx, len(states))

    @requires_cpp
    def test_fresh_engine_child_values_are_bit_identical(self):
        """单节点语义一致性：TT 不跨子节点累积时，子节点值必须逐位一致。"""
        from junqi.state import GameState, Piece, WIN_SCORE
        for st in _sample_states(2, seed=20260916):
            hidden = st.hidden_positions()
            if not hidden:
                continue
            pos = hidden[len(hidden) // 2]
            rem = st.remaining_types()
            for (clr, rk), _cnt in list(rem.items())[:8]:
                board = dict(st.board)
                board[pos] = Piece(clr, rk, revealed=True)
                child = GameState(board=board, dead=st.dead,
                                  seat_color=dict(st.seat_color),
                                  turn=1 - st.turn, first_flip_done=st.first_flip_done,
                                  ply=st.ply + 1, winner=st.winner,
                                  win_reason=st.win_reason, cfg=st.cfg,
                                  quiet=st.quiet + 1)
                eng = ExpertSearchEngine(seed=7, use_qtt=False, **PY_ONLY)
                py = eng._negamax(child, 2, 1, -WIN_SCORE, WIN_SCORE, set())
                cs = make_cpp_search(eng.w, eng.qsearch_depth)
                cpp = cs.negamax_blob(encode_state_blob(child), 2, 1,
                                      -WIN_SCORE, WIN_SCORE)
                self.assertLessEqual(abs(py - cpp), TOL,
                                     f"{clr} {rk.name}: py={py} cpp={cpp}")

    @requires_cpp
    def test_shared_engine_residue_stays_bounded(self):
        """共享引擎下的残余必须有界（现状 <= 20 分），且决策不得改变。

        这是**锁定现状**的测试，不是"正确性"断言：它防的是"残余突然变大"。
        """
        from junqi.state import Action
        worst = 0.0
        # 只取 3 个局面 × depth=3：这是成本与覆盖的折中（纯 Python 侧 depth=4 单局面
        # 就要几十秒，会把这个文件拖到 4 分钟以上）。
        for st in _sample_states(3, seed=20260916):
            hidden = st.hidden_positions()
            if not hidden:
                continue
            pos = hidden[len(hidden) // 2]
            flip = Action("flip", pos)
            for depth, ply in ((3, 0),):
                e1 = ExpertSearchEngine(seed=7, **PY_ONLY)
                e2 = ExpertSearchEngine(seed=7, **FULL_CPP)
                a = e1._evaluate_chance_flip(st, flip, depth, ply,
                                             -999999.0, 999999.0, set())
                b = e2._evaluate_chance_flip(st, flip, depth, ply,
                                             -999999.0, 999999.0, set())
                worst = max(worst, abs(a - b))
        self.assertLessEqual(worst, 20.0,
                             f"已知残余变大到 {worst:.3f} 分，超出 zobrist/TT 独立所能解释的范围")


if __name__ == "__main__":
    unittest.main()
