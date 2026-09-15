"""P4 切片 1：evaluate_expert 的 C++ 移植等价性与降级测试。

验收判据见 docs/05-ExecutionPlans/CPP_EXPERT_ENGINE_PORT_PLAN.md 第五节。
未编译 junqi_core 时，除"源码/生成表完整性"外的用例全部 skip（透明降级）。
"""
from __future__ import annotations

import random
import subprocess
import sys
import unittest
from pathlib import Path

from junqi.config import EvalWeights, RuleConfig
from junqi.core_bridge import (HAS_CPP_CORE, encode_board, encode_dead,
                               eval_expert_auto, make_cpp_weights)
from junqi.eval_expert import evaluate_expert
from junqi.rules import CAMPS, COMPOSITION, NEIGHBORS, Rank
from junqi.search import ExpertSearchEngine
from junqi.state import GameState, Piece, deal

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-9

requires_cpp = unittest.skipUnless(HAS_CPP_CORE, "junqi_core 未编译")


def _sample_states(n: int = 40, seed: int = 20260913):
    rng = random.Random(seed)
    cfg = RuleConfig()
    out = []
    for _ in range(n):
        target = rng.randint(8, 90)
        st = deal(rng, cfg)
        ok = True
        for _ in range(target):
            if st.is_terminal():
                ok = False
                break
            acts = st.legal_actions()
            if not acts:
                ok = False
                break
            st = st.apply(rng.choice(acts))
        if ok and st.my_color() is not None and not st.is_terminal():
            out.append(st)
    return out


class TestGeneratedTables(unittest.TestCase):
    """顺序敏感常量表必须与 Python 真源逐位一致。"""

    def test_generated_tables_are_up_to_date(self):
        """src_cpp/src/eval_expert_tables.cpp 必须与 Python 真源同步。"""
        rc = subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "gen_expert_tables.py"), "--check"],
            cwd=str(ROOT),
        )
        self.assertEqual(rc, 0, "顺序敏感表已过期，请运行 "
                                "python scripts/gen_expert_tables.py")

    def test_cpp_sources_registered_in_build(self):
        """新增源文件必须同时登记进 setup.py 与 CMakeLists.txt，否则不会参与编译。"""
        setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")
        cmake = (ROOT / "src_cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
        for rel in ("src/eval_expert.cpp", "src/eval_expert_tables.cpp"):
            self.assertIn(rel, setup_py, f"{rel} 未登记进 setup.py")
            self.assertIn(rel, cmake, f"{rel} 未登记进 CMakeLists.txt")

    @requires_cpp
    def test_road_neighbors_match_python_order(self):
        """C++ 邻接表的**顺序**必须与 Python NEIGHBORS 完全一致。

        evaluate_expert 里 my_reach[0] 直接取邻居列表首元素，顺序差异会改变
        语义（不只是浮点误差）。
        """
        import junqi_core
        tab = junqi_core.expert_road_neighbors()
        for r in range(12):
            for c in range(5):
                idx = r * 5 + c
                expect = [n[0] * 5 + n[1] for n in NEIGHBORS[(r, c)]]
                self.assertEqual(list(tab[idx]), expect,
                                 f"邻接表顺序不一致 @({r},{c})")

    @requires_cpp
    def test_camp_order_matches_python(self):
        import junqi_core
        expect = [r * 5 + c for r, c in
                  sorted(CAMPS, key=lambda x: 0 if x in ((3, 2), (8, 2)) else 1)]
        self.assertEqual(list(junqi_core.expert_camp_order()), expect)


class TestEvalEquivalence(unittest.TestCase):
    """C++ 与 Python 估值逐位等价（阈值 1e-9）。"""

    @requires_cpp
    def test_default_weights(self):
        cw = make_cpp_weights(EvalWeights())
        worst = 0.0
        for st in _sample_states(40):
            for seat in (0, 1):
                py = evaluate_expert(st, seat)
                cpp = eval_expert_auto(st, seat, None, False, cpp_w=cw)
                worst = max(worst, abs(py - cpp))
                self.assertLessEqual(abs(py - cpp), TOL,
                                     f"seat={seat} ply={st.ply}: {py} vs {cpp}")
        # 顺序差异的理论上界约 5.7e-14，留足余量断言其未退化
        self.assertLess(worst, 1e-11, f"worst={worst:.3e} 超出预期量级")

    @requires_cpp
    def test_apk_exponential_weights(self):
        w = EvalWeights.apk_weights()
        cw = make_cpp_weights(w)
        for st in _sample_states(25, seed=777):
            for seat in (0, 1):
                py = evaluate_expert(st, seat, w)
                cpp = eval_expert_auto(st, seat, w, False, cpp_w=cw)
                self.assertLessEqual(abs(py - cpp), TOL,
                                     f"apk seat={seat}: {py} vs {cpp}")

    @requires_cpp
    def test_ignore_rule_draw_flag(self):
        """ignore_rule_draw=True 分支必须与 Python 一致。"""
        cw = make_cpp_weights(EvalWeights())
        for st in _sample_states(20, seed=4242):
            py = evaluate_expert(st, 0, None, True)
            cpp = eval_expert_auto(st, 0, None, True, cpp_w=cw)
            self.assertLessEqual(abs(py - cpp), TOL, f"{py} vs {cpp}")

    @requires_cpp
    def test_custom_weights_roundtrip(self):
        w = EvalWeights(camp_occ=7.5, threat=0.4, fortress=30.0,
                        use_dynamic_bomb=False)
        cw = make_cpp_weights(w)
        self.assertAlmostEqual(cw.camp_occ, 7.5)
        self.assertAlmostEqual(cw.threat, 0.4)
        self.assertAlmostEqual(cw.fortress, 30.0)
        self.assertFalse(cw.use_dynamic_bomb)
        for rk, v in w.piece.items():
            self.assertAlmostEqual(cw.get_piece(int(rk)), float(v))
        for st in _sample_states(15, seed=99):
            py = evaluate_expert(st, 0, w)
            cpp = eval_expert_auto(st, 0, w, False, cpp_w=cw)
            self.assertLessEqual(abs(py - cpp), TOL, f"{py} vs {cpp}")


class TestSearchIntegration(unittest.TestCase):
    """搜索引擎接入 use_cpp_eval 后决策不变。"""

    @requires_cpp
    def test_use_cpp_eval_decision_identical(self):
        """决策（动作 + 分值）必须与纯 Python 一致。

        关于节点数：C++ 侧按棋盘 idx 升序遍历，Python 侧按 board dict 插入序，
        浮点累加顺序不同 ⇒ 估值差最大 5.7e-14。该量级**足以翻转根节点的并列
        比较**（`score > current_d_best_score`），从而让搜索遍历顺序/剪枝时机
        产生 <1% 的节点数差异 —— 但不会改变最终决策。
        因此这里只严格断言决策，节点数用宽松界（2%）防止出现量级性发散。
        """
        for st in _sample_states(6, seed=2026):
            py_eng = ExpertSearchEngine(seed=7, use_cpp_eval=False)
            cpp_eng = ExpertSearchEngine(seed=7, use_cpp_eval=True)
            a1, s1, st1 = py_eng.search(st, max_depth=2)
            a2, s2, st2 = cpp_eng.search(st, max_depth=2)
            self.assertEqual(str(a1), str(a2), f"决策不一致: {a1} vs {a2}")
            self.assertLessEqual(abs(s1 - s2), TOL, f"分值不一致: {s1} vs {s2}")
            for label, x, y in (("nodes", st1.nodes, st2.nodes),
                                ("qnodes", st1.qnodes, st2.qnodes)):
                denom = max(1, x)
                self.assertLess(abs(x - y) / denom, 0.02,
                                f"{label} 差异超 2%: {x} vs {y}")

    def test_cpp_eval_can_be_disabled(self):
        """use_cpp_eval=False 必须能完全旁路 C++（回滚路径）。"""
        st = _sample_states(1, seed=5)[0]
        eng = ExpertSearchEngine(seed=7, use_cpp_eval=False)
        self.assertFalse(eng.use_cpp_eval)
        act, score, _ = eng.search(st, max_depth=2)
        self.assertIsNotNone(act)


class TestBridgeSafety(unittest.TestCase):
    """桥接层的安全不变量。"""

    def test_dead_counts_are_not_derivable_from_board(self):
        """阵亡子**不可**由棋盘反推 —— 防止后人做这个"优化"。

        `_evaluate_chance_flip` 构造的子状态把某个暗子位置替换成采样身份，
        但 dead 不变，于是 board ∪ dead ≠ 完整编制。若 C++ 侧改用
        ``dead = COMPOSITION − board_counts`` 推导，会得到负数并静默算错。
        本用例固定该事实：确实存在派生阵亡数为负的情形。
        """
        rng = random.Random(7)
        st = deal(rng, RuleConfig())
        for _ in range(40):
            st = st.apply(rng.choice(st.legal_actions()))
        hidden = st.hidden_positions()
        self.assertTrue(hidden)
        pos = hidden[0]

        negatives = 0
        for (clr, rk), _cnt in st.remaining_types().items():
            b = dict(st.board)
            b[pos] = Piece(clr, rk, True)
            board_counts = {}
            for pc in b.values():
                board_counts[(pc.color, pc.rank)] = \
                    board_counts.get((pc.color, pc.rank), 0) + 1
            for color in ("r", "b"):
                for rank_, n in COMPOSITION.items():
                    if n - board_counts.get((color, rank_), 0) < 0:
                        negatives += 1
        self.assertGreater(negatives, 0,
                           "不变式居然成立 —— 需重新评估是否可省去 encode_dead")

    def test_encode_roundtrip(self):
        st = _sample_states(3, seed=11)[0]
        bb = encode_board(st)
        self.assertEqual(len(bb), 60)
        self.assertEqual(len(encode_dead(st)), len(st.dead))
        # 有子的格子必须非零，空格必须为零
        occupied = set()
        for (r, c) in st.board:
            occupied.add(r * 5 + c)
            self.assertNotEqual(bb[r * 5 + c] & 0x1F, 0)
        for i in range(60):
            if i not in occupied:
                self.assertEqual(bb[i], 0)

    def test_stale_pyd_falls_back_to_python(self):
        """旧版 pyd（缺 eval_expert_cpp）必须优雅降级而非崩溃。

        用户环境里可能残留改动前编译的 junqi_core：`import junqi_core` 会成功
        （HAS_CPP_CORE=True），但新符号不存在。此时必须退回 Python 估值。
        """
        import junqi.core_bridge as cb

        class StaleCore:
            def eval_expert_cpp(self, *a, **k):
                raise AttributeError("module 'junqi_core' has no attribute "
                                     "'eval_expert_cpp'")

        orig_core, orig_failed = cb.junqi_core, cb._CPP_EVAL_FAILED
        st = _sample_states(1, seed=3)[0]
        try:
            cb.junqi_core = StaleCore()
            cb._CPP_EVAL_FAILED = False
            got = cb.eval_expert_auto(st, 0)
            self.assertAlmostEqual(got, evaluate_expert(st, 0), delta=TOL)
            self.assertTrue(cb._CPP_EVAL_FAILED, "失败后应永久关闭 C++ 快路径")
        finally:
            cb.junqi_core = orig_core
            cb._CPP_EVAL_FAILED = orig_failed

    def test_state_with_unassigned_seat(self):
        """首翻前 seat_color 为 None，C++ 侧必须安全返回 0.0。"""
        st = GameState(board={(4, 0): Piece("r", Rank.PAI, False)}, cfg=RuleConfig())
        self.assertIsNone(st.my_color())
        if HAS_CPP_CORE:
            self.assertEqual(eval_expert_auto(st, 0), 0.0)
            self.assertEqual(eval_expert_auto(st, 1), 0.0)
        self.assertEqual(evaluate_expert(st, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
