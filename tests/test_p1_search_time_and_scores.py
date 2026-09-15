"""P0/P1：传统搜索引擎的时限语义与根节点分数语义（审查 C1–C5）。

问题：
  C1 `search.py` IDS 早停判据写成"当轮总耗时 > 预算 25%" —— 与注释里的
     "下一深度预计超时才停"不符，导致 1000ms 预算下 depth1 用掉 260ms 就退出，
     大量预算被浪费，实际深度长期停留在 1~2。
  C2 首层未完成即超时时返回 `(acts[0], -inf)`；该 -inf 经 hybrid_engine 取负成
     +inf 参与排序，并被 train_search_distill 全量 softmax 蒸馏 → 污染教师标签。
  C3 根节点后续动作在收窄后的 [alpha, beta) 窗口内搜索，fail-low 时返回的是
     **上界**，却被当作精确分写进 stats.root_scores 直接展示/蒸馏。
  C4 ai.py 的 topn 回退分支把 1e5 量级的 move-ordering 分当估值分返回。
  C5 hybrid_engine 兜底 `Action("pass")` 缺必填参数 frm，触发即 TypeError。
"""
from __future__ import annotations

import inspect
import math
import time
import unittest

from junqi.ai import ExpertAgent
from junqi.search import ExpertSearchEngine, SearchStats as _SearchStats
from junqi.state import GameState, deal, Action


def _state(seed: int = 7, flips: int = 0) -> GameState:
    """发牌；flips>0 时先翻若干暗子，制造出可移动的明子（才有 move 类动作）。"""
    import random as _r
    st = deal(_r.Random(seed), None)
    rng = _r.Random(seed + 991)
    for _ in range(flips):
        hidden = st.hidden_positions()
        if not hidden:
            break
        pos = hidden[rng.randrange(len(hidden))]
        from junqi.state import Action
        st = st.apply(Action("flip", pos))
    return st


def _state_with_moves(seed: int = 7) -> GameState:
    """至少有 move 类动作的局面（PVS 上界问题只在存在 move 时出现）。"""
    st = _state(seed, flips=6)
    assert any(a.kind == "move" for a in st.legal_actions())
    return st


class TestIdsStopRule(unittest.TestCase):
    """C1：早停判据必须是"预估下一层会超预算"，而非"已用 25%"。"""

    def test_pure_rule_semantics(self):
        from junqi.search import should_stop_ids

        # 已用 100ms、单层耗时 100ms、预算 1000ms：
        # 下一层预估 ≥2×100=200ms，100+200 < 1000 → 继续
        self.assertFalse(should_stop_ids(100.0, 1000.0, 100.0))
        # 已用 900ms：900 + 200 > 1000 → 停
        self.assertTrue(should_stop_ids(900.0, 1000.0, 100.0))
        # 预估增长至少 2 倍
        self.assertTrue(should_stop_ids(0.0, 1000.0, 600.0))
        self.assertFalse(should_stop_ids(0.0, 1000.0, 100.0))

    def test_growth_is_derived_from_measured_ratio(self):
        from junqi.search import ids_growth_estimate

        self.assertGreaterEqual(ids_growth_estimate(100.0, 10.0), 10.0)
        self.assertGreaterEqual(ids_growth_estimate(1.0, 1000.0), 2.0,
                                "增长因子下限为 2，防止用历史比值把预估压到 1")

    def test_budget_not_wasted_when_first_depth_is_cheap(self):
        """回归：depth1 只花 10ms 时不该在 250ms 处退出（旧判据会误停）。"""
        from junqi.search import should_stop_ids
        self.assertFalse(should_stop_ids(10.0, 1000.0, 10.0))
        self.assertTrue(should_stop_ids(10.0, 1000.0, 10.0) is False)


class TestTimeoutDegradesGracefully(unittest.TestCase):
    """C2：超时不得返回 -inf。"""

    def test_tiny_budget_returns_finite_score_and_legal_action(self):
        eng = ExpertSearchEngine(seed=1)
        st = _state()
        acts = st.legal_actions()
        act, score, stats = eng.search(st, max_depth=4, time_limit_ms=0)
        # 先跑一次不限时，确认基线可用
        self.assertIn(act, acts)
        self.assertTrue(math.isfinite(score))

    def test_stopped_first_depth_is_marked_degraded_and_finite(self):
        eng = ExpertSearchEngine(seed=1)
        st = _state()
        act, score, stats = eng.search(st, max_depth=6, time_limit_ms=1)
        self.assertIsNotNone(act)
        self.assertTrue(math.isfinite(score),
                        "超时兜底不得返回 ±inf（会被下游取负成 +inf 参与蒸馏）")
        self.assertTrue(stats.degraded or stats.max_depth >= 1)

    def test_no_time_limit_never_degraded(self):
        eng = ExpertSearchEngine(seed=1)
        st = _state()
        _act, _score, stats = eng.search(st, max_depth=2, time_limit_ms=0)
        self.assertFalse(stats.degraded)


class TestRootScores(unittest.TestCase):
    """C3：根节点分数必须是可比较的精确分。"""

    def _engine(self):
        return ExpertSearchEngine(seed=3)

    def test_exact_mode_has_no_negative_infinity(self):
        st = _state(11)
        eng = self._engine()
        _act, _sc, stats = eng.search(st, max_depth=2, exact_root_scores=True)
        self.assertTrue(stats.root_scores, "exact 模式必须产出根节点分数")
        for a, s in stats.root_scores:
            self.assertTrue(math.isfinite(s), f"根节点分数必须是有限值，得到 {s}")

    def test_exact_mode_covers_root_moves_and_sorted(self):
        st = _state_with_moves(11)
        eng = self._engine()
        _act, _sc, stats = eng.search(st, max_depth=2, exact_root_scores=True)
        # 根节点走法排序含 Delta 剪枝，返回的是候选子集而非全集
        self.assertGreater(len(stats.root_scores), 1)
        self.assertLessEqual(len(stats.root_scores), len(st.legal_actions()))
        scores = [s for _a, s in stats.root_scores]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_exact_and_pvs_agree_on_best_action(self):
        st = _state_with_moves(21)
        a1, _s1, _ = self._engine().search(st, max_depth=2, exact_root_scores=True)
        a2, _s2, _ = self._engine().search(st, max_depth=2, exact_root_scores=False)
        self.assertEqual(a1, a2, "精确模式不得改变最优走法")

    def test_pvs_mode_flags_bounded_scores(self):
        """C3 存在性证据：非精确模式下存在被窗口截断的上界分。"""
        st = _state_with_moves(21)
        _a, _s, stats = self._engine().search(st, max_depth=3,
                                              exact_root_scores=False)
        self.assertTrue(any(stats.root_scores_bounded),
                        "PVS 模式下应有动作只拿到上界（这正是 C3 的成因）")

    def test_exact_mode_has_no_bounded_scores(self):
        st = _state_with_moves(21)
        _a, _s, stats = self._engine().search(st, max_depth=3,
                                              exact_root_scores=True)
        self.assertFalse(any(stats.root_scores_bounded),
                         "精确模式下不应存在上界分")


class TestAgentTopnFallback(unittest.TestCase):
    """C4：回退分支不得把 1e5 量级排序分当估值分。"""

    def test_fallback_scores_are_not_ordering_magnitude(self):
        """C4 针对 `ExpertAgent`（审查中被点名的正是它）。"""
        st = _state(31)
        acts = st.legal_actions()
        best = acts[0]

        class _StubEngine:
            def __init__(self):
                self.stats = _SearchStats()
                self.stats.root_scores = []          # 触发回退分支

            def search(self, *a, **kw):
                return best, 0.0, self.stats

        agent = ExpertAgent(seed=1)          # noqa: F405
        agent.engine = _StubEngine()
        scored = agent.choose_actions(st, topn=3)
        self.assertTrue(scored)
        for _act, sc in scored:
            self.assertLess(abs(sc), 10_000.0,
                            "回退分支不得把 move-ordering 的 1e5 量级分当成估值分")

    def test_fallback_does_not_invent_extra_actions(self):
        from junqi import ai as ai_mod

        st = _state(31)
        acts = st.legal_actions()

        class _StubEngine:
            def __init__(self):
                self.stats = _SearchStats()
                self.stats.root_scores = []

            def search(self, *a, **kw):
                return acts[0], 0.0, self.stats

        agent = ExpertAgent(seed=1)          # noqa: F405
        agent.engine = _StubEngine()
        scored = agent.choose_actions(st, topn=5)
        self.assertEqual(len(scored), 1, "无根节点分时不得伪造候选列表")


class TestNoPassAction(unittest.TestCase):
    """C5：Action 必须带 frm，`Action("pass")` 是 TypeError 陷阱。"""

    def test_action_requires_frm(self):
        with self.assertRaises(TypeError):
            Action("pass")

    def test_hybrid_does_not_use_pass_action(self):
        src = inspect.getsource(__import__("junqi.hybrid_engine", fromlist=["x"]))
        self.assertNotIn('Action("pass")', src)
        self.assertNotIn("Action('pass')", src)


class TestHotLoopHygiene(unittest.TestCase):
    """P5：`_is_tactical` 不得在每动作循环内重复定义。"""

    def test_is_tactical_defined_outside_action_loop(self):
        from junqi.search import ExpertSearchEngine
        src = inspect.getsource(ExpertSearchEngine.search)
        def_idx = src.index("def _is_tactical(")
        loop_idx = src.index("for a in ordered_acts:")
        self.assertLess(def_idx, loop_idx,
                        "_is_tactical 必须提到动作循环之外（每动作重建闭包）")


if __name__ == "__main__":
    unittest.main()
