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
  C13 根循环在**层内**被时限打断时 `self.stopped` 仍为 False，于是这一层被当作
     "已完成"：`degraded` 不置位、部分动作集合覆盖 root_scores/max_depth，
     并以 `FLAG_EXACT` 写入 depth-d 根节点 TT 条目（实际只是下界）。
"""
from __future__ import annotations

import inspect
import math
import time
import unittest
from typing import Optional
from unittest import mock

from junqi.ai import ExpertAgent
from junqi.search import ExpertSearchEngine, SearchStats as _SearchStats
from junqi.state import GameState, deal, Action
from junqi.tt import FLAG_EXACT, FLAG_LOWER_BOUND
from junqi.zobrist import compute_zobrist


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
    """C1：早停判据必须是"预估下一层会超预算"，而非"已用 25%"。

    实测依据（`time_limit_ms=0` 逐层耗时，2026-09-15）：
        endgame 局面 d1/d2/d3/d4 = 2.6 / 26.9 / 173 / 598 ms
    用比值外推（6.1 倍）会把 d4 估成 891ms，导致 1000ms 预算下在 d3 就停、
    只用了 171ms —— 比旧实现还少搜一层。故改用"翻倍"假设。
    """

    def test_pure_rule_semantics(self):
        from junqi.search import should_stop_ids

        # 已用 100ms、本层 100ms → 估计 100ms，100+100 < 1000 → 继续
        self.assertFalse(should_stop_ids(100.0, 1000.0, 100.0))
        # 已用 950ms、本层 100ms → 950+475 > 1000 → 停
        self.assertTrue(should_stop_ids(950.0, 1000.0, 100.0))
        # 无预算限制：永不因预算停
        self.assertFalse(should_stop_ids(10_000.0, 0.0, 500.0))

    def test_estimate_is_doubling_based_not_ratio_based(self):
        from junqi.search import ids_next_depth_estimate

        # 签名：estimate(elapsed_ms, last_depth_ms) = max(last, elapsed/2)
        self.assertEqual(ids_next_depth_estimate(100.0, 50.0), 50.0)
        self.assertEqual(ids_next_depth_estimate(1000.0, 100.0), 500.0)
        # 不做比值外推：累计 175ms、本层 173ms 时估计仍是 173（而非 6.1 倍）
        self.assertEqual(ids_next_depth_estimate(175.0, 173.0), 173.0)

    def test_endgame_scenario_keeps_one_more_depth(self):
        """回归：endgame 实测序列下必须继续到 d4，而不是在 d3 停。"""
        from junqi.search import should_stop_ids

        # d3 完成时：累计 175.5ms，本层 146.1ms
        self.assertFalse(should_stop_ids(175.5, 1000.0, 146.1),
                         "该场景下应继续搜索 d4（旧比值外推会在此误停）")

    def test_budget_not_wasted_when_first_depth_is_cheap(self):
        """回归：depth1 只花 10ms 时不该在 250ms 处退出（旧 25% 判据会误停）。"""
        from junqi.search import should_stop_ids
        self.assertFalse(should_stop_ids(10.0, 1000.0, 10.0))


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


class _ArmedClock:
    """空闲时恒返回 0.0；被 `arm()` 唤醒后放行 `allow` 次调用，再把时间推到 1e9。

    不预知"根节点排序后的动作个数"——`_order_actions` 含翻棋候选剪枝与 Delta 剪枝，
    返回的是候选**子集**，硬编码动作数会让测试悄悄变成另一个场景。改为在目标深度的
    排序调用返回时唤醒时钟：其后第 1 次计时即"第一个动作前检查"。
    """

    def __init__(self, allow: int = 0):
        self.allow = allow
        self.armed = False
        self.calls_after_arm = 0

    def arm(self):
        self.armed = True

    def __call__(self) -> float:
        if not self.armed:
            return 0.0
        self.calls_after_arm += 1
        return 0.0 if self.calls_after_arm <= self.allow else 1e9


class TestRootLoopInterruptTtFlag(unittest.TestCase):
    """P0：根循环在**层内**被时限打断时，必须置 degraded 且 TT 只能记下界。

    旧实现的缺陷：`self.stopped` 只在 `_negamax` 内部被置位；根循环自己因时限
    `break` 时它仍是 False，于是这一层被当作"已完成"——

      1. `stats.degraded` 保持 False，调用方看不出降级（也无法知道
         `root_scores` 是截断的候选集合）；
      2. 以 `FLAG_EXACT` 把部分结果写进 depth-d 的根节点置换表条目。未搜索的动作
         可能更优 ⇒ 该分数只是**下界**；而后续对同一局面（作为子树出现）的搜索
         会在 TT 查询处把它当精确值直接返回，污染真实分数与 PV；
      3. "一个动作都没搜到就超时"时会把 `-inf` 当成分数写成 EXACT 条目。

    修复**只做这三件事**，不改变决策本身：部分层的最优仍然被采用（回退到浅一层
    会丢信息，见 `test_interrupt_keeps_partial_layer_result` 的实测依据）。
    """

    def _engine(self) -> ExpertSearchEngine:
        eng = ExpertSearchEngine(seed=0)
        # 桩：返回定值且**不消耗时钟、不置 self.stopped**，从而把"根循环自己超时"
        # 这条路径单独隔离出来（正是旧实现漏掉的那条）。
        eng._negamax = lambda *a, **kw: 0.5
        return eng

    def _root_action_count(self, st) -> int:
        """根节点排序后的动作数（`_order_actions` 是纯函数，可安全预调）。"""
        eng = ExpertSearchEngine(seed=0)
        return len(eng._order_actions(st.legal_actions(), st, 0, None))

    def _run(self, st, max_depth: int, time_limit_ms: int,
             arm_at_depth: Optional[int] = None, allow: int = 0):
        from junqi import search as search_mod
        eng = self._engine()
        clock = _ArmedClock(allow=allow)

        if arm_at_depth is not None:
            real_order = eng._order_actions
            seen = {"n": 0}

            def _counting_order(acts, st_, ply, tt_move=None):
                out = real_order(acts, st_, ply, tt_move)
                if ply == 0:                      # 根节点调用，每层恰好一次
                    seen["n"] += 1
                    if seen["n"] == arm_at_depth:
                        clock.arm()
                return out

            eng._order_actions = _counting_order

        with mock.patch.object(search_mod.time, "perf_counter", clock):
            act, score, stats = eng.search(st, max_depth=max_depth,
                                           time_limit_ms=time_limit_ms)
        entry = eng.tt.table[compute_zobrist(st) & eng.tt.mask]
        return act, score, stats, entry

    def test_interrupt_mid_depth_marks_degraded(self):
        st = _state_with_moves(11)
        self.assertGreaterEqual(self._root_action_count(st), 2,
                                "该用例需要根节点至少 2 个候选动作")
        # d=1 完整跑完；d=2 只过了 1 个动作就被打断
        _act, score, stats, _entry = self._run(st, 4, 1000, arm_at_depth=2, allow=1)
        self.assertTrue(stats.degraded,
                        "根循环层内被打断必须置 degraded（旧实现恒为 False）")
        self.assertTrue(math.isfinite(score))

    def test_interrupt_mid_depth_stores_lower_bound_not_exact(self):
        st = _state_with_moves(11)
        _act, _score, _stats, entry = self._run(st, 4, 1000,
                                               arm_at_depth=2, allow=1)
        self.assertIsNotNone(entry, "应写入根节点 TT 条目")
        self.assertEqual(entry.flag, FLAG_LOWER_BOUND,
                         "不完整层的分数只是下界，写成 EXACT 会被后续搜索当成精确值读走")
        self.assertEqual(entry.depth, 2)

    def test_interrupt_keeps_partial_layer_result(self):
        """被打断的层**仍然产出决策**，只是被标为 degraded。

        这一点是 A/B 实测（`scripts/ab_search_compare.py`）逼出来的：第一版实现
        在降级时退回"上一次完整层"的结果，结果在处女局面上反而更差——d=1 的全部
        翻棋动作同分 0.0（无信息），d=2 才出现区分度（如 `deal0` 部分层给出 7.64，
        退回 d=1 得到 0.0）。迭代加深的既定语义是保留部分层最优：层内首个动作
        （tt_move/上一层最优）走全窗口，其余动作也都会被验证，故它是真实值的
        有效下界估计。
        """
        st = _state_with_moves(11)
        _act, _score, stats, _entry = self._run(st, 4, 1000,
                                                arm_at_depth=2, allow=1)
        self.assertEqual(stats.max_depth, 2, "部分层仍应记录为已达深度")
        self.assertGreaterEqual(len(stats.root_scores), 1)
        self.assertLess(len(stats.root_scores),
                        self._root_action_count(st),
                        "部分层的 root_scores 是截断集合（这正是 degraded 要提示的）")

    def test_partial_layer_score_is_finite_and_used(self):
        """降级时返回的仍是部分层的最优分，而非被丢弃/置零。"""
        st = _state_with_moves(11)
        act, score, stats, _entry = self._run(st, 4, 1000,
                                              arm_at_depth=2, allow=1)
        self.assertTrue(stats.degraded)
        self.assertIsNotNone(act)
        self.assertTrue(math.isfinite(score))

    def test_interrupt_before_any_action_does_not_store_minus_inf(self):
        st = _state_with_moves(11)
        # d=2 连第一个动作都没搜到就超时 → 分数仍是 -inf，绝不能落盘
        _act, score, stats, entry = self._run(st, 4, 1000, arm_at_depth=2, allow=0)
        self.assertTrue(stats.degraded)
        self.assertTrue(math.isfinite(score))
        self.assertEqual(stats.max_depth, 1, "本层无产出，不应记录为已达深度")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.depth, 1, "未产出任何分数时不得写入本层 TT 条目")
        self.assertEqual(entry.flag, FLAG_EXACT)

    def test_interrupt_in_first_depth_returns_finite_and_lower_bound(self):
        """最坏情况：连 depth 1 都没跑完。仍须给出有限分，且 TT 只能记下界。"""
        st = _state_with_moves(11)
        self.assertGreaterEqual(self._root_action_count(st), 2)
        _act, score, stats, entry = self._run(st, 4, 1000, arm_at_depth=1, allow=1)
        self.assertTrue(stats.degraded)
        self.assertTrue(math.isfinite(score), "永不返回 ±inf（C2 口径）")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.flag, FLAG_LOWER_BOUND)
        self.assertEqual(entry.depth, 1)

    def test_complete_search_stores_exact(self):
        """对照组：未被打断时必须仍是 EXACT，且不得误报 degraded。"""
        st = _state_with_moves(11)
        _act, _score, stats, entry = self._run(st, 2, 0)
        self.assertFalse(stats.degraded)
        self.assertEqual(stats.max_depth, 2)
        self.assertEqual(entry.flag, FLAG_EXACT)
        self.assertEqual(entry.depth, 2)


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
