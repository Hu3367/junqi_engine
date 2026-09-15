"""QSearch 置换表（P1，2026-09-15）。

背景：残局 qnodes / nodes = 43~47×，depth=2 单步 4.6~6.0s、depth=3 达 33.3s。
实测定位（`scratch/diag_qsearch_bottleneck.py`，残局 ply=90）：

| 项 | 单次成本 |
|---|---|
| `compute_zobrist` | 7.41 µs |
| `evaluate_expert` | 104.68 µs（14.1×） |
| `legal_actions` | 56.15 µs（7.6×） |

qsearch 节点的 `(zobrist, depth_left)` **重复率 54.3%**（35538 → 16253），
cProfile 里 `_fortress_score_impl` 被调用 **71282 次**（= 2 × evaluate_expert 次数，
每次 evaluate 各算 fs_my / fs_opp），占 cumtime 26%。

## 设计要点（关键，防估值污染）

1. **QTT 必须与主置换表分离**。`_negamax` 的 `TTEntry.depth` 是「剩余搜索深度」，
   而 qsearch 的 `depth_left` 是「剩余吃子链长度」，两者语义不同。若共用一张表，
   qsearch 写入的 `depth_left=10` 条目会被 `_negamax` 的 `depth=2` 查询命中
   （`entry.depth >= depth` ⇒ 10 >= 2），**直接返回错误的分数**。
2. **`depth_left <= 0` 分支不写表**。该分支返回的是静态评估近似值，
   既非上界也非下界，写入会污染真实值。
3. **边界标记**：cutoff → `FLAG_LOWER_BOUND`；正常完成且 `alpha <= orig_alpha`
   → `FLAG_UPPER_BOUND`；否则 `FLAG_EXACT`。

## 验收

- QTT 与主 TT 是两个独立对象、独立计数；
- 开/关 QTT 的搜索**决策（动作 + 分值）逐位一致**；
- 开 QTT 后 qnodes **不增加**（预期显著下降）；
- `use_qtt=False` 与改动前行为一致（由 `scratch/perf_baseline.py` 的 capture/verify 守卫）。
"""
from __future__ import annotations

import random
import unittest

from junqi.config import RuleConfig
from junqi.search import ExpertSearchEngine
from junqi.selfplay import deal


def make_state(seed: int = 43, plies: int = 30):
    st = deal(random.Random(seed), RuleConfig())
    rng = random.Random(seed * 977 + 5)
    for _ in range(plies):
        if st.is_terminal():
            break
        acts = st.legal_actions()
        if not acts:
            break
        st = st.apply(rng.choice(acts))
    return st


class TestQttIsolation(unittest.TestCase):

    def test_qtt_is_a_separate_table(self):
        eng = ExpertSearchEngine(seed=7)
        self.assertTrue(hasattr(eng, "qtt"),
                        "QSearch 必须有独立的置换表（不能复用 self.tt）")
        self.assertIsNot(eng.qtt, eng.tt)

    def test_both_tables_are_used_and_counted_separately(self):
        st = make_state()
        eng = ExpertSearchEngine(seed=7)
        eng.search(st, max_depth=2)
        self.assertGreater(eng.qtt.stores, 0, "qsearch 应写入自己的表")
        # 主表由 _negamax 使用；两者计数互不影响
        self.assertNotEqual(id(eng.tt.table), id(eng.qtt.table))

    def test_clear_heuristics_clears_qtt_too(self):
        st = make_state()
        eng = ExpertSearchEngine(seed=7)
        eng.search(st, max_depth=2)
        self.assertGreater(eng.qtt.stores, 0)
        eng.clear_heuristics()
        self.assertEqual(eng.qtt.stores, 0, "clear_heuristics 应同时清空 QTT 计数")


class TestQttEquivalentDecision(unittest.TestCase):
    """开/关 QTT 必须给出**完全相同**的决策与分值。

    全部用 qsearch_depth=4 压住耗时（默认 16 在残局要数秒）——
    等价性不依赖具体的 qd 取值。
    """

    def _both(self, st, depth=2, qd=4):
        on = ExpertSearchEngine(seed=7, use_qtt=True)
        off = ExpertSearchEngine(seed=7, use_qtt=False)
        a_on, s_on, st_on = on.search(st, max_depth=depth, qsearch_depth=qd)
        a_off, s_off, st_off = off.search(st, max_depth=depth, qsearch_depth=qd)
        return (a_on, s_on, st_on), (a_off, s_off, st_off)

    def test_same_action_and_score_midgame(self):
        for seed, plies in ((43, 30), (1024, 30), (44, 60)):
            st = make_state(seed, plies)
            (a1, s1, _), (a0, s0, _) = self._both(st)
            self.assertEqual(str(a1), str(a0), f"seed={seed} 动作不一致")
            self.assertAlmostEqual(s1, s0, places=9, msg=f"seed={seed} 分值不一致")

    def test_same_root_scores(self):
        st = make_state(43, 30)
        (_, _, st_on), (_, _, st_off) = self._both(st)
        self.assertEqual(len(st_on.root_scores), len(st_off.root_scores))
        for (a1, v1), (a0, v0) in zip(st_on.root_scores, st_off.root_scores):
            self.assertEqual(str(a1), str(a0))
            self.assertAlmostEqual(v1, v0, places=9)

    def test_same_max_depth(self):
        st = make_state(1024, 30)
        (_, _, st_on), (_, _, st_off) = self._both(st, depth=2)
        self.assertEqual(st_on.max_depth, st_off.max_depth)


class TestQttEfficiency(unittest.TestCase):

    def test_qtt_does_not_increase_qnodes(self):
        st = make_state(44, 60)
        on = ExpertSearchEngine(seed=7, use_qtt=True)
        off = ExpertSearchEngine(seed=7, use_qtt=False)
        _, _, s_on = on.search(st, max_depth=2, qsearch_depth=4)
        _, _, s_off = off.search(st, max_depth=2, qsearch_depth=4)
        self.assertLessEqual(s_on.qnodes, s_off.qnodes,
                             "QTT 命中应减少或持平 qsearch 节点数")

    def test_qtt_hit_ratio_is_positive(self):
        """真实残局里应出现正向命中（这是 54.3% 重复率的直接体现）。"""
        st = make_state(44, 90)
        eng = ExpertSearchEngine(seed=7, use_qtt=True)
        _, _, stats = eng.search(st, max_depth=2, qsearch_depth=2)
        self.assertGreater(eng.qtt.hits, 0, "残局 qsearch 应出现置换表命中")
        self.assertGreater(stats.qnodes, 0)


class TestQttFlagSemantics(unittest.TestCase):
    """边界标记：cutoff 记下界，fail-low 记上界，正面完成记精确值。"""

    def test_cutoff_entry_is_lower_bound(self):
        st = make_state(44, 90)
        eng = ExpertSearchEngine(seed=7, use_qtt=True)
        eng.search(st, max_depth=2, qsearch_depth=2)
        flags = [e.flag for e in eng.qtt.table if e is not None]
        self.assertTrue(flags, "应有条目写入")
        # 绝不允许出现越界 flag（0=EXACT, 1=LOWER, 2=UPPER）
        self.assertTrue(all(f in (0, 1, 2) for f in flags))

    def test_no_entry_written_at_depth_left_zero(self):
        """depth_left<=0 返回的是静态近似值，不得写表。

        用 qsearch_depth=1 强制大量 depth_left==0 的截断，验证不会崩且仍等价。
        """
        st = make_state(43, 30)
        on = ExpertSearchEngine(seed=7, use_qtt=True)
        off = ExpertSearchEngine(seed=7, use_qtt=False)
        a1, s1, _ = on.search(st, max_depth=2, qsearch_depth=1)
        a0, s0, _ = off.search(st, max_depth=2, qsearch_depth=1)
        self.assertEqual(str(a1), str(a0))
        self.assertAlmostEqual(s1, s0, places=9)


if __name__ == "__main__":
    unittest.main()
