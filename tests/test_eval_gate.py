"""P0 评测门控模块单元测试：统计基元正确性 + 端到端冒烟。"""
from __future__ import annotations

import math
import unittest

from junqi.eval_gate import (_score_from_perspective, paired_z_test, run_gate,
                             sprt_trinomial, wilson_ci)


class TestStatPrimitives(unittest.TestCase):

    def test_wilson_ci_known_values(self):
        """Wilson 区间对已知小样本案例给出教科书值。"""
        # 0/100：95% 上界约 3.7%（Wilson 单侧经典值）
        lo, hi = wilson_ci(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0370, delta=0.003)

        # 50/100：区间应对称覆盖 0.5 且宽约 ±0.098
        lo, hi = wilson_ci(50, 100)
        self.assertAlmostEqual(lo, 0.4038, delta=0.01)
        self.assertAlmostEqual(hi, 0.5962, delta=0.01)

        # n=0 安全回退
        lo, hi = wilson_ci(0, 0)
        self.assertEqual((lo, hi), (0.0, 1.0))

    def test_wilson_ci_weighted_score(self):
        """加权得分（含 0.5 和棋）也能给出正确区间：30胜20和30负 => 得分率 0.5。"""
        lo, hi = wilson_ci(30 + 0.5 * 20, 80)
        self.assertAlmostEqual(lo, 0.400, delta=0.015)
        self.assertAlmostEqual(hi, 0.614, delta=0.015)

    def test_paired_z_test_balanced_vs_dominant(self):
        """均衡配对 z≈0；一边倒配对 |z| 大且显著。"""
        from junqi.eval_gate import paired_z_test
        balanced = paired_z_test([1.0] * 50)
        self.assertAlmostEqual(balanced["z"], 0.0, delta=1e-9)
        self.assertGreater(balanced["p_two_sided"], 0.9)

        dominant = paired_z_test([2.0] * 40)  # 每个 seed 两局全胜
        self.assertGreater(dominant["z"], 10.0)
        self.assertLess(dominant["p_two_sided"], 1e-6)

        # 空输入安全
        empty = paired_z_test([])
        self.assertEqual(empty["n"], 0)
        self.assertEqual(empty["p_two_sided"], 1.0)

    def test_sprt_trinomial_decisions(self):
        """SPRT：压倒性战绩接受 H1，均势继续，压倒性劣势接受 H0。"""
        from junqi.eval_gate import sprt_trinomial
        h1 = sprt_trinomial(wins=80, draws=10, losses=10)
        self.assertEqual(h1["decision"], "accept_h1")

        balanced = sprt_trinomial(wins=33, draws=34, losses=33)
        self.assertEqual(balanced["decision"], "continue")

        h0 = sprt_trinomial(wins=5, draws=20, losses=75)
        self.assertEqual(h0["decision"], "accept_h0")

        # LLR 单调性：战绩越好 LLR 越大
        l_weak = sprt_trinomial(20, 30, 50)["llr"]
        l_strong = sprt_trinomial(50, 30, 20)["llr"]
        self.assertGreater(l_strong, l_weak)


class TestGateEndToEnd(unittest.TestCase):

    def test_run_gate_smoke_random_vs_random(self):
        """端到端冒烟：random vs random 少量种子，报告结构完整且可复现。"""
        from junqi.eval_gate import format_gate_report, run_gate
        rep = run_gate("random", "greedy", seeds=[11, 22, 33],
                       max_plies=120)
        self.assertEqual(rep["n_games"], 6)
        self.assertEqual(rep["n_seeds"], 3)
        self.assertEqual(sum(rep["totals"].values()), 6)
        # 每局得分必为 1/0.5/0 => 每组配对合计 ∈ {0,0.5,1,1.5,2}
        # 结构检查：区间有序、原因拆分总数一致
        lo, hi = rep["score_rate_wilson"]
        self.assertLessEqual(lo, hi)
        total_reasons = sum(sum(c.values()) for c in rep["reason_breakdown"].values())
        self.assertEqual(total_reasons, 6)
        for seat in ("as_first", "as_second"):
            self.assertEqual(rep["seat_split"][seat]["games"], 3)
        # 晋级判据在均衡对局下不应触发
        self.assertFalse(rep["promote"])
        text = format_gate_report(rep)
        self.assertIn("得分率", text)
        self.assertIn("SPRT", text)
        self.assertIn("promote = False", text)

    def test_score_from_perspective(self):
        """视角换算：先手/后手/和棋三种情形。"""
        from junqi.eval_gate import _score_from_perspective
        rec = {"a": "x", "b": "y", "winner": 0, "reason": "flag"}
        self.assertEqual(_score_from_perspective(rec, "x"), (1.0, "flag"))
        self.assertEqual(_score_from_perspective(rec, "y"), (0.0, "flag"))
        rec2 = {"a": "x", "b": "y", "winner": -1, "reason": "repetition"}
        self.assertEqual(_score_from_perspective(rec2, "x"), (0.5, "repetition"))
        rec3 = {"a": "x", "b": "y", "winner": None, "reason": None}
        self.assertEqual(_score_from_perspective(rec3, "y")[0], 0.5)


if __name__ == "__main__":
    unittest.main()
