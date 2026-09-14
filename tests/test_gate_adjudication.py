"""裁决式判分验收（P3 修订 2026-09-14，仅评测口径）。

背景：同源模型间 72-92% 对局判和，专家对专家也 88% 和棋（见实验报告 §8.9/§8.10）。
官方得分率判据在高和棋率下**效应量被稀释约 12 倍**（8% 决胜负率 → 真实 +65 Elo 只
表现为得分率 0.50→0.525，被 Wilson 半宽 ±0.069 淹没）。裁决式判分把和棋局按终局
专家估值判出胜负，使每个对局都携带强度信息（n=200 全为有效样本）。

本文件验收：
1. `adjudicate_record` 的胜负映射（含座位互换）与 margin 语义；
2. `play_game` 在记录中写入 final_eval0/final_eval1（公开信息口径）；
3. `run_gate` 报告含 adjudicated 段，且不改动官方 promote 判据。
"""
from __future__ import annotations

import unittest

from junqi.config import RuleConfig
from junqi.eval_gate import adjudicate_record, run_gate
from junqi.selfplay import play_game

SPEC = "nn_mcts_20"
SPEC_B = "expert2"          # 与 SPEC 必须不同：用于验证"A 在后手座位"的胜负映射


class TestAdjudicateRecord(unittest.TestCase):

    def test_decisive_result_uses_real_winner(self):
        self.assertEqual(adjudicate_record({"winner": 0, "a": SPEC}, SPEC), 1.0)
        self.assertEqual(adjudicate_record({"winner": 1, "a": SPEC}, SPEC), 0.0)
        # 座位互换：A 是后手（记录里 a=对手 spec），winner=0 表示 A 负、winner=1 表示 A 胜
        self.assertEqual(adjudicate_record({"winner": 0, "a": SPEC_B}, SPEC), 0.0)
        self.assertEqual(adjudicate_record({"winner": 1, "a": SPEC_B}, SPEC), 1.0)

    def test_draw_adjudicated_by_final_eval(self):
        base = {"winner": -1, "a": SPEC}
        self.assertEqual(adjudicate_record({**base, "final_eval0": 40.0,
                                            "final_eval1": -40.0}, SPEC), 1.0)
        self.assertEqual(adjudicate_record({**base, "final_eval0": -40.0,
                                            "final_eval1": 40.0}, SPEC), 0.0)
        # 座位互换：A 是座位 1
        swapped = {"winner": -1, "a": "other_spec", "final_eval0": 40.0,
                   "final_eval1": -40.0}
        self.assertEqual(adjudicate_record(swapped, SPEC), 0.0)

    def test_margin_band_counts_as_draw(self):
        rec = {"winner": -1, "a": SPEC, "final_eval0": 10.0, "final_eval1": -5.0}
        self.assertEqual(adjudicate_record(rec, SPEC, margin=0.0), 1.0)
        self.assertEqual(adjudicate_record(rec, SPEC, margin=20.0), 0.5)

    def test_missing_evals_fall_back_to_draw(self):
        rec = {"winner": -1, "a": SPEC, "final_eval0": None, "final_eval1": None}
        self.assertEqual(adjudicate_record(rec, SPEC), 0.5)
        self.assertEqual(adjudicate_record({"winner": -1, "a": SPEC}, SPEC), 0.5)


class TestPlayGameRecordsFinalEval(unittest.TestCase):

    def test_record_contains_public_eval_for_both_seats(self):
        cfg = RuleConfig(max_plies=8, no_capture_draw_plies=8)
        rec = play_game("random", "random", seed=5, cfg=cfg, device="cpu")
        self.assertIn("final_eval0", rec)
        self.assertIn("final_eval1", rec)
        self.assertIsNotNone(rec["final_eval0"])

    def test_quiet_limit_draw_still_yields_nonzero_eval(self):
        """回归：规则限步判和的终局曾被 is_dead_draw 的限步分支清零（实测 74/80 局），
        使裁决判分在恰好需要它的场合退化为 0.5。现在应跳过限步分支取有效估值。"""
        cfg = RuleConfig(max_plies=60, no_capture_draw_plies=10)
        rec = play_game("random", "random", seed=7, cfg=cfg, device="cpu")
        self.assertEqual(rec["reason"], "no_capture")
        total = abs(rec["final_eval0"]) + abs(rec["final_eval1"])
        self.assertGreater(total, 0.0,
                           "限步判和局必须给出非零估值（否则裁决无从判定）")
        self.assertIsNotNone(rec["last_live_eval_used"])

    def test_default_dead_draw_behaviour_unchanged(self):
        """默认路径（不跳过限步）必须保持原语义，专家引擎热路径不受影响。"""
        from junqi.analysis import is_dead_draw
        from junqi.state import deal
        import random as _r
        cfg = RuleConfig(no_capture_draw_plies=5)
        st = deal(_r.Random(3), cfg)
        while st.quiet < 5 and not st.is_terminal():
            acts = st.legal_actions()
            if not acts:
                break
            st = st.apply(acts[0])
        self.assertGreaterEqual(st.quiet, 5)
        self.assertTrue(is_dead_draw(st)[0], "默认应因限步判死")
        self.assertFalse(is_dead_draw(st, ignore_quiet_limit=True)[0],
                         "跳过限步后不应仍因限步判死")


class TestGateAdjudicatedReport(unittest.TestCase):

    def test_run_gate_reports_adjudicated_section(self):
        rep = run_gate("random", "random", seeds=[1, 2], workers=1, max_plies=12)
        self.assertEqual(rep["n_games"], 4)
        adj = rep["adjudicated"]
        self.assertEqual(adj["n_games"], 4)
        self.assertEqual(sum(adj["totals"].values()), 4)
        self.assertAlmostEqual(adj["score_rate"],
                               (adj["totals"]["wins"] + 0.5 * adj["totals"]["draws"]) / 4,
                               places=9)
        self.assertIn("score_rate_wilson", adj)
        # 官方判据不受影响：仍基于官方得分率与 SPRT
        self.assertIn("promote", rep)
        self.assertIn("score_rate", rep)

    def test_same_spec_name_attributes_both_seat_directions(self):
        """回归：两侧 spec 同名（所有正式门控的用法，仅权重不同）时，原实现靠 spec 名
        推断候选座位，导致"候选执后手"的那一局被反向计分（历史报告 as_second 恒为 0 局），
        把真实差异系统性压回 0.5。座位必须由构造显式决定。"""
        rep = run_gate("random", "random", seeds=[1, 2, 3, 4], workers=1, max_plies=30)
        self.assertEqual(rep["n_games"], 8)
        self.assertEqual(rep["seat_split"]["as_first"]["games"], 4)
        self.assertEqual(rep["seat_split"]["as_second"]["games"], 4,
                         "候选执后手的对局必须被正确归属（修复前恒为 0）")

    def test_score_at_seat_is_seat_explicit(self):
        from junqi.eval_gate import _score_at_seat
        for seat in (0, 1):
            self.assertEqual(_score_at_seat({"winner": seat}, seat)[0], 1.0)
            self.assertEqual(_score_at_seat({"winner": 1 - seat}, seat)[0], 0.0)
        self.assertEqual(_score_at_seat({"winner": -1}, 0)[0], 0.5)


if __name__ == "__main__":
    unittest.main()
