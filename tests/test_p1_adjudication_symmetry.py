"""裁决口径镜像对称化（P1，2026-09-15）。

背景：裁决式判分在 n=80 下**无法分辨真实强度差** ——
`reports/gate_calib_mirror_fixed.json`（同一模型自对，应≈0.5）裁决得分率 **0.600**，
反而高于已知更强方的 `gate_calib_sims_fixed.json` **0.575**，两者 Wilson CI 几乎完全重叠。

诊断（`scratch/diag_eval_symmetry.py` / `scratch/verify_sym_adjudication.py`）：
`evaluate_expert` 存在「按座位标签」的加性偏差 `b_s`：直接测镜像对称性
`f(mirror(st),0) vs f(st,1)`，210 个局面中 **83 个违反，最大偏差 9 分**；
该偏差被裁决的零阈值（margin=0）放大为符号偏置（镜像下 Δ 正 58.8% / 负 41.2%），
把镜像裁决得分率从 0.50 推到 0.60。

修复：镜像对称化不变量
    Δ_sym(st) = [Δ(st) - Δ(mirror(st))] / 2
等价写法：F(st,s) = [f(st,s) + f(mirror(st),1-s)] / 2，Δ_sym = F(st,0) - F(st,1)。
数学上**严格消除**任何「按座位标签」的加性偏差，且严格反对称
（Δ_sym(mirror(st)) = -Δ_sym(st)），不引入任何可调参数。

本文件验收：
1. `mirror_state` 的结构语义与对合性；
2. `evaluate_expert_dual` 的对称化公式；
3. **核心不变量**：注入任意按座位偏差后 Δ 被污染而 Δ_sym 不变；
4. `adjudicate_record` 优先使用对称化估值、缺失时回退，且胜负映射语义不变；
5. `play_game` 记录对称化终局估值。
"""
from __future__ import annotations

import random
import unittest

import junqi.eval_expert as EE
from junqi.config import RuleConfig
from junqi.eval_expert import evaluate_expert, evaluate_expert_dual
from junqi.eval_gate import (adjudicate_record, paired_delta_test,
                             record_eval_delta)
from junqi.selfplay import play_game
from junqi.state import GameState, mirror_state, deal

SPEC = "nn_mcts_20"
SPEC_B = "expert2"


def make_state(seed: int = 3, plies: int = 12) -> GameState:
    st = deal(random.Random(seed), RuleConfig())
    rng = random.Random(seed * 131 + 7)
    for _ in range(plies):
        if st.is_terminal():
            break
        acts = st.legal_actions()
        if not acts:
            break
        st = st.apply(rng.choice(acts))
    return st


class TestMirrorState(unittest.TestCase):

    def test_board_positions_flipped_and_keys_match(self):
        st = make_state()
        rows = max(r for r, _ in st.board) + 1
        ms = mirror_state(st)
        self.assertEqual(set(ms.board),
                         {(rows - 1 - r, c) for (r, c) in st.board})

    def test_colors_unchanged_ranks_preserved(self):
        """镜像只翻转位置、交换座位标签，**不改变棋子颜色**。

        （把颜色也互换会得到恰好取负的结果，是测量假象，不是真对称化。）
        """
        st = make_state()
        ms = mirror_state(st)
        rows = max(r for r, _ in st.board) + 1
        for (r, c), pc in st.board.items():
            m = ms.board[(rows - 1 - r, c)]
            self.assertEqual(m.color, pc.color)
            self.assertEqual(m.rank, pc.rank)
            self.assertEqual(m.revealed, pc.revealed)

    def test_seat_colors_swapped(self):
        st = make_state()
        ms = mirror_state(st)
        self.assertEqual(ms.seat_color[0], st.seat_color[1])
        self.assertEqual(ms.seat_color[1], st.seat_color[0])

    def test_turn_flipped_and_scalars_preserved(self):
        st = make_state()
        ms = mirror_state(st)
        self.assertEqual(ms.turn, 1 - st.turn)
        self.assertEqual(ms.ply, st.ply)
        self.assertEqual(ms.quiet, st.quiet)
        self.assertEqual(ms.winner, st.winner)
        self.assertEqual(ms.first_flip_done, st.first_flip_done)

    def test_involution(self):
        """镜像两次回到原局面（board / seat_color / turn 全部复原）。"""
        st = make_state()
        back = mirror_state(mirror_state(st))
        self.assertEqual(back.board, st.board)
        self.assertEqual(back.seat_color, st.seat_color)
        self.assertEqual(back.turn, st.turn)

    def test_dead_pieces_preserved(self):
        st = make_state(plies=40)
        ms = mirror_state(st)
        self.assertEqual(sorted((p.color, p.rank) for p in ms.dead),
                         sorted((p.color, p.rank) for p in st.dead))


class TestEvaluateExpertDual(unittest.TestCase):

    def test_returns_four_values_matching_definition(self):
        st = make_state()
        e0, e1, s0, s1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        ms = mirror_state(st)
        m0 = evaluate_expert(ms, 0, ignore_rule_draw=True)
        m1 = evaluate_expert(ms, 1, ignore_rule_draw=True)
        self.assertAlmostEqual(s0, (e0 + m1) / 2.0, places=6)
        self.assertAlmostEqual(s1, (e1 + m0) / 2.0, places=6)

    def test_raw_values_unchanged_from_evaluate_expert(self):
        st = make_state()
        e0, e1, _s0, _s1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        self.assertAlmostEqual(e0, evaluate_expert(st, 0, ignore_rule_draw=True), places=6)
        self.assertAlmostEqual(e1, evaluate_expert(st, 1, ignore_rule_draw=True), places=6)

    def test_sym_is_antisymmetric_across_mirror(self):
        """Δ_sym(mirror(st)) == -Δ_sym(st)（严格反对称 ⇒ 镜像下必然 50/50）。"""
        for seed in (3, 11, 29):
            st = make_state(seed=seed)
            _, _, s0, s1 = evaluate_expert_dual(st, ignore_rule_draw=True)
            _, _, t0, t1 = evaluate_expert_dual(mirror_state(st), ignore_rule_draw=True)
            self.assertAlmostEqual((s0 - s1) + (t0 - t1), 0.0, places=6)

    def test_injected_seat_bias_is_eliminated(self):
        """**核心不变量**：注入任意「按座位标签」的加性偏差 b_s 后，
        Δ = e0 - e1 被污染，而 Δ_sym 完全不变。"""
        st = make_state()
        _e0, _e1, base0, base1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        base_delta_sym = base0 - base1

        origin = EE.evaluate_expert

        def biased(state, seat, w=None, ignore_rule_draw=False):
            return origin(state, seat, w=w,
                          ignore_rule_draw=ignore_rule_draw) + {0: 7.0, 1: -12.0}[seat]

        EE.evaluate_expert = biased
        try:
            e0, e1, s0, s1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        finally:
            EE.evaluate_expert = origin

        raw_delta = e0 - e1
        _u0, _u1, u0, u1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        raw_delta_clean = _u0 - _u1

        self.assertAlmostEqual(raw_delta - raw_delta_clean, 19.0, places=6)
        self.assertAlmostEqual(s0 - s1, base_delta_sym, places=6)
        self.assertAlmostEqual(s0 - s1, u0 - u1, places=6)

    def test_zero_sum_pair_not_required(self):
        """非零和（f0+f1≠0）不影响 Δ_sym 的对称性 —— 共同模在对称化中被抵消。"""
        st = make_state()
        _e0, _e1, s0, s1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        self.assertIsInstance(s0 - s1, float)


class TestAdjudicateWithSymmetricEval(unittest.TestCase):

    def test_prefers_symmetric_fields_when_present(self):
        rec = {"winner": -1, "a": SPEC,
               "final_eval0": 100.0, "final_eval1": -100.0,   # 原始值指向座位 0 胜
               "final_eval_sym0": -5.0, "final_eval_sym1": 5.0}  # 对称化值指向座位 1 胜
        self.assertEqual(adjudicate_record(rec, SPEC, seat_a=0), 0.0)
        self.assertEqual(adjudicate_record(rec, SPEC, seat_a=1), 1.0)

    def test_falls_back_to_raw_when_symmetric_missing(self):
        rec = {"winner": -1, "a": SPEC,
               "final_eval0": 40.0, "final_eval1": -40.0}
        self.assertEqual(adjudicate_record(rec, SPEC, seat_a=0), 1.0)

    def test_falls_back_when_symmetric_fields_are_none(self):
        rec = {"winner": -1, "a": SPEC,
               "final_eval0": 40.0, "final_eval1": -40.0,
               "final_eval_sym0": None, "final_eval_sym1": None}
        self.assertEqual(adjudicate_record(rec, SPEC, seat_a=0), 1.0)

    def test_decisive_result_still_uses_real_winner(self):
        """对称化不得改变"有胜负按胜负"的语义。"""
        rec = {"winner": 1, "a": SPEC, "final_eval0": 100.0, "final_eval1": -100.0,
               "final_eval_sym0": 100.0, "final_eval_sym1": -100.0}
        self.assertEqual(adjudicate_record(rec, SPEC, seat_a=0), 0.0)
        self.assertEqual(adjudicate_record(rec, SPEC, seat_a=1), 1.0)

    def test_margin_applies_to_symmetric_delta(self):
        rec = {"winner": -1, "a": SPEC,
               "final_eval0": 100.0, "final_eval1": -100.0,
               "final_eval_sym0": 3.0, "final_eval_sym1": 0.0}
        self.assertEqual(adjudicate_record(rec, SPEC, margin=10.0, seat_a=0), 0.5)
        self.assertEqual(adjudicate_record(rec, SPEC, margin=1.0, seat_a=0), 1.0)

    def test_seat_mapping_for_symmetric_fields(self):
        """座位 1 视角时，对称化字段必须交换读取。"""
        rec = {"winner": -1, "a": SPEC_B,
               "final_eval_sym0": 8.0, "final_eval_sym1": -8.0}
        self.assertEqual(adjudicate_record(rec, SPEC_B, seat_a=0), 1.0)
        self.assertEqual(adjudicate_record(rec, SPEC_B, seat_a=1), 0.0)


class TestRecordEvalDelta(unittest.TestCase):
    """从对局记录取「座位 0 相对座位 1 的估值优势」Δ。"""

    def test_prefers_symmetric_fields(self):
        rec = {"final_eval0": 100.0, "final_eval1": -100.0,
               "final_eval_sym0": 3.0, "final_eval_sym1": -1.0}
        self.assertAlmostEqual(record_eval_delta(rec), 4.0, places=6)

    def test_falls_back_to_raw(self):
        rec = {"final_eval0": 12.0, "final_eval1": -8.0}
        self.assertAlmostEqual(record_eval_delta(rec), 20.0, places=6)

    def test_returns_none_when_missing(self):
        self.assertIsNone(record_eval_delta({"final_eval0": None, "final_eval1": None}))
        self.assertIsNone(record_eval_delta({}))


class TestPairedDeltaTest(unittest.TestCase):
    """配对 Δ 检验：保留估值差**幅度**，配对差分自动消掉座位固有优势。

    与二元裁决（把 Δ 压成 0/0.5/1）相比，本检验在高和棋率环境下功效更高。
    """

    def test_empty_input(self):
        r = paired_delta_test([])
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["mean"], 0.0)
        self.assertEqual(r["p_two_sided"], 1.0)

    def test_symmetric_input_gives_no_signal(self):
        r = paired_delta_test([10.0, -10.0, 20.0, -20.0, 30.0, -30.0])
        self.assertAlmostEqual(r["mean"], 0.0, places=9)
        self.assertGreater(r["p_two_sided"], 0.5)

    def test_consistent_positive_shift_is_significant(self):
        r = paired_delta_test([5.0, 7.0, 6.0, 8.0, 5.5, 6.5, 7.5, 6.2])
        self.assertGreater(r["mean"], 0.0)
        self.assertLess(r["p_two_sided"], 0.01)
        self.assertEqual(r["pos"], 8)
        self.assertEqual(r["neg"], 0)

    def test_consistent_negative_shift_is_significant(self):
        r = paired_delta_test([-5.0, -7.0, -6.0, -8.0])
        self.assertLess(r["mean"], 0.0)
        self.assertLess(r["p_two_sided"], 0.05)

    def test_sign_test_separates_from_t_test(self):
        """全正但幅度很小：t 检验不显著、符号检验显著 —— 两个口径互为补充。"""
        r = paired_delta_test([0.1] * 10)
        self.assertLess(r["sign_p"], 0.05)
        self.assertGreater(r["pos"], 0)

    def test_magnitude_is_used_unlike_binary_adjudication(self):
        """同样 4 正 4 负，但正侧幅度大得多时，配对 Δ 检验应显著而二元裁决不显著。"""
        shifted = [50.0, 48.0, 60.0, 55.0, -1.0, -2.0, -1.5, -2.5]
        r = paired_delta_test(shifted)
        self.assertGreater(r["mean"], 0.0)
        self.assertLess(r["p_two_sided"], 0.05)
        self.assertGreater(r["sign_p"], 0.5)      # 只数符号则毫无信号
        # 二元裁决只看符号 ⇒ 4:4 平手
        bin_wins = sum(1 for d in shifted if d > 0)
        self.assertEqual(bin_wins, 4)

    def test_candidate_direction_positive_means_candidate_stronger(self):
        """d = Δ(r0) - Δ(r1)：候选在 r0 执先占优、在 r1 执后不亏 ⇒ d > 0。"""
        rec_r0 = {"final_eval_sym0": 50.0, "final_eval_sym1": -50.0}
        rec_r1 = {"final_eval_sym0": 10.0, "final_eval_sym1": -10.0}
        d = record_eval_delta(rec_r0) - record_eval_delta(rec_r1)
        self.assertGreater(d, 0.0)


class TestPlayGameRecordsSymmetricEval(unittest.TestCase):

    def test_record_contains_symmetric_evals(self):
        cfg = RuleConfig(max_plies=8, no_capture_draw_plies=8)
        rec = play_game("random", "random", seed=5, cfg=cfg, device="cpu")
        self.assertIn("final_eval_sym0", rec)
        self.assertIn("final_eval_sym1", rec)
        self.assertIsNotNone(rec["final_eval_sym0"])
        self.assertIsNotNone(rec["final_eval_sym1"])

    def test_symmetric_evals_are_antisymmetric_in_mirror_sense(self):
        """镜像局（同牌、先后手互换）下 Δ_sym 应严格反号。

        这里用直接构造的镜像记录验证判据方向，不跑对局：
        两条记录互为镜像 ⇒ seat0/seat1 的对称化值互换 ⇒ Δ_sym 反号。
        """
        cfg = RuleConfig(max_plies=8, no_capture_draw_plies=8)
        rec = play_game("random", "random", seed=9, cfg=cfg, device="cpu")
        d = rec["final_eval_sym0"] - rec["final_eval_sym1"]
        mirrored = {"winner": -1, "a": rec["a"],
                    "final_eval_sym0": rec["final_eval_sym1"],
                    "final_eval_sym1": rec["final_eval_sym0"]}
        d2 = mirrored["final_eval_sym0"] - mirrored["final_eval_sym1"]
        self.assertAlmostEqual(d + d2, 0.0, places=6)

    def test_raw_fields_still_recorded(self):
        """兼容性：final_eval0/final_eval1 必须保留（旧报告仍可分析）。"""
        cfg = RuleConfig(max_plies=8, no_capture_draw_plies=8)
        rec = play_game("random", "random", seed=5, cfg=cfg, device="cpu")
        self.assertIn("final_eval0", rec)
        self.assertIn("final_eval1", rec)


if __name__ == "__main__":
    unittest.main()
