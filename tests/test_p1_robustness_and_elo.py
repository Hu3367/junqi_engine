"""P3：故障可观测性与 Elo 口径（审查 C8 / C9）。

  C8 `selfplay.play_game` 的专家估值采样失败被 `except Exception: pass` 静默吞掉；
     终局估值失败也只把 final_eval 置 None。后果是门控的"裁决式判分"会静默退化为
     0.5，而报告上看不出任何异常（审查 C8）。同一段代码还把 `evaluate_expert`
     多调了两次——仅为计算一个布尔标志。
  C9 `train_rl.py` 用 `current_elo += 16.0 * (ov_score - 0.5) * 2` 更新 Elo，
     这是与对手无关的随机游走，却被写进 `elo_history.jsonl` 当实力曲线使用。
"""
from __future__ import annotations

import inspect
import math
import os
import re
import unittest
from unittest import mock

from junqi.config import RuleConfig
from junqi.selfplay import play_game


class TestEloUpdate(unittest.TestCase):
    """C9：Elo 必须由"对基准的得分率"按标准公式换算。"""

    def test_even_score_leaves_rating_unchanged(self):
        from junqi.train_rl import elo_update_from_score

        self.assertAlmostEqual(elo_update_from_score(1500.0, 0.5, 40), 1500.0,
                               places=6)

    def test_higher_score_raises_rating(self):
        from junqi.train_rl import elo_update_from_score

        self.assertGreater(elo_update_from_score(1500.0, 0.6, 40), 1500.0)

    def test_lower_score_lowers_rating(self):
        from junqi.train_rl import elo_update_from_score

        self.assertLess(elo_update_from_score(1500.0, 0.4, 40), 1500.0)

    def test_extreme_scores_stay_finite(self):
        from junqi.train_rl import elo_update_from_score

        for s in (0.0, 1.0):
            v = elo_update_from_score(1500.0, s, 40)
            self.assertTrue(math.isfinite(v), f"得分率 {s} 产生了非有限 Elo")
            self.assertLess(abs(v - 1500.0), 1000.0)

    def test_formula_is_standard_elo(self):
        from junqi.train_rl import elo_update_from_score

        # 得分率 0.75、对手 1500 => +400*log10(3) ≈ +190.85，夹紧不生效
        expected = 1500.0 + 400.0 * math.log10(0.75 / 0.25)
        self.assertAlmostEqual(elo_update_from_score(1500.0, 0.75, 1000),
                               expected, places=3)

    def test_old_random_walk_removed(self):
        src = open("junqi/train_rl.py", encoding="utf-8").read()
        import re as _re
        # 只检查可执行代码，不检查解释性注释/文档
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        self.assertIsNone(
            _re.search(r"current_elo\s*\+=\s*16\.0", code),
            "旧的随机游走式 Elo 更新必须删除")


class TestPlayGameEvalFailures(unittest.TestCase):
    """C8：专家估值失败必须可观测。"""

    def test_failures_are_counted_and_reported(self):
        def boom(*a, **kw):
            raise RuntimeError("估值器故障")

        with mock.patch("junqi.eval_expert.evaluate_expert", boom):
            rec = play_game("random", "random", 7, RuleConfig(max_plies=25))
        self.assertIn("eval_failures", rec,
                      "对局记录必须暴露估值失败次数（否则门控退化不可见）")
        self.assertGreater(rec["eval_failures"], 0)
        self.assertIsNone(rec["final_eval0"])

    def test_clean_game_reports_zero_failures(self):
        rec = play_game("random", "random", 7, RuleConfig(max_plies=25))
        self.assertEqual(rec.get("eval_failures"), 0)

    def test_eval_not_called_more_than_twice_per_sample(self):
        """终局标志位不得靠重复调用估值器换取（原实现多调了两次）。"""
        src = inspect.getsource(play_game)
        n = len(re.findall(r"evaluate_expert\(", src))
        self.assertLessEqual(n, 4,
                             f"play_game 内 evaluate_expert 调用点 {n} 处，"
                             f"预期 ≤4（每处采样 2 次）")


class TestBufferLoadIsLoud(unittest.TestCase):
    """C8：经验池加载失败不得静默。"""

    def test_corrupt_pickle_warns(self):
        import tempfile

        from junqi.train_rl import StratifiedReplayBuffer

        buf = StratifiedReplayBuffer(capacity=30)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_buffer.pkl")
            with open(path, "wb") as f:
                f.write(b"this is definitely not a pickle")
            import io
            from contextlib import redirect_stderr, redirect_stdout

            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                buf.load(path)           # 不得抛出
        self.assertIn("经验池", out.getvalue(),
                      "加载失败必须打印可诊断的告警")


if __name__ == "__main__":
    unittest.main()
