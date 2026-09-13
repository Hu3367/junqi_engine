"""P2 搜索蒸馏管线单元测试：教师软分布正确性 + 端到端小规模冒烟。"""
from __future__ import annotations

import os
import random
import tempfile
import unittest

import numpy as np

from junqi.config import RuleConfig
from junqi.state import deal


class TestTeacherSoftTargets(unittest.TestCase):

    def test_soft_targets_sum_to_one_and_peak_at_best(self):
        """教师软分布：合法动作上归一化，且质量集中在教师最优动作。"""
        from junqi.train_search_distill import teacher_soft_targets
        from junqi.encoder import legal_action_mask

        rng = random.Random(7)
        st = deal(rng, RuleConfig())
        targets, top1 = teacher_soft_targets(st, depth=1, time_limit_ms=200,
                                             temperature=80.0, seed=7)
        mask = legal_action_mask(st)
        # 质量只落在合法动作上且归一
        self.assertAlmostEqual(float(targets[~mask].sum()), 0.0, places=5)
        self.assertAlmostEqual(float(targets.sum()), 1.0, places=4)
        # top1 必须是合法动作且概率最高
        self.assertGreaterEqual(top1, 0)
        self.assertTrue(mask[top1])
        self.assertAlmostEqual(float(targets.max()), float(targets[top1]), places=6)
        self.assertGreater(float(targets.max()), 1.0 / max(int(mask.sum()), 1))

    def test_terminal_scores_dominate_softmax(self):
        """教师根节点出现制胜分（±WIN_SCORE）时软分布应饱和到该动作。"""
        from junqi.train_search_distill import teacher_soft_targets
        from junqi.rules import Rank
        from junqi.state import Action, GameState, Piece

        cfg = RuleConfig()
        st = GameState(board={
            (1, 1): Piece("r", Rank.GONG, True),
            (0, 1): Piece("b", Rank.QI, True),
        }, dead=[Piece("b", Rank.LEI)] * 3, seat_color={0: "r", 1: "b"},
            turn=0, cfg=cfg)
        root_scores = [
            (Action("move", (1, 1), (0, 1)), 999_000.0),  # 一步吃旗
            (Action("move", (1, 1), (2, 1)), 0.0),
        ]
        targets, top1 = teacher_soft_targets(st, root_scores=root_scores,
                                             temperature=80.0)
        flag_idx = None
        from junqi.encoder import action_to_index
        flag_idx = action_to_index(Action("move", (1, 1), (0, 1)))
        self.assertEqual(top1, flag_idx)
        self.assertGreater(float(targets[flag_idx]), 0.99)


class TestDistillEndToEnd(unittest.TestCase):

    def test_train_search_distill_smoke(self):
        """端到端冒烟：极小规模（6 局面 x 1 epoch）产出可加载的候选权重。"""
        from junqi.train_search_distill import train_search_distill
        from junqi.net import JunqiNet

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "search_distilled_smoke.pt")
            res = train_search_distill(base_model="models/bc_best.pt",
                                       out_path=out, n_states=6, epochs=1,
                                       batch_size=4, depth=1, time_limit_ms=150,
                                       val_ratio=0.34, seed=11, device="cpu")
            self.assertEqual(res["states"], 6)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(res["out_path"]))
            # 候选权重可加载且结构完整
            net = JunqiNet.load_from_file(out, device="cpu")
            self.assertIsNotNone(net)
            self.assertGreaterEqual(res["val_kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
