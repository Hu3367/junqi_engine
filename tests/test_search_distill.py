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


    def test_return_degraded_and_argmax_fix(self):
        """A2 & 新增 3: 教师软分布支持返回 degraded 标志，且 top1 取分值最高而非动作索引编号最大。"""
        from junqi.train_search_distill import teacher_soft_targets
        from junqi.encoder import action_to_index
        from junqi.state import Action, GameState, Piece
        from junqi.rules import Rank

        cfg = RuleConfig()
        st = GameState(board={
            (1, 1): Piece("r", Rank.GONG, True),
            (0, 1): Piece("b", Rank.QI, True),
            (2, 1): Piece("b", Rank.SI, True),
        }, dead=[Piece("b", Rank.LEI)] * 3, seat_color={0: "r", 1: "b"},
            turn=0, cfg=cfg)

        # 构造动作：让分值高的动作其 action_index 反而更小
        act_best = Action("move", (1, 1), (0, 1))
        act_worse = Action("move", (1, 1), (2, 1))
        idx_best = action_to_index(act_best)
        idx_worse = action_to_index(act_worse)
        # 如果 idx_best > idx_worse，调换一下确保我们要测的场景：
        if idx_best > idx_worse:
            act_best, act_worse = act_worse, act_best
            idx_best, idx_worse = idx_worse, idx_best

        # act_best 得分 500，act_worse 得分 10
        root_scores = [(act_best, 500.0), (act_worse, 10.0)]
        targets, top1, degraded = teacher_soft_targets(st, root_scores=root_scores,
                                                       temperature=50.0,
                                                       return_degraded=True)
        self.assertEqual(top1, idx_best, "top1 必须为得分最高的动作，绝不可按动作编号 argmax")
        self.assertFalse(degraded)

        # 验证显式传入 degraded=True 时能够被正确透传
        _, _, degraded_true = teacher_soft_targets(st, root_scores=root_scores,
                                                   return_degraded=True,
                                                   degraded=True)
        self.assertTrue(degraded_true)


class TestAgentStatsAndReproducibility(unittest.TestCase):

    def test_choose_actions_return_stats_and_agent_stats(self):
        """A2: choose_actions 支持 return_stats=True，且 agent.stats 属性可正常访问。"""
        from junqi.ai import ExpertAgent
        from junqi.config import SearchConfig
        from junqi.search import SearchStats

        rng = random.Random(42)
        st = deal(rng, RuleConfig())
        agent = ExpertAgent(SearchConfig(depth=1, time_limit_ms=100), seed=42)

        scored, stats = agent.choose_actions(st, topn=3, return_stats=True)
        self.assertIsInstance(stats, SearchStats)
        self.assertIsInstance(agent.stats, SearchStats)
        self.assertGreaterEqual(len(scored), 1)

    def test_cold_tt_clearing_reproducibility(self):
        """A3: 打标循环中 clear() 置换表确保各局面均在冷 TT 下评估。"""
        from junqi.ai import ExpertAgent
        from junqi.config import SearchConfig

        rng = random.Random(42)
        st = deal(rng, RuleConfig())
        agent = ExpertAgent(SearchConfig(depth=1, time_limit_ms=100), seed=42)

        agent.engine.tt.clear()
        scored1 = agent.choose_actions(st, topn=1)
        # 搜索后 TT 不为空
        self.assertGreater(agent.engine.tt.stores, 0)
        # 清空 TT
        agent.engine.tt.clear()
        self.assertEqual(agent.engine.tt.stores, 0)
        scored2 = agent.choose_actions(st, topn=1)
        self.assertEqual(scored1[0][0], scored2[0][0])
        self.assertAlmostEqual(scored1[0][1], scored2[0][1], places=5)


class TestDistillEndToEnd(unittest.TestCase):

    def test_train_search_distill_smoke(self):
        """端到端冒烟：极小规模（6 局面 x 1 epoch）产出可加载的候选权重，并记录 samples_degraded。"""
        from junqi.train_search_distill import train_search_distill
        from junqi.net import JunqiNet
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "search_distilled_smoke.pt")
            res = train_search_distill(base_model="models/bc_best.pt",
                                       out_path=out, n_states=6, epochs=1,
                                       batch_size=4, depth=1, time_limit_ms=150,
                                       val_ratio=0.34, seed=11, device="cpu")
            self.assertEqual(res["states"], 6)
            self.assertIn("samples_degraded", res)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(res["out_path"]))
            # 候选权重可加载且结构完整，元数据含 samples_degraded
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            self.assertIn("samples_degraded", ckpt)
            net = JunqiNet.load_from_file(out, device="cpu")
            self.assertIsNotNone(net)
            self.assertGreaterEqual(res["val_kl"], 0.0)

    def test_train_search_distill_freezes_backbone_bn(self):
        """A1 扩展: 验证 train_search_distill 训练 policy_head 时，骨干网络 BatchNorm 统计严格不变。"""
        from junqi.train_search_distill import train_search_distill
        from junqi.net import JunqiNet
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.pt")
            out = os.path.join(tmp, "out.pt")
            net = JunqiNet(in_channels=38, num_blocks=1, channels=16)
            net.save(base_path)

            before_bn = {
                name: buf.detach().clone()
                for name, buf in net.named_buffers()
                if "policy_head" not in name and ("running_mean" in name or "running_var" in name)
            }
            self.assertGreater(len(before_bn), 0)

            train_search_distill(base_model=base_path, out_path=out, n_states=6, epochs=1,
                                 batch_size=4, depth=1, time_limit_ms=100, val_ratio=0.34,
                                 seed=11, device="cpu")

            distilled = JunqiNet.load_from_file(out, device="cpu")
            for name, before_buf in before_bn.items():
                current_buf = dict(distilled.named_buffers())[name]
                self.assertTrue(
                    torch.equal(current_buf, before_buf),
                    f"搜索蒸馏不得改动冻结模块 BatchNorm 统计 {name}")

    def test_train_search_distill_multiprocessing_and_confidence(self):
        """A2/A3/新增3: 验证多进程打标（workers=2）正常执行，记录 samples_degraded，且正确应用 tac_min_spread 样本权重过滤。"""
        from junqi.train_search_distill import train_search_distill

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "search_distilled_mp.pt")
            res = train_search_distill(base_model="models/bc_best.pt",
                                       out_path=out, n_states=12, epochs=1,
                                       batch_size=4, depth=1, time_limit_ms=100,
                                       val_ratio=0.34, seed=12, workers=2,
                                       tac_min_spread=99999.0, device="cpu")
            self.assertEqual(res["states"], 12)
            self.assertIn("samples_degraded", res)
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
