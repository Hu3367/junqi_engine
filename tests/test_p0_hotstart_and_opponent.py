"""P0  regressions：训练闭环的两处致命缺陷（审查 R1 / R2）。

审查报告见 reviews/CODE_REVIEW_2026-09-15.md。

R1 热启动 optimizer 脱钩
    train_rl.run_training 先构造 net 再构造 optimizer，随后
    `net = JunqiNet.load_from_file(...)` 重新绑定变量名 → optimizer 仍持有旧网络
    的参数对象；新网络反传后旧参数 grad is None，AdamW 全部跳过 → 训练全程
    权重零更新。修复要求：热启动必须**就地**载入权重，重绑时同步重建 optimizer。

R2 自博弈 "best 对手" 退化为随机网络
    `torch.load(net.save(path))` 得到的是包装字典 {"model_state": ...}，
    而 worker 用 `load_state_dict(payload, strict=False)` 直接载入 → 键名无一
    匹配、静默通过 → net1 保持随机初始化。OPP_MIX 中 best 约占 25%。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import torch

from junqi.net import JunqiNet
from junqi.train_rl import (build_opponent_net, load_weights_into_net,
                            warmstart_candidate)

LR = 1e-3
WD = 1e-4


def _tiny_net(seed: int = 0) -> JunqiNet:
    """轻量网络：主干参数与动作头无关，保证单测秒级完成。"""
    torch.manual_seed(seed)
    return JunqiNet(in_channels=38, num_blocks=1, channels=8)


def _param_set(net: JunqiNet) -> set:
    return {id(p) for p in net.parameters()}


class TestUnwrapStateDict(unittest.TestCase):
    """R2 前置：torch.load(net.save(...)) 的产物必须能被规范成裸 state_dict。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.net = _tiny_net(seed=1)
        self.path = os.path.join(self.dir, "best.pt")
        self.net.save(self.path)
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_saved_file_is_wrapped_dict(self):
        """固化事实：net.save() 写出的是包装字典，键不是模块参数名。"""
        self.assertIsInstance(self.payload, dict)
        self.assertIn("model_state", self.payload)
        self.assertNotIn("in_conv.0.weight", self.payload)

    def test_legacy_direct_load_silently_loads_nothing(self):
        """缺陷证据：旧写法 strict=False 静默通过，一个键都没载入。"""
        victim = _tiny_net(seed=2)
        res = victim.load_state_dict(self.payload, strict=False)
        self.assertTrue(res.missing_keys, "旧写法必须表现为『全部键缺失』")
        self.assertTrue(res.unexpected_keys, "旧写法必须表现为『全部键多余』")

    def test_unwrap_yields_real_module_keys(self):
        sd = JunqiNet.unwrap_state_dict(self.payload)
        self.assertIn("in_conv.0.weight", sd)
        self.assertEqual(sd["in_conv.0.weight"].shape,
                         self.net.state_dict()["in_conv.0.weight"].shape)

    def test_load_weights_into_net_restores_exact_weights(self):
        """回归核心： helper 载入后权重必须与源网络逐位一致。"""
        victim = _tiny_net(seed=3)
        ok = load_weights_into_net(victim, self.payload, role="单测")
        self.assertTrue(ok, "net.save() 的包装字典必须能正确载入")
        src = self.net.state_dict()
        dst = victim.state_dict()
        for k in src:
            self.assertTrue(torch.equal(src[k], dst[k]), f"权重不一致: {k}")


class TestBuildOpponentNet(unittest.TestCase):
    """R2 端到端：worker 视角构造对手网络。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.net = _tiny_net(seed=11)
        self.path = os.path.join(self._tmp.name, "opp.pt")
        self.net.save(self.path)
        self.payload = torch.load(self.path, map_location="cpu", weights_only=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_opponent_net_carries_real_weights(self):
        opp = build_opponent_net(self.payload, torch.device("cpu"))
        self.assertIsNotNone(opp, "对手网络不应退化为 None")
        self.assertFalse(opp.training)
        src, dst = self.net.state_dict(), opp.state_dict()
        for k in src:
            self.assertTrue(torch.equal(src[k], dst[k]), f"对手权重不一致: {k}")

    def test_garbage_payload_returns_none_not_random_net(self):
        """宁可降级为『无对手』，也绝不能返回一个随机初始化的假对手。"""
        opp = build_opponent_net({"totally": "wrong"}, torch.device("cpu"))
        self.assertIsNone(opp)


class TestWarmstartCandidate(unittest.TestCase):
    """R1：热启动不得让 optimizer 与网络脱钩。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "warm.pt")
        _tiny_net(seed=21).save(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def _fresh_pair(self):
        net = _tiny_net(seed=22)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        return net, opt

    def test_warmstart_keeps_same_net_object(self):
        net, opt = self._fresh_pair()
        net2, opt2 = warmstart_candidate(net, opt, self.path, device="cpu",
                                         lr=LR, weight_decay=WD)
        self.assertIs(net2, net, "同架构下热启动必须就地载入，不得重建对象")
        self.assertIs(opt2, opt)

    def test_optimizer_still_bound_to_net_after_warmstart(self):
        """optimizer 的每个 param_groups 项都必须仍属于当前 net。"""
        net, opt = self._fresh_pair()
        net2, opt2 = warmstart_candidate(net, opt, self.path, device="cpu",
                                         lr=LR, weight_decay=WD)
        ids = _param_set(net2)
        flat = [p for g in opt2.param_groups for p in g["params"]]
        self.assertTrue(flat, "optimizer 不得为空")
        for p in flat:
            self.assertIn(id(p), ids, "optimizer 持有游离于 net 之外的参数")

    def test_training_step_actually_updates_weights(self):
        """端到端回归：热启动后做一步优化，权重必须真的变化。"""
        torch.manual_seed(7)
        net, opt = self._fresh_pair()
        net, opt = warmstart_candidate(net, opt, self.path, device="cpu",
                                       lr=LR, weight_decay=WD)
        before = {k: v.detach().clone() for k, v in net.state_dict().items()}

        net.train()
        x = torch.randn(2, net.in_channels, 12, 5)
        mask = torch.zeros(2, net.action_size, dtype=torch.bool)
        mask[:, :5] = True
        logits, _ = net(x, legal_mask=mask)
        loss = logits[:, :5].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        changed = [k for k, v in net.state_dict().items()
                   if not torch.equal(before[k], v)]
        self.assertTrue(changed, "热启动后优化步必须改变权重（R1 回归）")

    def test_architecture_mismatch_rebuilds_optimizer(self):
        """检查点主干超参不一致时允许重建 net，但必须同时重建 optimizer。"""
        mismatch = os.path.join(self._tmp.name, "mismatch.pt")
        torch.save({"model_state": {
                        "in_conv.0.weight": torch.randn(16, 38, 3, 3),
                        "blocks.0.conv1.weight": torch.randn(16, 16, 3, 3),
                        "blocks.1.conv1.weight": torch.randn(16, 16, 3, 3)},
                    "num_blocks": 2, "channels": 16},
                   mismatch)
        net, opt = self._fresh_pair()          # num_blocks=1, channels=8
        net2, opt2 = warmstart_candidate(net, opt, mismatch, device="cpu",
                                         lr=LR, weight_decay=WD)
        self.assertIsNot(net2, net, "架构不一致时应重建网络")
        self.assertIsNot(opt2, opt, "重建网络后必须重建优化器（否则回到 R1）")
        ids = _param_set(net2)
        for g in opt2.param_groups:
            for p in g["params"]:
                self.assertIn(id(p), ids,
                              "重建网络后 optimizer 必须指向新网络的参数")


if __name__ == "__main__":
    unittest.main()
