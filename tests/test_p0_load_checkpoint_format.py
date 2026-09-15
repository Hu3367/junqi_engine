"""P0：`load_from_file` 必须认识检查点格式，且不得静默载入零个权重。

2026-09-15 修复的隐患（与审查 R2 同族）：
`save_checkpoint()` 写出的是 `{"net":..., "optimizer":..., "epoch":...}`，
而 `load_from_file` 只判 `"model_state"` 与"裸 dict"两个分支 ——
传入 `models/candidate_latest.pt` 时会把整个检查点 dict 当 state_dict，
`load_state_dict(strict=False)` **键名无一匹配、静默载入零个权重**，
于是得到一个随机初始化的网络。评测/策略构造路径一旦踩到，
会得到"看似正常、实为随机"的对手或候选。
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import torch

from junqi.net import JunqiNet


def _tiny(seed: int = 0) -> JunqiNet:
    torch.manual_seed(seed)
    return JunqiNet(in_channels=38, num_blocks=1, channels=8)


class TestLoadFromFileFormats(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.net = _tiny(seed=7)

    def tearDown(self):
        self._tmp.cleanup()

    def _p(self, name):
        return os.path.join(self.dir, name)

    def test_save_wrapper_format(self):
        p = self._p("wrapped.pt")
        self.net.save(p)
        loaded = JunqiNet.load_from_file(p, device="cpu")
        src, dst = self.net.state_dict(), loaded.state_dict()
        for k in src:
            self.assertTrue(torch.equal(src[k], dst[k]), f"包装格式载入不一致: {k}")

    def test_checkpoint_format_is_recognized(self):
        """回归：检查点（{"net":...}）过去会被静默载入零个权重。"""
        p = self._p("ckpt.pt")
        torch.save({"net": self.net.state_dict(),
                    "optimizer": {}, "epoch": 3,
                    "in_channels": 38, "num_blocks": len(self.net.blocks),
                    "channels": self.net.in_conv[0].out_channels}, p)

        buf = io.StringIO()
        with redirect_stdout(buf):
            loaded = JunqiNet.load_from_file(p, device="cpu")

        self.assertNotIn("未载入", buf.getvalue(),
                         "检查点格式应被正确识别，不应触发'大量键未载入'告警")
        src, dst = self.net.state_dict(), loaded.state_dict()
        for k in src:
            self.assertTrue(torch.equal(src[k], dst[k]),
                            f"检查点格式载入不一致: {k}")

    def test_checkpoint_load_matches_manual_extraction(self):
        p = self._p("ckpt2.pt")
        torch.save({"net": self.net.state_dict(), "optimizer": {}, "epoch": 1,
                    "in_channels": 38, "num_blocks": len(self.net.blocks),
                    "channels": self.net.in_conv[0].out_channels}, p)
        a = JunqiNet.load_from_file(p, device="cpu")
        b = _tiny(seed=999)
        b.load_state_dict(torch.load(p, map_location="cpu",
                                     weights_only=False)["net"])
        for k in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[k], b.state_dict()[k]))

    def test_checkpoint_without_arch_meta_uses_defaults(self):
        """老检查点没有架构元信息：默认 128/6 架构仍能正常载入。"""
        p = self._p("legacy_ckpt.pt")
        big = JunqiNet(in_channels=38)          # 默认 6 块 / 128 通道
        torch.save({"net": big.state_dict(), "optimizer": {}, "epoch": 4}, p)
        loaded = JunqiNet.load_from_file(p, device="cpu")
        src, dst = big.state_dict(), loaded.state_dict()
        for k in src:
            self.assertTrue(torch.equal(src[k], dst[k]))

    def test_arch_mismatch_raises_loudly(self):
        """架构不符时必须抛错，绝不静默给出随机网络。"""
        p = self._p("mismatch.pt")
        torch.save({"net": self.net.state_dict(), "optimizer": {}, "epoch": 1},
                   p)                            # 小网络，但无架构元信息
        with self.assertRaises(RuntimeError):
            JunqiNet.load_from_file(p, device="cpu")

    def test_wrong_file_warns_loudly(self):
        """给非权重文件时必须告警，而不是静默返回随机网络。"""
        p = self._p("junk.pt")
        torch.save({"totally": "wrong", "epoch": 2}, p)
        buf = io.StringIO()
        with redirect_stdout(buf):
            JunqiNet.load_from_file(p, device="cpu")
        self.assertIn("未载入", buf.getvalue(),
                      "疑似非权重文件必须打印告警（不得静默）")
        self.assertIn("随机", buf.getvalue())

    def test_raw_module_still_supported(self):
        p = self._p("module.pt")
        torch.save(_tiny(seed=3), p)
        loaded = JunqiNet.load_from_file(p, device="cpu")
        self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main()
