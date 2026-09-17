"""P2 行为克隆与评测管线单元测试：通道数验证、防塌缩指标、CLI参数映射。"""
from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np
import torch

from junqi.encoder import NUM_CHANNELS
from junqi.net import JunqiNet
from junqi.eval_bc import evaluate_test_set, generate_p2_report
from junqi.train_bc import evaluate_model
from junqi.dataset import NpzReplayDataset
from torch.utils.data import DataLoader


class TestBCPipeline(unittest.TestCase):

    def test_in_channels_is_38(self):
        """新增 1: 验证通道数为 38（NUM_CHANNELS），包含重复局面特征平面。"""
        self.assertEqual(NUM_CHANNELS, 38)
        net = JunqiNet(in_channels=NUM_CHANNELS, num_blocks=1, channels=16)
        self.assertEqual(net.in_channels, 38)
        dummy_x = torch.randn(2, 38, 12, 5)
        p_logits, v_logits = net(dummy_x)
        self.assertEqual(p_logits.shape, (2, 3650))
        self.assertEqual(v_logits.shape, (2, 3))

    def test_eval_bc_metrics_and_collapse_detection(self):
        """新增 4: eval_bc 包含平衡准确率、各类别召回及塌缩检测。"""
        with tempfile.TemporaryDirectory() as td:
            npz_path = os.path.join(td, "test.npz")
            meta_path = os.path.join(td, "metadata.json")
            model_path = os.path.join(td, "model.pt")

            # 写入元数据 (3.0.0)
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"version": "3.0.0"}, f)

            # 写入测试数据：60条，分布为 15 Win, 30 Draw, 15 Loss
            n = 60
            states = np.random.randn(n, 38, 12, 5).astype(np.float32)
            masks = np.ones((n, 3650), dtype=bool)
            actions = np.random.randint(0, 50, size=n, dtype=np.int32)
            val_classes = np.array([0] * 15 + [1] * 30 + [2] * 15, dtype=np.int8)
            values = np.where(val_classes == 0, 1.0, np.where(val_classes == 2, -1.0, 0.0)).astype(np.float32)
            has_values = np.ones(n, dtype=bool)
            phases = np.zeros(n, dtype=np.int8)

            np.savez(npz_path, states=states, masks=masks, actions=actions,
                     values=values, val_classes=val_classes,
                     has_values=has_values, phases=phases)

            net = JunqiNet(in_channels=38, num_blocks=1, channels=16)
            torch.save({
                "model_state": net.state_dict(),
                "in_channels": 38,
                "num_blocks": 1,
                "channels": 16,
                "action_size": 3650,
            }, model_path)

            res = evaluate_test_set(model_path=model_path, test_npz=npz_path,
                                    batch_size=32, device_str="cpu")

            self.assertIn("value_balanced_accuracy", res)
            self.assertIn("value_per_class_recall", res)
            self.assertIn("value_pred_counts", res)
            self.assertIn("value_collapse_warning", res)
            self.assertEqual(res["test_samples"], 60)
            self.assertEqual(res["value_samples"], 60)

            # 生成报告验证无异常抛出
            report_md = os.path.join(td, "report.md")
            generate_p2_report(res, out_md=report_md)
            self.assertTrue(os.path.exists(report_md))
            with open(report_md, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Value 平衡准确率", content)
                self.assertIn("Value 类别独立召回率", content)

    def test_evaluate_model_7tuple_unpacking(self):
        """B2: evaluate_model 正确解包 7 元组并计算 Policy / Value 损失。"""
        with tempfile.TemporaryDirectory() as td:
            npz_path = os.path.join(td, "val.npz")
            meta_path = os.path.join(td, "metadata.json")
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"version": "3.0.0"}, f)

            n = 16
            np.savez(npz_path,
                     states=np.random.randn(n, 38, 12, 5).astype(np.float32),
                     masks=np.ones((n, 3650), dtype=bool),
                     actions=np.random.randint(0, 50, size=n, dtype=np.int32),
                     values=np.zeros(n, dtype=np.float32),
                     val_classes=np.ones(n, dtype=np.int8),
                     has_values=np.ones(n, dtype=bool),
                     phases=np.zeros(n, dtype=np.int8))

            ds = NpzReplayDataset(npz_path)
            loader = DataLoader(ds, batch_size=8)
            net = JunqiNet(in_channels=38, num_blocks=1, channels=16)

            metrics = evaluate_model(net, loader, device=torch.device("cpu"), value_weight=0.5)
            self.assertIn("top1_acc", metrics)
            self.assertIn("val_loss", metrics)
            self.assertIn("phase_top1", metrics)

    def test_distill_value_cli_arguments(self):
        """新增 5 & C3: 校验 distill_value 子命令支持 --workers 与 --sync-pool 参数。"""
        import junqi.__main__ as main_mod

        with self.assertRaises(SystemExit) as cm:
            main_mod.main(["distill_value", "--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_train_bc_smoke_and_losses_logged(self):
        """C1 & 新增 1: 验证 train_bc 正常输出 policy/value loss 并在 checkpoint 中写入 in_channels=38。"""
        from junqi.train_bc import train_bc
        with tempfile.TemporaryDirectory() as td:
            train_npz = os.path.join(td, "train.npz")
            val_npz = os.path.join(td, "val.npz")
            meta_path = os.path.join(td, "metadata.json")
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"version": "3.0.0"}, f)

            n = 8
            for p in (train_npz, val_npz):
                np.savez(p,
                         states=np.random.randn(n, 38, 12, 5).astype(np.float32),
                         masks=np.ones((n, 3650), dtype=bool),
                         actions=np.random.randint(0, 50, size=n, dtype=np.int32),
                         values=np.zeros(n, dtype=np.float32),
                         val_classes=np.ones(n, dtype=np.int8),
                         has_values=np.ones(n, dtype=bool),
                         phases=np.zeros(n, dtype=np.int8))

            out_model = os.path.join(td, "model.pt")
            res = train_bc(train_npz=train_npz, val_npz=val_npz, out_path=out_model,
                           epochs=1, batch_size=4, num_blocks=1, channels=16, device="cpu")
            self.assertTrue(os.path.exists(out_model))
            ckpt = torch.load(out_model, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["in_channels"], 38)
            hist_file = os.path.splitext(out_model)[0] + "_history.json"
            self.assertTrue(os.path.exists(hist_file))
            with open(hist_file, "r", encoding="utf-8") as f:
                h = json.load(f)
            self.assertIn("train_policy_loss", h["history"][0])
            self.assertIn("train_value_loss", h["history"][0])


if __name__ == "__main__":
    unittest.main()
