"""P2 阶段：复盘数据行为克隆（Behavior Cloning）训练管线。

严格遵守 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md 规范：
1. 训练公共视角 Policy，只在合法动作上做归一化，学习人类翻棋节奏与走子习惯；
2. Value 严格仅在 has_value=True（明确胜负/规则和棋）上反向传播，不污染未终局数据；
3. 输出带元数据签名的标准权重文件 models/bc_best.pt。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import NpzReplayDataset
from .net import JunqiNet


def evaluate_model(model: JunqiNet, val_loader: DataLoader,
                   device: torch.device, value_weight: float = 0.5) -> dict:
    """在验证集上全面评估 Policy 和 Value 指标。"""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total_samples = 0
    value_samples = 0

    phase_counts = {0: 0, 1: 0, 2: 0}
    phase_top1 = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 7:
                states, masks, actions, values, val_classes, has_values, phases = batch_data
                val_classes = val_classes.to(device)
            else:
                states, masks, actions, values, has_values, phases = batch_data
                val_classes = None

            states = states.to(device)
            masks = masks.to(device)
            actions = actions.to(device)
            values = values.to(device)
            has_values = has_values.to(device)
            phases = phases.to(device)

            batch_sz = actions.size(0)
            total_samples += batch_sz

            logits, pred_val = model(states, legal_mask=masks)
            loss_p = F.cross_entropy(logits, actions)
            total_policy_loss += loss_p.item() * batch_sz

            # Value Loss
            loss_v = torch.tensor(0.0, device=device)
            if has_values.sum() > 0:
                if pred_val.shape[-1] == 3:
                    if val_classes is None:
                        # 动态构造 3 分类目标
                        target_c = torch.where(values > 0.5, torch.tensor(0, device=device),
                                               torch.where(values < -0.5, torch.tensor(2, device=device),
                                                           torch.tensor(1, device=device)))
                    else:
                        target_c = val_classes
                    loss_v = F.cross_entropy(pred_val[has_values], target_c[has_values])
                else:
                    pred_v_sub = pred_val.squeeze(-1)[has_values]
                    target_v_sub = values[has_values]
                    loss_v = F.mse_loss(pred_v_sub, target_v_sub)
                n_v = has_values.sum().item()
                value_samples += n_v
                total_value_loss += loss_v.item() * n_v

            loss = loss_p + value_weight * loss_v
            total_loss += loss.item() * batch_sz

            # Top-1
            pred_top1 = logits.argmax(dim=-1)
            is_top1 = (pred_top1 == actions)
            correct_top1 += is_top1.sum().item()

            # Top-3
            _, top3_indices = logits.topk(3, dim=-1)
            is_top3 = (top3_indices == actions.unsqueeze(-1)).any(dim=-1)
            correct_top3 += is_top3.sum().item()

            # 分阶段统计
            for p_idx in (0, 1, 2):
                mask_p = (phases == p_idx)
                p_cnt = mask_p.sum().item()
                if p_cnt > 0:
                    phase_counts[p_idx] += p_cnt
                    phase_top1[p_idx] += (is_top1 & mask_p).sum().item()

    avg_loss = total_loss / total_samples if total_samples else 0.0
    avg_p_loss = total_policy_loss / total_samples if total_samples else 0.0
    avg_v_loss = total_value_loss / value_samples if value_samples else 0.0
    top1_acc = correct_top1 / total_samples if total_samples else 0.0
    top3_acc = correct_top3 / total_samples if total_samples else 0.0

    return {
        "val_loss": avg_loss,
        "policy_loss": avg_p_loss,
        "value_loss": avg_v_loss,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "phase_top1": {
            "opening": phase_top1[0] / phase_counts[0] if phase_counts[0] else 0.0,
            "midgame": phase_top1[1] / phase_counts[1] if phase_counts[1] else 0.0,
            "endgame": phase_top1[2] / phase_counts[2] if phase_counts[2] else 0.0,
        }
    }


def train_bc(train_npz: str = "datasets/p1_v1/train.npz",
             val_npz: str = "datasets/p1_v1/val.npz",
             out_path: str = "models/bc_best.pt",
             epochs: int = 20, batch_size: int = 256,
             lr: float = 1e-3, value_weight: float = 0.5,
             num_blocks: int = 6, channels: int = 128,
             device: Optional[str] = None, seed: int = 2026) -> dict:
    """执行行为克隆主训练循环。"""
    # 1. 种子与设备
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    dev = torch.device(device_str)
    print(f"[Train BC] 训练设备: {dev}, 批大小: {batch_size}, 轮数: {epochs}, 初始学习率: {lr}")

    # 2. 数据加载
    train_ds = NpzReplayDataset(train_npz)
    val_ds = NpzReplayDataset(val_npz)
    print(f"[Train BC] 训练集样本数: {len(train_ds)}, 验证集样本数: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=(dev.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=(dev.type == "cuda"))

    # 3. 初始化网络与优化器
    model = JunqiNet(in_channels=36, num_blocks=num_blocks, channels=channels).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    best_top1 = 0.0
    best_metrics = {}
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_p_loss = 0.0
        train_v_loss = 0.0
        train_correct_top1 = 0
        train_total = 0
        v_samples = 0

        ep_start = time.time()

        for batch_data in train_loader:
            if len(batch_data) == 7:
                states, masks, actions, values, val_classes, has_values, _ = batch_data
                val_classes = val_classes.to(dev)
            else:
                states, masks, actions, values, has_values, _ = batch_data
                val_classes = None

            states = states.to(dev)
            masks = masks.to(dev)
            actions = actions.to(dev)
            values = values.to(dev)
            has_values = has_values.to(dev)

            optimizer.zero_grad()
            logits, pred_val = model(states, legal_mask=masks)

            # Policy Loss
            loss_p = F.cross_entropy(logits, actions)

            # Value Loss (仅在有效胜负样本上计算)
            loss_v = torch.tensor(0.0, device=dev)
            if has_values.sum() > 0:
                if pred_val.shape[-1] == 3:
                    if val_classes is None:
                        target_c = torch.where(values > 0.5, torch.tensor(0, device=dev),
                                               torch.where(values < -0.5, torch.tensor(2, device=dev),
                                                           torch.tensor(1, device=dev)))
                    else:
                        target_c = val_classes
                    loss_v = F.cross_entropy(pred_val[has_values], target_c[has_values])
                else:
                    loss_v = F.mse_loss(pred_val.squeeze(-1)[has_values], values[has_values])
                n_v = has_values.sum().item()
                v_samples += n_v
                train_v_loss += loss_v.item() * n_v

            loss = loss_p + value_weight * loss_v
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_sz = actions.size(0)
            train_total += batch_sz
            train_loss += loss.item() * batch_sz
            train_p_loss += loss_p.item() * batch_sz
            train_correct_top1 += (logits.argmax(dim=-1) == actions).sum().item()

        scheduler.step()
        ep_duration = time.time() - ep_start

        # 评估验证集
        val_metrics = evaluate_model(model, val_loader, dev, value_weight=value_weight)

        t_top1 = train_correct_top1 / train_total if train_total else 0.0
        t_loss = train_loss / train_total if train_total else 0.0
        v_top1 = val_metrics["top1_acc"]
        v_top3 = val_metrics["top3_acc"]
        v_loss = val_metrics["val_loss"]

        log_entry = {
            "epoch": epoch,
            "train_loss": t_loss,
            "train_top1": t_top1,
            "val_loss": v_loss,
            "val_top1": v_top1,
            "val_top3": v_top3,
            "val_phase_top1": val_metrics["phase_top1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": ep_duration
        }
        history.append(log_entry)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.1f}s) - "
              f"Train Loss: {t_loss:.4f}, Top1: {t_top1*100:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Top1: {v_top1*100:.2f}%, Top3: {v_top3*100:.2f}% | "
              f"Phase Top1: (开:{val_metrics['phase_top1']['opening']*100:.1f}%, "
              f"中:{val_metrics['phase_top1']['midgame']*100:.1f}%, "
              f"残:{val_metrics['phase_top1']['endgame']*100:.1f}%)")

        if v_top1 > best_top1:
            best_top1 = v_top1
            best_metrics = val_metrics
            best_metrics["epoch"] = epoch
            # 保存最佳权重
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_blocks": num_blocks,
                "channels": channels,
                "in_channels": 36,
                "action_size": 3650,
                "epoch": epoch,
                "val_top1": v_top1,
                "val_top3": v_top3,
                "val_metrics": val_metrics,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, out_path)
            print(f"  --> [Checkpointed] 保存新最佳模型至 {out_path} (Val Top1: {v_top1*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[Train BC] 训练完成！总耗时: {total_time:.1f}s, 最佳 Val Top1: {best_top1*100:.2f}% (Epoch {best_metrics.get('epoch', 0)})")

    # 保存训练历史
    hist_path = os.path.splitext(out_path)[0] + "_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"best_metrics": best_metrics, "history": history}, f, indent=2, ensure_ascii=False)

    return {
        "best_val_top1": best_top1,
        "best_metrics": best_metrics,
        "total_time_sec": total_time,
        "model_path": out_path
    }


def main():
    parser = argparse.ArgumentParser(description="军棋翻棋行为克隆 (BC) 训练")
    parser.add_argument("--train-npz", default="datasets/p1_v1/train.npz", help="训练集 npz 路径")
    parser.add_argument("--val-npz", default="datasets/p1_v1/val.npz", help="验证集 npz 路径")
    parser.add_argument("--out", default="models/bc_best.pt", help="输出模型路径")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--value-weight", type=float, default=0.5, help="Value 损失权重")
    parser.add_argument("--blocks", type=int, default=6, help="ResNet 残差块数量")
    parser.add_argument("--channels", type=int, default=128, help="网络隐藏通道数")
    parser.add_argument("--device", default=None, help="训练设备 (cuda:0 / cpu)")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")

    args = parser.parse_args()
    train_bc(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_weight=args.value_weight,
        num_blocks=args.blocks,
        channels=args.channels,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
