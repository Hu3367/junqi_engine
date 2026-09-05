"""双头策略-价值深度神经网络（Policy-Value ResNet）。

输入: 棋盘多通道张量 [Batch, 31, 12, 5]
输出:
  - Policy Logits: [Batch, 3650] 动作分布
  - Value: [Batch, 1] 局面胜率估值 (-1.0 ~ +1.0)
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (ACTION_SPACE_SIZE, NUM_CHANNELS, action_to_index,
                    encode_state, encode_state_np, legal_action_mask)
from .state import Action, GameState


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class JunqiNet(nn.Module):
    def __init__(self, in_channels: int = NUM_CHANNELS, num_blocks: int = 6,
                 channels: int = 128, action_size: int = ACTION_SPACE_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.action_size = action_size

        # 特征提取主干 (Backbone)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # 策略头 (Policy Head)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 12 * 5, action_size),
        )

        # 价值头 (Value Head - Win/Draw/Loss 三分类)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 12 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # [Win, Draw, Loss] Logits
        )

        # S2 辅助回归头：监督 material_diff（编码器通道 25，analysis.py 确定性可算）。
        # 属监督信号而非奖励塑形（不改变终局目标），为主干提供密集锚定信号，
        # 防止价值头在稀疏终局标签下塌缩（KataGo scoreLead 类比）。
        # 旧检查点无此键，加载一律走 strict=False 兼容。
        self.aux_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 12 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),   # 输出 [-1, 1]，与 material_diff 值域一致
        )

    def _align_channels(self, x: torch.Tensor) -> torch.Tensor:
        """输入通道数对齐（裁剪/补零），兼容旧版 36/31 通道模型。"""
        if x.shape[1] > self.in_channels:
            x = x[:, :self.in_channels, :, :]
        elif x.shape[1] < self.in_channels:
            pad = torch.zeros((x.shape[0], self.in_channels - x.shape[1], x.shape[2], x.shape[3]),
                              dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        return x

    def forward(self, x: torch.Tensor,
                legal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算。
        x: [B, C, 12, 5]
        legal_mask: [B, Action_Size] 布尔或 0/1 张量
        返回: (logits [B, Action_Size], value_logits [B, 3] 或 value [B, 1])
        """
        x = self._align_channels(x)
        feat = self.in_conv(x)
        for block in self.blocks:
            feat = block(feat)

        logits = self.policy_head(feat)
        if legal_mask is not None:
            # 过滤非法走法
            inv_mask = ~legal_mask.bool()
            logits = logits.masked_fill(inv_mask, -1e9)

        value_logits = self.value_head(feat)
        return logits, value_logits

    def forward_with_aux(self, x: torch.Tensor,
                         legal_mask: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练专用前向：额外输出辅助回归值 aux [B]（material_diff 监督目标）。
        推理链路仍用 forward（保持 2 元组接口不变）。"""
        x = self._align_channels(x)
        feat = self.in_conv(x)
        for block in self.blocks:
            feat = block(feat)

        logits = self.policy_head(feat)
        if legal_mask is not None:
            inv_mask = ~legal_mask.bool()
            logits = logits.masked_fill(inv_mask, -1e9)

        value_logits = self.value_head(feat)
        aux = self.aux_head(feat).squeeze(-1)
        return logits, value_logits, aux

    # ------------------------------------------------------------- 实用推理方法

    @torch.no_grad()
    def predict_state(self, state: GameState, seat: int | None = None,
                      world: Optional[dict] = None,
                      history_counts: Optional[dict] = None,
                      device: torch.device | str = "cpu") -> Tuple[dict[Action, float], float]:
        """单状态便捷推理：输出合法走法概率字典及当前局面的标量期望估值 (-1.0 ~ +1.0)。"""
        self.eval()
        tensor = encode_state(state, seat=seat, world=world,
                              history_counts=history_counts, device=device)
        mask_np = legal_action_mask(state)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

        logits, val_out = self.forward(tensor, legal_mask=mask_t)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if val_out.shape[-1] == 3:
            val_probs = F.softmax(val_out, dim=-1).squeeze(0).cpu().numpy()
            v = float(val_probs[0] - val_probs[2])  # P(win) - P(loss)
        else:
            v = float(torch.tanh(val_out).item()) if not hasattr(self.value_head[-1], 'tanh') else float(val_out.item())

        policy = {}
        for act in state.legal_actions():
            idx = action_to_index(act)
            policy[act] = float(probs[idx])
        return policy, v

    @torch.no_grad()
    def predict_probabilities(self, state: GameState, seat: int | None = None,
                              world: Optional[dict] = None,
                              history_counts: Optional[dict] = None,
                              device: torch.device | str = "cpu") -> Tuple[dict[Action, float], dict[str, float]]:
        """输出策略分布及完整的胜/和/负概率字典 {"win": p, "draw": p, "loss": p}。"""
        self.eval()
        tensor = encode_state(state, seat=seat, world=world,
                              history_counts=history_counts, device=device)
        mask_np = legal_action_mask(state)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

        logits, val_out = self.forward(tensor, legal_mask=mask_t)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if val_out.shape[-1] == 3:
            v_p = F.softmax(val_out, dim=-1).squeeze(0).cpu().numpy()
            v_dict = {"win": float(v_p[0]), "draw": float(v_p[1]), "loss": float(v_p[2])}
        else:
            # 兼容旧 1 维 value
            v_val = float(val_out.item())
            p_win = max(0.0, v_val)
            p_loss = max(0.0, -v_val)
            p_draw = max(0.0, 1.0 - p_win - p_loss)
            v_dict = {"win": p_win, "draw": p_draw, "loss": p_loss}

        policy = {act: float(probs[action_to_index(act)]) for act in state.legal_actions()}
        return policy, v_dict

    @torch.no_grad()
    def predict_batch(self, items: List[Tuple],
                      device: torch.device | str = "cpu") -> List[Tuple[dict[Action, float], float]]:
        """批量状态推理：items 为 [(state, seat, world), ...] 或 [(state, seat, world, history_counts), ...] 列表。
        一次性打包张量与掩码做 GPU 前向推理，返回 [(policy_dict, value), ...]。
        """
        if not items:
            return []
        self.eval()
        tensors = []
        masks = []
        for item in items:
            st = item[0]
            seat = item[1]
            world = item[2]
            hist = item[3] if len(item) > 3 else None
            tensors.append(encode_state_np(st, seat=seat, world=world, history_counts=hist))
            masks.append(legal_action_mask(st))

        batch_tensor = torch.from_numpy(np.stack(tensors)).float().to(device)
        batch_mask = torch.from_numpy(np.stack(masks)).bool().to(device)

        logits, val_out = self.forward(batch_tensor, legal_mask=batch_mask)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        if val_out.shape[-1] == 3:
            val_probs = F.softmax(val_out, dim=-1).cpu().numpy()
            vals = val_probs[:, 0] - val_probs[:, 2]
        else:
            vals = val_out.squeeze(1).cpu().numpy() if val_out.dim() > 1 else val_out.cpu().numpy()

        results = []
        for i, item in enumerate(items):
            st = item[0]
            p_map = {act: float(probs[i, action_to_index(act)]) for act in st.legal_actions()}
            v_i = float(vals[i]) if hasattr(vals, '__len__') else float(vals)
            results.append((p_map, v_i))
        return results

    # ------------------------------------------------------------- 模型持久化

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "in_channels": self.in_channels,
            "num_blocks": len(self.blocks),
            "channels": self.in_conv[0].out_channels,
        }, path)

    @classmethod
    def load_from_file(cls, path: str, device: torch.device | str = "cpu",
                       in_channels: int = NUM_CHANNELS, num_blocks: int = 6,
                       channels: int = 128) -> "JunqiNet":
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, dict) and "model_state" in obj:
            b = obj.get("num_blocks", num_blocks)
            c = obj.get("channels", channels)
            in_c = obj.get("in_channels", in_channels)
            sd = obj["model_state"]
        elif isinstance(obj, dict):
            sd = obj
            b = num_blocks
            c = channels
            in_c = in_channels
        else:
            net = obj
            net.to(device)
            net.eval()
            return net

        net = cls(in_channels=in_channels, num_blocks=b, channels=c)
        adapted_sd = {}
        for k, v in sd.items():
            if k == "in_conv.0.weight" and v.shape[1] < in_channels:
                new_w = torch.zeros((v.shape[0], in_channels, v.shape[2], v.shape[3]), dtype=v.dtype)
                new_w[:, :v.shape[1], :, :] = v
                adapted_sd[k] = new_w
            elif k == "value_head.6.weight" and v.shape[0] == 1:
                new_w = torch.zeros((3, v.shape[1]), dtype=v.dtype)
                new_w[0, :] = v[0, :]
                new_w[2, :] = -v[0, :]
                adapted_sd[k] = new_w
            elif k == "value_head.6.bias" and v.shape[0] == 1:
                new_b = torch.zeros(3, dtype=v.dtype)
                new_b[0] = v[0]
                new_b[2] = -v[0]
                adapted_sd[k] = new_b
            else:
                adapted_sd[k] = v

        net.load_state_dict(adapted_sd, strict=False)
        net.to(device)
        net.eval()
        return net
