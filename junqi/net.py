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


def _mask_from_actions(acts) -> "np.ndarray":
    """由已算出的合法动作列表直接生成掩码，省掉一次 legal_actions() 遍历。"""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for a in acts:
        mask[action_to_index(a)] = True
    return mask


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

    # ------------------------------------------------------------- 模式切换短路
    #
    # 批次 4（2026-09-14）：自博弈热路径每次单状态推理都调用 self.eval()，而
    # nn.Module.train/eval 会**递归遍历全部子模块**并逐个 __setattr__ 改 training 标志。
    # 实测代价（cProfile，1 局 301 手 / sims=20）：369,812 次子模块遍历、约 8.7s 累计
    # （占单局墙钟 123.7s 的约 7%，另叠加 module.__setattr__ 4.8s）。
    # 模式未变时直接返回；语义与 nn.Module.train 一致（仅在已是目标模式时跳过遍历）。
    def train(self, mode: bool = True):
        if self.training == mode:
            return self
        return super().train(mode)

    def eval(self):
        return self.train(False)

    # ------------------------------------------------------------- 实用推理方法

    @torch.no_grad()
    def predict_state(self, state: GameState, seat: int | None = None,
                      world: Optional[dict] = None,
                      history_counts: Optional[dict] = None,
                      device: torch.device | str = "cpu",
                      acts: Optional[list] = None,
                      mask: Optional["torch.Tensor"] = None) -> Tuple[dict[Action, float], float]:
        """单状态便捷推理：输出合法走法概率字典及当前局面的标量期望估值 (-1.0 ~ +1.0)。

        P2 优化（审查 P2，2026-09-15）：原本内部调用两次 `state.legal_actions()`
        （一次生成掩码、一次组装 policy 字典）。MCTS 叶子评估这条热路径上，
        调用方通常已经算过合法动作，可通过 `acts` / `mask` 传入避免重复遍历棋盘。
        """
        self.eval()
        tensor = encode_state(state, seat=seat, world=world,
                              history_counts=history_counts, device=device)
        if mask is None:
            if acts is None:
                acts = state.legal_actions()
            mask_np = _mask_from_actions(acts)
            mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        acts = acts if acts is not None else state.legal_actions()

        logits, val_out = self.forward(tensor, legal_mask=mask)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if val_out.shape[-1] == 3:
            val_probs = F.softmax(val_out, dim=-1).squeeze(0).cpu().numpy()
            v = float(val_probs[0] - val_probs[2])  # P(win) - P(loss)
        else:
            v = float(torch.tanh(val_out).item()) if not hasattr(self.value_head[-1], 'tanh') else float(val_out.item())

        policy = {}
        for act in acts:
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

    # ------------------------------------------------- 权重读取与就地载入

    @staticmethod
    def unwrap_state_dict(obj) -> dict:
        """把 checkpoint 产物规范成裸 state_dict。

        `self.save()` 写出的文件带有 {"model_state", "in_channels", ...} 包装，
        而 `torch.load` 只是把它原样读回。历史上曾有调用点把这个包装字典直接
        喂给 `load_state_dict(..., strict=False)`：键名无一匹配、STATIC 静默通过，
        使自博弈 25% 的 "best 对手" 退化为随机初始化网络（审查 R2）。
        所有载入路径必须先过本方法。
        """
        if isinstance(obj, dict):
            if isinstance(obj.get("model_state"), dict):
                return obj["model_state"]
            return obj
        return obj.state_dict()          # 直接 pickle 出来的 nn.Module

    @classmethod
    def adapt_state_dict(cls, state_dict: dict, in_channels: int) -> dict:
        """按目标网络的输入通道/价值头维度改写旧格式 state_dict。"""
        adapted = {}
        for k, v in state_dict.items():
            if k == "in_conv.0.weight" and v.shape[1] < in_channels:
                new_w = torch.zeros((v.shape[0], in_channels, v.shape[2], v.shape[3]),
                                    dtype=v.dtype)
                new_w[:, :v.shape[1], :, :] = v
                adapted[k] = new_w
            elif k == "value_head.6.weight" and v.shape[0] == 1:
                new_w = torch.zeros((3, v.shape[1]), dtype=v.dtype)
                new_w[0, :] = v[0, :]
                new_w[2, :] = -v[0, :]
                adapted[k] = new_w
            elif k == "value_head.6.bias" and v.shape[0] == 1:
                new_b = torch.zeros(3, dtype=v.dtype)
                new_b[0] = v[0]
                new_b[2] = -v[0]
                adapted[k] = new_b
            else:
                adapted[k] = v
        return adapted

    @classmethod
    def load_state_dict_into(cls, net: "JunqiNet", state_dict) -> Tuple[list, list]:
        """把权重**就地**载入已存在的 net，返回 (missing_keys, unexpected_keys)。

        与 `load_from_file` 的区别：本方法不新建对象，因此在其之前构造的
        optimizer/LR scheduler 仍与 net 保持绑定。热启动链路必须使用本方法，
        禁止 `net = JunqiNet.load_from_file(...)` 式的变量名重绑（审查 R1）。
        """
        sd = cls.adapt_state_dict(cls.unwrap_state_dict(state_dict), net.in_channels)
        res = net.load_state_dict(sd, strict=False)
        return list(res.missing_keys), list(res.unexpected_keys)

    @classmethod
    def load_from_file(cls, path: str, device: torch.device | str = "cpu",
                       in_channels: int = NUM_CHANNELS, num_blocks: int = 6,
                       channels: int = 128) -> "JunqiNet":
        """从磁盘加载并**新建**一个网络（供推理/策略构造使用）。

        兼容三种文件形态：
          · `net.save()` 的包装字典 `{"model_state": ...}`
          · `save_checkpoint()` 的完整检查点 `{"net": ..., "optimizer": ...}`
          · 裸 state_dict 或直接 pickle 的 nn.Module

        2026-09-15 修复：原实现不认检查点形态（只判 `"model_state"`），
        遇到 `candidate_latest.pt` 会把它整个 dict 当 state_dict 喂给
        `load_state_dict(strict=False)` —— **键名无一匹配、静默载入零个权重**，
        得到一个随机初始化的网络。这与审查 R2（自博弈"best 对手"实为随机网络）
        是同一类失效模式，只是发生在推理/评测路径上。
        """
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, dict) and "model_state" in obj:
            b = obj.get("num_blocks", num_blocks)
            c = obj.get("channels", channels)
            sd = obj["model_state"]
        elif isinstance(obj, dict) and isinstance(obj.get("net"), dict):
            # 完整训练检查点（save_checkpoint 产物）
            b = obj.get("num_blocks", num_blocks)
            c = obj.get("channels", channels)
            sd = obj["net"]
        elif isinstance(obj, dict):
            sd = obj
            b = num_blocks
            c = channels
        else:
            net = obj
            net.to(device)
            net.eval()
            return net

        net = cls(in_channels=in_channels, num_blocks=b, channels=c)
        adapted = cls.adapt_state_dict(sd, in_channels)
        res = net.load_state_dict(adapted, strict=False)
        # 兜底告警：若绝大多数权重都没载入，说明给错了文件（例如把检查点/日志
        # 当权重传进来）。保持 strict=False 的宽容语义，但绝不静默。
        expected = len(net.state_dict())
        if expected and len(res.missing_keys) > expected * 0.5:
            print(f"⚠️ JunqiNet.load_from_file: {path} 有 {len(res.missing_keys)}/"
                  f"{expected} 个权重键未载入 —— 该文件可能不是模型权重，"
                  f"得到的网络接近随机初始化", flush=True)
        net.to(device)
        net.eval()
        return net
