"""P4.2 在线多世界采样与混合决策引擎 (Hybrid Online Decision Engine)。

核心机制：
1. 结合贝叶斯暗子信念跟踪 (BeliefTracker) 与并行确定化多世界采样 (PIMC)；
2. 批量将 K 个采样世界打包输入深度神经网络 (JunqiNet) 进行 GPU 张量推理；
3. 执行轻量战术硬约束拦截（一步吃旗、困毙判定、自杀送子剪枝、循环判和惩罚）；
4. 综合输出动作决策分布与三分类胜率概率。
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .belief import BeliefTracker
from .config import RuleConfig
from .encoder import action_to_index, encode_state_np, legal_action_mask
from .net import JunqiNet
from .rules import Rank, battle, is_camp, is_hq, other
from .state import Action, GameState, Piece, position_key


class HybridDecisionEngine:
    """P4.2 多世界采样与神经网络混合决策引擎。"""

    def __init__(self, model_path: str = "models/best.pt",
                 k_worlds: int = 4,
                 device: Optional[str] = None,
                 seed: Optional[int] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.k_worlds = max(1, k_worlds)
        self.rng = random.Random(seed)
        self.tracker = BeliefTracker(seed=seed)

        if model_path and os.path.exists(model_path):
            self.net = JunqiNet.load_from_file(model_path, device=device)
        else:
            self.net = JunqiNet().to(device)
            self.net.eval()

    def reset(self):
        """重置内部信念跟踪器。"""
        self.tracker = BeliefTracker(seed=self.rng.randint(0, 2**31 - 1))

    def evaluate_position(self, state: GameState,
                          tracker: Optional[BeliefTracker] = None,
                          history_counts: Optional[dict] = None) -> dict:
        """评估当前局面的胜/和/负概率、期望标量值与关键推荐走法。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            if state.winner == -1:
                return {"win": 0.0, "draw": 1.0, "loss": 0.0, "value": 0.0, "best_action": None}
            elif state.winner == state.turn:
                return {"win": 1.0, "draw": 0.0, "loss": 0.0, "value": 1.0, "best_action": None}
            else:
                return {"win": 0.0, "draw": 0.0, "loss": 1.0, "value": -1.0, "best_action": None}

        # 智能复用或同步信念跟踪器
        active_tracker = tracker or self.tracker
        hidden_set = set(state.hidden_positions())
        if not active_tracker.beliefs or active_tracker.hidden_positions != hidden_set:
            active_tracker.reset(state)

        worlds = active_tracker.sample_k_worlds(state, k=self.k_worlds, rng=self.rng)

        batch_tensors = []
        mask_np = legal_action_mask(state)
        for w in worlds:
            tensor = encode_state_np(state, seat=state.turn, world=w,
                                     history_counts=history_counts)
            batch_tensors.append(tensor)

        batch_t = torch.from_numpy(np.stack(batch_tensors)).float().to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).expand(len(worlds), -1).bool().to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits, val_out = self.net(batch_t, legal_mask=mask_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            if val_out.shape[-1] == 3:
                val_probs = F.softmax(val_out, dim=-1).cpu().numpy()
                mean_win = float(np.mean(val_probs[:, 0]))
                mean_draw = float(np.mean(val_probs[:, 1]))
                mean_loss = float(np.mean(val_probs[:, 2]))
                scalar_val = mean_win - mean_loss
            else:
                scalar_val = float(np.mean(val_out.cpu().numpy()))
                mean_win = max(0.0, scalar_val)
                mean_loss = max(0.0, -scalar_val)
                mean_draw = max(0.0, 1.0 - mean_win - mean_loss)

        # 跨世界动作概率均值
        mean_probs = np.mean(probs, axis=0)
        action_scores = {}
        for a in acts:
            idx = action_to_index(a)
            action_scores[a] = float(mean_probs[idx])

        # 战术硬规则加权
        scored_actions = self._apply_tactical_rules(state, action_scores, history_counts)
        best_act = scored_actions[0][0] if scored_actions else acts[0]

        return {
            "win": mean_win,
            "draw": mean_draw,
            "loss": mean_loss,
            "value": scalar_val,
            "best_action": best_act,
            "action_scores": scored_actions,
        }

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: Optional[Set] = None,
                       history_counts: Optional[dict] = None) -> List[Tuple[Action, float]]:
        """实现 Agent 接口：返回排序后的 [(action, score)] 列表。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 1.0)]

        eval_info = self.evaluate_position(state, history_counts=history_counts)
        scored = eval_info.get("action_scores", [])

        # 若存在 avoid 惩罚
        if avoid:
            adjusted = []
            for a, s in scored:
                if a.kind == "move":
                    pk = position_key(state.apply(a))
                    if pk in avoid:
                        s -= 100.0
                adjusted.append((a, s))
            adjusted.sort(key=lambda t: t[1], reverse=True)
            scored = adjusted

        return scored[:topn]

    def select_action(self, state: GameState,
                      avoid: Optional[Set] = None,
                      history_counts: Optional[dict] = None) -> Action:
        """返回最优单步决策动作。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid,
                                     history_counts=history_counts)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else Action("pass")

    def _apply_tactical_rules(self, state: GameState,
                              policy_probs: Dict[Action, float],
                              history_counts: Optional[dict] = None) -> List[Tuple[Action, float]]:
        """战术硬约束与启发式安全规则修正。"""
        acts = list(policy_probs.keys())
        scored: List[Tuple[Action, float]] = []

        my_clr = state.my_color()

        for a in acts:
            score = policy_probs.get(a, 0.0)

            if a.kind == "move":
                tgt = state.board.get(a.to)
                mover = state.board.get(a.frm)

                # 1. 一步吃旗制胜 (Instant Flag Capture)
                if tgt is not None and tgt.revealed and tgt.rank == Rank.QI:
                    score += 10000.0

                # 2. 困毙终结 (Immobilization Win)
                nxt = state.apply(a)
                if nxt.is_terminal() and nxt.winner == state.turn:
                    score += 5000.0
                elif nxt.is_terminal() and nxt.winner == -1:
                    # 避免在均势/优势下主动走成和棋
                    score -= 50.0

                # 3. 规避重复局面判和 (Repetition Avoidance)
                if history_counts:
                    pk = position_key(nxt)
                    c = history_counts.get(pk, 0)
                    if c >= 2:
                        score -= 200.0
                    elif c == 1:
                        score -= 10.0

                # 4. 自杀/撞大子初级防护
                if tgt is not None and tgt.revealed and mover is not None:
                    # 已知防守方大于攻击方（且非工兵挖雷、非炸弹）
                    if mover.rank > tgt.rank and mover.rank != Rank.ZHA and not (mover.rank == Rank.GONG and tgt.rank == Rank.LEI):
                        score -= 100.0

            scored.append((a, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
