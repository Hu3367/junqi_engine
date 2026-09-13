"""P4.2 在线多世界采样与混合决策引擎 (Hybrid Online Decision Engine)。

核心机制：
1. 结合贝叶斯暗子信念跟踪 (BeliefTracker) 与并行确定化多世界采样 (PIMC)；
2. 批量将 K 个采样世界打包输入深度神经网络 (JunqiNet) 进行 GPU 张量推理；
3. 规则层只保留硬判定：一步吃旗终局捷径、结构性死锁和棋前置拦截、
   重复局面根节点规避（价值量纲罚分）；
4. 动作定价 = QSearch 战术搜索估值 + 神经网络对数先验融合，
   彻底移除 ±15/±150/±400 等加性硬打分常数（2026-09-13 复盘分析报告
   根因 2.1"量纲崩溃"与 2.3"规则踩踏"的根本性修复，归属阶段 P2）。

价值量纲说明（对齐 HybridAgent / ai.py）：
  - 战术估值与终局分同处搜索分值空间（±WIN_SCORE / 百分级物质分）；
  - 先验项 = clip(log P(a|s) * prior_log_scale, -prior_band, 0)，影响上界
    prior_band=15 分 < 最小子力单位（排长 18 分），仅作为近平手动作的
    平局打破器，永不翻转搜索已定价的真实战术摆动；
  - 重复和棋规避使用 REP_PENALTY=150（高于常规战术分、低于胜负分），
    与 ai.py 的 Agent 根节点规避同口径。
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .analysis import is_dead_draw
from .belief import BeliefTracker
from .config import EvalWeights
from .encoder import action_to_index, encode_state_np, legal_action_mask
from .net import JunqiNet
from .rules import Rank
from .search import ExpertSearchEngine
from .state import WIN_SCORE, Action, GameState, position_key


class HybridDecisionEngine:
    """P4.2 多世界采样与神经网络混合决策引擎。"""

    def __init__(self, model_path: str = "models/best.pt",
                 k_worlds: int = 4,
                 device: Optional[str] = None,
                 seed: Optional[int] = None,
                 weights: Optional[EvalWeights] = None,
                 candidate_k: int = 12,
                 tactical_depth: int = 2,
                 prior_weight: float = 0.25,
                 prior_log_scale: float = 5.0,
                 prior_band: float = 15.0,
                 rep_penalty: float = 150.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.k_worlds = max(1, k_worlds)
        self.rng = random.Random(seed)
        self.tracker = BeliefTracker(seed=seed)
        self.weights = weights or EvalWeights()

        # P2：QSearch 战术定价器（吃子截断 + Star1 几率节点），与教师搜索同源
        self.engine = ExpertSearchEngine(weights=self.weights, seed=seed)
        self.candidate_k = max(4, candidate_k)
        self.tactical_depth = max(1, tactical_depth)
        self.prior_weight = float(prior_weight)
        self.prior_log_scale = float(prior_log_scale)
        self.prior_band = float(prior_band)
        self.rep_penalty = float(rep_penalty)

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

        # 结构性必和死锁前置拦截（规则层硬判定：直接返回 Draw）
        is_draw, _ = is_dead_draw(state)
        if is_draw:
            scored_acts = [(a, 0.0) for a in acts]
            return {
                "win": 0.0,
                "draw": 1.0,
                "loss": 0.0,
                "value": 0.0,
                "best_action": acts[0] if acts else None,
                "action_scores": scored_acts,
            }

        # 智能复用或同步信念跟踪器
        active_tracker = tracker or self.tracker
        hidden_set = set(state.hidden_positions())
        if not hidden_set:
            worlds = [{}]
        else:
            if not active_tracker.beliefs or active_tracker.hidden_positions != hidden_set:
                active_tracker.reset(state)
            worlds = active_tracker.sample_k_worlds(state, k=self.k_worlds, rng=self.rng)

        mask_np = legal_action_mask(state)

        # 1. 公共状态单次前向推理 -> 计算合规的公共 Policy（严格遵守 AGENTS.md：不得偷看暗子真实身份）
        public_tensor = encode_state_np(state, seat=state.turn, world=None,
                                        history_counts=history_counts)
        public_t = torch.from_numpy(public_tensor).unsqueeze(0).float().to(self.device)
        public_mask_t = torch.from_numpy(mask_np).unsqueeze(0).bool().to(self.device)

        self.net.eval()
        with torch.no_grad():
            public_logits, _ = self.net(public_t, legal_mask=public_mask_t)
            public_probs = F.softmax(public_logits, dim=-1).squeeze(0).cpu().numpy()

        policy_probs: Dict[Action, float] = {}
        for a in acts:
            idx = action_to_index(a)
            policy_probs[a] = float(public_probs[idx])

        # 2. 多世界采样批量前向推理 -> 仅计算 Value 头（期望胜率与价值评估）
        batch_tensors = []
        for w in worlds:
            tensor = encode_state_np(state, seat=state.turn, world=w,
                                     history_counts=history_counts)
            batch_tensors.append(tensor)

        batch_t = torch.from_numpy(np.stack(batch_tensors)).float().to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).expand(len(worlds), -1).bool().to(self.device)

        with torch.no_grad():
            _, val_out = self.net(batch_t, legal_mask=mask_t)

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

        # 3. QSearch 战术定价 + 对数先验融合（P2 重构：无加性硬打分常数）
        scored_actions = self._price_actions(state, policy_probs, history_counts)
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
        """实现 Agent 接口：返回排序后的 [(action, score)] 列表（搜索分值量纲）。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 1.0)]

        eval_info = self.evaluate_position(state, history_counts=history_counts)
        scored = eval_info.get("action_scores", [])
        if not scored and acts:
            scored = [(acts[0], 0.0)]

        # 重复和棋根节点规避（基线 §4.1：与 ai.py Agent 的 REP_PENALTY 同口径）
        if avoid:
            adjusted = []
            for a, s in scored:
                if a.kind == "move" and position_key(state.apply(a)) in avoid:
                    s -= self.rep_penalty
                adjusted.append((a, s))
            adjusted.sort(key=lambda t: t[1], reverse=True)
            scored = adjusted

        if not scored and acts:
            scored = [(acts[0], 0.0)]

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

    # ------------------------------------------------------------- 动作定价

    def _price_actions(self, state: GameState,
                       policy_probs: Dict[Action, float],
                       history_counts: Optional[dict] = None) -> List[Tuple[Action, float]]:
        """P2 动作定价：QSearch 战术搜索估值 + 神经网络对数先验融合。

        规则层硬判定（不经评分）：
          1. 一步吃旗必胜 -> 终局捷径直接返回；
          2. 走后立即终局的动作 -> ±WIN_SCORE / 0；
        其余动作的"好与坏"全部交给战术搜索定价，先验仅做平局打破。
        """
        # 终局捷径：一步吃旗制胜，直接选为决策动作（Terminal Win Shortcut）
        for a in policy_probs:
            if a.kind == "move":
                tgt = state.board.get(a.to)
                if tgt is not None and tgt.revealed and tgt.rank == Rank.QI:
                    return [(a, float(WIN_SCORE))]

        # 候选集：策略 Top-K + 全部吃明子走法（战术验证不可漏吃子线）
        ranked = sorted(policy_probs.items(), key=lambda t: t[1], reverse=True)
        cand_set = {a for a, _ in ranked[:self.candidate_k]}
        for a in policy_probs:
            if a.kind == "move":
                tgt = state.board.get(a.to)
                if tgt is not None and tgt.revealed:
                    cand_set.add(a)

        scored: List[Tuple[Action, float]] = []
        for a in cand_set:
            if a.kind == "move":
                nxt = state.apply(a)
                if nxt.is_terminal():
                    if nxt.winner == state.turn:
                        val = float(WIN_SCORE)
                    elif nxt.winner == -1:
                        val = 0.0
                    else:
                        val = -float(WIN_SCORE)
                else:
                    # 对手视角浅层搜索（叶子由 QSearch 吃子截断，
                    # 彻底消除 1-ply 反杀检测的地平线盲区）。
                    # as_evaluator=True：跳过教师"唯一合法走法返 0 分"的决策捷径，
                    # 返回强制应着的真实搜索分（子局必胜/必败线不被误估为 0）
                    _, opp_score, _ = self.engine.search(
                        nxt, max_depth=max(1, self.tactical_depth - 1),
                        as_evaluator=True)
                    val = -opp_score
            else:
                # 翻棋几率节点：Star1 期望估值（含深层截断解析回退）
                val = self.engine._evaluate_chance_flip(
                    state, a, self.tactical_depth, 0, -WIN_SCORE, WIN_SCORE, set())

            # 重复局面规避（价值量纲罚分：高于常规战术分、低于胜负分）
            if history_counts and a.kind == "move":
                c = history_counts.get(position_key(state.apply(a)), 0)
                if c >= 2:
                    val -= self.rep_penalty
                elif c == 1:
                    val -= self.rep_penalty * 0.1

            # 对数先验融合：先验影响被 prior_band 截断（默认 15 分 < 最小子力单位
            # "排长 18 分"），保证先验只做近平手动作的平局打破器，
            # 永不翻转搜索已定价的真实子力/战术摆动（量纲崩溃的核心防线）
            p = max(float(policy_probs.get(a, 0.0)), 1e-6)
            prior_term = max(-self.prior_band,
                             min(0.0, math.log(p) * self.prior_log_scale))
            total = (1.0 - self.prior_weight) * val + self.prior_weight * prior_term
            scored.append((a, total))

        # 非候选动作（策略尾部且非吃子）：不做战术搜索，按先验-only 评分补齐，
        # 保持 choose_actions 返回完整合法动作排序的既有契约（如 scan_replays 依赖）
        priced = {a for a, _ in scored}
        for a, p_raw in policy_probs.items():
            if a in priced:
                continue
            p = max(float(p_raw), 1e-6)
            prior_term = max(-self.prior_band,
                             min(0.0, math.log(p) * self.prior_log_scale))
            total = self.prior_weight * prior_term
            if history_counts and a.kind == "move":
                c = history_counts.get(position_key(state.apply(a)), 0)
                if c >= 2:
                    total -= self.rep_penalty
            scored.append((a, total))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
