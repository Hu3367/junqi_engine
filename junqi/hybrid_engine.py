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
from .rules import (ATTACKER_WINS, BOTH_DIE, CAMPS, NEIGHBORS, Rank,
                    battle, is_camp, is_hq, other)
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

        # 结构性必和死锁前置拦截
        from .analysis import is_dead_draw
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

        action_scores = {}
        for a in acts:
            idx = action_to_index(a)
            action_scores[a] = float(public_probs[idx])

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
        if not scored and acts:
            scored = [(acts[0], 0.0)]

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

                # 4. 1-ply 战术反扑威胁与反杀检测 (Counter-Attack / Poisoned Piece Trap)
                # 检查若执行此动作，下一手敌方明子是否能在 a.to 处反杀我方 mover
                fatal_threat = False
                is_threatened = False
                for opp_a in nxt.legal_actions():
                    if opp_a.kind == "move" and opp_a.to == a.to:
                        opp_pc = nxt.board.get(opp_a.frm)
                        if opp_pc is not None and opp_pc.revealed and mover is not None:
                            b_res = battle(opp_pc.rank, mover.rank)
                            if b_res in (ATTACKER_WINS, BOTH_DIE):
                                is_threatened = True
                                if b_res == ATTACKER_WINS or opp_pc.rank <= mover.rank:
                                    fatal_threat = True
                                    break

                leaves_camp = is_camp(a.frm) and not is_camp(a.to)
                enters_camp = not is_camp(a.frm) and is_camp(a.to)
                camp_to_camp = is_camp(a.frm) and is_camp(a.to)

                # 检查放弃行营后，行营是否次手直接面临敌方入侵占领 (丢营风险)
                camp_invadable = False
                if leaves_camp:
                    for opp_a in nxt.legal_actions():
                        if opp_a.kind == "move" and opp_a.to == a.frm:
                            camp_invadable = True
                            break

                # 行营据点战略奖励：积极占领空行营或营间机动
                if enters_camp:
                    score += 25.0
                elif camp_to_camp:
                    score += 15.0

                # 5. 吃子战术硬门保护与白送/诱杀陷阱严格拦截（基于 rules.battle）
                from .config import EvalWeights
                weights = EvalWeights()

                if tgt is not None and tgt.revealed and mover is not None:
                    res = battle(mover.rank, tgt.rank)
                    piece_val = weights.piece.get(tgt.rank, 30.0)
                    mover_val = weights.piece.get(mover.rank, 30.0)

                    if res == "defender_wins":
                        # 真正的自杀：小子撞大子或非工兵撞地雷
                        score -= 500.0
                    elif res == "attacker_wins":
                        if fatal_threat:
                            # 诱杀陷阱：吃子后次手立即被敌方反杀！
                            # 若吃小亏大（如工兵挖雷被连长反吃、或大子吃小子被反吃），或弃营被反杀，重度惩罚
                            if piece_val < mover_val or leaves_camp:
                                score -= 300.0
                            else:
                                score += max(0.0, (piece_val - mover_val) * 0.5)
                        else:
                            # 安全吃子：军长吃排长、营长吃排长、工兵挖雷等
                            score += 50.0 + piece_val * 0.5
                            if mover.rank == Rank.GONG and tgt.rank == Rank.LEI:
                                score += 30.0
                    elif res == "both_die":
                        # 同归于尽：炸弹兑高价值大子（司令/军长/师长）给予战术优先
                        if mover.rank == Rank.ZHA and tgt.rank in (Rank.SI, Rank.JUN, Rank.SHI):
                            score += 80.0
                        elif mover.rank == Rank.ZHA and tgt.rank not in (Rank.QI, Rank.SI, Rank.JUN, Rank.SHI):
                            # 严禁炸弹主动撞廉价小子 (排/连/营/团/工兵) 自爆贱卖
                            score -= 350.0
                        elif fatal_threat:
                            score -= 100.0
                else:
                    # 静止/普通走步：若无故走入敌方明子火力网白送吃
                    if fatal_threat:
                        score -= 400.0

                # 6. 行营战略庇护特权与“占营优于吃小子”原则 (Camp Hegemony)
                # 行营是不可侵犯的绝对避难所与控制据点。离开行营意味着丧失庇护！
                if leaves_camp:
                    if fatal_threat:
                        # 任何离开行营后次手在目标格面临被反杀的走法，无论目的为何均顶格严惩
                        score -= 400.0
                    if tgt is not None and tgt.revealed:
                        # 离开行营去吃小子（排长、连长、营长或地雷）
                        # 核心棋理：实战占营比吃一个小子更重要，严防为了贪吃廉价小子而主动弃营
                        if tgt.rank not in (Rank.QI, Rank.SI, Rank.JUN):
                            if mover.rank <= Rank.YING:
                                score -= 150.0  # 小子/中子弃营吃小子，得不偿失
                            elif fatal_threat or camp_invadable:
                                score -= 150.0  # 大子弃营吃小子但面临反扑或丢营
                            else:
                                score -= 20.0   # 安全出击但放弃据点折损
                    else:
                        # 无目的离开行营（闲走弃营）
                        score -= 80.0

                    if camp_invadable:
                        # 弃营且次手即被敌军入驻占领（丢营）
                        score -= 100.0

            elif a.kind == "flip":
                # 7. 据点辐射拓荒翻棋激励 (Adjacent Flip Incentive)
                # 若翻棋位置邻接己方已占领的行营，给予据点辐射战术加分，破除“龟缩拒翻”死循环
                is_camp_adj = False
                for c in CAMPS:
                    if a.frm in NEIGHBORS[c]:
                        occ = state.board.get(c)
                        if occ is not None and occ.revealed and occ.color == my_clr:
                            is_camp_adj = True
                            break
                if is_camp_adj:
                    score += 15.0

            scored.append((a, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
