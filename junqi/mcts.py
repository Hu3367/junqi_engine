"""神经网络引导的蒙特卡洛树搜索（MCTS with Policy-Value Net - V2.2 P0 修复版）。

核心修复：
1. 修复 D1 缺陷：确定化世界沿模拟路径传播，叶子节点以世界模式（world != None）编码送网；
2. 根节点采用 K=4 采样世界平均先验；
3. Dirichlet 探索噪声修复（自适应 alpha = 10.0 / num_actions）；
4. 支持返回叶子节点世界模式样本供 Value 学习；
5. P0 修复：终局价值统一为“走子者（叶子轮次方）视角”（_terminal_value），
   修复 apply() 切换 turn 导致的胜局反号（强制吃旗被传播成 -1）；
6. P0 修复：选择阶段按当前采样世界的合法动作过滤子节点（ISMCTS 可用性语义），
   展开阶段子节点取并集不覆盖，消除跨世界树复用造成的非法模拟。
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .encoder import action_to_index, encode_state_np
from .net import JunqiNet
from .state import Action, GameState, position_key


class MCTSNode:
    """MCTS 树节点。"""
    __slots__ = ("prior", "visit_count", "value_sum", "children", "is_expanded")

    def __init__(self, prior: float = 0.0):
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[Action, MCTSNode] = {}
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 0.6) -> float:
        u = c_puct * self.prior * math.sqrt(max(1, parent_visits)) / (1 + self.visit_count)
        return self.q_value + u


def _terminal_value(state: GameState) -> float:
    """终局价值：走子者（叶子轮次方）视角，+1 胜 / -1 负 / 0 和。

    关键约定：GameState.apply() 在判定胜者后会将 turn 切换给对手，
    因此终局态下 winner == 1 - state.turn（走子者）。反向传播约定
    leaf_turn = 1 - state.turn，与此处视角严格一致。
    修复回归：旧版误用 state.turn 判视角，一步吃旗被传播成 -1。
    """
    if state.winner is None or state.winner == -1:
        return 0.0
    mover = 1 - state.turn
    return 1.0 if state.winner == mover else -1.0


class MCTS:
    """PUCT 树搜索控制器（参数严格对齐非完全信息翻棋黄金探索区间）。"""

    def __init__(self, net: JunqiNet, simulations: int = 100,
                 c_puct: float = 0.6, dirichlet_eps: float = 0.20,
                 device: torch.device | str = "cpu"):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    def search(self, root_state: GameState, temperature: float = 1.0,
               add_noise: bool = False, rng: Optional[random.Random] = None,
               history_counts: Optional[dict] = None,
               avoid: Optional[Set] = None) -> Tuple[Action, np.ndarray, Dict[Action, float], List[Tuple[np.ndarray, int]]]:
        """执行 MCTS 搜索。
        返回: (选定动作, 访问概率分布 pi_vec [3650], 动作访问分布 dict, 叶子状态样本列表 [(world_state_arr, seat)])
        """
        rng = rng or random.Random()
        root = MCTSNode(prior=1.0)
        acts = root_state.legal_actions()

        if not acts or root_state.is_terminal():
            raise ValueError("无合法走法或已终局")
        if len(acts) == 1:
            pi_dict = {acts[0]: 1.0}
            pi_vec = np.zeros(3650, dtype=np.float32)
            pi_vec[action_to_index(acts[0])] = 1.0
            return acts[0], pi_vec, pi_dict, []

        has_hidden = bool(root_state.hidden_positions())

        # 1. 根节点展开：采用 K=4 个采样世界平均先验
        if has_hidden:
            sample_k = 4
            root_worlds = [root_state.sample_world(rng, reveal=False) for _ in range(sample_k)]
        else:
            sample_k = 1
            root_worlds = [None]

        # 批量推理计算 K 个世界的策略先验（传入 history_counts 激活重复通道 36/37）
        batch_items = [(root_state, root_state.turn, w, history_counts) for w in root_worlds]
        batch_results = self.net.predict_batch(batch_items, device=self.device)

        acc_priors = {a: 0.0 for a in acts}
        for policy_map, _ in batch_results:
            for a in acts:
                acc_priors[a] += policy_map.get(a, 1e-4)

        for a in acts:
            prior_val = acc_priors[a] / sample_k
            # 若动作导致命中 avoid 集合，大幅压低先验促使搜索其它路径
            if avoid and position_key(root_state.apply(a)) in avoid:
                prior_val = 1e-6
            root.children[a] = MCTSNode(prior=prior_val)
        root.is_expanded = True

        # 2. 添加 Dirichlet 探索噪声
        # P0 修复（§3.1.5）：不再使用全局 np.random，改由实验种子派生的 rng 派生
        # 独立的 numpy Generator，保证固定种子下噪声可复现。
        if add_noise and len(root.children) > 1:
            num_acts = len(root.children)
            alpha_val = 0.15
            alphas = [alpha_val] * num_acts
            np_rng = np.random.default_rng(rng.randrange(1 << 31))
            dir_samples = np_rng.dirichlet(alphas)
            for (act, child), d in zip(root.children.items(), dir_samples):
                child.prior = (1.0 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * float(d)

        leaf_samples: List[Tuple[np.ndarray, int]] = []

        # 3. 模拟主循环
        for _ in range(self.simulations):
            # 每次模拟采样一个确定化世界并在搜索路径中完整传播
            if has_hidden:
                world = root_state.sample_world(rng, reveal=False)
                sim_state = root_state.instantiate(world)
            else:
                world = None
                sim_state = root_state

            node = root
            state = sim_state
            sim_history = Counter(history_counts) if history_counts else Counter()
            sim_history[position_key(state)] += 1
            search_path: List[Tuple[MCTSNode, Action, int]] = []

            # 3.1 选择阶段 (Select)
            # P0 修复：子节点是在其他采样世界中展开的，某世界下合法的明子动作
            # 在另一世界可能不存在。选择时必须按当前世界的合法动作过滤，
            # 否则 GameState.apply()（不校验合法性）会产生非法模拟并污染统计。
            while node.is_expanded and not state.is_terminal():
                parent_visits = node.visit_count
                legal_acts = set(state.legal_actions())
                best_act, best_child = None, None
                best_score = -float("inf")

                for act, child in node.children.items():
                    if act not in legal_acts:
                        continue      # 该动作在当前世界非法，跳过（可用性过滤）
                    score = child.ucb_score(parent_visits, self.c_puct)
                    if score > best_score:
                        best_score = score
                        best_act = act
                        best_child = child

                if best_act is None:
                    # 当前世界在此节点无可用子动作：跳出选择，按叶子处理并补展开
                    break

                search_path.append((node, best_act, state.turn))
                state = state.apply(best_act)
                pk = position_key(state)
                sim_history[pk] += 1
                if sim_history[pk] >= 3:
                    state.winner, state.win_reason = -1, "repetition"
                node = best_child

            # 3.2 展开与评估阶段 (Expand & Evaluate)
            value = 0.0
            if state.is_terminal():
                # P0 修复：终局价值统一走子者视角（原公式因 apply 切换 turn 而反号）
                value = _terminal_value(state)
            else:
                acts_curr = state.legal_actions()
                if acts_curr:
                    policy_map, v = self.net.predict_state(state, seat=state.turn, world=world,
                                                           history_counts=sim_history, device=self.device)
                    # P0 修复：子节点取跨世界并集——已存在的子节点保留统计量与先验，
                    # 仅补充当前世界新增的合法动作；禁止覆盖（会丢弃其他世界的访问统计）。
                    for a in acts_curr:
                        if a not in node.children:
                            node.children[a] = MCTSNode(prior=policy_map.get(a, 1e-4))
                    node.is_expanded = True
                    value = v

                    # 收集世界模式叶子样本（上限抽样）
                    if len(leaf_samples) < 8:
                        leaf_arr = encode_state_np(state, seat=state.turn, world=world, history_counts=sim_history)
                        leaf_samples.append((leaf_arr, state.turn))

            # 3.3 反向传播阶段 (Backpropagate)
            for p_node, act, turn in reversed(search_path):
                leaf_turn = state.turn if not state.is_terminal() else (1 - state.turn)
                step_val = value if turn == leaf_turn else -value
                p_node.visit_count += 1
                child_node = p_node.children[act]
                child_node.visit_count += 1
                child_node.value_sum += step_val

        # 4. 计算动作访问分布（Q 值视图）
        acts_list = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in acts_list], dtype=np.float32)
        q_vals = np.array([root.children[a].q_value for a in acts_list], dtype=np.float32)

        # 严格 avoid 过滤：当存在非 avoid 动作时，彻底封锁导致第 3 次重复的走法
        if avoid:
            avoid_indices = [i for i, a in enumerate(acts_list) if position_key(root_state.apply(a)) in avoid]
            if len(avoid_indices) < len(acts_list):
                for idx in avoid_indices:
                    visits[idx] = 0.0
                    q_vals[idx] = -999.0

        if temperature <= 1e-3:
            # 极低温度：Q 值优先、访问量决胜。
            best_idx = int(np.lexsort((visits, q_vals))[-1])
            probs = np.zeros_like(visits)
            probs[best_idx] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            probs = visits_temp / (total if total > 0 else 1.0)
            if probs.sum() == 0:
                probs[np.argmax(q_vals)] = 1.0

        pi_vec = np.zeros(3650, dtype=np.float32)
        pi_dict = {}
        for a, p in zip(acts_list, probs):
            pi_dict[a] = float(p)
            pi_vec[action_to_index(a)] = float(p)

        chosen_act = rng.choices(acts_list, weights=probs.tolist(), k=1)[0]
        return chosen_act, pi_vec, pi_dict, leaf_samples
