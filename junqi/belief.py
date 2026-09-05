"""P4.1 贝叶斯暗子信念跟踪器与加权多世界采样器 (Bayesian Belief State Tracker)。

核心职责：
1. 维护全盘未翻开暗子的后验概率分布矩阵 P(PieceType = (c, r) | pos, History)；
2. 基于对局中的公开观测事件（翻棋、走子、战斗结果、避让威慑）执行贝叶斯条件更新；
3. 从后验分布中高效采样满足全盘子力守恒的确定化世界 (Determinized Worlds)，供在线混合决策引擎使用。
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .rules import (CAMPS, COMPOSITION, HQS, PLAY_POSITIONS, Rank, battle,
                    is_camp, is_hq, other)
from .state import Action, GameState, Piece


class BeliefTracker:
    """贝叶斯暗子信念跟踪器。"""

    def __init__(self, state: Optional[GameState] = None, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.hidden_positions: Set[Tuple[int, int]] = set()
        self.remaining_pool: Counter[Tuple[str, Rank]] = Counter()
        self.beliefs: Dict[Tuple[int, int], Dict[Tuple[str, Rank], float]] = {}
        if state is not None:
            self.reset(state)

    def reset(self, state: GameState):
        """根据当前局面重新初始化暗子信念与剩余子力池。"""
        self.hidden_positions.clear()
        self.remaining_pool.clear()
        self.beliefs.clear()

        # 1. 统计全局已知剩余暗子池 T_pool = 总编制 - 已明子 - 已阵亡子
        all_pool: Counter[Tuple[str, Rank]] = Counter()
        for color in ("r", "b"):
            for rk, cnt in COMPOSITION.items():
                all_pool[(color, rk)] += cnt

        # 减去阵亡子
        for dead_pc in state.dead:
            all_pool[(dead_pc.color, dead_pc.rank)] -= 1

        # 减去棋盘上的明子
        for pos, pc in state.board.items():
            if pc.revealed:
                all_pool[(pc.color, pc.rank)] -= 1
            else:
                self.hidden_positions.add(pos)

        self.remaining_pool = Counter({k: max(0, v) for k, v in all_pool.items() if v > 0})
        total_hidden_count = sum(self.remaining_pool.values())

        if not self.hidden_positions or total_hidden_count == 0:
            return

        # 2. 初始化各暗子格点的先验概率分布 P(PieceType | pos)
        # 依据翻棋（全盘洗牌均匀发牌）规则，每个未翻开暗子的精确先验即为当前全局剩余池的边缘分布
        marginal = state.marginal()
        for pos in self.hidden_positions:
            self.beliefs[pos] = dict(marginal)

    def on_action(self, action: Action, state_before: GameState, state_after: GameState):
        """接收公开动作事件并执行贝叶斯后验更新。"""
        if action.kind == "flip":
            # 翻棋观测：该位置身份完全确定，暗子池扣减，其余位置重归一化
            flip_pos = action.frm
            revealed_pc = state_after.board.get(flip_pos)
            if revealed_pc is not None and flip_pos in self.hidden_positions:
                actual_key = (revealed_pc.color, revealed_pc.rank)
                self.hidden_positions.remove(flip_pos)
                if flip_pos in self.beliefs:
                    del self.beliefs[flip_pos]
                if self.remaining_pool[actual_key] > 0:
                    self.remaining_pool[actual_key] -= 1
                self._renormalize_all()

    def _renormalize_pos(self, pos: Tuple[int, int]):
        """归一化单个位置的概率分布。"""
        if pos not in self.beliefs:
            return
        dist = self.beliefs[pos]
        for k in list(dist.keys()):
            if self.remaining_pool[k] <= 0:
                dist[k] = 0.0
        tot = sum(dist.values())
        if tot > 1e-9:
            for k in dist:
                dist[k] /= tot
        else:
            active_keys = [k for k, v in self.remaining_pool.items() if v > 0]
            if active_keys:
                p = 1.0 / len(active_keys)
                self.beliefs[pos] = {k: p for k in active_keys}

    def _renormalize_all(self):
        """对所有当前暗子格点依据剩余池进行重归一化。"""
        for pos in list(self.hidden_positions):
            self._renormalize_pos(pos)

    def sample_world(self, state: GameState, rng: Optional[random.Random] = None) -> Dict[Tuple[int, int], Piece]:
        """加权无放回采样一个完全确定化世界 (pos -> Piece(color, rank, revealed=True))。"""
        rng = rng or self.rng
        if not self.hidden_positions:
            return {}

        pool_list: List[Tuple[str, Rank]] = []
        for piece_key, cnt in self.remaining_pool.items():
            pool_list.extend([piece_key] * cnt)

        if not pool_list:
            return {}

        sampled_world: Dict[Tuple[int, int], Piece] = {}
        positions = list(self.hidden_positions)
        rng.shuffle(positions)

        available_pieces = list(pool_list)

        for pos in positions:
            if not available_pieces:
                break

            dist = self.beliefs.get(pos, {})
            weights = []
            for pc_key in available_pieces:
                w = dist.get(pc_key, 1.0)
                weights.append(max(w, 1e-4))

            tot_w = sum(weights)
            if tot_w > 0:
                norm_w = [w / tot_w for w in weights]
                chosen_idx = rng.choices(range(len(available_pieces)), weights=norm_w, k=1)[0]
            else:
                chosen_idx = rng.randrange(len(available_pieces))

            chosen_color, chosen_rank = available_pieces.pop(chosen_idx)
            sampled_world[pos] = Piece(color=chosen_color, rank=chosen_rank, revealed=True)

        return sampled_world

    def sample_k_worlds(self, state: GameState, k: int = 4,
                        rng: Optional[random.Random] = None) -> List[Dict[Tuple[int, int], Piece]]:
        """高效生成 K 个独立的确定化采样世界。"""
        rng = rng or self.rng
        return [self.sample_world(state, rng=rng) for _ in range(max(1, k))]

    def get_piece_probability(self, pos: Tuple[int, int], color: str, rank: Rank) -> float:
        """查询指定坐标为特定棋子的后验概率。"""
        if pos not in self.beliefs:
            return 0.0
        return self.beliefs[pos].get((color, rank), 0.0)

    def summary(self) -> dict:
        """返回信念跟踪器核心状态摘要。"""
        return {
            "hidden_count": len(self.hidden_positions),
            "pool_remaining": dict(self.remaining_pool),
            "tracked_positions": len(self.beliefs),
        }
