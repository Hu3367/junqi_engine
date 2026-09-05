"""置换表 (Transposition Table)：用于 Alpha-Beta / Expectiminimax 剪枝与 PV 着法复用。

采用基于哈希槽位与深度优先替换策略（Depth-preferred replacement）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .state import Action

# 置换表条目标记
FLAG_EXACT = 0       # 精确值 (PV 节点)
FLAG_LOWER_BOUND = 1  # 下界 (Cut 节点，score >= beta)
FLAG_UPPER_BOUND = 2  # 上界 (All 节点，score <= alpha)


@dataclass(slots=True)
class TTEntry:
    key: int
    depth: int
    score: float
    flag: int
    best_action: Optional[Action] = None


class TranspositionTable:
    """定长置换表。"""

    def __init__(self, size_power: int = 18):
        """
        参数:
            size_power: 2^size_power 个槽位。默认 18 = 262,144 个条目。
        """
        self.size = 1 << size_power
        self.mask = self.size - 1
        self.table: list[Optional[TTEntry]] = [None] * self.size
        self.hits = 0
        self.lookups = 0
        self.stores = 0

    def clear(self):
        self.table = [None] * self.size
        self.hits = 0
        self.lookups = 0
        self.stores = 0

    def store(self, key: int, depth: int, score: float, flag: int,
              best_action: Optional[Action] = None):
        """存储或更新置换表条目。"""
        self.stores += 1
        idx = key & self.mask
        existing = self.table[idx]

        if existing is None or existing.key != key or depth >= existing.depth:
            self.table[idx] = TTEntry(
                key=key,
                depth=depth,
                score=score,
                flag=flag,
                best_action=best_action if best_action is not None else (existing.best_action if existing and existing.key == key else None),
            )
        elif existing.best_action is None and best_action is not None:
            existing.best_action = best_action

    def lookup(self, key: int, depth: int, alpha: float, beta: float
               ) -> tuple[Optional[float], Optional[Action]]:
        """查询置换表。

        返回:
            (cutoff_score, best_action)
            如果命中且满足深度/边界条件，cutoff_score 为 float；否则为 None。
            best_action 在命中时返回（可用于走法排序，无论深度是否满足）。
        """
        self.lookups += 1
        idx = key & self.mask
        entry = self.table[idx]

        if entry is None or entry.key != key:
            return None, None

        best_act = entry.best_action

        # 深度满足时尝试剪枝
        if entry.depth >= depth:
            self.hits += 1
            if entry.flag == FLAG_EXACT:
                return entry.score, best_act
            elif entry.flag == FLAG_LOWER_BOUND and entry.score >= beta:
                return entry.score, best_act
            elif entry.flag == FLAG_UPPER_BOUND and entry.score <= alpha:
                return entry.score, best_act

        return None, best_act
