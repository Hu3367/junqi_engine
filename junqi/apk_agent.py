"""官方 APK (libjunqi.so) 逆向原生引擎 1:1 复刻假想敌代理 (ApkNativeAgent)。

设计依据与逆向对齐规范（阶段 P4）：
1. 估值核心 (0x124094)：
   - 采用官方 2 的幂次等比子力梯度：司令 2560、军长 1280、师长 640、旅长 320、
     团长 160、营长 80、连长 40、排长 30、工兵 80（高权值）、地雷 70（护旗额外+80）。
2. 动态炸弹定价 (0x600ca)：
   - 炸弹价值根据敌方全盘（明子+暗子期望池）存活最大军衔按 floor(max / 3) 动态折算。
3. 搜索与剪枝体系 (0x5e574 / 0x5e378 / 0x5e0a8)：
   - 采用 PVS (Principal Variation Search / NegaScout) 零窗口探测；
   - 64 位 Zobrist 哈希与置换表 (Transposition Table) 剪枝；
   - MVV-LVA、行营特权与安全翻棋启发式走法排序。
4. 严格遵守非透视信息屏障 (0x2f9f6)：
   - 决策前将所有暗子视为代号 13（暗子掩码），仅利用公共先验与残局剩余池，严禁窥探真实暗子身份。
5. 难度档位对齐：
   - beginner (初级): 深度 2 ply，超时截断 100ms
   - intermediate (中级): 深度 3 ply，超时截断 300ms
   - advanced (高级): 深度 4 ply，超时截断 1000ms
"""
from __future__ import annotations

import math
import random
from typing import Optional

from .config import EvalWeights, RuleConfig, SearchConfig
from .search import ExpertSearchEngine
from .state import Action, GameState


# 官方预设难度参数表
APK_LEVEL_CONFIGS = {
    "beginner": {"depth": 2, "time_limit_ms": 100},
    "intermediate": {"depth": 3, "time_limit_ms": 300},
    "advanced": {"depth": 4, "time_limit_ms": 1000},
}


class ApkNativeAgent:
    """官方 APK 原生传统博弈引擎假想敌 (P4 Benchmark Opponent)"""

    def __init__(self, level: str = "advanced",
                 weights: Optional[EvalWeights] = None,
                 tt_size_power: int = 18,
                 seed: Optional[int] = None):
        if level not in APK_LEVEL_CONFIGS:
            raise ValueError(f"未知难度档位 '{level}'，可选: {list(APK_LEVEL_CONFIGS.keys())}")
        self.level = level
        self.cfg_dict = APK_LEVEL_CONFIGS[level]
        self.depth = self.cfg_dict["depth"]
        self.time_limit_ms = self.cfg_dict["time_limit_ms"]

        # 使用 1:1 对齐 APK 的估值体系与动态炸弹
        self.w = weights or EvalWeights.apk_weights()
        self.engine = ExpertSearchEngine(weights=self.w, tt_size_power=tt_size_power, seed=seed)
        self.rng = random.Random(seed)

    @property
    def nodes(self) -> int:
        return self.engine.stats.nodes

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: Optional[set] = None) -> list[tuple[Action, float]]:
        """执行博弈树搜索并返回最优走法列表 [(action, score)]。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 0.0)]

        best_act, best_score, stats = self.engine.search(
            state,
            max_depth=self.depth,
            time_limit_ms=self.time_limit_ms,
            avoid=avoid,
        )

        if topn <= 1 or best_act is None:
            return [(best_act or acts[0], best_score)]

        if stats.root_scores:
            return stats.root_scores[:topn]

        scored = []
        for a in acts:
            score = self.engine._score_action(a, state, 0, best_act)
            if a == best_act:
                score += 100_000.0
            scored.append((a, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:topn]

    def select_action(self, state: GameState,
                      avoid: Optional[set] = None) -> Action:
        """返回单步最优决策动作 (符合统一 Agent 规范)。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else Action("pass")
