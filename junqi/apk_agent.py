"""官方 APK (libjunqi.so) 逆向原生引擎 1:1 复刻假想敌代理 (ApkNativeAgent)。

设计依据与逆向对齐规范（阶段 P1 / P4）：
1. 真实二人翻棋原生体系 (0x57000 - 0x5b000)：
   - 0x59f90: 二人翻棋 60 格静态估值核 (含 0x124094 2的幂次等比价值表与行营/地雷护旗判定)；
   - 0x5a678: 二人翻棋静态搜索 (Quiescence Search)，纯吃子交火线展开与 Delta 剪枝；
   - 0x59e80: 明子走法生成器，在对抗树深层绝不递归展开暗子翻棋；
   - 0x5ac3a: 迭代加深 (IDS) 动态时间预算早停 (耗时超 25% 提前退出保持完整 PV)。
2. 估值核心 (0x124094 与 0x59f90)：
   - 采用官方 2 的幂次等比子力梯度：司令 2560、军长 1280、师长 640、旅长 320、
     团长 160、营长 80、连长 40、排长 30、工兵 80（高权值）、地雷 70（护旗额外+80）。
3. 严格遵守非透视公共信息屏障：
   - 决策仅利用已知明子公共信息与暗子剩余池，严禁窥探真实暗子身份。
4. 难度档位对齐 (0x3404c / 0x34078 / 0x5ac3a)：
   - beginner (初级): 深度 2 ply，超时截断 100ms，QSearch 深度 8，抖动 Jitter=±30
   - intermediate (中级): 深度 3 ply，超时截断 300ms，QSearch 深度 12，抖动 Jitter=±10
   - advanced (高级): 深度 4 ply，超时截断 1000ms，QSearch 深度 16，抖动 Jitter=±0.5
"""
from __future__ import annotations

import random
from typing import Optional, List, Tuple

from .apk_engine import ApkSearchEngine, APK_LEVEL_SPECS, eval_apk_pure
from .config import EvalWeights, RuleConfig, SearchConfig
from .state import Action, GameState


# 官方预设难度参数表 (对齐 0x3404c / 0x34078 / 0x5ac3a)
APK_LEVEL_CONFIGS = {
    "beginner": {"depth": 2, "time_limit_ms": 100, "qsearch_depth": 8},
    "intermediate": {"depth": 3, "time_limit_ms": 300, "qsearch_depth": 12},
    "advanced": {"depth": 4, "time_limit_ms": 1000, "qsearch_depth": 16},
}


class ApkNativeAgent:
    """官方 APK 原生传统博弈引擎假想敌 (P4 Benchmark Opponent)"""

    def __init__(self, level: str = "advanced",
                 weights: Optional[EvalWeights] = None,
                 tt_size_power: int = 18,
                 seed: Optional[int] = None,
                 depth: Optional[int] = None,
                 time_limit_ms: Optional[int] = None,
                 qsearch_depth: Optional[int] = None):
        if level not in APK_LEVEL_CONFIGS:
            raise ValueError(f"未知难度档位 '{level}'，可选: {list(APK_LEVEL_CONFIGS.keys())}")
        self.level = level
        self.cfg_dict = APK_LEVEL_CONFIGS[level]
        self.depth = depth if depth is not None else self.cfg_dict["depth"]
        self.time_limit_ms = time_limit_ms if time_limit_ms is not None else self.cfg_dict["time_limit_ms"]
        self.qsearch_depth = qsearch_depth if qsearch_depth is not None else self.cfg_dict.get("qsearch_depth", 16)

        self.w = weights or EvalWeights.apk_weights()
        # 使用 1:1 对齐 APK 的原生极简搜索引擎
        self.engine = ApkSearchEngine(tt_size_power=tt_size_power, seed=seed)
        self.rng = random.Random(seed)

    @property
    def nodes(self) -> int:
        return self.engine.stats.nodes

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: Optional[set] = None) -> List[Tuple[Action, float]]:
        """执行博弈树搜索并返回最优走法列表 [(action, score)]。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 0.0)]

        best_act, best_score, stats = self.engine.search(
            state,
            level=self.level,
            depth=self.depth,
            time_limit_ms=self.time_limit_ms,
            qsearch_depth=self.qsearch_depth,
            avoid=avoid,
        )

        if topn <= 1 or best_act is None:
            return [(best_act or acts[0], best_score)]

        if stats.root_scores:
            return stats.root_scores[:topn]

        return [(best_act or acts[0], best_score)]

    def select_action(self, state: GameState,
                      avoid: Optional[set] = None) -> Optional[Action]:
        """返回单步最优决策动作 (符合统一 Agent 规范)。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else None
