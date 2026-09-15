"""军棋翻棋推演引擎 - v2.0 重构版

两人陆战棋 · 翻棋（翻明棋）模式：
50 枚棋子洗乱背面朝上铺满全盘，轮流"翻开一枚暗子"或"移动一枚己方明子"，
吃掉对方军旗或使对方无子可动即获胜。
"""

__version__ = "2.0.0"

import sys as _sys

# 主要类和功能
#
# 诊断增强（2026-09-15）：包级导入会连带 `import torch`（经 ai / hybrid_engine）。
# 虚拟环境已移出工程目录后，"用了没装 torch 的解释器"成为最常见的一类启动失败，
# 而原生报错只是一行 `ModuleNotFoundError: No module named 'torch'`，看不出该换哪个
# 解释器。这里把第三方依赖缺失转成可操作的提示，并保留原始异常链。
_THIRD_PARTY_DEPS = {"torch", "numpy", "networkx", "PIL", "yaml", "matplotlib"}

try:
    from .ai import Agent, ExpertAgent, HybridAgent, NNAgent, evaluate, evaluate_expert
    from .config import EvalWeights, RuleConfig, SearchConfig
    from .rules import Rank
    from .search import ExpertSearchEngine
    from .state import Action, GameState, Piece, deal
    from .tt import TranspositionTable
    from .zobrist import compute_zobrist
    from .belief import BeliefTracker
    from .hybrid_engine import HybridDecisionEngine
except ModuleNotFoundError as _exc:
    if _exc.name not in _THIRD_PARTY_DEPS:
        raise
    raise ModuleNotFoundError(
        f"junqi 缺少第三方依赖 `{_exc.name}`。\n"
        f"当前解释器：{_sys.executable}\n"
        f"这通常意味着没有使用项目虚拟环境（虚拟环境自 2026-09-15 起位于工程同级目录）。\n"
        f"正确用法：\n"
        f'  ..{chr(92)}venv_junqi_engine{chr(92)}Scripts{chr(92)}python.exe -m junqi <子命令>\n'
        f"  或先激活：..{chr(92)}venv_junqi_engine{chr(92)}Scripts{chr(92)}Activate.ps1\n"
        f"  或使用工程根目录的封装脚本：run.bat <子命令>\n"
        f"详见 README「环境准备（必须）」。"
    ) from _exc

__all__ = [
    "Agent",
    "ExpertAgent",
    "HybridAgent",
    "HybridDecisionEngine",
    "BeliefTracker",
    "NNAgent",
    "evaluate",
    "evaluate_expert",
    "EvalWeights",
    "RuleConfig",
    "SearchConfig",
    "Rank",
    "ExpertSearchEngine",
    "Action",
    "GameState",
    "Piece",
    "deal",
    "TranspositionTable",
    "compute_zobrist",
]
