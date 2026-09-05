"""军棋翻棋推演引擎 - v2.0 重构版

两人陆战棋 · 翻棋（翻明棋）模式：
50 枚棋子洗乱背面朝上铺满全盘，轮流"翻开一枚暗子"或"移动一枚己方明子"，
吃掉对方军旗或使对方无子可动即获胜。
"""

__version__ = "2.0.0"

# 主要类和功能
from .ai import Agent, ExpertAgent, HybridAgent, NNAgent, evaluate, evaluate_expert
from .config import EvalWeights, RuleConfig, SearchConfig
from .rules import Rank
from .search import ExpertSearchEngine
from .state import Action, GameState, Piece, deal
from .tt import TranspositionTable
from .zobrist import compute_zobrist
from .belief import BeliefTracker
from .hybrid_engine import HybridDecisionEngine

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
