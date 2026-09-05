"""
Expert Engine - 军棋翻棋专家引擎 V1.0

这是一个能够稳定执行棋理、搜索局面、产生高质量决策与标签的 Teacher / Oracle。

架构分层：
- Expert-0: 规则验证层 (Rule Validator)
- Expert-1: 战术分析层 (Tactical Analyst)
- Expert-2: 棋理评估层 (Principle Evaluator)
  ├─ Hidden Piece Belief (信息推理)
  ├─ Mobility Calculator (空间控制)
  ├─ Tempo Tracker (Tempo 经济学)
  ├─ Threat Detection (威胁体系)
  └─ Conditional Value (条件子力价值)
- Expert-3: 搜索优化层 (Search Optimizer) ⭐ NEW
  ├─ Monte Carlo Tree Search (MCTS)
  ├─ Partial Information MCTS (PIMC/ISMCTS)
  └─ Candidate Filtering

作者：Junqi Engine Team
日期：2026-09-05
版本：V1.0
"""

from .rule_validator import RuleValidator
from .tactical_analyzer import TacticalAnalyzer
from .hidden_piece_belief import BeliefSystem, InformationReasoner, HiddenPieceBelief
from .mobility_calculator import MobilityCalculator, SpaceAdvantageCalculator, StrategicNodeAnalyzer
from .tempo_tracker import TempoTracker, TempoAssessment
from .threat_detection import ThreatDetectionEngine, Threat, ThreatTier
from .conditional_value import ConditionalPieceValueEvaluator, PieceValueReport
from .search_optimizer import MonteCarloTreeSearch, PartialInformationMCTS, CandidateFilter
from .expert_engine import ExpertEngine

__all__ = [
    # Core Experts
    'RuleValidator',
    'TacticalAnalyzer', 
    'ThreatDetectionEngine',
    'ConditionalPieceValueEvaluator',
    
    # Principle Layers (Expert-2)
    'BeliefSystem',
    'InformationReasoner',
    'HiddenPieceBelief',
    'MobilityCalculator',
    'SpaceAdvantageCalculator',
    'StrategicNodeAnalyzer',
    'TempoTracker',
    'TempoAssessment',
    'PieceValueReport',
    
    # Search Layer (Expert-3) ⭐ NEW
    'MonteCarloTreeSearch',
    'PartialInformationMCTS',
    'CandidateFilter',
    
    # Main Interface
    'ExpertEngine'
]
