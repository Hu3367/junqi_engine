"""
Expert-2: Tempo Layer - Tempo 经济学

实现棋理体系第 4 章的核心概念：
- Tempo gain/loss tracking (得失跟踪)
- Phase-based strategy (分阶段策略)
- Information-tempo tradeoff (信息-tempo 权衡)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.append('.')

from junqi.state import GameState, Move
from junqi.expert.tactical_analyzer import ThreatAssessment

class TempoFlowType(Enum):
    """Tempo 流动类型"""
    GAIN = "gain"           # 获得先手
    LOSS = "loss"           # 损失先手
    FORCED = "forced"       # 被迫花费
    RESERVE = "reserve"     # 储备先手

@dataclass
class TempoEvent:
    """单个 Tempo 事件"""
    flow_type: TempoFlowType
    amount: float  # +/-
    description: str
    move: Optional[Move] = None

@dataclass 
class TempoAssessment:
    """Tempo 评估报告"""
    current_balance: float       # 当前 tempo 差值
    phase: str                   # 游戏阶段：opening/midgame/endgame
    advantage_level: str         # 'none', 'small', 'moderate', 'large'
    recommended_strategy: str
    urgency: str                 # 'low', 'medium', 'high', 'critical'

class TempoTracker:
    """
    Tempo 经济追踪器
    
    核心思想：tempo 是军棋中最稀缺的资源，需要精确计算和运用
    """
    
    def __init__(self):
        self.history: List[TempoEvent] = []
        self.current_balance = 0.0
        
    def record_move(self, move: Move, state_before: GameState,
                   state_after: GameState) -> TempoEvent:
        """
        记录一步移动的 tempo 影响
        
        分析四个维度：
        1. Tempo Gain: 是否获得了先手？
        2. Tempo Loss: 是否损失了先手？
        3. Forced Tempo: 是否是被动应对？
        4. Tempo Reserve: 是否为未来创造了条件？
        """
        events = []
        
        # 1. 检查是否有吃子（通常意味着 tempo gain）
        if self._is_capture(move, state_before, state_after):
            material_gain = self._calculate_material_gain(move, state_after)
            events.append(TempoEvent(
                flow_type=TempoFlowType.GAIN,
                amount=min(material_gain * 0.3, 1.5),
                description=f"Captured piece: +{material_gain:.1f} material",
                move=move
            ))
        
        # 2. 检查是否是逃跑（通常是 tempo loss）
        if self._is_retreat(move):
            retreat_distance = self._calculate_retreat_distance(move)
            events.append(TempoEvent(
                flow_type=TempoFlowType.LOSS,
                amount=-retreat_distance * 0.5,
                description=f"Retreated {retreat_distance} steps: -{retreat_distance*0.5:.1f}",
                move=move
            ))
        
        # 3. 检查是否是被迫防守
        if self._is_forced_defense(move, state_before, state_after):
            events.append(TempoEvent(
                flow_type=TempoFlowType.FORCED,
                amount=-1.0,
                description="Forced defensive move: -1.0",
                move=move
            ))
        
        # 4. 检查是否占领了关键位置（tempo reserve）
        if self._occupies_key_position(move):
            key_value = self._evaluate_key_position_value(move.to_pos)
            events.append(TempoEvent(
                flow_type=TempoFlowType.RESERVE,
                amount=key_value * 0.5,
                description=f"Occupied strategic position: +{key_value*0.5:.1f} reserve",
                move=move
            ))
        
        # 记录所有事件
        for event in events:
            self.history.append(event)
            self.current_balance += event.amount
        
        return events[0] if events else TempoEvent(
            TempoFlowType.GAIN, 0.0, "Neutral move", move
        )
    
    def get_current_assessment(self, game_state: GameState) -> TempoAssessment:
        """获取当前的 Tempo 评估"""
        phase = self._determine_phase(game_state.turn_count)
        
        # 评估当前平衡
        balance = self.current_balance
        
        if abs(balance) < 0.3:
            advantage = 'none'
            strategy = 'maintain_balance_and_see_kingship'
            urgency = 'low'
        elif abs(balance) < 0.8:
            advantage = 'small'
            strategy = 'cautious_expansion_if_positive_else_defensive'
            urgency = 'medium'
        elif abs(balance) < 1.5:
            advantage = 'moderate'
            strategy = 'aggressive_play_if_positive_or_counterattack'
            urgency = 'high'
        else:
            advantage = 'large'
            strategy = 'maximum_pressure_if_positive_or_desperation'
            urgency = 'critical'
        
        return TempoAssessment(
            current_balance=balance,
            phase=phase,
            advantage_level=advantage,
            recommended_strategy=strategy,
            urgency=urgency
        )
    
    def calculate_tempo_with_threats(self, threats: List[ThreatAssessment]) -> float:
        """
        根据威胁计算被迫花费的 Tempo
        
        Tier 威胁:
        - IMMEDIATE: 必须立即回应 (-1.0)
        - URGENT: 应该尽快回应 (-0.5)
        - STRATEGIC: 可以延后处理 (-0.2)
        """
        total_forced = 0.0
        
        for threat in threats:
            if threat.threat_type in ['flag_threat', 'immediate_capture']:
                total_forced += 1.0
            elif threat.threat_type in ['material_loss', 'mine_threat']:
                total_forced += 0.5
            else:
                total_forced += 0.2
        
        return -total_forced
    
    def should_spend_on_information(self, info_gain: float, 
                                   current_balance: float) -> bool:
        """
        判断是否值得用 tempo 换取信息
        
        原则:
        - tempo 优势时：阈值降低（愿意花钱买信息）
        - tempo 劣势时：阈值提高（只有高价值才侦察）
        """
        # 设置动态阈值
        if current_balance > 0.5:
            threshold = 0.3
        elif current_balance < -0.5:
            threshold = 0.8
        else:
            threshold = 0.5
        
        return info_gain > threshold
    
    def _is_capture(self, move: Move, state_before: GameState,
                   state_after: GameState) -> bool:
        """判断是否是吃子移动"""
        # TODO: 实现完整的捕获检测逻辑
        return False
    
    def _calculate_material_gain(self, move: Move, state: GameState) -> float:
        """计算材料收益"""
        # TODO: 实现
        return 0.0
    
    def _is_retreat(self, move: Move) -> bool:
        """判断是否是撤退"""
        # 简化版：向后移动
        return move.to_pos[0] > move.from_pos[0]
    
    def _calculate_retreat_distance(self, move: Move) -> int:
        """计算后退距离"""
        return move.to_pos[0] - move.from_pos[0]
    
    def _is_forced_defense(self, move: Move, state_before: GameState,
                          state_after: GameState) -> bool:
        """判断是否是 forced defense"""
        # TODO: 检查是否存在 immediate threat
        return False
    
    def _occupies_key_position(self, move: Move) -> bool:
        """判断是否占据了战略要地"""
        key_positions = [
            (5,0), (5,4), (6,0), (6,4),  # 中央铁路枢纽
            (3,2), (8,2),                  # 行营集群中心
        ]
        return move.to_pos in key_positions
    
    def _evaluate_key_position_value(self, position: Tuple[int, int]) -> float:
        """评估战略位置的价值"""
        base_values = {
            (5,0): 0.95, (5,4): 0.95,
            (6,0): 0.95, (6,4): 0.95,
            (3,2): 0.80, (8,2): 0.80,
        }
        return base_values.get(position, 0.5)
    
    def _determine_phase(self, turn_count: int) -> str:
        """确定当前游戏阶段"""
        if turn_count < 15:
            return 'opening'
        elif turn_count < 35:
            return 'midgame'
        else:
            return 'endgame'
    
    def reset(self):
        """重置 tempo 历史"""
        self.history = []
        self.current_balance = 0.0


def tempo_analysis_summary(tracker: TempoTracker, game_state: GameState) -> str:
    """生成 Tempo 分析的简洁摘要"""
    assessment = tracker.get_current_assessment(game_state)
    
    lines = [
        f"Tempo Balance: {assessment.current_balance:+.1f}",
        f"Phase: {assessment.phase}",
        f"Advantage Level: {assessment.advantage_level}",
        f"Strategy: {assessment.recommended_strategy}",
        f"Urgency: {assessment.urgency}"
    ]
    
    return "\n".join(lines)
