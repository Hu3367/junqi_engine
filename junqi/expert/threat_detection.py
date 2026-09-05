"""
Expert-2: Threat Layer - 威胁体系

实现棋理体系第 3 章的核心概念：
- 7-tier threat classification (T1-T7)
- Threat response priority matrix
- Confidence scoring under incomplete information
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math
import sys
sys.path.append('.')

from junqi.state import GameState, Piece, Move
from junqi.expert.rule_validator import RuleValidator
from junqi.expert.hidden_piece_belief import BeliefSystem

class ThreatTier(Enum):
    """威胁等级枚举"""
    T1_IMMEDIATE_CAPTURE = "T1_immediate_capture"      # 立即吃子
    T2_CHAIN_ATTACK = "T2_chain_attack"                 # 连环攻击
    T3_FLAG_ZONE = "T3_flag_zone"                       # 旗区威胁
    T4_RAIL_INVASION = "T4_rail_invasion"               # 铁路入侵
    T5_ENCIRCLEMENT = "T5_encirclement"                 # 包围威胁
    T6_INFO_TRAP = "T6_info_trap"                       # 诱导暴露
    T7_DEFERRED = "T7_deferred"                         # 延迟威胁

@dataclass
class Threat:
    """威胁对象"""
    tier: ThreatTier
    severity: float       # 严重程度 0-1.0
    confidence: float     # 置信度 0-1.0
    source_piece: Optional[Piece]
    target_piece: Optional[Piece]
    time_to_crisis: int   # 几步内会发生危机
    description: str
    recommended_response: str

class ThreatDetectionEngine:
    """
    威胁检测引擎
    
    实现七类威胁的完整检测系统
    """
    
    # 旗区路径定义
    FLAG_ZONE_PATHS = [
        [(10, 0), (9, 0), (8, 0)],  # 左路
        [(10, 4), (9, 4), (8, 4)],  # 中路
        [(10, 8), (9, 8), (8, 8)],  # 右路
    ]
    
    # 铁路网格
    RAIL_GRID = {
        (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (9, 2), (9, 3), (9, 4), (9, 5), (9, 6),
        (2, 0), (3, 0), (4, 0),
        (2, 8), (3, 8), (4, 8),
    }
    
    CENTRAL_JUNCTIONS = [(5, 0), (5, 4), (6, 0), (6, 4)]
    
    def __init__(self, rule_validator: RuleValidator, belief_system: Optional[BeliefSystem] = None):
        self.validator = rule_validator
        self.belief_system = belief_system
    
    def detect_all_threats(self, state: GameState, color: int) -> List[Threat]:
        """
        检测所有类型的威胁
        
        Returns:
            List[Threat]: 按优先级排序的所有威胁
        """
        all_threats = []
        
        # T1: 立即吃子威胁
        all_threats.extend(self._detect_t1_immediate_captures(state, color))
        
        # T2: 连环攻击威胁
        all_threats.extend(self._detect_t2_chain_attacks(state, color))
        
        # T3: 旗区威胁
        all_threats.extend(self._detect_t3_flag_zone_threats(state, color))
        
        # T4: 铁路入侵威胁
        all_threats.extend(self._detect_t4_rail_invasion(state, color))
        
        # T5: 包围威胁
        all_threats.extend(self._detect_t5_encirclement(state, color))
        
        # T6: 诱导暴露威胁
        all_threats.extend(self._detect_t6_info_traps(state, color))
        
        # T7: 延迟威胁
        all_threats.extend(self._detect_t7_deferred_threats(state, color))
        
        # 按优先级排序
        sorted_threats = self._sort_by_priority(all_threats)
        
        return sorted_threats
    
    def _detect_t1_immediate_captures(self, state: GameState, 
                                       color: int) -> List[Threat]:
        """
        T1: 立即吃子威胁
        
        敌方明棋可在 1-2 步内吃掉我方明棋
        """
        threats = []
        
        my_pieces = state.get_pieces(color)
        enemy_pieces = state.get_pieces(1 - color)
        
        for my_piece in my_pieces:
            if not my_piece.is_revealed or my_piece.is_hidden:
                continue
            
            for enemy_piece in enemy_pieces:
                if not enemy_piece.is_revealed or enemy_piece.is_hidden:
                    continue
                
                # 检查是否可以被吃掉
                if self._can_enemy_capture(my_piece, enemy_piece, state):
                    severity = self._calculate_t1_severity(my_piece, enemy_piece)
                    
                    threats.append(Threat(
                        tier=ThreatTier.T1_IMMEDIATE_CAPTURE,
                        severity=severity,
                        confidence=1.0,  # 完全确定
                        source_piece=enemy_piece,
                        target_piece=my_piece,
                        time_to_crisis=1,
                        description=f"{enemy_piece.type} can capture {my_piece.type}",
                        recommended_response=self._get_t1_response(my_piece, enemy_piece)
                    ))
        
        return threats
    
    def _detect_t2_chain_attacks(self, state: GameState,
                                 color: int) -> List[Threat]:
        """
        T2: 连环攻击威胁
        
        单个敌方行动会依次触发多个威胁
        """
        threats = []
        
        # Simplified detection: look for patterns where one attack
        # enables another
        my_pieces = state.get_pieces(color)
        enemy_pieces = state.get_pieces(1 - color)
        
        # Check for potential chain attacks
        for enemy_piece in enemy_pieces:
            if not enemy_piece.is_revealed:
                continue
            
            # Find all targets this piece can attack
            immediate_targets = []
            for my_piece in my_pieces:
                if my_piece.is_revealed and not my_piece.is_hidden:
                    if self._can_enemy_capture(my_piece, enemy_piece, state):
                        immediate_targets.append(my_piece)
            
            # If multiple targets, this could be a chain/fork attack
            if len(immediate_targets) >= 2:
                total_severity = sum(
                    self.config.PIECE_RANKS.get(t.type, 0) for t in immediate_targets
                ) / 10.0
                
                threats.append(Threat(
                    tier=ThreatTier.T2_CHAIN_ATTACK,
                    severity=min(total_severity, 1.0),
                    confidence=0.85,  # High but not certain
                    source_piece=enemy_piece,
                    target_piece=None,  # Multiple targets
                    time_to_crisis=2,
                    description=f"{enemy_piece.type} threatens {len(immediate_targets)} pieces",
                    recommended_response="defend_multiple_targets_or_counterattack"
                ))
        
        return threats
    
    def _detect_t3_flag_zone_threats(self, state: GameState,
                                     color: int) -> List[Threat]:
        """
        T3: 旗区威胁
        
        敌方对我军旗保护区的任何入侵行为
        """
        threats = []
        
        # Find flag position (simplified)
        flag_pos = self._find_my_flag(state, color)
        if not flag_pos:
            return threats
        
        enemy_pieces = state.get_pieces(1 - color)
        
        # Check which approach paths are threatened
        for path in self.FLAG_ZONE_PATHS:
            closest_enemy = None
            min_distance = float('inf')
            
            for pos in path:
                piece = state.get_piece_at(pos)
                if piece and piece.color == 1 - color:
                    dist = self._count_steps_to(pos, flag_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_enemy = piece
            
            if closest_enemy and min_distance <= 3:
                # Calculate severity based on distance and piece value
                severity = max(0.7, 1.0 - min_distance * 0.15)
                
                threats.append(Threat(
                    tier=ThreatTier.T3_FLAG_ZONE,
                    severity=severity,
                    confidence=1.0,
                    source_piece=closest_enemy,
                    target_piece=None,
                    time_to_crisis=min_distance,
                    description=f"Enemy approaching flag via path at distance {min_distance}",
                    recommended_response="immediate_flag_defense_required"
                ))
        
        return threats
    
    def _detect_t4_rail_invasion(self, state: GameState,
                                 color: int) -> List[Threat]:
        """
        T4: 铁路入侵威胁
        
        敌方通过铁路网快速机动至我后方
        """
        threats = []
        
        enemy_pieces = state.get_pieces(1 - color)
        
        for piece in enemy_pieces:
            if not piece.is_revealed or piece.type not in self.config.RAIL_PIECES:
                continue
            
            # Check if piece is controlling rail grid
            if piece.position in self.RAIL_GRID:
                # Assess potential targets in rear area
                rear_areas = [(row, col) for row in range(7, 10) 
                             for col in range(9)]
                
                for target in rear_areas:
                    if self._can_reach_via_rail(piece.position, target, state):
                        severity = self._assess_rail_threat_severity(target, state, color)
                        
                        threats.append(Threat(
                            tier=ThreatTier.T4_RAIL_INVASION,
                            severity=severity,
                            confidence=0.75,  # Possible but not certain
                            source_piece=piece,
                            target_piece=None,
                            time_to_crisis=3,  # Typically 3 moves via rail
                            description=f"Enemy rail piece threatening rear area",
                            recommended_response="block_rail_access_or_counterattack"
                        ))
                        break  # One threat per rail piece
        
        return threats
    
    def _detect_t5_encirclement(self, state: GameState,
                                color: int) -> List[Threat]:
        """
        T5: 包围威胁
        
        敌方逐步压缩我活动空间
        """
        threats = []
        
        my_pieces = state.get_pieces(color)
        
        for piece in my_pieces:
            if not piece.is_revealed:
                continue
            
            # Calculate current mobility
            current_mobility = len(self._get_reachable_squares(state, piece.position, 1))
            
            # Look for enemy encirclement pattern
            nearby_enemies = self._count_nearby_enemies(piece.position, state, radius=3)
            
            if nearby_enemies >= 3 and current_mobility <= 3:
                severity = min(1.0, nearby_enemies * 0.2)
                
                threats.append(Threat(
                    tier=ThreatTier.T5_ENCIRCLEMENT,
                    severity=severity,
                    confidence=0.7,
                    source_piece=None,
                    target_piece=piece,
                    time_to_crisis=3,
                    description=f"{piece.type} being surrounded by {nearby_enemies} enemies",
                    recommended_response="break_encirclement_or_escape"
                ))
        
        return threats
    
    def _detect_t6_info_traps(self, state: GameState,
                              color: int) -> List[Threat]:
        """
        T6: 诱导暴露威胁
        
        对手故意放置诱饵棋子引诱我翻开暗子
        """
        threats = []
        
        if not self.belief_system:
            return threats
        
        enemy_pieces = state.get_pieces(1 - color)
        
        for piece in enemy_pieces:
            if not piece.is_revealed:
                continue
            
            # Check for suspicious patterns
            suspicious_factors = []
            
            # Factor 1: High-value piece in isolated position
            if piece.value > 7 and self._is_isolated(piece.position, state):
                suspicious_factors.append(0.3)
            
            # Factor 2: Exposed to multiple attacks but still alive
            attack_count = self._count_exposure_level(piece.position, state, color)
            if attack_count >= 2:
                suspicious_factors.append(0.3)
            
            # Factor 3: Positioned on expected capture path
            if self._on_expected_capture_path(piece.position, state, color):
                suspicious_factors.append(0.2)
            
            # If enough suspicious factors, it's likely a bait
            if sum(suspicious_factors) >= 0.5:
                threats.append(Threat(
                    tier=ThreatTier.T6_INFO_TRAP,
                    severity=sum(suspicious_factors),
                    confidence=0.6,  # Bayesian probability
                    source_piece=piece,
                    target_piece=None,
                    time_to_crisis=1,  # Immediate trap if I fall for it
                    description=f"Suspicious enemy piece that may be a trap",
                    recommended_response="avoid_flipping_neighboring_hidden_pieces"
                ))
        
        return threats
    
    def _detect_t7_deferred_threats(self, state: GameState,
                                    color: int) -> List[Threat]:
        """
        T7: 延迟威胁
        
        当前无害但未来会被触发的隐患
        """
        threats = []
        
        my_pieces = state.get_pieces(color)
        
        for piece in my_pieces:
            if not piece.is_revealed:
                continue
            
            # Simulate future positions
            future_trajectories = self._predict_piece_trajectory(piece.position, state, steps=5)
            
            for trajectory in future_trajectories:
                # Check if trajectory leads into danger zones
                danger_score = self._assess_trajectory_danger(trajectory, state, color)
                
                if danger_score > 0.5:
                    threats.append(Threat(
                        tier=ThreatTier.T7_DEFERRED,
                        severity=danger_score,
                        confidence=0.5,  # Lower confidence as it's predictive
                        source_piece=None,
                        target_piece=piece,
                        time_to_crisis=4,  # When crisis will hit
                        description=f"Piece trajectory leading to dangerous position",
                        recommended_response="reposition_before_crises_arrives"
                    ))
        
        return threats
    
    def get_priority_matrix(self, threats: List[Threat]) -> Dict[str, List[Threat]]:
        """
        生成威胁响应优先级矩阵
        
        Tier assignments:
        IMMEDIATE CRISIS (< 2 steps):
        - T3 Flag zone threats
        - T1 immediate captures of high-value pieces
        - T2 chain attacks causing ≥2 losses
        
        URGENT PROTECTION (3-5 steps):
        - T1 captures of medium-value pieces
        - T4 rail invasion near阵地
        - T5 encirclement with Mobility ≤ 2
        
        STRATEGIC ADJUSTMENT (> 5 steps):
        - T6 info traps
        - Remote T4 rail invasions
        - Information disadvantage accumulation
        """
        matrix = {
            'IMMEDIATE_CRISIS': [],
            'URGENT_PROTECTION': [],
            'STRATEGIC_ADJUSTMENT': []
        }
        
        for threat in threats:
            if threat.tier == ThreatTier.T3_FLAG_ZONE:
                matrix['IMMEDIATE_CRISIS'].append(threat)
            elif threat.tier == ThreatTier.T1_IMMEDIATE_CAPTURE:
                if threat.target_piece and threat.target_piece.value > 8:
                    matrix['IMMEDIATE_CRISIS'].append(threat)
                else:
                    matrix['URGENT_PROTECTION'].append(threat)
            elif threat.tier == ThreatTier.T2_CHAIN_ATTACK:
                if threat.severity > 0.7:
                    matrix['IMMEDIATE_CRISIS'].append(threat)
                else:
                    matrix['URGENT_PROTECTION'].append(threat)
            elif threat.tier == ThreatTier.T4_RAIL_INVASION:
                if threat.time_to_crisis <= 3:
                    matrix['URGENT_PROTECTION'].append(threat)
                else:
                    matrix['STRATEGIC_ADJUSTMENT'].append(threat)
            elif threat.tier == ThreatTier.T5_ENCIRCLEMENT:
                if threat.time_to_crisis <= 3:
                    matrix['URGENT_PROTECTION'].append(threat)
                else:
                    matrix['STRATEGIC_ADJUSTMENT'].append(threat)
            elif threat.tier in [ThreatTier.T6_INFO_TRAP, ThreatTier.T7_DEFERRED]:
                matrix['STRATEGIC_ADJUSTMENT'].append(threat)
        
        return matrix
    
    def _sort_by_priority(self, threats: List[Threat]) -> List[Threat]:
        """按优先级排序威胁"""
        priority_order = {
            ThreatTier.T3_FLAG_ZONE: 0,
            ThreatTier.T1_IMMEDIATE_CAPTURE: 1,
            ThreatTier.T2_CHAIN_ATTACK: 2,
            ThreatTier.T4_RAIL_INVASION: 3,
            ThreatTier.T5_ENCIRCLEMENT: 4,
            ThreatTier.T6_INFO_TRAP: 5,
            ThreatTier.T7_DEFERRED: 6,
        }
        
        return sorted(threats, key=lambda t: (priority_order[t.tier], -t.severity))
    
    # Helper methods
    
    def _can_enemy_capture(self, my_piece: Piece, enemy_piece: Piece, 
                          state: GameState) -> bool:
        """判断敌方是否可以吃掉我的棋子"""
        if enemy_piece.type == '工兵':
            return my_piece.type == '地雷'
        elif enemy_piece.type == '炸弹':
            return True
        else:
            att_rank = self.config.PIECE_RANKS.get(enemy_piece.type, 0)
            def_rank = self.config.PIECE_RANKS.get(my_piece.type, 0)
            return att_rank >= def_rank
    
    def _calculate_t1_severity(self, my_piece: Piece, enemy_piece: Piece) -> float:
        """计算 T1 威胁的严重程度"""
        base_severity = my_piece.value / 10.0
        
        # Bonus for flag proximity
        flag_dist = self._distance_to_flag(my_piece.position)
        if flag_dist <= 2:
            base_severity *= 1.5
        
        return min(base_severity, 1.0)
    
    def _get_t1_response(self, my_piece: Piece, enemy_piece: Piece) -> str:
        """获取 T1 威胁的应对建议"""
        if my_piece.value > 8:
            return "protect_high_value_piece_at_all_cost"
        elif my_piece.type == '地雷':
            return "consider_bomb_protection_or_reposition"
        else:
            return "evaluate_retreat_vs_counterattack"
    
    def _find_my_flag(self, state: GameState, color: int) -> Optional[Tuple[int, int]]:
        """找到己方军旗位置（简化版）"""
        # TODO: 实现完整的军旗查找
        return (9, 4)  # Default center position
    
    def _count_steps_to(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        """计算两点之间的步数"""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def _can_reach_via_rail(self, start: Tuple[int, int], end: Tuple[int, int],
                           state: GameState) -> bool:
        """判断是否可以通过铁路到达"""
        # Simplified rail reachability check
        if start[0] == end[0]:  # Same row
            for col in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                if (start[0], col) not in self.RAIL_GRID:
                    return False
            return True
        elif start[1] == end[1]:  # Same column
            for row in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                if (row, start[1]) not in self.RAIL_GRID:
                    return False
            return True
        return False
    
    def _assess_rail_threat_severity(self, target: Tuple[int, int],
                                     state: GameState, color: int) -> float:
        """评估铁路威胁的严重程度"""
        piece = state.get_piece_at(target)
        
        if piece:
            return piece.value / 10.0
        
        # Check if it's a critical area
        if target[0] >= 8:  # Near flag area
            return 0.9
        
        return 0.5
    
    def _count_nearby_enemies(self, pos: Tuple[int, int], state: GameState,
                             radius: int) -> int:
        """统计附近的敌人数量"""
        count = 0
        for piece in state.get_pieces(1 - state.current_turn):
            if not piece.is_revealed:
                continue
            
            dist = abs(pos[0] - piece.position[0]) + abs(pos[1] - piece.position[1])
            if dist <= radius:
                count += 1
        
        return count
    
    def _is_isolated(self, pos: Tuple[int, int], state: GameState) -> bool:
        """判断位置是否孤立"""
        allies = sum(1 for p in state.pieces 
                    if p.color == state.current_turn and 
                    self._distance_to(p.position, pos) <= 2)
        return allies == 0
    
    def _count_exposure_level(self, pos: Tuple[int, int], state: GameState,
                             color: int) -> int:
        """计算暴露程度（被多少敌人可以攻击到）"""
        exposure = 0
        for piece in state.get_pieces(1 - color):
            if piece.is_revealed:
                if self._can_reach_from(piece.position, pos, state):
                    exposure += 1
        return exposure
    
    def _on_expected_capture_path(self, pos: Tuple[int, int],
                                  state: GameState, color: int) -> bool:
        """判断是否在预期的吃子路径上"""
        # Simplified heuristic
        return pos[0] < 5 and pos[1] in [2, 4, 6]  # Front line assumption
    
    def _predict_piece_trajectory(self, start: Tuple[int, int],
                                 state: GameState, steps: int) -> List[List[Tuple[int, int]]]:
        """预测棋子可能的轨迹（简化版）"""
        trajectories = []
        
        # Generate simple forward paths
        for dc in range(-2, 3):
            trajectory = []
            current = start
            
            for step in range(steps):
                trajectory.append(current)
                current = (current[0], current[1] + dc)
            
            if len(trajectory) == steps:
                trajectories.append(trajectory)
        
        return trajectories
    
    def _assess_trajectory_danger(self, trajectory: List[Tuple[int, int]],
                                  state: GameState, color: int) -> float:
        """评估轨迹的危险程度"""
        danger_sum = 0.0
        
        for pos in trajectory:
            # Check if position is exposed to enemies
            for piece in state.get_pieces(1 - color):
                if piece.is_revealed:
                    if self._can_reach_from(piece.position, pos, state):
                        danger_sum += 0.2
        
        return min(danger_sum, 1.0)
    
    def _distance_to(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """计算距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _can_reach_from(self, start: Tuple[int, int], end: Tuple[int, int],
                       state: GameState) -> bool:
        """判断能否从起点到达终点"""
        dist = self._distance_to(start, end)
        return dist <= 3  # Simplified reachability
    
    def _distance_to_flag(self, pos: Tuple[int, int]) -> int:
        """计算到军旗的距离"""
        flag_pos = (9, 4)
        return self._distance_to(pos, flag_pos)


# Import config
import sys
sys.path.append('.')
from junqi.config import RuleConfig