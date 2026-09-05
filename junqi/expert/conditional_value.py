"""
Expert-2: Conditional Piece Value - 条件子力价值

实现棋理体系第 5 章的核心概念：
- Five-dimension dynamic valuation
  1. Position quality multiplier
  2. Offensive pressure contribution  
  3. Defensive necessity analysis
  4. Future mobility bonus
  5. Special ability activation
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('.')

from junqi.state import GameState, Piece
from junqi.config import RuleConfig
from junqi.expert.mobility_calculator import MobilityCalculator, SpaceAdvantageCalculator

@dataclass
class PieceValueReport:
    """棋子价值报告"""
    base_value: float          # 基础价值
    position_multiplier: float # 位置乘数
    offensive_contribution: float  # 进攻贡献
    defensive_necessity: float   # 防守必要性
    mobility_bonus: float      # 机动加成
    special_ability_activation: float  # 特殊能力激活
    
    @property
    def dynamic_value(self) -> float:
        """计算动态价值"""
        return (self.base_value * self.position_multiplier * 
               (1 + self.offensive_contribution) *
               (1 + self.defensive_necessity) *
               self.mobility_bonus *
               self.special_ability_activation)


class ConditionalPieceValueEvaluator:
    """
    条件子力价值评估器
    
    核心思想：棋子价值是动态的，取决于当前局面
    """
    
    def __init__(self, config: RuleConfig):
        self.config = config
        self.mobility_calc = MobilityCalculator(config)
    
    def evaluate_piece(self, piece: Piece, board: GameState,
                      game_context: Dict) -> PieceValueReport:
        """
        评估单个棋子的动态价值
        
        Args:
            piece: 要评估的棋子
            board: 当前棋盘状态
            game_context: 游戏上下文信息
            
        Returns:
            PieceValueReport: 包含五维度分析的完整报告
        """
        
        # 获取基础价值
        base_value = self.config.PIECE_RANKS.get(piece.type, 0)
        
        # 维度 1: 位置质量评分
        position_multiplier = self._evaluate_position_quality(
            piece, board, game_context
        )
        
        # 维度 2: 威胁贡献度
        offensive_contribution = self._evaluate_offensive_pressure(
            piece, board
        )
        
        # 维度 3: 防守必要性
        defensive_necessity = self._evaluate_defensive_necessity(
            piece, board, game_context
        )
        
        # 维度 4: 机动潜力
        mobility_bonus = self._evaluate_future_mobility(
            piece, board, game_context
        )
        
        # 维度 5: 特殊能力激活度
        special_activation = self._evaluate_special_capabilities(
            piece, board
        )
        
        return PieceValueReport(
            base_value=base_value,
            position_multiplier=position_multiplier,
            offensive_contribution=offensive_contribution,
            defensive_necessity=defensive_necessity,
            mobility_bonus=mobility_bonus,
            special_ability_activation=special_activation
        )
    
    def _evaluate_position_quality(self, piece: Piece, board: GameState,
                                  context: Dict) -> float:
        """
        维度 1: 位置质量评分
        
        Factors:
        - Center score (0.3)
        - Safety score (0.4)
        - Strategic importance (0.3)
        """
        pos = piece.position
        
        # Center score: closer to center is better
        center_row = board.height // 2
        center_col = board.width // 2
        center_dist = abs(pos[0] - center_row) + abs(pos[1] - center_col)
        max_dist = board.height + board.width
        center_score = 1.0 - (center_dist / max_dist)
        
        # Safety score: distance from enemies
        safety_score = self._calculate_safety_score(pos, board, piece.color)
        
        # Strategic score: occupying key positions
        strategic_score = self._calculate_strategic_importance(pos)
        
        # Weighted average, then normalize to [0.5, 1.5]
        weighted_avg = (center_score * 0.3 + 
                       safety_score * 0.4 + 
                       strategic_score * 0.3)
        
        # Normalize to range [0.5, 1.5] with mean 1.0
        multiplier = 0.5 + weighted_avg
        
        return multiplier
    
    def _calculate_safety_score(self, pos: Tuple[int, int],
                               board: GameState, color: int) -> float:
        """计算安全性评分"""
        nearby_enemies = 0
        danger_levels = []
        
        for enemy in board.get_pieces(1 - color):
            if enemy.is_revealed:
                dist = abs(pos[0] - enemy.position[0]) + abs(pos[1] - enemy.position[1])
                
                if dist <= 2:
                    nearby_enemies += 1
                    # Higher value enemies are more dangerous
                    danger_levels.append(enemy.value / 10.0)
        
        # Base safety
        safety = 1.0 - (nearby_enemies * 0.15)
        
        # Penalize based on enemy strength
        if danger_levels:
            avg_danger = sum(danger_levels) / len(danger_levels)
            safety *= (1.0 - avg_danger * 0.3)
        
        # Bonus if near own pieces (protection)
        nearby_friends = sum(1 for f in board.get_pieces(color)
                           if not f.is_hidden and 
                           abs(pos[0] - f.position[0]) + abs(pos[1] - f.position[1]) <= 2)
        
        safety += min(nearby_friends * 0.1, 0.3)
        
        return max(0.3, min(1.0, safety))
    
    def _calculate_strategic_importance(self, pos: Tuple[int, int]) -> float:
        """计算战略重要性"""
        strategic_positions = {
            # Central rail junctions (highest value)
            (5, 0): 0.95, (5, 4): 0.95, (6, 0): 0.95, (6, 4): 0.95,
            
            # Flag approach paths
            (1, 0): 0.90, (1, 4): 0.90, (10, 0): 0.90, (10, 4): 0.90,
            
            # Frontline crossings
            (5, 2): 0.85, (6, 2): 0.85,
            
            # Camp cluster centers
            (3, 2): 0.80, (8, 2): 0.80,
        }
        
        return strategic_positions.get(pos, 0.3)
    
    def _evaluate_offensive_pressure(self, piece: Piece,
                                     board: GameState) -> float:
        """
        维度 2: 威胁贡献度
        
        Calculates how much attacking pressure this piece exerts
        """
        if piece.is_hidden:
            return 0.0
        
        # Find all targets this piece can threaten
        potential_threats = []
        
        # Check Mobility-2 reachable squares
        m2_reachable = self.mobility_calc.calculate_mobility_2(
            board, piece.position, piece.type
        )
        
        for target_pos in m2_reachable:
            target = board.get_piece_at(target_pos)
            if target and target.color != piece.color:
                # Calculate threat strength
                threat_strength = self._calculate_attack_strength(
                    piece, target
                )
                potential_threats.append(threat_strength)
        
        if not potential_threats:
            return 0.0
        
        # Average threat strength + maximum threat bonus
        avg_threat = sum(potential_threats) / len(potential_threats)
        max_threat = max(potential_threats)
        
        return avg_threat * 0.6 + max_threat * 0.4
    
    def _calculate_attack_strength(self, attacker: Piece,
                                  defender: Piece) -> float:
        """计算攻击强度"""
        att_rank = self.config.PIECE_RANKS.get(attacker.type, 0)
        def_rank = self.config.PIECE_RANKS.get(defender.type, 0)
        
        if attacker.type == '工兵' and defender.type == '地雷':
            return 1.0  # Guaranteed kill
        
        if attacker.type == '炸弹':
            return 0.8  # Can trade with anything
        
        # Standard comparison
        if att_rank > def_rank:
            return (att_rank - def_rank) / 10.0
        elif att_rank == def_rank:
            return 0.5  # Draw possibility
        else:
            return max(0.0, (att_rank - def_rank) / 10.0) * 0.5
    
    def _evaluate_defensive_necessity(self, piece: Piece, board: GameState,
                                     context: Dict) -> float:
        """
        维度 3: 防守必要性
        
        Determines if this piece plays a critical defensive role
        """
        necessity = 0.0
        
        # Check proximity to flag
        my_flag_row = context.get('my_flag_row', 9)
        flag_distance = abs(piece.position[0] - my_flag_row)
        
        if piece.can_attack() and flag_distance <= 3:
            # Pieces near flag that can attack are valuable defensively
            necessity += 0.5 + (3 - flag_distance) * 0.1
        
        # Check if protecting mines
        if piece.type == '工兵':
            mine_proximity = self._distance_to_nearest_mine(piece.position, board)
            if mine_proximity <= 2:
                necessity += 0.8
        
        # Check if是唯一 protector of something valuable
        if self._is_critical_protector(piece, board):
            necessity += 0.6
        
        return min(necessity, 1.0)
    
    def _evaluate_future_mobility(self, piece: Piece, board: GameState,
                                 context: Dict) -> float:
        """
        维度 4: 机动潜力
        
        Assesses how much mobility this piece will gain/lose
        """
        if piece.is_hidden:
            return 0.5  # Unknown pieces have uncertain mobility
        
        current_m1 = len(self.mobility_calc.calculate_mobility_1(
            board, piece.position, piece.type
        ))
        
        current_m3 = len(self.mobility_calc.calculate_mobility_3(
            board, piece.position, piece.type
        ))
        
        # Growth ratio: higher means more potential
        growth_ratio = current_m3 / max(current_m1, 1)
        
        # Base mobility factor
        mobility_factor = min(2.0, growth_ratio * 0.5 + 0.5)
        
        # Rail access bonus
        rail_bonus = 1.5 if piece.position in self.mobility_calc.RAIL_GRID else 1.0
        
        # Check if currently blocked (negative modifier)
        block_penalty = 0.0
        if self._is_blocked(piece.position, board, piece.type):
            block_penalty = -0.3
        
        return max(0.5, (mobility_factor * rail_bonus + block_penalty))
    
    def _evaluate_special_capabilities(self, piece: Piece,
                                       board: GameState) -> float:
        """
        维度 5: 特殊能力激活度
        
        Evaluates how activated/special capabilities are for this piece type
        """
        if piece.type == '工兵':
            # Worker bombs have special abilities: flying, digging
            can_fly = self._can_fly_from_here(piece.position, board)
            can_dig = self._can_dig_mines(piece.position, board)
            
            fly_mult = 1.5 if can_fly else 1.0
            dig_mult = 2.0 if can_dig else 1.0
            
            return fly_mult * dig_mult
        
        elif piece.type == '炸弹':
            # Bombs are more valuable when there are high-value targets nearby
            nearby_high_value = self._count_nearby_high_value_targets(
                piece.position, board
            )
            
            return min(2.0, 1.0 + nearby_high_value * 0.5)
        
        elif piece.type in ['司令', '军长']:
            # Large pieces value depends on free killing opportunities
            free_kills = self._count_free_killing_opportunities(
                piece.position, board
            )
            
            return min(1.5, 1.0 + free_kills * 0.1)
        
        elif piece.type == '地雷':
            # Mines value depends on protection level
            protection_level = self._calculate_mine_protection(piece.position, board)
            
            # Better protected mines are more valuable
            return 1.0 + protection_level * 0.5
        
        return 1.0  # Regular pieces
    
    def get_conditional_value_summary(self, state: GameState,
                                     color: int) -> Dict[str, float]:
        """
        获取所有己方棋子的条件价值汇总
        
        Useful for overall position evaluation
        """
        values = {}
        
        for piece in state.get_pieces(color):
            if piece.is_revealed:
                report = self.evaluate_piece(piece, state, {})
                key = f"{piece.type}_{piece.position}"
                values[key] = report.dynamic_value
        
        # Aggregate statistics
        if not values:
            return {'total': 0.0, 'average': 0.0, 'max': 0.0}
        
        value_list = list(values.values())
        
        return {
            'total': sum(value_list),
            'average': sum(value_list) / len(value_list),
            'max': max(value_list),
            'min': min(value_list),
            'distribution': values
        }
    
    def compare_with_enemy(self, state: GameState, my_color: int) -> Dict:
        """
        比较双方棋子的动态价值
        
        Returns comprehensive comparison including advantages/disadvantages
        """
        my_values = self.get_conditional_value_summary(state, my_color)
        enemy_values = self.get_conditional_value_summary(
            state, 1 - my_color
        )
        
        material_advantage = (my_values['total'] - enemy_values['total']) / \
                           max(enemy_values['total'], 1)
        
        # Who has stronger individual pieces?
        my_max = my_values['max']
        enemy_max = enemy_values['max']
        peak_advantage = (my_max - enemy_max) / max(enemy_max, 1)
        
        return {
            'material_advantage_ratio': material_advantage,
            'peak_advantage_ratio': peak_advantage,
            'my_total': my_values['total'],
            'enemy_total': enemy_values['total'],
            'my_average': my_values['average'],
            'enemy_average': enemy_values['average'],
            'my_best_piece': max(my_values['distribution'].items(), 
                                key=lambda x: x[1])[0] if my_values['distribution'] else None,
            'enemy_best_piece': max(enemy_values['distribution'].items(), 
                                   key=lambda x: x[1])[0] if enemy_values['distribution'] else None
        }
    
    # Helper methods
    
    def _is_blocked(self, pos: Tuple[int, int], board: GameState,
                   piece_type: str) -> bool:
        """判断棋子是否被阻挡"""
        # Check if surrounded by own pieces or cannot move effectively
        occupied_count = sum(1 for p in board.pieces
                            if not p.is_hidden and 
                            abs(p.position[0] - pos[0]) + abs(p.position[1] - pos[1]) <= 2)
        
        return occupied_count >= 4
    
    def _distance_to_nearest_mine(self, pos: Tuple[int, int],
                                  board: GameState) -> int:
        """找到最近的地雷距离"""
        min_dist = float('inf')
        
        for piece in board.pieces:
            if piece.type == '地雷' and piece.is_revealed:
                dist = abs(pos[0] - piece.position[0]) + abs(pos[1] - piece.position[1])
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 999
    
    def _is_critical_protector(self, piece: Piece, board: GameState) -> bool:
        """判断是否是关键的守护者"""
        # Simplified: checking if piece protects flag area
        if piece.position[0] >= 7:  # Near flag zone
            return True
        return False
    
    def _can_fly_from_here(self, pos: Tuple[int, int],
                          board: GameState) -> bool:
        """判断是否可以飞行"""
        # Simplified flight rule
        return pos[0] <= 3 or pos[0] >= 6  # Within certain rows
    
    def _can_dig_mines(self, pos: Tuple[int, int],
                      board: GameState) -> bool:
        """判断是否可以挖雷"""
        for piece in board.pieces:
            if piece.type == '地雷' and piece.is_revealed:
                dist = abs(pos[0] - piece.position[0]) + abs(pos[1] - piece.position[1])
                if dist <= 2:
                    return True
        return False
    
    def _count_nearby_high_value_targets(self, pos: Tuple[int, int],
                                        board: GameState) -> int:
        """统计附近的高价值目标"""
        count = 0
        high_value_threshold = 7
        
        for piece in board.get_pieces(1 - board.current_turn):
            if piece.is_revealed and piece.value >= high_value_threshold:
                dist = abs(pos[0] - piece.position[0]) + abs(pos[1] - piece.position[1])
                if dist <= 3:
                    count += 1
        
        return count
    
    def _count_free_killing_opportunities(self, pos: Tuple[int, int],
                                         board: GameState) -> int:
        """计算自由击杀机会数量"""
        count = 0
        
        for piece in board.get_pieces(1 - board.current_turn):
            if piece.is_revealed:
                if self.config.PIECE_RANKS.get(piece.type, 0) < \
                   self.config.PIECE_RANKS.get('司令', 10):
                    dist = abs(pos[0] - piece.position[0]) + abs(pos[1] - piece.position[1])
                    if dist <= 2:
                        count += 1
        
        return count
    
    def _calculate_mine_protection(self, pos: Tuple[int, int],
                                  board: GameState) -> float:
        """计算地雷的保护程度"""
        # Count protecting pieces around the mine
        protectors = 0
        
        for piece in board.pieces:
            if not piece.is_hidden and piece.color == board.current_turn:
                dist = abs(pos[0] - piece.position[0]) + abs(pos[1] - piece.position[1])
                if dist <= 2:
                    protectors += 1
        
        # Each protector adds ~0.2, capped at 1.0
        return min(protectors * 0.2, 1.0)


# Configuration import
from junqi.config import RuleConfig