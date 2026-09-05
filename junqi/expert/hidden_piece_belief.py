"""
Expert-2: Information Layer - 不完全信息推理

实现棋理体系第 1 章的核心概念：
- HiddenPieceBelief (贝叶斯信念建模)
- Entropy calculation (信息熵计算)  
- Reveal value quantification (信息价值量化)
- Elimination reasoning (排除法推理)
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import math
import sys
sys.path.append('.')

from junqi.state import GameState
from junqi.config import RuleConfig

@dataclass
class HiddenPieceBelief:
    """单个暗子的信念分布"""
    
    position: Tuple[int, int]
    belief: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.belief:
            # 初始化时为均匀分布
            self.belief = {}
    
    def get_probability(self, piece_type: str) -> float:
        """获取某个棋子类型的概率"""
        return self.belief.get(piece_type, 0.0)
    
    def calculate_entropy(self) -> float:
        """
        计算信息熵
        H(P) = -Σ p(x) * log₂(p(x))
        
        Returns:
            float: 熵值，越大表示越不确定
        """
        entropy = 0.0
        
        for prob in self.belief.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def get_most_likely_pieces(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """获取最可能的 Top N 个棋子类型"""
        sorted_pieces = sorted(
            self.belief.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_pieces[:top_n]
    
    def get_confidence_score(self) -> float:
        """
        计算当前信念的确定性评分
        
        Returns:
            float: 0 = 完全未知，1 = 几乎确定
        """
        max_prob = max(self.belief.values()) if self.belief else 0.0
        return max_prob
    
    def update_with_observation(self, observation_type: str, 
                               observed_piece: Optional[str] = None):
        """
        根据观察事件更新信念
        
        Args:
            observation_type: 'piece_revealed' | 'piece_killed' | 'behavior_pattern'
            observed_piece: 如果已知具体棋子类型
        """
        # TODO: 实现具体的更新逻辑
        pass


class BeliefSystem:
    """
    全局信念系统
    
    维护所有暗子的信念分布
    """
    
    def __init__(self, config: RuleConfig, board_state: GameState):
        self.config = config
        self.board = board_state
        self.hidden_beliefs: Dict[Tuple[int, int], HiddenPieceBelief] = {}
        
        # 剩余牌池跟踪
        self.remaining_pool: Dict[str, int] = {}
        
        self._initialize()
    
    def _initialize(self):
        """初始化信念系统和剩余牌池"""
        # 初始状态：所有位置都是等概率的
        total_hidden = 0
        pool_counts = {rank: count for rank, count in 
                      self.config.INITIAL_POOL.items()}
        
        # 找出所有暗子位置
        hidden_positions = self._find_hidden_pieces()
        total_hidden = len(hidden_positions) * 25  # 假设每个玩家 25 个暗子
        
        # 为每个暗子位置创建信念
        for pos in hidden_positions:
            self.hidden_beliefs[pos] = HiddenPieceBelief(
                position=pos,
                belief={rank: count/total_hidden 
                       for rank, count in pool_counts.items()}
            )
        
        # 初始化剩余牌池
        self.remaining_pool = pool_counts.copy()
    
    def _find_hidden_pieces(self) -> List[Tuple[int, int]]:
        """找到棋盘上所有的暗子位置"""
        hidden = []
        
        for row in range(self.board.height):
            for col in range(self.board.width):
                piece = self.board.get_piece_at((row, col))
                if piece and piece.is_hidden:
                    hidden.append((row, col))
        
        return hidden
    
    def get_belief(self, position: Tuple[int, int]) -> Optional[HiddenPieceBelief]:
        """获取某个位置的信念分布"""
        return self.hidden_beliefs.get(position)
    
    def update_after_flip(self, position: Tuple[int, int], revealed_type: str):
        """
        翻出某个暗子后更新信念
        
        关键洞察：
        翻出一个棋子会同时影响所有其他位置的概率分布！
        """
        # 从剩余牌池中移除已翻出的棋子
        if revealed_type in self.remaining_pool:
            self.remaining_pool[revealed_type] -= 1
            if self.remaining_pool[revealed_type] <= 0:
                del self.remaining_pool[revealed_type]
        
        # 更新翻开的这个位置
        belief = self.hidden_beliefs[position]
        if belief:
            belief.belief[revealed_type] = 1.0
            for other_type in belief.belief:
                if other_type != revealed_type:
                    belief.belief[other_type] = 0.0
        
        # 重新归一化其他所有位置的概率
        self._renormalize_all_beliefs()
    
    def update_after_kill(self, killed_type: str):
        """
        某个棋子被吃掉时更新信念
        
        被吃掉的棋子同样要从牌池中移除
        """
        if killed_type in self.remaining_pool:
            self.remaining_pool[killed_type] -= 1
            if self.remaining_pool[killed_type] <= 0:
                del self.remaining_pool[killed_type]
        
        self._renormalize_all_beliefs()
    
    def _renormalize_all_beliefs(self):
        """
        重新归一化所有信念分布
        
        这是因为牌池减少了某些棋子类型
        """
        remaining_total = sum(self.remaining_pool.values())
        
        if remaining_total == 0:
            return
        
        for belief in self.hidden_beliefs.values():
            # 获取剩余牌池中的所有类型
            available_types = set(belief.belief.keys()) & set(self.remaining_pool.keys())
            
            new_sum = 0.0
            for piece_type in available_types:
                prob = self.remaining_pool[piece_type] / remaining_total
                belief.belief[piece_type] = prob
                new_sum += prob
            
            # 如果没有可用类型，保持最小概率
            if new_sum == 0:
                num_types = len(belief.belief)
                for piece_type in belief.belief:
                    belief.belief[piece_type] = 1.0 / num_types
    
    def detect_impossible_pieces(self, color: int) -> Dict[Tuple[int, int], List[str]]:
        """
        识别某些暗子"绝对不可能是哪些棋子"
        
        Example:
        - 敌方司令已经出现 → 所有敌方暗子都不是司令
        - 敌方地雷已全部暴露 → 所有暗子都不可能是地雷
        """
        impossible = {}
        
        # 找出已知的敌方棋子
        known_enemy_pieces = {
            piece.type for piece in self.board.get_pieces(1 - color)
            if not piece.is_hidden
        }
        
        # 对于每个暗子位置，标记不可能的棋子
        for pos, belief in self.hidden_beliefs.items():
            impossible_pieces = []
            
            # 检查哪些棋子已经完全暴露
            for piece_type in known_enemy_pieces:
                if piece_type in self.remaining_pool:
                    continue
                
                # 这个棋子已经完全可见，所以暗子不可能是它
                impossible_pieces.append(piece_type)
                belief.belief[piece_type] = 0.0
            
            if impossible_pieces:
                impossible[pos] = impossible_pieces
        
        return impossible
    
    def calculate_information_gain(self, target_pos: Tuple[int, int]) -> float:
        """
        计算翻出某个暗子的预期信息增益
        
        Information Gain = 翻前熵 - 翻后期望熵
        """
        belief = self.hidden_beliefs.get(target_pos)
        if not belief:
            return 0.0
        
        # 翻前的熵
        before_entropy = belief.calculate_entropy()
        
        # 翻后的期望熵（基于当前信念分布模拟）
        after_entropy = 0.0
        total_prob = 0.0
        
        for piece_type, prob in belief.belief.items():
            if prob > 0:
                # 假设翻出这个类型后的新熵 = 0（完全确定）
                after_entropy += prob * 0
                total_prob += prob
        
        expected_after_entropy = after_entropy
        
        return before_entropy - expected_after_entropy
    
    def get_highest_uncertainty_positions(self, top_n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """
        获取不确定性最高的暗子位置
        
        Returns:
            List of [(position, entropy), ...]
        """
        positions_with_entropy = [
            (pos, belief.calculate_entropy())
            for pos, belief in self.hidden_beliefs.items()
        ]
        
        # 按熵值排序
        positions_with_entropy.sort(key=lambda x: x[1], reverse=True)
        
        return positions_with_entropy[:top_n]
    
    def assess_reveal_value(self, position: Tuple[int, int], 
                           strategic_context: Dict) -> float:
        """
        评估翻出某个暗子的综合价值
        
        Value = Information_Gain × Strategic_Multiplier
        
        Strategic factors:
        - 靠近军旗的位置价值更高
        - 位于前线 vs 后方
        - 是否是关键路径
        """
        info_gain = self.calculate_information_gain(position)
        
        # 计算战略乘数
        strategic_mult = self._calculate_strategic_multiplier(position, strategic_context)
        
        return info_gain * strategic_mult
    
    def _calculate_strategic_multiplier(self, position: Tuple[int, int],
                                       context: Dict) -> float:
        """计算战略重要性乘数"""
        multiplier = 1.0
        
        row, col = position
        
        # 靠近军旗：大幅增加
        my_flag_row = context.get('my_flag_row', 9)
        distance_to_flag = abs(row - my_flag_row)
        
        if distance_to_flag <= 2:
            multiplier *= 2.0
        elif distance_to_flag <= 4:
            multiplier *= 1.5
        
        # 铁路枢纽
        is_rail_junction = (row, col) in [(5,0), (5,4), (6,0), (6,4)]
        if is_rail_junction:
            multiplier *= 1.3
        
        # 前线 vs 后方
        is_frontline = 3 <= row <= 6
        if is_frontline:
            multiplier *= 1.2
        
        return min(multiplier, 3.0)  # 上限 3 倍


class InformationReasoner:
    """
    信息推理引擎
    
    整合贝叶斯信念和排除法推理
    """
    
    def __init__(self, belief_system: BeliefSystem):
        self.belief_system = belief_system
    
    def analyze_information_advantage(self, color: int) -> Dict:
        """
        分析双方在信息层面的优劣
        
        Returns:
            {'advantage': float, 'clarity_score': float, 'recent_gains': int}
        """
        # 计算己方对敌方的了解程度
        my_clarity = self._calculate_clarity(1 - color)
        
        # 计算敌方对我的了解程度
        enemy_clarity = self._calculate_clarity(color)
        
        advantage = my_clarity - enemy_clarity
        
        return {
            'advantage': advantage,
            'clarity_score': my_clarity,
            'information_gap': abs(enemy_clarity - my_clarity)
        }
    
    def _calculate_clarity(self, target_color: int) -> float:
        """
        计算对某方暗子的了解程度
        
        Returns:
            float: 0 = 完全未知，1 = 完全清楚
        """
        total_entropy = 0.0
        count = 0
        
        for pos, belief in self.belief_system.hidden_beliefs.items():
            # 只统计目标颜色的暗子（简化版本，实际需要颜色标记）
            total_entropy += belief.calculate_entropy()
            count += 1
        
        if count == 0:
            return 0.0
        
        avg_entropy = total_entropy / count
        
        # 转换为清晰度（熵越低越清晰）
        max_entropy = math.log2(len(self.belief_system.config.PIECE_RANKS))
        clarity = 1.0 - (avg_entropy / max_entropy)
        
        return max(0.0, min(1.0, clarity))
    
    def generate_scouting_priorities(self, game_state: GameState,
                                    color: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        生成侦察优先级列表
        
        Returns:
            List of [(position, priority_score), ...]
        """
        priorities = []
        
        # 获取不确定性最高的位置
        uncertain_positions = self.belief_system.get_highest_uncertainty_positions(10)
        
        for pos, entropy in uncertain_positions:
            # 计算综合价值
            strategic_info = {
                'my_flag_row': self._get_my_flag_position(game_state, color)[0],
            }
            
            reveal_value = self.belief_system.assess_reveal_value(pos, strategic_info)
            
            # 综合评分
            priority = entropy * 0.6 + reveal_value * 0.4
            
            priorities.append((pos, priority))
        
        # 排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return priorities
    
    def _get_my_flag_position(self, state: GameState, color: int) -> Tuple[int, int]:
        """获取己方军旗位置（简化版）"""
        # TODO: 实现完整的军旗查找逻辑
        return (9, 4)  # 默认中心位置


# 实用工具函数

def calculate_bayesian_likelihood(observation: str, hypothesis: str,
                                 prior: float, evidence_strength: float) -> float:
    """
    贝叶斯更新辅助函数
    
    P(H|E) = P(E|H) × P(H) / P(E)
    """
    from scipy.stats import binom
    
    likelihood = evidence_strength
    posterior = likelihood * prior / (likelihood * prior + (1 - likelihood) * (1 - prior))
    
    return posterior
