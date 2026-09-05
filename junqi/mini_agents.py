"""
Mini-Junqi Expert Agent - 专家代理系统

提供多种层级的 AI Agent:
- RandomAgent: 随机走子
- HeuristicAgent: 基于规则的启发式 Agent
- MiniExpertAgent: 完整的 Mini-Junqi 专家模型
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import random
import numpy as np

from .mini_junqi import MiniGameState, MiniMove, MoveType, MiniPiece


class BaseAgent(ABC):
    """基础 Agent 接口"""
    
    @abstractmethod
    def select_move(self, state: MiniGameState, legal_moves: List[MiniMove]) -> MiniMove:
        """选择移动"""
        pass
    
    def name(self) -> str:
        return "BaseAgent"


class RandomAgent(BaseAgent):
    """随机选择 Agent"""
    
    def select_move(self, state: MiniGameState, legal_moves: List[MiniMove]) -> MiniMove:
        if not legal_moves:
            return None
        return random.choice(legal_moves)
    
    def name(self) -> str:
        return "RandomAgent"


class HeuristicAgent(BaseAgent):
    """启发式规则 Agent - 使用棋理规则做决策"""
    
    def __init__(self, color: int, strategy: str = "balanced"):
        self.color = color
        self.strategy = strategy  # "aggressive", "defensive", "balanced"
        
    def select_move(self, state: MiniGameState, legal_moves: List[MiniMove]) -> MiniMove:
        if not legal_moves:
            return None
        
        # 为每个移动评分
        scored_moves = []
        
        for move in legal_moves:
            score = self._score_move(move, state)
            scored_moves.append((move, score))
        
        # 按分数排序，选择最佳移动
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # 有一定概率选择非最优移动 (模拟人类的不完美)
        if random.random() < 0.1 and len(scored_moves) > 2:
            return scored_moves[random.randint(0, min(3, len(scored_moves)-1))][0]
        
        return scored_moves[0][0]
    
    def _score_move(self, move: MiniMove, state: MiniGameState) -> float:
        """单个移动的评分"""
        score = 0.0
        
        # 吃子奖励
        if move.move_type == MoveType.CAPTURE:
            target = state.get_piece_at(move.to_pos)
            if target:
                score += target.rank_value * 0.5
        
        # 翻棋信息价值
        if move.move_type == MoveType.FLIP:
            info_value = self._calculate_reveal_value(move.from_pos, state)
            score += info_value * 0.8
        
        # 行营控制奖励
        if state.is_bunker(move.to_pos):
            score += 0.3
        
        # 靠近军旗奖励
        enemy_flag = state.flag_position(1 - self.color)
        if enemy_flag:
            dist_to_flag = self._manhattan_distance(move.to_pos, enemy_flag)
            score += (5 - dist_to_flag) * 0.1
        
        # 策略调整
        if self.strategy == "aggressive":
            score *= 1.2
        elif self.strategy == "defensive":
            # 防守型：避免高风险移动
            if move.move_type == MoveType.CAPTURE:
                score *= 0.7
        
        return score
    
    def _calculate_reveal_value(self, pos: Tuple[int, int], 
                               state: MiniGameState) -> float:
        """计算翻棋的信息价值"""
        # 简化版：位置越关键价值越高
        center_dist = abs(pos[0] - 2) + abs(pos[1] - 2)
        
        # 边缘位置通常有更高价值
        is_edge = pos[0] in [0, 4] or pos[1] in [0, 4]
        edge_bonus = 0.2 if is_edge else 0.0
        
        return max(0.3, 1.0 - center_dist * 0.2) + edge_bonus
    
    def _manhattan_distance(self, pos1: Tuple[int, int], 
                           pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def name(self) -> str:
        return f"HeuristicAgent({self.strategy})"


class MiniExpertAgent(BaseAgent):
    """
    Mini-Junqi 专家 Agent
    
    集成完整棋理系统的专家级代理
    基于大规模对局数据统计得出最优策略
    """
    
    def __init__(self, color: int, config_path: Optional[str] = None):
        self.color = color
        self.config = self._load_config(config_path)
        
        # 加载统计数据库 (将在训练阶段生成)
        self.stats_db = {}
        
    def _load_config(self, path: Optional[str]) -> Dict:
        """加载配置"""
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def select_move(self, state: MiniGameState, legal_moves: List[MiniMove]) -> MiniMove:
        if not legal_moves:
            return None
        
        # 综合评估每个移动
        evaluations = []
        
        for move in legal_moves:
            eval_score = self._evaluate_move(move, state)
            evaluations.append((move, eval_score))
        
        # 按评估分数排序
        evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # 采样选择 (允许一定探索性)
        top_k = min(3, len(evaluations))
        top_moves = evaluations[:top_k]
        
        # 使用 softmax 采样
        probs = np.exp([e[1] for e in top_moves])
        probs /= probs.sum()
        
        selected_idx = random.choices(range(len(top_moves)), weights=probs)[0]
        return top_moves[selected_idx][0]
    
    def _evaluate_move(self, move: MiniMove, state: MiniGameState) -> float:
        """评估单个移动的综合价值"""
        score = 0.0
        
        # 1. 战术价值 (40%)
        tactical_score = self._evaluate_tactics(move, state)
        score += tactical_score * 0.4
        
        # 2. 战略价值 (30%)
        strategic_score = self._evaluate_strategy(move, state)
        score += strategic_score * 0.3
        
        # 3. 风险控制 (20%)
        risk_score = self._evaluate_risk(move, state)
        score += risk_score * 0.2
        
        # 4. 长期潜力 (10%)
        future_score = self._evaluate_future_potential(move, state)
        score += future_score * 0.1
        
        return score
    
    def _evaluate_tactics(self, move: MiniMove, state: MiniGameState) -> float:
        """评估战术价值"""
        score = 0.0
        
        # 吃子优先
        if move.move_type == MoveType.CAPTURE:
            target = state.get_piece_at(move.to_pos)
            if target:
                value_ratio = target.rank_value / 5.0  # 归一化
                score += value_ratio * 0.8
        
        # 翻棋信息收集
        elif move.move_type == MoveType.FLIP:
            flip_value = self._calculate_flip_quality(move.from_pos, state)
            score += flip_value * 0.6
        
        # 占领关键点
        elif move.to_pos in [(2, 2), (1, 1), (1, 3)]:
            score += 0.4
        
        return score
    
    def _evaluate_strategy(self, move: MiniMove, state: MiniGameState) -> float:
        """评估战略价值"""
        score = 0.0
        
        # 行营控制
        if state.is_bunker(move.to_pos):
            bunker_importance = self._bunker_importance(move.to_pos)
            score += bunker_importance * 0.5
        
        # 向敌方军旗逼近
        enemy_flag = state.flag_position(1 - self.color)
        if enemy_flag:
            old_dist = self._manhattan_distance(move.from_pos, enemy_flag)
            new_dist = self._manhattan_distance(move.to_pos, enemy_flag)
            
            if new_dist < old_dist:
                score += (old_dist - new_dist) * 0.3
        
        # 保护己方军旗
        my_flag = state.flag_position(self.color)
        if my_flag:
            flag_safety_before = self._flag_safety(my_flag, state)
            
            # 如果移动到更接近军旗的位置，可能提高安全性
            if self._closer_to_flag(move.to_pos, my_flag):
                score += 0.2
        
        return score
    
    def _evaluate_risk(self, move: MiniMove, state: MiniGameState) -> float:
        """评估风险 (希望最大化安全度)"""
        safety = 1.0
        
        # 如果被吃子后的损失大，降低分数
        if move.move_type == MoveType.CAPTURE:
            piece = state.get_piece_at(move.from_pos)
            if piece and piece.can_attack():
                # 估算被反吃的概率
                counter_attack_risk = self._counter_attack_probability(
                    move.to_pos, state
                )
                safety -= counter_attack_risk * 0.5
        
        return safety
    
    def _evaluate_future_potential(self, move: MiniMove, 
                                  state: MiniGameState) -> float:
        """评估长期潜力"""
        potential = 0.0
        
        # Mobility-3 评估 (未来可达性)
        mobility_improvement = self._mobility_change(move.from_pos, move.to_pos, state)
        potential += mobility_improvement * 0.3
        
        # Tempo gain/loss
        tempo_impact = self._tempo_impact(move, state)
        potential += tempo_impact * 0.2
        
        return potential
    
    def _calculate_flip_quality(self, pos: Tuple[int, int],
                              state: MiniGameState) -> float:
        """计算翻棋质量"""
        # 基于统计数据判断该位置的翻棋价值
        position_key = str(pos)
        
        if position_key in self.stats_db:
            stats = self.stats_db[position_key]
            avg_win_rate_after_flip = stats.get('avg_winrate_after_flip', 0.5)
            return avg_win_rate_after_flip * 0.8
        
        # 默认值：边缘位置略高
        is_corner = pos[0] in [0, 4] and pos[1] in [0, 4]
        is_center = abs(pos[0] - 2) <= 1 and abs(pos[1] - 2) <= 1
        
        if is_corner:
            return 0.7
        elif is_center:
            return 0.5
        else:
            return 0.6
    
    def _bunker_importance(self, pos: Tuple[int, int]) -> float:
        """行营重要性评分"""
        bunkers_priority = {
            (2, 2): 0.35,   # 中心
            (1, 1): 0.25,   # 前线左
            (1, 3): 0.25,   # 前线右
            (3, 1): 0.15,   # 后方左
            (3, 3): 0.15,   # 后方右
        }
        return bunkers_priority.get(pos, 0.0)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], 
                           pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _closer_to_flag(self, pos: Tuple[int, int], 
                       flag: Tuple[int, int]) -> bool:
        # 简化检查
        return self._manhattan_distance(pos, flag) <= 3
    
    def _flag_safety(self, flag_pos: Tuple[int, int],
                    state: MiniGameState) -> float:
        """评估军旗安全性"""
        safety = 0.0
        
        # 检查周围格子的保护情况
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                neighbor = (flag_pos[0] + dr, flag_pos[1] + dc)
                if neighbor != flag_pos:
                    piece = state.get_piece_at(neighbor)
                    if piece and piece.color == state.current_player:
                        safety += 0.15
        
        return min(1.0, safety)
    
    def _counter_attack_probability(self, pos: Tuple[int, int],
                                   state: MiniGameState) -> float:
        """估计被反攻击的概率"""
        # 简化：检查附近是否有敌方明子
        risky_count = 0
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                neighbor = (pos[0] + dr, pos[1] + dc)
                if neighbor != pos:
                    piece = state.get_piece_at(neighbor)
                    if piece and piece.color != state.current_player and piece.revealed:
                        risky_count += 1
        
        return min(1.0, risky_count * 0.2)
    
    def _mobility_change(self, from_pos: Tuple[int, int],
                        to_pos: Tuple[int, int],
                        state: MiniGameState) -> float:
        """评估移动带来的机动性变化"""
        # 简化版
        from_mobility = self._count_reachable(from_pos, state)
        to_mobility = self._count_reachable(to_pos, state)
        
        return (to_mobility - from_mobility) * 0.05
    
    def _count_reachable(self, pos: Tuple[int, int],
                       state: MiniGameState) -> int:
        """计算从某位置可达的格子数"""
        count = 0
        for p in state.all_positions():
            if p != pos:
                dist = self._manhattan_distance(pos, p)
                if dist <= 2:
                    count += 1
        return count
    
    def _tempo_impact(self, move: MiniMove, state: MiniGameState) -> float:
        """评估 Tempo 影响"""
        # 简化版
        if move.move_type == MoveType.FLIP:
            return 0.1  # 翻棋通常略微失去 tempo
        elif move.move_type == MoveType.CAPTURE:
            return 0.2  # 吃子获得 tempo
        return 0.0
    
    def name(self) -> str:
        return "MiniExpertAgent"


# 工厂函数
def create_agent(agent_type: str, color: int, **kwargs) -> BaseAgent:
    """创建 Agent 的工厂函数"""
    if agent_type == "random":
        return RandomAgent()
    elif agent_type == "heuristic":
        strategy = kwargs.get('strategy', 'balanced')
        return HeuristicAgent(color, strategy)
    elif agent_type == "expert":
        config_path = kwargs.get('config_path')
        return MiniExpertAgent(color, config_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
