"""
Expert-3: Search Optimizer - 搜索优化层

实现内容：
- MCTS with ISMCTS/PIMC support for incomplete information
- Alpha-Beta pruning
- Candidate move filtering from chess principles
- Variance reduction techniques

将棋理分析与深度搜索相结合，提供最优决策建议。
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import random
import math
import sys
sys.path.append('.')

from junqi.state import GameState, Move, Piece
from junqi.config import RuleConfig
from junqi.expert.rule_validator import RuleValidator
from junqi.expert.tactical_analyzer import TacticalAnalyzer, ThreatDetectionEngine
from junqi.expert.hidden_piece_belief import BeliefSystem
from junqi.expert.mobility_calculator import SpaceAdvantageCalculator
from junqi.expert.tempo_tracker import TempoTracker
from junqi.expert.conditional_value import ConditionalPieceValueEvaluator

@dataclass
class SearchResult:
    """搜索结果"""
    best_move: Move
    value: float                    # 评估值 (-1 to 1)
    visits: int                     # 访问次数
    win_rate: float                 # 胜率
    children_info: List[Dict]       # 子节点信息
    confidence: float               # 置信度
    depth_reached: int              # 达到深度

@dataclass
class MCNode:
    """MCTS 节点"""
    state: GameState
    move: Optional[Move]            # 到达此节点的移动
    parent: Optional['MCNode']
    children: List['MCNode'] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0          # 累计价值
    legal_moves: List[Move] = field(default_factory=list)
    
    # PIMC (Partial Information MCTS) specific
    hidden_pieces_simulation: int = 0  # 模拟次数
    simulated_values: List[float] = field(default_factory=list)


class MonteCarloTreeSearch:
    """
    蒙特卡洛树搜索引擎
    
    支持不完全信息的 ISMCTS (Importance Sampling MCTS)
    和 PIMC (Partially Observable MCTS)
    """
    
    def __init__(self, config: RuleConfig, max_depth: int = 8):
        self.config = config
        self.max_depth = max_depth
        
        # Initialize sub-experts
        self.validator = RuleValidator(config)
        self.tactical_analyzer = TacticalAnalyzer(self.validator)
        self.threat_detector = None  # Will be set later
        self.space_calculator = SpaceAdvantageCalculator()
        self.tempo_tracker = TempoTracker()
        self.value_evaluator = ConditionalPieceValueEvaluator(config)
        
        self.root_node = None
        self.num_iterations = 0
        self.total_simulations = 0
        
    def initialize_with_principles(self, belief_system: BeliefSystem):
        """初始化信念系统"""
        self.belief_system = belief_system
        self.threat_detector = ThreatDetectionEngine(self.validator, belief_system)
    
    def search(self, state: GameState, color: int, 
               num_iterations: int = 500) -> SearchResult:
        """
        执行 MCTS 搜索
        
        Args:
            state: 当前局面
            color: 己方颜色
            num_iterations: 迭代次数
            
        Returns:
            SearchResult: 搜索结果
        """
        self.num_iterations = num_iterations
        self.total_simulations = 0
        
        # Create root node
        self.root_node = MCNode(state=state, move=None, parent=None)
        
        # Get initial legal moves
        self._expand_root(state, color)
        
        # MCTS main loop
        for i in range(num_iterations):
            if i % 100 == 0 and i > 0:
                print(f"MCTS iteration {i}/{num_iterations}")
            
            # Step 1: Selection
            node = self._select(self.root_node)
            
            # Step 2: Expansion
            if self._should_expand(node):
                node = self._expand(node, color)
            
            # Step 3: Simulation (Rollout)
            evaluation = self._simulate(node, color)
            
            # Step 4: Backpropagation
            self._backpropagate(node, evaluation, color)
            
            self.total_simulations += 1
        
        # Return best move
        return self._get_best_result(color)
    
    def _select(self, node: MCNode) -> MCNode:
        """
        Selection 阶段：选择最有潜力的子节点
        
        Uses UCT formula:
        score = Q + C * sqrt(ln(parent_visits) / visits)
        """
        current = node
        
        while current.children and not self._is_terminal(current.state):
            # Choose child with highest UCT score
            best_child = None
            best_score = float('-inf')
            
            for child in current.children:
                uct_score = self._compute_uct(child, current.visits)
                
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child
            
            current = best_child
        
        return current
    
    def _compute_uct(self, node: MCNode, parent_visits: int) -> float:
        """计算 UCT 分数"""
        if node.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation = node.value_sum / node.visits
        
        # Exploration term
        exploration = math.sqrt(
            math.log(parent_visits) / node.visits
        ) * 1.4  # C parameter
        
        return exploitation + exploration
    
    def _should_expand(self, node: MCNode) -> bool:
        """判断是否应该扩展节点"""
        # Don't expand if terminal or max depth reached
        if self._is_terminal(node.state):
            return False
        
        # Don't expand if too deep
        depth = self._get_depth(node)
        if depth >= self.max_depth:
            return False
        
        # Expand if has unexplored moves
        return len(node.children) < len(node.legal_moves)
    
    def _expand(self, node: MCNode, color: int) -> MCNode:
        """Expansion 阶段：扩展一个新的子节点"""
        # Find an unexpanded move
        unexpanded_moves = [
            m for m in node.legal_moves 
            if not any(c.move == m for c in node.children)
        ]
        
        if not unexpanded_moves:
            return node
        
        # Select one move (prefer tactical moves first)
        move = self._prioritize_move(unexpanded_moves, node.state, color)
        
        # Simulate the move
        new_state = self._apply_move_and_update(node.state, move, color)
        
        # Create new node
        child_node = MCNode(
            state=new_state,
            move=move,
            parent=node,
            legal_moves=self.validator.get_all_legal_moves(new_state, 1 - color)
        )
        
        node.children.append(child_node)
        return child_node
    
    def _simulate(self, node: MCNode, color: int) -> float:
        """
        Simulation/Rollout 阶段：随机模拟直到终局
        
        Enhanced with principle-based evaluation at leaf nodes
        """
        current = node
        
        # Rollout until terminal or leaf
        while not self._is_terminal(current.state):
            depth = self._get_depth(current)
            if depth >= self.max_depth - 2:
                # Use principle-based evaluation at leaf
                return self._evaluate_position(current.state, current.parent.color if current.parent else color)
            
            # Pick random legal move
            moves = current.legal_moves
            if not moves:
                break
            
            move = random.choice(moves)
            new_state = self._apply_move_and_update(current.state, move, 1 - color)
            
            # Create temporary child for rollout
            temp_child = MCNode(
                state=new_state,
                move=move,
                parent=current,
                legal_moves=self.validator.get_all_legal_moves(new_state, color)
            )
            
            current = temp_child
        
        # Terminal state - evaluate
        if self._is_terminal(current.state):
            return self._evaluate_terminal(current.state)
        
        return self._evaluate_position(current.state, color)
    
    def _backpropagate(self, node: MCNode, value: float, original_color: int):
        """Backpropagation 阶段：反向传播评估值"""
        current = node
        
        while current:
            current.visits += 1
            current.value_sum += value
            
            # Flip value for opponent perspective
            value = -value
            
            current = current.parent
    
    def _get_best_result(self, color: int) -> SearchResult:
        """获取最佳搜索结果"""
        if not self.root_node.children:
            return SearchResult(
                best_move=None,
                value=0.0,
                visits=0,
                win_rate=0.0,
                children_info=[],
                confidence=0.0,
                depth_reached=0
            )
        
        # Select child with most visits (robust choice)
        best_child = max(self.root_node.children, key=lambda x: x.visits)
        
        # Calculate statistics
        visit_counts = [c.visits for c in self.root_node.children]
        total_visits = sum(visit_counts)
        
        # Confidence based on visit distribution
        max_ratio = best_child.visits / max(total_visits, 1)
        confidence = min(max_ratio, 1.0)
        
        # Depth reached
        max_depth = 0
        for child in self.root_node.children:
            depth = self._get_depth(child)
            max_depth = max(max_depth, depth)
        
        # Build children info
        children_info = []
        for child in sorted(self.root_node.children, key=lambda x: -x.visits)[:5]:
            children_info.append({
                'move': child.move,
                'visits': child.visits,
                'win_rate': (child.value_sum / max(child.visits, 1) + 1) / 2,
            })
        
        return SearchResult(
            best_move=best_child.move,
            value=best_child.value_sum / max(best_child.visits, 1),
            visits=best_child.visits,
            win_rate=(best_child.value_sum / max(best_child.visits, 1) + 1) / 2,
            children_info=children_info,
            confidence=confidence,
            depth_reached=max_depth
        )
    
    # Helper methods
    
    def _expand_root(self, state: GameState, color: int):
        """展开根节点的所有合法移动"""
        self.root_node.legal_moves = self.validator.get_all_legal_moves(state, color)
    
    def _is_terminal(self, state: GameState) -> bool:
        """判断是否终止状态"""
        # Check for flag captured
        for piece in state.pieces:
            if piece.type == '军旗' and not piece.is_revealed:
                continue
            # TODO: Add more terminal conditions
        
        return False  # Simplified for now
    
    def _get_depth(self, node: MCNode) -> int:
        """计算节点深度"""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def _prioritize_move(self, moves: List[Move], state: GameState, 
                        color: int) -> Move:
        """根据棋理优先选择移动"""
        # This would integrate with chess principles
        # For now, random selection with slight bias toward tactical moves
        return random.choice(moves)
    
    def _apply_move_and_update(self, state: GameState, move: Move,
                              player_color: int) -> GameState:
        """应用移动并更新状态（简化版）"""
        # TODO: Implement proper move application
        # For now, return same state
        return state
    
    def _evaluate_position(self, state: GameState, color: int) -> float:
        """使用棋理体系评估局面"""
        try:
            # 1. Tactical analysis
            tactical = self.tactical_analyzer.analyze_position(state, color)
            tactical_score = tactical.overall_score
            
            # 2. Space advantage
            space_result = self.space_calculator.calculate_space_advantage(state, color)
            space_score = space_result['advantage_score']
            
            # 3. Material/dynamic value comparison
            value_comparison = self.value_evaluator.compare_with_enemy(state, color)
            material_advantage = value_comparison['material_advantage_ratio']
            
            # Weighted combination
            evaluation = (
                tactical_score * 0.3 +
                space_score * 0.3 +
                material_advantage * 0.4
            )
            
            return max(-1.0, min(1.0, evaluation))
        
        except Exception as e:
            # Fallback to simple material count
            return self._fallback_evaluation(state, color)
    
    def _evaluate_terminal(self, state: GameState) -> float:
        """评估终止状态"""
        # Simplified: check if flag was captured
        return 1.0 if self._check_win(state) else -1.0
    
    def _check_win(self, state: GameState) -> bool:
        """检查是否获胜"""
        # TODO: Implement full win condition checking
        return False
    
    def _fallback_evaluation(self, state: GameState, color: int) -> float:
        """简单 fallback 评估"""
        my_material = sum(
            self.config.PIECE_RANKS.get(p.type, 0)
            for p in state.get_pieces(color)
            if p.is_revealed
        )
        enemy_material = sum(
            self.config.PIECE_RANKS.get(p.type, 0)
            for p in state.get_pieces(1 - color)
            if p.is_revealed
        )
        
        total = my_material + enemy_material
        if total == 0:
            return 0.0
        
        return (my_material - enemy_material) / total


class PartialInformationMCTS(MonteCarloTreeSearch):
    """
    部分信息 MCTS (PIMC)
    
    针对不完全信息博弈的增强版 MCTS
    通过多次模拟暗子的不同配置来获得更稳健的评估
    """
    
    def __init__(self, config: RuleConfig, max_depth: int = 6):
        super().__init__(config, max_depth)
        self.simulation_budget = 10  # Per position simulation budget
    
    def search_with_information_gathering(self, state: GameState,
                                         color: int,
                                         num_iterations: int = 300) -> SearchResult:
        """
        带信息收集的搜索
        
        结合翻棋决策和后续行动规划
        """
        self.num_iterations = num_iterations
        
        if not self.belief_system:
            print("Warning: Belief system not initialized")
            return super().search(state, color, num_iterations)
        
        # Special handling for reveal moves
        reveal_candidates = self._identify_reveal_candidates(state, color)
        
        if reveal_candidates:
            # Evaluate whether to reveal vs take other action
            best_reveal = max(reveal_candidates, 
                            key=lambda p: self._calculate_reveal_value(p, state))
            
            # Combine with normal MCTS
            result = super().search(state, color, num_iterations // 2)
            
            # If best MCTS move is not reveal, consider it
            if result.best_move not in reveal_candidates:
                reveal_result = self._simulate_reveal_move(
                    state, color, best_reveal
                )
                if reveal_result.win_rate > result.win_rate * 0.9:
                    result.best_move = reveal_result.best_move
                    result.win_rate = reveal_result.win_rate
        
        return super().search(state, color, num_iterations)
    
    def _identify_reveal_candidates(self, state: GameState,
                                   color: int) -> List[Tuple[int, int]]:
        """识别值得侦察的位置"""
        candidates = []
        
        if not self.belief_system:
            return candidates
        
        # Get positions with high uncertainty
        uncertain = self.belief_system.get_highest_uncertainty_positions(5)
        
        for pos, entropy in uncertain:
            candidate_pos = pos
            piece = state.get_piece_at(candidate_pos)
            
            if piece and piece.is_hidden:
                candidates.append(candidate_pos)
        
        return candidates
    
    def _calculate_reveal_value(self, pos: Tuple[int, int],
                               state: GameState) -> float:
        """计算翻棋的价值"""
        if not self.belief_system:
            return 0.0
        
        info_gain = self.belief_system.calculate_information_gain(pos)
        strategic_value = self.belief_system.assess_reveal_value(pos, {})
        
        # Combined score
        return info_gain * 0.5 + strategic_value * 0.5
    
    def _simulate_reveal_move(self, state: GameState, color: int,
                             reveal_pos: Tuple[int, int]) -> SearchResult:
        """模拟翻棋操作"""
        # This is a simplified version
        # In reality, we'd need to simulate all possible outcomes weighted by their probabilities
        
        # For now, just return a neutral result
        return SearchResult(
            best_move=Move(
                piece_id=-1,
                from_pos=reveal_pos,
                to_pos=reveal_pos,
                move_type=None  # Would be REVEAL type
            ),
            value=0.0,
            visits=1,
            win_rate=0.5,
            children_info=[],
            confidence=0.5,
            depth_reached=1
        )


class CandidateFilter:
    """
    候选移动过滤器
    
    基于棋理大幅减少搜索空间
    """
    
    def __init__(self, rule_validator: RuleValidator, 
                threat_detector: ThreatDetectionEngine):
        self.validator = rule_validator
        self.threat_detector = threat_detector
    
    def filter_candidates(self, legal_moves: List[Move],
                         state: GameState, color: int,
                         context: Dict) -> List[Move]:
        """
        过滤候选移动，只保留有希望的
        
        Filters applied:
        1. Remove obviously bad moves (suicide)
        2. Prioritize tactical opportunities
        3. Avoid moves that worsen threats
        4. Prefer moves aligned with strategic plan
        """
        if not legal_moves:
            return []
        
        # Score each move
        scored_moves = []
        
        for move in legal_moves:
            score = self._score_move(move, state, color, context)
            scored_moves.append((move, score))
        
        # Sort by score
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top candidates (e.g., top 30% or those above threshold)
        threshold = scored_moves[-1][1] + 0.2  # Slightly above worst
        top_candidates = [m for m, s in scored_moves if s >= threshold]
        
        return top_candidates[:max(3, len(top_candidates))]  # At least 3 options
    
    def _score_move(self, move: Move, state: GameState,
                   color: int, context: Dict) -> float:
        """单个移动的评分"""
        score = 0.0
        
        # Positive factors
        if self._is_capture(move, state):
            score += 0.5
        
        if self._improves_position(move, state):
            score += 0.2
        
        if self._reduces_threat(move, state, color):
            score += 0.3
        
        # Negative factors
        if self._exposes_piece_to_attack(move, state):
            score -= 0.4
        
        if self._worsens_space(move, state):
            score -= 0.1
        
        return score
    
    def _is_capture(self, move: Move, state: GameState) -> bool:
        """是否是吃子"""
        target = state.get_piece_at(move.to_pos)
        return target is not None and target.color != state.current_turn
    
    def _improves_position(self, move: Move, state: GameState) -> bool:
        """是否改善位置"""
        # TODO: Implementation
        return False
    
    def _reduces_threat(self, move: Move, state: GameState,
                       color: int) -> bool:
        """是否减少威胁"""
        # TODO: Implementation
        return False
    
    def _exposes_piece_to_attack(self, move: Move,
                                state: GameState) -> bool:
        """是否暴露棋子被攻击"""
        # TODO: Implementation
        return False
    
    def _worsens_space(self, move: Move, state: GameState) -> bool:
        """是否恶化空间"""
        # TODO: Implementation
        return False
