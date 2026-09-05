"""
Expert Engine: 统一的 Teacher / Oracle 接口

整合四个专家层级，提供决策和标签输出能力。
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
sys.path.append('.')

from junqi.state import GameState, Move
from junqi.config import RuleConfig

from .rule_validator import RuleValidator
from .tactical_analyzer import TacticalAnalyzer, TacticalReport
from .hidden_piece_belief import BeliefSystem, InformationReasoner
from .mobility_calculator import SpaceAdvantageCalculator
from .tempo_tracker import TempoTracker

@dataclass
class Decision:
    """单一决策结果"""
    move: Move
    policy_value: float      # 该动作的概率值
    value_estimate: float    # 局面评估值
    labels: Dict[str, Any]   # 战术和战略标签

@dataclass
class FullEvaluation:
    """完整局面评估"""
    best_move: Move
    policy_distribution: List[float]
    value_score: float
    tactical_labels: Dict[str, Any]
    strategic_labels: Dict[str, Any]
    confidence: float

class ExpertEngine:
    """
    专家引擎 V0
    
    Architecture:
    
    Expert-0: Rule Validator (规则验证器)
        ↓
    Expert-1: Tactical Analyzer (战术分析器)  
        ↓
    Expert-2: Principle Evaluator (棋理评估器)
        ├─ Hidden Piece Belief (信息推理)
        ├─ Mobility Calculator (空间控制)
        └─ Tempo Tracker (Tempo 经济学)
        ↓
    Expert-3: Search Optimizer (搜索优化器) - TODO
        ↓
    Unified Output: Policy + Value + Labels
    """
    
    def __init__(self, config: RuleConfig):
        self.config = config
        
        # Initialize all experts
        self.rule_validator = RuleValidator(config)
        self.tactical_analyzer = TacticalAnalyzer(self.rule_validator)
        
        # Board state for belief system
        self.belief_system = None
        self.information_reasoner = None
        
        self.space_calculator = SpaceAdvantageCalculator()
        self.tempo_tracker = TempoTracker()
    
    def initialize_belief_system(self, board_state: GameState):
        """初始化信念系统"""
        self.belief_system = BeliefSystem(self.config, board_state)
        self.information_reasoner = InformationReasoner(self.belief_system)
    
    def evaluate_position(self, game_state: GameState, 
                         color: int) -> FullEvaluation:
        """
        全面评估当前局面
        
        This is the main API call from the Teacher/Oracle perspective.
        
        Returns comprehensive evaluation including:
        - Best move recommendation
        - Policy distribution over candidate moves
        - Value estimate (win probability or expected outcome)
        - Tactical labels (threats, opportunities)
        - Strategic labels (information advantage, space control, etc.)
        """
        
        # Step 1: Generate legal moves
        legal_moves = self.rule_validator.get_all_legal_moves(game_state, color)
        
        if not legal_moves:
            return FullEvaluation(
                best_move=None,
                policy_distribution=[],
                value_score=-1.0 if len(legal_moves) == 0 else 0.0,
                tactical_labels={},
                strategic_labels={'stalemate_or_loss': True},
                confidence=0.0
            )
        
        # Step 2: Tactical analysis
        tactical_report = self.tactical_analyzer.analyze_position(game_state, color)
        
        # Step 3: Principle-based evaluation
        info_advantage = self._evaluate_information(game_state, color)
        space_advantage = self.space_calculator.calculate_space_advantage(game_state, color)
        tempo_assessment = self.tempo_tracker.get_current_assessment(game_state)
        
        # Step 4: Score each move
        move_scores = []
        
        for move in legal_moves:
            score = self._score_move(move, game_state, tactical_report, 
                                   info_advantage, space_advantage, tempo_assessment)
            move_scores.append((move, score))
        
        # Step 5: Sort by score and pick best
        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_move, best_score = move_scores[0]
        
        # Step 6: Convert to policy distribution (softmax)
        policy_values = [score for _, score in move_scores]
        normalized_policy = self._softmax(policy_values)
        
        # Step 7: Calculate overall value estimate
        value_estimate = self._calculate_value_estimate(
            best_move, tactical_report, info_advantage, space_advantage
        )
        
        # Step 8: Compile labels
        tactical_labels = self._extract_tactical_labels(tactical_report)
        strategic_labels = self._extract_strategic_labels(
            info_advantage, space_advantage, tempo_assessment
        )
        
        return FullEvaluation(
            best_move=best_move,
            policy_distribution=normalized_policy[:10],  # Top 10 moves
            value_score=value_estimate,
            tactical_labels=tactical_labels,
            strategic_labels=strategic_labels,
            confidence=self._calculate_confidence(best_score, legal_moves)
        )
    
    def _score_move(self, move: Move, game_state: GameState,
                   tactical: TacticalReport, info: Dict,
                   space: Dict, tempo: TempoAssessment) -> float:
        """单个移动评分"""
        score = 0.0
        
        # 1. Tactical value (40%)
        tactical_score = self._score_tactical_value(move, tactical)
        score += tactical_score * 0.4
        
        # 2. Information value (15%)
        info_value = self._score_information_gain(move)
        score += info_value * 0.15
        
        # 3. Space improvement (20%)
        space_improvement = self._score_space_benefit(move, space)
        score += space_improvement * 0.2
        
        # 4. Tempo consideration (15%)
        tempo_impact = self._score_tempo_impact(move, tempo)
        score += tempo_impact * 0.15
        
        # 5. Risk assessment (-10% penalty for risky moves)
        risk_penalty = self._assess_move_risk(move, game_state)
        score -= risk_penalty * 0.1
        
        return score
    
    def _score_tactical_value(self, move: Move, tactical: TacticalReport) -> float:
        """评估战术价值"""
        score = 0.0
        
        # 检查是否是吃子
        capture_opps = tactical.capture_opportunities
        for opp in capture_opps:
            if opp['attacker'].id == move.piece_id and \
               opp['target'].position == move.to_pos:
                score += opp['expected_gain'] * opp['priority']
                break
        
        # 检查是否能消除威胁
        threat_reduction = self._check_threat_reduction(move, tactical.enemy_threats)
        score += threat_reduction
        
        # 检查是否缓解被迫移动
        forced_moves = tactical.forced_moves
        if move in forced_moves:
            score += 0.5  # 化解了被迫移动有分
        
        return score
    
    def _score_information_gain(self, move: Move) -> float:
        """评估信息增益价值（翻棋）"""
        if move.move_type.name != 'REVEAL':  # 假设翻棋的类型
            return 0.0
        
        if self.belief_system:
            gain = self.belief_system.calculate_information_gain(move.to_pos)
            reveal_value = self.belief_system.assess_reveal_value(move.to_pos, {})
            return min(reveal_value, 1.5)
        
        return 0.0
    
    def _score_space_benefit(self, move: Move, space_info: Dict) -> float:
        """评估空间提升"""
        # 简化版：检查是否移动到中心或关键位置
        key_positions = [
            (5,0), (5,4), (6,0), (6,4),  # 铁路枢纽
            (3,2), (8,2),                  # 行营中心
        ]
        
        if move.to_pos in key_positions:
            return 0.8
        
        return 0.0
    
    def _score_tempo_impact(self, move: Move, tempo: TempoAssessment) -> float:
        """评估 tempo 影响"""
        current_balance = tempo.current_balance
        
        # 如果我有 tempo 优势，应该保持压力 (+0.3)
        if current_balance > 0.5:
            return 0.3
        
        # 如果我 tempo 劣势，需要扭转 (-0.3 for defensive)
        elif current_balance < -0.5:
            return -0.3
        
        return 0.0
    
    def _assess_move_risk(self, move: Move, game_state: GameState) -> float:
        """评估移动风险"""
        risk = 0.0
        
        # 检查是否会暴露重要棋子到危险位置
        target = game_state.get_piece_at(move.to_pos)
        if target and target.is_revealed:
            att_rank = self.config.PIECE_RANKS.get(target.type, 0)
            # 如果暴露后会被轻易吃掉
            if att_rank >= 7:  # 师长及以上
                risk += 0.5
        
        return risk
    
    def _calculate_value_estimate(self, best_move: Move,
                                 tactical: TacticalReport,
                                 info: Dict, space: Dict) -> float:
        """计算整体局面估值 (-1 to 1)"""
        score = 0.0
        
        # Tactical contribution
        score += tactical.overall_score * 0.3
        
        # Information contribution
        info_adv = info.get('advantage', 0)
        score += info_adv * 0.2
        
        # Space contribution
        space_adv = space.get('advantage_score', 0)
        score += space_adv * 0.4
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _extract_tactical_labels(self, tactical: TacticalReport) -> Dict:
        """提取战术标签"""
        return {
            'has_capture_opportunity': len(tactical.capture_opportunities) > 0,
            'capture_count': len(tactical.capture_opportunities),
            'under_threat': len(tactical.enemy_threats) > 0,
            'threat_severity': sum(t.severity for t in tactical.enemy_threats),
            'forced_to_move': len(tactical.forced_moves) > 0,
            'fork_opportunities': len(tactical.forks),
            'tactical_score': tactical.overall_score
        }
    
    def _extract_strategic_labels(self, info: Dict, space: Dict,
                                 tempo: TempoAssessment) -> Dict:
        """提取战略标签"""
        return {
            'information_advantage': info.get('advantage', 0),
            'space_advantage': space.get('advantage_score', 0),
            'tempo_balance': tempo.current_balance,
            'recommended_strategy': tempo.recommended_strategy,
            'urgency_level': tempo.urgency,
            'phase': tempo.phase
        }
    
    def _calculate_confidence(self, best_score: float,
                            legal_moves: List[Move]) -> float:
        """计算决策置信度"""
        if len(legal_moves) <= 1:
            return 0.5  # 只有一种选择，没有真正的决策
        
        # Score variance indicates how clear the decision is
        scores = [score for _, score in [(m, self._score_move(m, None, None, None, None, None)) 
                                          for m in legal_moves]]
        
        score_range = max(scores) - min(scores) if scores else 0
        confidence = min(score_range * 2, 1.0)
        
        return max(0.3, confidence)
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Softmax 归一化"""
        import math
        
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        
        return [e / total for e in exp_scores]
    
    def generate_training_sample(self, game_state: GameState,
                                color: int) -> Dict[str, Any]:
        """
        生成一个高质量的训练样本
        
        Includes:
        - State encoding (input)
        - Policy (action probabilities)
        - Value (outcome prediction)
        - All labels (multi-task training)
        """
        evaluation = self.evaluate_position(game_state, color)
        
        # Create state encoding here (simplified)
        state_encoding = self._encode_state(game_state)
        
        return {
            'state': state_encoding,
            'policy': evaluation.policy_distribution,
            'value': evaluation.value_score,
            'tactical_labels': evaluation.tactical_labels,
            'strategic_labels': evaluation.strategic_labels,
            'best_move': self._move_to_string(evaluation.best_move),
            'confidence': evaluation.confidence
        }
    
    def _encode_state(self, state: GameState) -> Dict:
        """编码游戏状态为向量/张量"""
        # TODO: 实现完整的状态编码逻辑
        return {
            'board_repr': str(state.board),
            'turn': state.turn_count,
            'active_color': state.current_turn
        }
    
    def _move_to_string(self, move: Move) -> str:
        """将移动转换为字符串表示"""
        if move is None:
            return "None"
        
        return f"{move.from_pos}->{move.to_pos}"
