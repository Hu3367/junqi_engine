"""
Expert-1: 战术分析层 (Tactical Analyzer)

负责：
- 立即吃子识别
- 强制吃子判断
- 连续吃链条发现
- 反吃风险检测
- 必杀/直接威胁检测

解决："棋盘上现在有什么立即发生的事情？"
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.append('.')

from junqi.state import GameState, Piece
from .move_adapter import Move, MoveType
from junqi.expert.rule_validator import RuleValidator

@dataclass
class TacticalOpportunity:
    """战术机会"""
    type: str  # 'capture', 'fork', 'discovered_attack', 'sacrifice'
    attacker: Piece
    target: Piece
    expected_gain: float  # 预期收益（子力价值差）
    risk_level: str  # 'low', 'medium', 'high'
    description: str

@dataclass
class ThreatAssessment:
    """威胁评估"""
    severity: float  # 0-1.0
    threat_type: str  # 'immediate_capture', 'flag_threat', 'material_loss'
    source_piece: Optional[Piece]
    target_piece: Optional[Piece]
    time_to_threat: int  # 几步内会发生
    recommended_response: str

class TacticalAnalyzer:
    """战术分析器 - Expert-1"""
    
    def __init__(self, rule_validator: RuleValidator):
        self.validator = rule_validator
        
    def analyze_position(self, state: GameState, color: int) -> TacticalReport:
        """
        全面分析当前局面的战术特征
        
        Returns:
            TacticalReport: 包含所有战术信息的报告
        """
        # 1. 检测所有可能的吃子机会
        captures = self.detect_capture_opportunities(state, color)
        
        # 2. 检测对手的吃子威胁
        enemy_threats = self.detect_enemy_threats(state, color)
        
        # 3. 检查是否有强制移动（必须应对的威胁）
        forced_moves = self.identify_forced_moves(state, color, enemy_threats)
        
        # 4. 检测双攻击/多重威胁
        forks = self.detect_forks(state, color)
        
        # 5. 计算整体战术评分
        tactical_score = self.calculate_tactical_score(
            captures, enemy_threats, forced_moves, forks
        )
        
        return TacticalReport(
            capture_opportunities=captures,
            enemy_threats=enemy_threats,
            forced_moves=forced_moves,
            forks=forks,
            overall_score=tactical_score
        )
    
    def detect_capture_opportunities(self, state: GameState, color: int) -> List[Dict]:
        """
        检测所有可以吃子的机会
        
        Returns:
            List[Dict]: [{'attacker': Piece, 'target': Piece, 'expected_outcome': Dict, 'position': tuple}]
        """
        opportunities = []
        
        my_pieces = state.get_pieces(color)
        enemy_pieces = state.get_pieces(1 - color)
        
        for my_piece in my_pieces:
            if my_piece.is_hidden or not my_piece.is_revealed:
                continue
                
            for enemy_piece in enemy_pieces:
                if enemy_piece.is_hidden or not enemy_piece.is_revealed:
                    continue
                    
                if self._can_capture(my_piece, enemy_piece, state):
                    outcome = self._simulate_attack(my_piece, enemy_piece)
                    
                    # 计算期望收益
                    value_diff = self._calculate_value_difference(
                        my_piece, enemy_piece, outcome
                    )
                    
                    opportunities.append({
                        'type': 'capture',
                        'attacker': my_piece,
                        'target': enemy_piece,
                        'outcome': outcome,
                        'position': (my_piece.position, enemy_piece.position),
                        'expected_gain': value_diff,
                        'priority': self._calculate_capture_priority(value_diff, my_piece, enemy_piece)
                    })
        
        # 按优先级排序
        opportunities.sort(key=lambda x: x['priority'], reverse=True)
        
        return opportunities
    
    def detect_enemy_threats(self, state: GameState, color: int) -> List[ThreatAssessment]:
        """
        检测对手的所有威胁
        
        包括：
        - 立即吃子威胁
        - 旗区威胁
        - 材料损失威胁
        """
        threats = []
        enemy_color = 1 - color
        
        my_pieces = state.get_pieces(color)
        enemy_pieces = state.get_pieces(enemy_color)
        
        for enemy_piece in enemy_pieces:
            if enemy_piece.is_hidden or not enemy_piece.is_revealed:
                continue
            
            for my_piece in my_pieces:
                if my_piece.is_hidden or not my_piece.is_revealed:
                    continue
                
                # 检查是否可以被吃掉
                if self._can_capture(enemy_piece, my_piece, state):
                    # 计算威胁等级
                    threat = self._assess_threat(
                        enemy_piece, my_piece, state, color
                    )
                    threats.append(threat)
        
        # 按严重程度排序
        threats.sort(key=lambda x: x.severity, reverse=True)
        
        return threats
    
    def identify_forced_moves(self, state: GameState, color: int, 
                             enemy_threats: List[ThreatAssessment]) -> List[Move]:
        """
        识别所有强制性的移动
        
        强制性移动包括：
        - 必须逃跑的被将军棋子
        - 必须保护的受威胁高价值棋子
        - 必须防守的关键位置
        """
        forced = []
        
        # 如果有很严重的威胁（如军旗），可能需要强制回应
        critical_threats = [t for t in enemy_threats if t.severity > 0.8]
        
        if critical_threats:
            # 为每个关键威胁找到应对移动
            for threat in critical_threats:
                response_moves = self.find_response_moves(state, color, threat)
                forced.extend(response_moves[:2])  # 最多两个响应
            
        return list(set(forced))  # 去重
    
    def detect_forks(self, state: GameState, color: int) -> List[Dict]:
        """
        检测双攻击/多重威胁
        
        Fork: 一个棋子同时攻击多个目标
        """
        forks = []
        
        my_pieces = state.get_pieces(color)
        enemy_pieces = state.get_pieces(1 - color)
        
        for my_piece in my_pieces:
            if my_piece.is_hidden or not my_piece.is_revealed:
                continue
            
            # 找出该棋子可以攻击的所有敌方棋子
            targets = []
            for enemy_piece in enemy_pieces:
                if not enemy_piece.is_hidden and enemy_piece.is_revealed:
                    if self._can_capture(my_piece, enemy_piece, state):
                        targets.append(enemy_piece)
            
            # 如果一次可以攻击多个，就是 fork
            if len(targets) >= 2:
                forks.append({
                    'piece': my_piece,
                    'targets': targets,
                    'fork_type': 'direct_capture',
                    'potential_gain': sum(
                        self.config.PIECE_RANKS.get(t.type, 0) for t in targets
                    )
                })
        
        return forks
    
    def calculate_tactical_score(self, captures: List, enemy_threats: List,
                                forced_moves: List, forks: List) -> float:
        """
        计算整体战术分数
        
        Returns:
            float: -1.0 to 1.0
                  > 0: 有战术优势
                  < 0: 有战术劣势
        """
        score = 0.0
        
        # 我的进攻机会
        attack_score = sum(c['expected_gain'] * c['priority'] for c in captures) / 10.0
        score += min(attack_score, 0.3)
        
        # 我的 fork 机会
        fork_score = len(forks) * 0.1
        score += min(fork_score, 0.2)
        
        # 对手的威胁
        defense_cost = sum(t.severity for t in enemy_threats) / 5.0
        score -= min(defense_cost, 0.4)
        
        # 被迫移动惩罚
        forced_penalty = len(forced_moves) * 0.05
        score -= min(forced_penalty, 0.1)
        
        return max(-1.0, min(1.0, score))
    
    def _can_capture(self, attacker: Piece, defender: Piece, state: GameState) -> bool:
        """判断是否可以吃掉对方"""
        if attacker.type == '工兵':
            return defender.type == '地雷'
        elif attacker.type == '炸弹':
            return True
        else:
            att_rank = self.validator.config.PIECE_RANKS.get(attacker.type, 0)
            def_rank = self.validator.config.PIECE_RANKS.get(defender.type, 0)
            return att_rank >= def_rank
    
    def _simulate_attack(self, attacker: Piece, defender: Piece) -> Dict:
        """模拟攻击结果"""
        if attacker.type == '工兵' and defender.type == '地雷':
            return {
                'result': 'mine_dug',
                'attacker_survives': True,
                'defender_eliminated': True,
                'material_gain': self.validator.config.PIECE_RANKS.get('地雷', 0)
            }
        elif attacker.type == '炸弹':
            return {
                'result': 'mutual_destruction',
                'attacker_survives': False,
                'defender_eliminated': True,
                'material_gain': self.validator.config.PIECE_RANKS.get(defender.type, 0) - 
                                self.validator.config.PIECE_RANKS.get(attacker.type, 0)
            }
        else:
            att_rank = self.validator.config.PIECE_RANKS.get(attacker.type, 0)
            def_rank = self.validator.config.PIECE_RANKS.get(defender.type, 0)
            
            if att_rank > def_rank:
                return {
                    'result': 'capture',
                    'attacker_survives': True,
                    'defender_eliminated': True,
                    'material_gain': def_rank
                }
            elif att_rank == def_rank:
                return {
                    'result': 'exchange',
                    'attacker_survives': False,
                    'defender_eliminated': False,
                    'material_gain': 0
                }
            else:
                return {
                    'result': 'fail',
                    'attacker_survives': True,
                    'defender_eliminated': False,
                    'material_gain': 0
                }
    
    def _calculate_value_difference(self, attacker: Piece, defender: Piece,
                                   outcome: Dict) -> float:
        """计算期望价值差"""
        gain = outcome.get('material_gain', 0)
        
        if not outcome['attacker_survives']:
            loss = self.validator.config.PIECE_RANKS.get(attacker.type, 0)
            return gain - loss
        
        return gain
    
    def _calculate_capture_priority(self, expected_gain: float, attacker: Piece,
                                   target: Piece) -> float:
        """计算吃子的优先级"""
        priority = abs(expected_gain)
        
        # 高价值目标加分
        if target.type in ['司令', '军长', '师长']:
            priority *= 1.5
        
        # 安全吃子额外加分
        if self._is_safe_capture(attacker, target):
            priority *= 1.2
        
        return priority
    
    def _is_safe_capture(self, attacker: Piece, target: Piece) -> bool:
        """判断是否是安全吃子"""
        # 简单的安全检查：敌人不能用更高价值的棋子反吃
        att_rank = self.validator.config.PIECE_RANKS.get(attacker.type, 0)
        def_rank = self.validator.config.PIECE_RANKS.get(target.type, 0)
        
        return att_rank > def_rank
    
    def _assess_threat(self, aggressor: Piece, victim: Piece, state: GameState,
                      defense_color: int) -> ThreatAssessment:
        """评估单个威胁的严重程度"""
        base_severity = 0.0
        
        # 根据棋子价值调整
        victim_value = self.validator.config.PIECE_RANKS.get(victim.type, 0)
        base_severity = victim_value / 10.0
        
        # 如果是军旗，直接最高威胁
        if victim.type == '军旗':
            return ThreatAssessment(
                severity=1.0,
                threat_type='flag_threat',
                source_piece=aggressor,
                target_piece=victim,
                time_to_threat=1,
                recommended_response='immediate_defense'
            )
        
        # 根据距离调整时间
        # TODO: 计算最短路径
        
        # 判断威胁类型
        if aggressor.type == '工兵':
            threat_type = 'mine_threat' if victim.type == '地雷' else 'light_infantry'
        elif aggressor.type == '炸弹':
            threat_type = 'bomb_threat'
        else:
            threat_type = 'material_loss'
        
        return ThreatAssessment(
            severity=min(base_severity, 1.0),
            threat_type=threat_type,
            source_piece=aggressor,
            target_piece=victim,
            time_to_threat=1,
            recommended_response='evaluate_response'
        )
    
    def find_response_moves(self, state: GameState, color: int,
                           threat: ThreatAssessment) -> List[Move]:
        """寻找应对特定威胁的移动"""
        responses = []
        
        # 方法 1: 逃跑
        if threat.target_piece:
            escape_moves = self._find_escape_moves(state, color, threat.target_piece)
            responses.extend(escape_moves)
        
        # 方法 2: 拦截
        if threat.source_piece:
            block_moves = self._find_intercept_moves(state, color, threat.source_piece)
            responses.extend(block_moves)
        
        # 方法 3: 反威胁
        counter_threat_moves = self._find_counter_threat_moves(state, color, threat)
        responses.extend(counter_threat_moves)
        
        return responses
    
    def _find_escape_moves(self, state: GameState, color: int, 
                          piece: Piece) -> List[Move]:
        """寻找逃跑路线"""
        moves = []
        
        # 获取合法移动并过滤出安全的
        all_moves = self.validator.get_all_legal_moves(state, color)
        
        for move in all_moves:
            if move.piece_id == piece.id:
                # 检查目标位置是否安全
                target = state.get_piece_at(move.to_pos)
                if target is None or target.color == color:
                    moves.append(move)
        
        return moves[:5]  # 最多返回 5 个选择
    
    def _find_intercept_moves(self, state: GameState, color: int,
                             target_piece: Piece) -> List[Move]:
        """寻找拦截路线"""
        moves = []
        # TODO: 实现具体的拦截逻辑
        return moves
    
    def _find_counter_threat_moves(self, state: GameState, color: int,
                                  threat: ThreatAssessment) -> List[Move]:
        """寻找反威胁移动"""
        moves = []
        # TODO: 实现反威胁逻辑
        return moves


@dataclass
class TacticalReport:
    """战术分析报告"""
    capture_opportunities: List[Dict]
    enemy_threats: List[ThreatAssessment]
    forced_moves: List[Move]
    forks: List[Dict]
    overall_score: float  # -1 to 1
    
    def summary(self) -> str:
        """生成简洁总结"""
        lines = [
            f"Tactical Score: {self.overall_score:.2f}",
            f"Capture Opportunities: {len(self.capture_opportunities)}",
            f"Enemy Threats: {len(self.enemy_threats)}",
            f"Forced Moves Required: {len(self.forced_moves)}",
            f"Forks Detected: {len(self.forks)}"
        ]
        return "\n".join(lines)
