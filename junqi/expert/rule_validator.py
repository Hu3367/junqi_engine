"""
Expert-0: 规则验证层 (Rule Validator)

负责：
1. 100% 正确识别合法走法
2. 100% 正确识别攻击关系
3. 100% 正确处理翻棋
4. 100% 正确处理工兵/炸弹/地雷

这是所有上层功能的基础，必须绝对可靠。
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.append('.')

from junqi.state import GameState, Piece
from junqi.config import RuleConfig
from .move_adapter import Move, MoveType

@dataclass
class ValidationReport:
    """验证报告"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
@dataclass
class AttackResult:
    """攻击结果"""
    attacker: Piece
    defender: Piece
    winner: Piece
    is_explodable: bool = False
    mine_detected: bool = False
    
class RuleValidator:
    """规则验证器 - Expert-0"""
    
    def __init__(self, config: RuleConfig):
        self.config = config
        
    def validate_move(self, state: GameState, move: Move, color: int) -> ValidationReport:
        """
        验证一个移动是否合法
        
        检查项：
        1. 棋子是否存在且属于当前玩家
        2. 目标格子是否可达
        3. 是否符合棋子移动规则（公路/铁路/行营/大本营）
        4. 战斗结算是否正确
        5. 是否有特殊规则违反（循环局面、三手规则等）
        
        Returns:
            ValidationReport: 验证结果
        """
        errors = []
        warnings = []
        
        # 1. 检查棋子是否存在
        piece = state.get_piece_at(move.from_pos)
        if piece is None:
            errors.append(f"No piece at {move.from_pos}")
            return ValidationReport(False, errors, warnings)
            
        # 2. 检查棋子是否属于当前玩家
        if piece.color != color:
            errors.append(f"Piece at {move.from_pos} belongs to opponent")
            return ValidationReport(False, errors, warnings)
            
        # 3. 检查是否是暗子（暗子不能主动移动）
        if piece.is_hidden:
            errors.append("Cannot move hidden piece")
            return ValidationReport(False, errors, warnings)
            
        # 4. 检查目标位置合法性
        target_valid, target_error = self._validate_target_position(state, move, piece)
        if not target_valid:
            errors.append(target_error)
            return ValidationReport(False, errors, warnings)
            
        # 5. 检查移动路径合法性（铁路滑行）
        path_valid, path_error = self._validate_path(state, move, piece)
        if not path_valid:
            errors.append(path_error)
            return ValidationReport(False, errors, warnings)
            
        # 6. 检查战斗可行性（如果是攻击移动）
        if move.move_type in [MoveType.CAPTURE, MoveType.EXPLODE]:
            attack_valid, attack_result = self._validate_attack(state, move, piece)
            if not attack_valid:
                errors.append(attack_result)
                return ValidationReport(False, errors, warnings)
                
        # 7. 检查特殊限制（行营容量、大本营限制等）
        special_valid, special_error = self._validate_special_rules(state, move, piece)
        if not special_valid:
            errors.append(special_error)
            return ValidationReport(False, errors, warnings)
            
        # 8. 检查是否会形成长期循环（简单版本）
        cycle_valid, cycle_error = self._check_repetition(state, move)
        if not cycle_valid:
            warnings.append(cycle_error)
            
        return ValidationReport(True, [], warnings)
    
    def _validate_target_position(self, state: GameState, move: Move, piece: Piece) -> Tuple[bool, str]:
        """验证目标位置是否合法"""
        target_pos = move.to_pos
        
        # 检查目标是否在棋盘内
        if not state.is_valid_position(target_pos):
            return False, f"Target position {target_pos} out of bounds"
            
        # 检查目标是否是行营（任何棋子都可以进入行营）
        if state.board.is_bunker(target_pos):
            # 检查行营是否已被占用
            if state.get_piece_at(target_pos) is not None:
                # 只能是吃子行为
                if move.move_type != MoveType.CAPTURE:
                    return False, f"Bunker at {target_pos} is occupied"
            return True, ""
            
        # 检查目标是否是大本营（只有司令和军长可以进入己方大本营）
        if state.board.is_base(target_pos):
            if piece.type not in ['司令', '军长']:
                return False, f"Only Marshal and Army Commander can enter base"
            if not state.board.is_my_base(target_pos, piece.color):
                return False, f"Cannot enter opponent's base"
                
        # 检查目标是否有己方棋子（不允许自吃）
        target_piece = state.get_piece_at(target_pos)
        if target_piece is not None and target_piece.color == piece.color:
            return False, f"Cannot move to square occupied by own piece"
            
        return True, ""
    
    def _validate_path(self, state: GameState, move: Move, piece: Piece) -> Tuple[bool, str]:
        """验证移动路径是否合法"""
        from_pos = move.from_pos
        to_pos = move.to_pos
        
        # 如果不是铁路移动，检查是否是相邻格子
        if not piece.can_slide_on_rail() or not state.board.is_rail(from_pos):
            # 公路移动：只能移动到相邻格子
            dr = abs(from_pos[0] - to_pos[0])
            dc = abs(from_pos[1] - to_pos[1])
            if dr + dc != 1:
                return False, "Non-rail pieces can only move to adjacent squares"
            return True, ""
            
        # 铁路移动：检查是否可以滑行到目标
        if not self._is_rail_connection(from_pos, to_pos, state.board):
            return False, f"No rail connection between {from_pos} and {to_pos}"
            
        # 检查路径上是否有阻挡（除了目标位置）
        path_cells = self._get_rail_path(from_pos, to_pos, state.board)
        for cell in path_cells[1:-1]:  # 跳过起点和终点
            if state.get_piece_at(cell) is not None:
                return False, f"Path blocked by piece at {cell}"
                
        return True, ""
    
    def _is_rail_connection(self, start: Tuple[int, int], end: Tuple[int, int], board) -> bool:
        """检查两点之间是否有铁路连接"""
        if start[0] == end[0]:  # 同一行
            min_col = min(start[1], end[1])
            max_col = max(start[1], end[1])
            for col in range(min_col, max_col + 1):
                if not board.is_rail((start[0], col)):
                    return False
            return True
        elif start[1] == end[1]:  # 同一列
            min_row = min(start[0], end[0])
            max_row = max(start[0], end[0])
            for row in range(min_row, max_row + 1):
                if not board.is_rail((row, start[1])):
                    return False
            return True
        else:
            return False
    
    def _get_rail_path(self, start: Tuple[int, int], end: Tuple[int, int], board) -> List[Tuple[int, int]]:
        """获取铁路路径上的所有格子"""
        path = []
        
        if start[0] == end[0]:  # 同一行
            min_col = min(start[1], end[1])
            max_col = max(start[1], end[1])
            for col in range(min_col, max_col + 1):
                path.append((start[0], col))
        elif start[1] == end[1]:  # 同一列
            min_row = min(start[0], end[0])
            max_row = max(start[0], end[0])
            for row in range(min_row, max_row + 1):
                path.append((row, start[1]))
                
        return path
    
    def _validate_attack(self, state: GameState, move: Move, attacker: Piece) -> Tuple[bool, str]:
        """验证攻击是否可行"""
        target_piece = state.get_piece_at(move.to_pos)
        
        if target_piece is None:
            return False, "No target to attack"
            
        # 检查等级关系
        if attacker.type == '工兵':
            if target_piece.type == '地雷':
                return True, "工兵可以挖雷"
            elif target_piece.type == '炸弹':
                return False, "工兵不能吃炸弹"
            else:
                # 工兵只能吃地雷和飞，不能吃其他棋子
                return False, "工兵只能 dig mines or fly"
                
        if attacker.type == '炸弹':
            # 炸弹可以和任何棋子同归于尽
            return True, "炸弹可以与任何棋子同归于尽"
            
        # 普通战斗：比较等级
        attacker_rank = self.config.PIECE_RANKS.get(attacker.type, 0)
        defender_rank = self.config.PIECE_RANKS.get(target_piece.type, 0)
        
        if attacker_rank > defender_rank:
            return True, "Attacker wins"
        elif attacker_rank == defender_rank:
            return True, "Draw - both eliminated"
        else:
            return False, f"Attacker loses: {attacker.type} vs {target_piece.type}"
    
    def _validate_special_rules(self, state: GameState, move: Move, piece: Piece) -> Tuple[bool, str]:
        """验证特殊规则"""
        # 检查行营容量（每个行营同一时间只能有一个棋子）
        if state.board.is_bunker(move.to_pos):
            existing = state.get_piece_at(move.to_pos)
            if existing is not None and move.move_type != MoveType.CAPTURE:
                return False, f"Bunker already occupied"
                
        # 检查大本营限制（每个大本营最多一个棋子）
        if state.board.is_base(move.to_pos):
            # 这个检查需要在完整的GameState中实现
            pass
            
        return True, ""
    
    def _check_repetition(self, state: GameState, move: Move) -> Tuple[bool, str]:
        """检查是否形成长期循环（简化版）"""
        # TODO: 实现完整的三次重复局面检测
        # 这里只做一个简单的启发式警告
        return True, ""
    
    def get_all_legal_moves(self, state: GameState, color: int) -> List[Move]:
        """
        生成所有合法的移动
        
        这是 Expert-0 的核心功能之一，必须 100% 正确
        """
        legal_moves = []
        
        # 遍历所有棋子
        for piece in state.get_pieces(color):
            if piece.is_hidden:
                continue
                
            # 尝试所有可能的目标位置
            for row in range(state.board.height):
                for col in range(state.board.width):
                    target_pos = (row, col)
                    
                    # 生成候选移动
                    candidate_move = Move(
                        piece_id=piece.id,
                        from_pos=piece.position,
                        to_pos=target_pos,
                        move_type=MoveType.MOVE
                    )
                    
                    # 验证合法性
                    report = self.validate_move(state, candidate_move, color)
                    
                    if report.is_valid:
                        legal_moves.append(candidate_move)
                        
        return legal_moves
    
    def detect_capture_opportunities(self, state: GameState, color: int) -> List[Dict]:
        """
        检测所有可以吃子的机会
        
        Returns:
            List[Dict]: 每个元素包含{'piece', 'target', 'expected_outcome'}
        """
        opportunities = []
        
        my_pieces = state.get_pieces(color)
        enemy_pieces = state.get_enemy_pieces(color)
        
        for my_piece in my_pieces:
            if my_piece.is_hidden:
                continue
                
            for enemy_piece in enemy_pieces:
                if enemy_piece.is_hidden:
                    continue
                    
                # 检查是否可以攻击
                if self._can_capture(my_piece, enemy_piece, state):
                    outcome = self._simulate_attack(my_piece, enemy_piece)
                    opportunities.append({
                        'attacker': my_piece,
                        'target': enemy_piece,
                        'outcome': outcome,
                        'position': (my_piece.position, enemy_piece.position)
                    })
                    
        return opportunities
    
    def _can_capture(self, attacker: Piece, defender: Piece, state: GameState) -> bool:
        """判断是否能够吃掉对方"""
        if attacker.type == '工兵':
            return defender.type == '地雷'
        elif attacker.type == '炸弹':
            return True
        else:
            att_rank = self.config.PIECE_RANKS.get(attacker.type, 0)
            def_rank = self.config.PIECE_RANKS.get(defender.type, 0)
            return att_rank >= def_rank
    
    def _simulate_attack(self, attacker: Piece, defender: Piece) -> Dict:
        """模拟攻击结果"""
        att_rank = self.config.PIECE_RANKS.get(attacker.type, 0)
        def_rank = self.config.PIECE_RANKS.get(defender.type, 0)
        
        if attacker.type == '工兵' and defender.type == '地雷':
            return {'result': 'mine_dug', 'attacker_survives': True, 'defender_eliminated': True}
        elif attacker.type == '炸弹':
            return {'result': 'mutual_destruction', 'attacker_survives': False, 'defender_eliminated': True}
        elif att_rank > def_rank:
            return {'result': 'capture', 'attacker_survives': True, 'defender_eliminated': True}
        elif att_rank == def_rank:
            return {'result': 'draw', 'attacker_survives': False, 'defender_eliminated': False}
        else:
            return {'result': 'fail', 'attacker_survives': True, 'defender_eliminated': False}
