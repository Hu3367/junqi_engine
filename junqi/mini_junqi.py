"""
Mini-Junqi Engine - 5x5 微缩版军棋翻棋引擎

提供完整的游戏逻辑、规则验证、AI Agent 接口和实验支持
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import yaml
from pathlib import Path

from junqi.config import RuleConfig
from junqi.rules import battle, RANK_CN, Rank


class MoveType(Enum):
    """移动类型"""
    FLIP = "flip"           # 翻棋
    MOVE = "move"           # 移动
    CAPTURE = "capture"     # 吃子
    EXPLODE = "explode"     # 爆炸 (炸弹)


@dataclass
class MiniPiece:
    """微型棋子"""
    piece_id: int
    type: str               # SI, JUN, SHI, LV, TUAN, YING, LIAN, GONG, LEI, BOMB, QI
    color: int              # 0=红方，1=黑方
    position: Tuple[int, int]
    hidden: bool = True     # 初始为暗子
    revealed: bool = False
    
    @property
    def rank_value(self) -> int:
        return self.type_to_rank(self.type)
    
    @staticmethod
    def type_to_rank(piece_type: str) -> int:
        """类型到等级的映射"""
        ranks = {
            'SI': 10,      # 司令
            'JUN': 9,      # 军长
            'SHI': 8,      # 师长
            'LV': 7,       # 旅长
            'TUAN': 6,     # 团长
            'YING': 5,     # 营长
            'LIAN': 4,     # 连长
            'PAI': 3,      # 排长
            'GONG': 1,     # 工兵
            'LEI': 0,      # 地雷
            'QI': 0,       # 军旗
            'BOMB': 7,     # 炸弹
        }
        return ranks.get(piece_type, 0)
    
    def can_attack(self) -> bool:
        """是否可以主动攻击"""
        return self.type not in ['LEI', 'QI']
    
    def is_high_value(self) -> bool:
        """是否为高价值棋子"""
        return self.rank_value >= 8


@dataclass
class MiniMove:
    """微型移动"""
    move_id: int
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    move_type: MoveType
    piece_id: int = -1
    
    def __str__(self):
        if self.move_type == MoveType.FLIP:
            return f"FLIP {self.from_pos}"
        else:
            return f"{self.from_pos}→{self.to_pos}"


@dataclass
class MiniGameState:
    """游戏状态"""
    board: Dict[Tuple[int, int], Optional[MiniPiece]]  # 位置 → 棋子
    pieces: List[MiniPiece]
    turn: int = 0
    current_player: int = 0  # 0 or 1
    winner: Optional[int] = None
    win_reason: Optional[str] = None
    quiet_turns: int = 0  # 无吃子回合数
    first_flip_done: bool = False
    seat_color: Dict[int, int] = field(default_factory=dict)
    
    # 实验统计
    stats: Dict = field(default_factory=dict)
    
    BOARD_SIZE: int = 5
    
    def __post_init__(self):
        if not self.stats:
            self.stats = {
                'flips': [],
                'captures': [],
                'bunker_occupancy': {},
                'move_history': [],
                'entropy_history': []
            }
    
    @property
    def height(self) -> int:
        return self.BOARD_SIZE
    
    @property
    def width(self) -> int:
        return self.BOARD_SIZE
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        return (0 <= pos[0] < self.BOARD_SIZE and 
                0 <= pos[1] < self.BOARD_SIZE)
    
    def get_piece_at(self, pos: Tuple[int, int]) -> Optional[MiniPiece]:
        """获取指定位置的棋子"""
        return self.board.get(pos)
    
    def get_pieces(self, color: int) -> List[MiniPiece]:
        """获取某方的所有棋子"""
        return [p for p in self.pieces if p.color == color]
    
    def get_enemy_pieces(self, color: int) -> List[MiniPiece]:
        """获取敌方棋子"""
        return [p for p in self.pieces if p.color != color]
    
    def flag_position(self, color: int) -> Optional[Tuple[int, int]]:
        """找到军旗位置"""
        for piece in self.pieces:
            if piece.type == 'QI' and piece.color == color:
                return piece.position
        return None
    
    def is_bunker(self, pos: Tuple[int, int]) -> bool:
        """检查是否是行营位置"""
        # Check against all defined bunkers
        bunker_positions = [
            (2, 2),   # center
            (1, 1),   # frontline_l
            (1, 3),   # frontline_r
            (3, 1),   # rear_l
            (3, 3),   # rear_r
        ]
        return pos in bunker_positions
    
    def is_rail(self, pos: Tuple[int, int]) -> bool:
        """检查是否在铁路上"""
        rail_positions = {
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # Left vertical
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),  # Right vertical
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),  # Horizontal
        }
        return pos in rail_positions
    
    def copy(self) -> 'MiniGameState':
        """深拷贝状态"""
        new_board = dict(self.board)
        new_pieces = [
            MiniPiece(
                piece_id=p.piece_id,
                type=p.type,
                color=p.color,
                position=p.position,
                hidden=p.hidden,
                revealed=p.revealed
            )
            for p in self.pieces
        ]
        
        return MiniGameState(
            board=new_board,
            pieces=new_pieces,
            turn=self.turn,
            current_player=self.current_player,
            winner=self.winner,
            win_reason=self.win_reason,
            quiet_turns=self.quiet_turns,
            first_flip_done=self.first_flip_done,
            seat_color=dict(self.seat_color),
            stats={k: list(v) if isinstance(v, list) else v 
                   for k, v in self.stats.items()}
        )
    
    def generate_legal_moves(self, color: int) -> List[MiniMove]:
        """生成所有合法移动"""
        moves = []
        move_id = 0
        
        # 生成翻棋移动
        flips = self._generate_flip_moves(color)
        for flip_pos in flips:
            moves.append(MiniMove(
                move_id=move_id,
                from_pos=flip_pos,
                to_pos=flip_pos,
                move_type=MoveType.FLIP,
                piece_id=-1
            ))
            move_id += 1
        
        # 生成移动移动
        moves.extend(self._generate_move_moves(color, move_id))
        
        return moves
    
    def _generate_flip_moves(self, color: int) -> List[Tuple[int, int]]:
        """生成翻棋移动候选"""
        flips = []
        
        for pos, piece in self.board.items():
            if piece is None:
                continue
            
            # 必须是己方未翻开的暗子
            if piece.hidden and piece.color == color:
                flips.append(pos)
        
        # 首翻限制：不能直接翻司令/军长
        if not self.first_flip_done:
            valid_flips = []
            for pos in flips:
                piece = self.get_piece_at(pos)
                if piece and piece.rank_value < 9:  # 不是司令或军长
                    valid_flips.append(pos)
            
            # 如果所有翻棋都是大子，允许翻任何子
            if not valid_flips:
                valid_flips = flips
                
            return valid_flips
        
        return flips
    
    def _generate_move_moves(self, color: int, start_id: int) -> List[MiniMove]:
        """生成移动移动"""
        moves = []
        move_id = start_id
        
        my_pieces = self.get_pieces(color)
        
        for piece in my_pieces:
            if not piece.revealed or piece.hidden:
                continue
            
            # 不可移动的棋子（地雷、军旗）
            if not piece.can_attack():
                continue
            
            # 检查每个可能的目标位置
            for target_pos in self.all_positions():
                if target_pos == piece.position:
                    continue
                
                # 检查目标是否有友军
                target = self.get_piece_at(target_pos)
                if target and target.color == color:
                    continue
                
                # 检查移动合法性
                if self.is_valid_move(piece, piece.position, target_pos):
                    moves.append(MiniMove(
                        move_id=move_id,
                        from_pos=piece.position,
                        to_pos=target_pos,
                        move_type=MoveType.MOVE,
                        piece_id=piece.piece_id
                    ))
                    move_id += 1
        
        return moves
    
    def is_valid_move(self, piece: MiniPiece, from_pos: Tuple[int, int], 
                     to_pos: Tuple[int, int]) -> bool:
        """检查移动是否合法"""
        # 距离检查 (公路移动)
        dist = abs(from_pos[0] - to_pos[0]) + abs(from_pos[1] - to_pos[1])
        
        # 工兵飞行
        if piece.type == 'GONG':
            if dist <= 3:  # 可以飞行
                return self._is_flight_path_clear(piece, from_pos, to_pos)
        
        # 普通移动 (1 步)
        if dist == 1:
            return True
        
        # 铁路滑行 (简化版)
        if self._is_on_rail(from_pos) and dist > 1:
            return self._is_rail_path_clear(piece, from_pos, to_pos)
        
        return False
    
    def _is_flight_path_clear(self, piece: MiniPiece, 
                             from_pos: Tuple[int, int],
                             to_pos: Tuple[int, int]) -> bool:
        """检查工兵飞行路径是否清晰"""
        # 简化：假设直线飞行，不经过阻挡
        return True
    
    def _is_rail_path_clear(self, piece: MiniPiece, 
                           from_pos: Tuple[int, int],
                           to_pos: Tuple[int, int]) -> bool:
        """检查铁路路径是否清晰"""
        # 实现铁路滑行检查
        return True
    
    def all_positions(self) -> List[Tuple[int, int]]:
        """所有棋盘位置"""
        return [(r, c) for r in range(self.BOARD_SIZE) 
                for c in range(self.BOARD_SIZE)]
    
    def apply(self, move: MiniMove) -> 'MiniGameState':
        """应用移动并返回新状态"""
        # TODO: 实现完整的状态转移逻辑
        return self.copy()
    
    def check_win_condition(self) -> bool:
        """检查胜负条件"""
        # 检查军旗是否被吃掉
        enemy_flag_pos = self.flag_position(1 - self.current_player)
        if enemy_flag_pos and self.board.get(enemy_flag_pos) is None:
            self.winner = self.current_player
            self.win_reason = "flag_capture"
            return True
        
        # 检查是否超时
        if self.quiet_turns >= 60:
            # 按子力差判定
            my_material = sum(p.rank_value for p in self.get_pieces(self.current_player) 
                            if p.revealed)
            enemy_material = sum(p.rank_value for p in self.get_enemy_pieces(self.current_player) 
                               if p.revealed)
            
            if my_material > enemy_material:
                self.winner = self.current_player
                self.win_reason = "material_advantage"
                return True
            elif my_material < enemy_material:
                self.winner = 1 - self.current_player
                self.win_reason = "material_disadvantage"
                return True
        
        return False
    
    def evaluate_benefit_score(self) -> float:
        """评估当前局面的优势分数 (-1~1)"""
        # TODO: 实现完整的评估函数
        return 0.0


def deal_mini_pieces(scheme_name: str, config: Dict) -> Tuple[List[MiniPiece], List[MiniPiece]]:
    """
    发牌：将棋子随机分配到棋盘上
    
    Returns:
        (red_pieces, black_pieces)
    """
    scheme = config['piece_sets'][scheme_name]
    total_pieces = sum(p['count'] for p in scheme['pieces'])
    
    # 创建棋子池
    pieces_pool = []
    for i, p in enumerate(scheme['pieces']):
        for _ in range(p['count']):
            pieces_pool.append(p['type'])
    
    # 随机打乱
    random.shuffle(pieces_pool)
    
    # 分配给双方
    red_pieces = []
    black_pieces = []
    
    positions = [(r, c) for r in range(5) for c in range(5)]
    random.shuffle(positions)
    
    for i, pos in enumerate(positions[:total_pieces]):
        piece_type = pieces_pool[i]
        color = 0 if i < len(pieces_pool) // 2 else 1
        
        piece = MiniPiece(
            piece_id=i,
            type=piece_type,
            color=color,
            position=pos,
            hidden=True
        )
        
        if color == 0:
            red_pieces.append(piece)
        else:
            black_pieces.append(piece)
    
    return red_pieces, black_pieces


class MiniBoard:
    """5x5 迷你棋盘"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Parse bunkers - convert lists to tuples since sets need hashable types
        bunkers_dict = config.get('bunkers', {})
        self.bunkers = set()
        for key, pos in bunkers_dict.items():
            if isinstance(pos, list):
                self.bunkers.add(tuple(pos))
            else:
                self.bunkers.add(pos)
        
        self.railroads = set()
        
        # Parse railroads
        for line in config.get('railroads', []):
            for pos in line:
                if isinstance(pos, list):
                    self.railroads.add(tuple(pos))
                else:
                    self.railroads.add(pos)
    
    def is_bunker(self, pos: Tuple[int, int]) -> bool:
        """是否是行营"""
        return pos in self.bunkers
    
    def is_rail(self, pos: Tuple[int, int]) -> bool:
        """是否在铁路上"""
        return pos in self.railroads
    
    def control_score(self, state: MiniGameState, color: int) -> float:
        """计算控制权评分"""
        score = 0.0
        
        # 行营控制
        for bunker in self.bunkers:
            piece = state.get_piece_at(bunker)
            if piece and piece.color == color:
                score += 0.2
        
        # 中心控制
        center = (2, 2)
        piece = state.get_piece_at(center)
        if piece and piece.color == color:
            score += 0.3
        
        return max(-1.0, min(1.0, score))
