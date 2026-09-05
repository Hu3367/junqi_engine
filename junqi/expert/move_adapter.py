"""
Move 适配器 - 将 Action 转换为 Move 格式供 Expert Engine 使用

由于原项目使用 Action 而非 Move，这里提供兼容性层
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

# 定义 MoveType 枚举（与原项目的 Action kind 对应）
class MoveType(Enum):
    FLIP = "flip"
    MOVE = "move"
    CAPTURE = "capture"
    EXPLODE = "explode"

@dataclass
class Move:
    """
    Move 数据结构（适配 Expert Engine）
    
    基于原项目的 Action 创建
    """
    piece_id: int
    from_pos: Tuple[int, int]
    to_pos: Optional[Tuple[int, int]]
    move_type: Optional[MoveType] = None
    
    # 额外信息（来自 Action）
    action_kind: str = ""  # 'flip' or 'move'
    
    def __str__(self):
        if self.move_type == MoveType.FLIP:
            return f"FLIP {self.from_pos}"
        elif self.to_pos:
            return f"{self.from_pos}->{self.to_pos}"
        else:
            return f"MOVE {self.from_pos}"

def create_move_from_action(action, piece_id: int) -> Move:
    """从 Action 创建 Move"""
    return Move(
        piece_id=piece_id,
        from_pos=action.frm,
        to_pos=action.to if hasattr(action, 'to') else None,
        move_type=MoveType(action.kind) if hasattr(action, 'kind') else None,
        action_kind=action.kind if hasattr(action, 'kind') else ""
    )

def moves_to_actions(moves: list) -> list:
    """将 Move 列表转换回 Action 列表"""
    actions = []
    for m in moves:
        kind = m.action_kind or (m.move_type.value if m.move_type else "move")
        # Create a simple Action-like object
        class SimpleAction:
            def __init__(self, k, f, t):
                self.kind = k
                self.frm = f
                self.to = t
        actions.append(SimpleAction(kind, m.from_pos, m.to_pos))
    return actions
