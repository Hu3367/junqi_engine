"""状态张量编码与动作编解码模块（V2 升级版：36 通道双模式编码）。

支持：
1. 公共模式 (world=None)：真实玩家视角，暗子仅可见掩码；
2. 世界模式 (world={pos: Piece})：MCTS 采样世界视角，暗子按指派身份并入明子通道；
3. 3650 维动作空间与合法动作掩码生成。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .analysis import (camps_occupied, detect_phase, fortress_score,
                       hidden_count, material_diff)
from .rules import (CAMPS, COMPOSITION, HQS, PLAY_POSITIONS, ROWS, COLS,
                    Rank, is_camp, is_hq, is_rail, other)
from .state import Action, GameState, Piece

# 动作空间大小：50 (翻暗子) + 60*60 (移动起点到终点) = 3650
ACTION_SPACE_SIZE = 3650
NUM_CHANNELS = 38

# 军衔固定顺序
RANK_LIST = (
    Rank.SI, Rank.JUN, Rank.SHI, Rank.LV, Rank.TUAN, Rank.YING,
    Rank.LIAN, Rank.PAI, Rank.GONG, Rank.ZHA, Rank.LEI, Rank.QI,
)
RANK_TO_IDX = {r: i for i, r in enumerate(RANK_LIST)}
PLAY_POS_TO_IDX = {pos: i for i, pos in enumerate(PLAY_POSITIONS)}


# ------------------------------------------------------------- 动作编解码

def action_to_index(act: Action) -> int:
    """将 Action 编码为 0 ~ 3649 的唯一离散整数索引。"""
    if act.kind == "flip":
        idx = PLAY_POS_TO_IDX.get(act.frm)
        if idx is None:
            raise ValueError(f"非法翻子位置: {act.frm}")
        return idx
    else:
        frm_idx = act.frm[0] * COLS + act.frm[1]
        to_idx = act.to[0] * COLS + act.to[1]
        return 50 + frm_idx * (ROWS * COLS) + to_idx


def index_to_action(idx: int) -> Action:
    """将离散动作索引还原为 Action。"""
    if idx < 0 or idx >= ACTION_SPACE_SIZE:
        raise ValueError(f"动作索引超出范围: {idx}")
    if idx < 50:
        return Action("flip", PLAY_POSITIONS[idx])
    rem = idx - 50
    frm_idx = rem // (ROWS * COLS)
    to_idx = rem % (ROWS * COLS)
    frm = (frm_idx // COLS, frm_idx % COLS)
    to = (to_idx // COLS, to_idx % COLS)
    return Action("move", frm, to)


def legal_action_mask(state: GameState) -> np.ndarray:
    """生成长度为 ACTION_SPACE_SIZE 的布尔掩码（合法动作为 True，非法为 False）。"""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    acts = state.legal_actions()
    for a in acts:
        idx = action_to_index(a)
        mask[idx] = True
    return mask


# ------------------------------------------------------------- 36 通道状态空间张量化

def encode_state_np(state: GameState, seat: int | None = None,
                    world: Optional[Dict[tuple, Piece]] = None,
                    history_counts: Optional[Dict[tuple, int]] = None) -> np.ndarray:
    """将 GameState 转化为 shape 为 (38, 12, 5) 的 float32 numpy 数组。
    - world is None -> 公共模式（玩家真实视角）
    - world 给定    -> 世界模式（暗子身份并入明子通道）
    - history_counts: 历史局面出现频次字典，用于生成重复计数平面
    """
    if seat is None:
        seat = state.turn
    my_color = state.my_color(seat)
    opp_color = other(my_color) if my_color else None
    other_seat = 1 - seat

    tensor = np.zeros((NUM_CHANNELS, ROWS, COLS), dtype=np.float32)
    is_world_mode = world is not None

    # 通道 0~11: 己方明子 (12种军衔)
    # 通道 12~23: 敌方明子 (12种军衔)
    # 通道 24: 暗子掩码 (世界模式下为 0)
    for pos, pc in state.board.items():
        r, c = pos
        if pc.revealed:
            target_pc = pc
        elif is_world_mode and world and pos in world:
            target_pc = world[pos]
        else:
            tensor[24, r, c] = 1.0
            continue

        # 将 target_pc 写入对应通道
        if my_color is not None and target_pc.color == my_color:
            ch = RANK_TO_IDX[target_pc.rank]
            tensor[ch, r, c] = 1.0
        elif opp_color is not None and target_pc.color == opp_color:
            ch = 12 + RANK_TO_IDX[target_pc.rank]
            tensor[ch, r, c] = 1.0
        else:
            # 尚未定色时按绝对颜色 'r'=0, 'b'=12 编码
            ch = (0 if target_pc.color == "r" else 12) + RANK_TO_IDX[target_pc.rank]
            tensor[ch, r, c] = 1.0

    # 通道 25: 子力差平面 (广播)
    tensor[25, :, :] = material_diff(state, seat)

    # 通道 26: 己方死区完备度 (广播)
    tensor[26, :, :] = fortress_score(state, seat)

    # 通道 27: 对方死区完备度 (广播)
    tensor[27, :, :] = fortress_score(state, other_seat)

    # 通道 28: 铁路网拓扑
    for r in range(ROWS):
        for c in range(COLS):
            if is_rail((r, c)):
                tensor[28, r, c] = 1.0

    # 通道 29: 行营 (1.0) 与大本营 (0.5) 掩码
    for r, c in CAMPS:
        tensor[29, r, c] = 1.0
    for r, c in HQS:
        tensor[29, r, c] = 0.5

    # 通道 30: 工兵与地雷存活全局状态
    if my_color and opp_color:
        my_dead_gong = sum(1 for d in state.dead if d.color == my_color and d.rank == Rank.GONG)
        opp_dead_gong = sum(1 for d in state.dead if d.color == opp_color and d.rank == Rank.GONG)
        my_dead_lei = sum(1 for d in state.dead if d.color == my_color and d.rank == Rank.LEI)
        opp_dead_lei = sum(1 for d in state.dead if d.color == opp_color and d.rank == Rank.LEI)

        my_gong_ratio = (COMPOSITION[Rank.GONG] - my_dead_gong) / COMPOSITION[Rank.GONG]
        opp_gong_ratio = (COMPOSITION[Rank.GONG] - opp_dead_gong) / COMPOSITION[Rank.GONG]
        my_lei_ratio = (COMPOSITION[Rank.LEI] - my_dead_lei) / COMPOSITION[Rank.LEI]
        opp_lei_ratio = (COMPOSITION[Rank.LEI] - opp_dead_lei) / COMPOSITION[Rank.LEI]

        tensor[30, 0:6, :] = (my_gong_ratio + my_lei_ratio) * 0.5
        tensor[30, 6:12, :] = (opp_gong_ratio + opp_lei_ratio) * 0.5

    # 通道 31: 连续无吃子倒计时 quiet / 70.0
    max_quiet = state.cfg.no_capture_draw_plies or 70
    tensor[31, :, :] = min(1.0, state.quiet / max_quiet)

    # 通道 32: 总步数进度 ply / 1000.0
    max_plies = state.cfg.max_plies or 1000
    tensor[32, :, :] = min(1.0, state.ply / max_plies)

    # 通道 33: 暗子数量 hidden_count / 50.0
    tensor[33, :, :] = hidden_count(state) / 50.0

    # 通道 34: 阶段标号 phase / 2.0 (0.0=开局, 0.5=中盘, 1.0=尾盘)
    tensor[34, :, :] = detect_phase(state) / 2.0

    # 通道 35: 模式标志 (世界模式=1.0, 公共模式=0.0)
    tensor[35, :, :] = 1.0 if is_world_mode else 0.0

    # 通道 36: 当前局面出现过至少 1 次 (seen >= 1)
    # 通道 37: 当前局面出现过至少 2 次 (seen >= 2，再出现第 3 次将直接触发判和)
    if history_counts:
        from .state import position_key
        k = position_key(state)
        cnt = history_counts.get(k, 1)
        if cnt >= 1:
            tensor[36, :, :] = 1.0
        if cnt >= 2:
            tensor[37, :, :] = 1.0
    else:
        tensor[36, :, :] = 1.0

    return tensor


def encode_state(state: GameState, seat: int | None = None,
                 world: Optional[Dict[tuple, Piece]] = None,
                 history_counts: Optional[Dict[tuple, int]] = None,
                 device: torch.device | str = "cpu") -> torch.Tensor:
    """将 GameState 转化为 Tensor [1, 38, 12, 5]。"""
    arr = encode_state_np(state, seat=seat, world=world, history_counts=history_counts)
    t = torch.from_numpy(arr).unsqueeze(0).to(device)
    return t
