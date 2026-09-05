"""静态分析层：游戏阶段识别、子力差精确计算、死区/堡垒静态分析。

纯函数模块，只读 GameState，不修改任何现有逻辑。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .config import EvalWeights
from .rules import CAMPS, COMPOSITION, NEIGHBORS, Rank, other
from .state import GameState

# 阶段常量
PHASE_OPENING = 0
PHASE_MIDGAME = 1
PHASE_ENDGAME = 2

# 单方全子力最大值（司令100 + 军长90 + 师长75*2 + 旅长60*2 + 团长45*2 + 营长35*2 + 连长25*3 + 排长18*3 + 工兵42*3 + 炸弹52*2 + 地雷30*3 + 军旗50 = 1119）
MATERIAL_MAX = 1120.0
PIECE_VALUES = EvalWeights().piece


# ------------------------------------------------------------- 阶段识别

def hidden_count(state: GameState) -> int:
    """场上剩余暗子数量。"""
    return len(state.hidden_positions())


def camps_occupied(state: GameState) -> int:
    """CAMPS 中被棋子占据的格数（0~10）。"""
    return sum(1 for p in CAMPS if p in state.board)


def detect_phase(state: GameState) -> int:
    """阶段判定：
    - 暗子 >= 20：开局翻棋期 (PHASE_OPENING)
    - 6 <= 暗子 < 20 且 10 个行营未全被占：中期占营期 (PHASE_MIDGAME)
    - 其余（暗子 < 6，或 10 个行营全部有子）：尾盘期 (PHASE_ENDGAME)
    """
    h = hidden_count(state)
    c = camps_occupied(state)
    if h >= 20:
        return PHASE_OPENING
    if 6 <= h < 20 and c < 10:
        return PHASE_MIDGAME
    return PHASE_ENDGAME


# ------------------------------------------------------------- 子力差计算

def material_diff(state: GameState, seat: int) -> float:
    """己方存活子力 − 对方存活子力，归一化到 [-1.0, 1.0]。
    存活 = 棋盘明子 + 暗子池精确份额（来自 remaining_types()）。
    """
    my_color = state.my_color(seat)
    if my_color is None:
        return 0.0
    opp_color = other(my_color)

    my_mat = 0.0
    opp_mat = 0.0

    # 1. 棋盘明子
    for pc in state.board.values():
        if pc.revealed:
            v = PIECE_VALUES.get(pc.rank, 0.0)
            if pc.color == my_color:
                my_mat += v
            elif pc.color == opp_color:
                opp_mat += v

    # 2. 暗子池期望
    rem = state.remaining_types()
    for (clr, rk), count in rem.items():
        v = count * PIECE_VALUES.get(rk, 0.0)
        if clr == my_color:
            my_mat += v
        elif clr == opp_color:
            opp_mat += v

    diff = (my_mat - opp_mat) / MATERIAL_MAX
    return max(-1.0, min(1.0, float(diff)))


# ------------------------------------------------------------- 死区/堡垒分析

def fortress_score(state: GameState, seat: int) -> float:
    """评估 seat 方以军旗为核心的死区完备度，返回 [0.0, 1.0]。

    算法（BFS 可达性）：
    1. 若 seat 方军旗未翻开（场上无 revealed 己方 QI）-> 返回 0.0
    2. 定义永久墙（对方永远无法通过/拔除的格子）：
       a. 对方已翻开的明地雷（对方不能吃自己的子，非工兵攻雷攻方亡雷存，故为永久阻碍）；
       b. 己方已翻开的地雷，且对方工兵已全灭（remaining_types 中对方工兵为 0 且场上无对方明工兵）。
    3. 从所有对方明子与暗子占据格出发，沿 NEIGHBORS 公路邻接做 BFS，只能通过空格；
    4. 若军旗格不可达 -> 返回 1.0（死区已成）；
       否则返回软分：0.3 * (军旗四邻中永久墙及己方子数 / 军旗四邻总数)。
    """
    my_color = state.my_color(seat)
    if my_color is None:
        return 0.0
    opp_color = other(my_color)

    # 1. 寻找己方明军旗
    my_flag_pos = next(
        (p for p, pc in state.board.items()
         if pc.revealed and pc.color == my_color and pc.rank == Rank.QI),
        None
    )
    if my_flag_pos is None:
        return 0.0

    # 2. 判定对方工兵是否全灭
    rem = state.remaining_types()
    opp_hidden_eng = rem.get((opp_color, Rank.GONG), 0)
    opp_revealed_eng = sum(
        1 for pc in state.board.values()
        if pc.revealed and pc.color == opp_color and pc.rank == Rank.GONG
    )
    opp_engineers_alive = opp_hidden_eng + opp_revealed_eng

    # 永久墙判定
    perm_walls = set()
    for pos, pc in state.board.items():
        if pc.revealed:
            # 对方明地雷对对方自身是永久墙
            if pc.color == opp_color and pc.rank == Rank.LEI:
                perm_walls.add(pos)
            # 对方无工兵时，己方明地雷也是永久墙
            elif pc.color == my_color and pc.rank == Rank.LEI and opp_engineers_alive == 0:
                perm_walls.add(pos)

    # 3. 寻找所有敌方威胁出发源（对方明子与暗子）
    enemy_sources = [
        pos for pos, pc in state.board.items()
        if not pc.revealed or (pc.color == opp_color and pc.rank not in (Rank.LEI, Rank.QI))
    ]
    if not enemy_sources:
        return 1.0  # 对方完全无子可动

    # 4. 从 enemy_sources 沿空格做 BFS
    visited = set()
    queue = deque(enemy_sources)
    for p in enemy_sources:
        visited.add(p)

    flag_reached = False
    while queue:
        curr = queue.popleft()
        if curr == my_flag_pos:
            flag_reached = True
            break
        for nxt in NEIGHBORS[curr]:
            if nxt not in visited:
                # 只有当 nxt 是空格或者是目标军旗格时才可通行
                if nxt == my_flag_pos or nxt not in state.board:
                    visited.add(nxt)
                    queue.append(nxt)

    # 5. 结果评分
    if not flag_reached:
        return 1.0  # 死区已成

    # 未完全成死区时计算软分
    flag_adj = NEIGHBORS[my_flag_pos]
    if not flag_adj:
        return 0.0
    blocked_count = sum(
        1 for np in flag_adj
        if np in perm_walls or (state.board.get(np) is not None and state.board[np].revealed and state.board[np].color == my_color)
    )
    return round(0.3 * (blocked_count / len(flag_adj)), 3)
