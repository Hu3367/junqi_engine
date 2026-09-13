"""静态分析层：游戏阶段识别、子力差精确计算、死区/堡垒静态分析。

纯函数模块，只读 GameState，不修改任何现有逻辑。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .config import EvalWeights
from .rules import CAMPS, COMPOSITION, NEIGHBORS, Rank, other, is_camp
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

def _fortress_score_impl(state: GameState, seat: int) -> float:
    """fortress_score 原始实现（作为实例缓存的计算源）。

    评估 seat 方以军旗为核心的死区完备度，返回 [0.0, 1.0]。

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


def fortress_score(state: GameState, seat: int) -> float:
    """fortress_score 实例缓存包装：同一局面实例的 BFS 只算一次。

    evaluate_expert 每次调用需要 fortress(seat) - fortress(1-seat)，
    is_dead_draw 双向死区检查也各调用一次；热路径单局触发 ~63 万次，
    实例缓存后每个局面至多实算 2 次（双方各一）。
    缓存以座位颜色元组为守卫：首翻定色等 seat_color 变化自动失效。
    （board/dead 在构造后不可变，apply/翻棋几率节点均构造新实例。）"""
    seats = (state.seat_color.get(0), state.seat_color.get(1))
    cache = getattr(state, "_fortress_cache", None)
    if cache is not None and cache[0] == seats:
        val = cache[1].get(seat)
        if val is not None:
            return val
        val = _fortress_score_impl(state, seat)
        cache[1][seat] = val
        return val
    val = _fortress_score_impl(state, seat)
    state._fortress_cache = (seats, {seat: val})
    return val


# ------------------------------------------------------------- 理论必和死锁检测器

def is_dead_draw(state: GameState) -> tuple[bool, str]:
    """is_dead_draw 实例缓存包装：同一局面实例只实算一次。

    evaluate_expert 每次估值前置调用本函数（热路径单局 ~50 万次），
    结果仅取决于构造后不可变的 board/dead/quiet/cfg/seat_color，
    以 (座位色元组, quiet) 为守卫做实例缓存。
    """
    seats = (state.seat_color.get(0), state.seat_color.get(1))
    cache = getattr(state, "_dead_draw_cache", None)
    if cache is not None and cache[0] == seats and cache[1] == state.quiet:
        return cache[2]
    res = _is_dead_draw_impl(state)
    state._dead_draw_cache = (seats, state.quiet, res)
    return res


def _is_dead_draw_impl(state: GameState) -> tuple[bool, str]:
    """is_dead_draw 原始实现（作为实例缓存的计算源）。

    核心拓扑涵盖四大原型：
    1. 双无工兵死锁 (Zero Engineer Bilateral Lockout)
    2. 1v1 单大子追单小子死锁 (Single Chaser vs Camp/Corridor Fugitive)
    3. 双向军旗死区 (Bilateral Dead Fortress)
    4. 40步无吃子极限时钟逼近 (Near-Limit Quiet Moves)

    返回: (is_draw: bool, reason: str)
    """
    my_color = state.my_color()
    if my_color is None:
        return False, "color_not_assigned"
    opp_color = other(my_color)

    # 0. 规则层无吃子限步优先检测（达到 40 步限步直接判和）
    max_quiet = getattr(state.cfg, "no_capture_draw_plies", 40)
    if max_quiet > 0 and state.quiet >= max_quiet:
        return True, "quiet_moves_limit_reached"

    # 若场上仍有未翻开暗子，绝不轻易判定结构性死锁（翻棋永远拥有破局可能）
    hidden_pos = state.hidden_positions()
    if len(hidden_pos) > 0:
        return False, "too_many_hidden"

    dead_my = [p for p in state.dead if p.color == my_color]
    dead_opp = [p for p in state.dead if p.color == opp_color]

    my_gong_dead = sum(1 for p in dead_my if p.rank == Rank.GONG)
    opp_gong_dead = sum(1 for p in dead_opp if p.rank == Rank.GONG)

    my_gong_alive = COMPOSITION[Rank.GONG] - my_gong_dead
    opp_gong_alive = COMPOSITION[Rank.GONG] - opp_gong_dead

    # 双方棋盘现存明地雷
    my_mines = sum(1 for p in state.board.values() if p.revealed and p.color == my_color and p.rank == Rank.LEI)
    opp_mines = sum(1 for p in state.board.values() if p.revealed and p.color == opp_color and p.rank == Rank.LEI)

    # 存活可移动作战子力
    combat_my = [(pos, p) for pos, p in state.board.items()
                 if p.revealed and p.color == my_color and p.rank not in (Rank.QI, Rank.LEI)]
    combat_opp = [(pos, p) for pos, p in state.board.items()
                  if p.revealed and p.color == opp_color and p.rank not in (Rank.QI, Rank.LEI)]

    # -------------------------------------------------------------
    # 原型 1: 双无工兵死锁 (Zero Engineer Bilateral Deadlock)
    # -------------------------------------------------------------
    my_can_flag = True
    opp_can_flag = True

    # 规则：仅工兵能吃旗，或必须挖完地雷才能吃旗
    if state.cfg.flag_gong_only:
        if my_gong_alive == 0:
            my_can_flag = False
        if opp_gong_alive == 0:
            opp_can_flag = False

    if state.cfg.flag_needs_mines_cleared:
        if my_gong_alive == 0 and opp_mines > 0:
            my_can_flag = False
        if opp_gong_alive == 0 and my_mines > 0:
            opp_can_flag = False

    if not my_can_flag and not opp_can_flag:
        # 双方均永久失去吃旗可能！
        # 此时只能通过歼灭战（吃光所有可移动子）获胜
        if not combat_my and not combat_opp:
            return True, "both_sides_no_mobile_pieces"
        if not combat_my or not combat_opp:
            return False, "one_side_wiped"

        max_rank_my = max(p.rank for pos, p in combat_my)
        max_rank_opp = max(p.rank for pos, p in combat_opp)

        has_zha_my = any(p.rank == Rank.ZHA for pos, p in combat_my)
        has_zha_opp = any(p.rank == Rank.ZHA for pos, p in combat_opp)

        # 我方是否能消灭敌方最高战力？
        can_kill_opp_max = (max_rank_my >= max_rank_opp) or has_zha_my
        # 敌方是否能消灭我方最高战力？
        can_kill_my_max = (max_rank_opp >= max_rank_my) or has_zha_opp

        # A. 双方均无法消灭对方最高战力（例如双方仅剩完全对称的同级子力且无炸）
        if not can_kill_opp_max and not can_kill_my_max:
            ranks_my = sorted([p.rank for _, p in combat_my])
            ranks_opp = sorted([p.rank for _, p in combat_opp])
            if ranks_my == ranks_opp:
                return True, "bilateral_cannot_kill_supreme"

        # B. 一方有绝对无敌大子（如司令/师长），另一方无有效对抗子力
        # 严格机制死锁判定：
        # 1. 优势方必须仅有单单一颗压制大子（若有多颗大子如双师长，完全可以多路合围围剿）
        # 2. 弱势方不可有暴露在外的易俘子力（弱势方所有子力均已在行营内免死，或在极端长局 ply>=140 下至少2子在营对峙）
        long_standoff = (state.quiet >= 25 or state.ply >= 140)

        if not can_kill_opp_max:
            # 我方无法消灭对方大子 -> 我方绝无可能歼灭敌方
            opp_dominating = sum(1 for _, p in combat_opp if p.rank > max_rank_my)
            my_in_camps = sum(1 for pos, p in combat_my if is_camp(pos))
            all_in_camps = (my_in_camps == len(combat_my))
            if opp_dominating <= 1 and (all_in_camps or (my_in_camps >= 2 and long_standoff)):
                return True, "opp_has_invincible_si_but_cannot_annihilate"

        if not can_kill_my_max:
            # 对方无法消灭我方大子 -> 对方绝无可能歼灭我方
            my_dominating = sum(1 for _, p in combat_my if p.rank > max_rank_opp)
            opp_in_camps = sum(1 for pos, p in combat_opp if is_camp(pos))
            all_opp_in_camps = (opp_in_camps == len(combat_opp))
            if my_dominating <= 1 and (all_opp_in_camps or (opp_in_camps >= 2 and long_standoff)):
                return True, "my_has_invincible_si_but_cannot_annihilate"

    # -------------------------------------------------------------
    # 原型 2: 1v1 追逐死锁 (Single Combat Piece vs Single Combat Piece)
    # -------------------------------------------------------------
    if len(combat_my) == 1 and len(combat_opp) == 1 and len(hidden_pos) == 0 and len(state.dead) >= 30:
        pos_m, p_m = combat_my[0]
        pos_o, p_o = combat_opp[0]

        # 同级子力（如旅长对旅长）：若攻击则同归于尽
        if p_m.rank == p_o.rank:
            return True, "1v1_equal_rank_deadlock"

        # 劣势方与优势方
        superior_pos, inferior_pos = (pos_m, pos_o) if p_m.rank > p_o.rank else (pos_o, pos_m)

        # 劣势方身处行营中
        if is_camp(inferior_pos):
            return True, "1v1_inferior_in_camp"

        # 检查劣势方到最近行营的距离
        q = deque([(inferior_pos, 0)])
        visited = {inferior_pos}
        dist_to_camp = 999
        while q:
            curr, d = q.popleft()
            if is_camp(curr):
                dist_to_camp = d
                break
            for nxt in NEIGHBORS[curr]:
                if nxt not in visited and nxt != superior_pos:
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        if dist_to_camp <= 2:
            return True, "1v1_inferior_near_camp"

        # 底线往复走棋安全走廊 (row 0 或 row 11，且列在 1, 2, 3)
        if inferior_pos[0] in (0, 11) and inferior_pos[1] in (1, 2, 3):
            return True, "1v1_inferior_bottom_line_corridor"

    # -------------------------------------------------------------
    # 原型 3: 工兵灭绝下的地雷物理阻断子图 (Physical Partition Lockout)
    # -------------------------------------------------------------
    if not my_can_flag and not opp_can_flag and combat_my and combat_opp:
        # 当双方均无工兵且双方军旗皆不可达时，检查双方现存作战子力之间是否存在物理连通路径
        # （地雷与军旗为不可逾越的绝对障碍物）
        impassable = {pos for pos, p in state.board.items()
                      if p.revealed and p.rank in (Rank.LEI, Rank.QI)}
        q = deque([pos for pos, _ in combat_my])
        visited = set(q)
        can_reach_opp = False
        opp_positions = {pos for pos, _ in combat_opp}
        while q:
            curr = q.popleft()
            if curr in opp_positions:
                can_reach_opp = True
                break
            for nxt in NEIGHBORS[curr]:
                if nxt not in visited and nxt not in impassable:
                    visited.add(nxt)
                    q.append(nxt)
        if not can_reach_opp:
            return True, "mines_partition_board_disconnected"

    return False, "normal_play"

