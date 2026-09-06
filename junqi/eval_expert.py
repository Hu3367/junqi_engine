"""专家级军棋翻棋评估函数 (Expert Evaluation for Junqi Fanqi)。

参照计算机暗棋 (Chinese Dark Chess / 半棋) 顶级博弈专家的特征工程体系：
1. 动态子力与制霸矩阵 (Dynamic Dominance): 司令/军长/炸弹/工兵/地雷生存状态联动
2. 棋子机动力与死子惩罚 (Mobility & Trapped Analysis)
3. 行营战术支点与围杀控制 (Camp Tactical Dominance)
4. 暗子局部攻防安全指数 (Flip Safety Index)
5. 死区势能与和棋控制 (Fortress & Draw Energy)
6. 攻防时差与节奏控制 (Tempo Differential)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .analysis import fortress_score, is_dead_draw
from .config import EvalWeights
from .rules import (ATTACKER_WINS, BOTH_DIE, CAMPS, COMPOSITION, HQS,
                    NEIGHBORS, PLAY_POSITIONS, RANK_CN, Rank, battle,
                    is_camp, is_hq, is_rail, other)

if TYPE_CHECKING:
    from .state import GameState, Piece


# 基础子力标称分
DEFAULT_PIECE_VALUES: dict[Rank, float] = {
    Rank.SI: 100.0,
    Rank.JUN: 90.0,
    Rank.SHI: 75.0,
    Rank.LV: 60.0,
    Rank.TUAN: 45.0,
    Rank.YING: 35.0,
    Rank.LIAN: 25.0,
    Rank.PAI: 18.0,
    Rank.GONG: 42.0,
    Rank.ZHA: 52.0,
    Rank.LEI: 30.0,
    Rank.QI: 50.0,
}


def _get_alive_counts(state: GameState, my: Optional[str] = None) -> tuple[dict[tuple[str, Rank], int], dict[tuple[str, Rank], int]]:
    """统计双方当前存活总数（明子 + 暗子池期望份额 = 初始编制 - 阵亡子）。

    返回: (my_alive, opp_alive) 映射 (color, rank) -> count
    """
    if my is None:
        my = state.my_color()
    opp = other(my) if my else None

    my_counts: dict[tuple[str, Rank], int] = {(my, rk): COMPOSITION[rk] for rk in COMPOSITION} if my else {}
    opp_counts: dict[tuple[str, Rank], int] = {(opp, rk): COMPOSITION[rk] for rk in COMPOSITION} if opp else {}

    # 扣除阵亡子
    for pc in state.dead:
        if my and pc.color == my:
            my_counts[(pc.color, pc.rank)] -= 1
        elif opp and pc.color == opp:
            opp_counts[(pc.color, pc.rank)] -= 1

    return my_counts, opp_counts


def evaluate_expert(state: GameState, seat: int, w: Optional[EvalWeights] = None) -> float:
    """专家级公开信息估值函数（seat 视角）。

    严格遵守公共信息边界，综合子力动态制霸、机动力、行营、死区、威胁网络。
    """
    if w is None:
        w = EvalWeights()

    my = state.seat_color.get(seat)
    if my is None:
        # 首翻未定色时，公共局面完全对称
        return 0.0
    opp = other(my)

    # 0. 结构性必和死锁前置断言 (Dead Draw Assertion)
    # 若满足双无工兵死锁、1v1 追逐死锁或拓扑断绝，终局胜负期望严格为 0.0，杜绝虚假分值
    is_draw, _ = is_dead_draw(state)
    if is_draw:
        return 0.0

    # 1. 存活子力统计（用于动态制霸与物质分）
    my_counts, opp_counts = _get_alive_counts(state, my=my)

    revealed_mine: list[tuple[tuple[int, int], Piece]] = []
    revealed_opp: list[tuple[tuple[int, int], Piece]] = []

    for pos, pc in state.board.items():
        if pc.revealed:
            if pc.color == my:
                revealed_mine.append((pos, pc))
            else:
                revealed_opp.append((pos, pc))

    # 2. 动态制霸系数与物质估值
    # 2.1 司令/军长制霸 (Hegemony)
    opp_has_si = opp_counts.get((opp, Rank.SI), 0) > 0
    my_has_si = my_counts.get((my, Rank.SI), 0) > 0
    opp_has_jun = opp_counts.get((opp, Rank.JUN), 0) > 0
    my_has_jun = my_counts.get((my, Rank.JUN), 0) > 0
    opp_has_zha = opp_counts.get((opp, Rank.ZHA), 0) > 0
    my_has_zha = my_counts.get((my, Rank.ZHA), 0) > 0
    opp_has_gong = opp_counts.get((opp, Rank.GONG), 0) > 0
    my_has_gong = my_counts.get((my, Rank.GONG), 0) > 0

    # 动态子力价值加成 (以配置中的 piece 表为基准，兼顾默认线性表与 APK 等比表)
    my_piece_values = dict(w.piece if w and w.piece else DEFAULT_PIECE_VALUES)
    opp_piece_values = dict(w.piece if w and w.piece else DEFAULT_PIECE_VALUES)

    # 敌方无司令时，我方司令大幅增值且军长称霸
    if not opp_has_si and my_has_si:
        my_piece_values[Rank.SI] += 20.0  # 绝对王者
        if not opp_has_zha:
            my_piece_values[Rank.SI] += 15.0  # 敌无炸弹，司令无解
    if not opp_has_si and not opp_has_jun and my_has_jun:
        my_piece_values[Rank.JUN] += 15.0  # 军长称霸

    # 对称计算敌方制霸
    if not my_has_si and opp_has_si:
        opp_piece_values[Rank.SI] += 20.0
        if not my_has_zha:
            opp_piece_values[Rank.SI] += 15.0
    if not my_has_si and not my_has_jun and opp_has_jun:
        opp_piece_values[Rank.JUN] += 15.0

    # 2.2 二线梯队火力网接管与补偿 (2026-09-06 实证：司令先死逆转胜 50.2%，胜负均等)
    # 当司令阵亡，但拥有盘面已就位参战的军长、师长或炸弹时，二线火力网健全度可对冲单司令制霸劣势
    my_rev_jun = any(pc.rank == Rank.JUN for _, pc in revealed_mine)
    my_rev_shi = sum(1 for _, pc in revealed_mine if pc.rank == Rank.SHI)
    my_rev_zha = any(pc.rank == Rank.ZHA for _, pc in revealed_mine)
    my_echelon = 0.0
    if not my_has_si and opp_has_si:
        echelon_units = (
            (1.0 if my_rev_jun else 0.0) +
            (0.5 * min(2, my_rev_shi)) +
            (0.8 if my_rev_zha else 0.0)
        )
        my_echelon = min(1.0, echelon_units / 2.0) * getattr(w, "echelon_si_compensation", 18.0)

    opp_rev_jun = any(pc.rank == Rank.JUN for _, pc in revealed_opp)
    opp_rev_shi = sum(1 for _, pc in revealed_opp if pc.rank == Rank.SHI)
    opp_rev_zha = any(pc.rank == Rank.ZHA for _, pc in revealed_opp)
    opp_echelon = 0.0
    if not opp_has_si and my_has_si:
        opp_echelon_units = (
            (1.0 if opp_rev_jun else 0.0) +
            (0.5 * min(2, opp_rev_shi)) +
            (0.8 if opp_rev_zha else 0.0)
        )
        opp_echelon = min(1.0, opp_echelon_units / 2.0) * getattr(w, "echelon_si_compensation", 18.0)

    # 敌方工兵全灭时：我方地雷与军旗安全系数飙升（敌方无法挖雷吃旗）
    if not opp_has_gong and my_has_gong:
        my_piece_values[Rank.LEI] += 15.0
        my_piece_values[Rank.QI] += 25.0
    elif not my_has_gong and opp_has_gong:
        opp_piece_values[Rank.LEI] += 15.0
        opp_piece_values[Rank.QI] += 25.0
    elif not my_has_gong and not opp_has_gong:
        # 双无工兵：地雷与军旗无法移动也无法被拔，纯属死棋，不计入机动进攻物质分
        my_piece_values[Rank.LEI] = 10.0
        opp_piece_values[Rank.LEI] = 10.0
        my_piece_values[Rank.QI] = 10.0
        opp_piece_values[Rank.QI] = 10.0

    # 炸弹联动价值：支持原版 APK 动态定价公式 (0x600ca) 或传统定额加成
    if getattr(w, "use_dynamic_bomb", False):
        ratio = getattr(w, "bomb_ratio", 1.0 / 3.0)
        opp_combat_ranks = [rk for (clr, rk), cnt in opp_counts.items()
                            if cnt > 0 and rk not in (Rank.LEI, Rank.QI, Rank.ZHA)]
        if opp_combat_ranks:
            opp_max_rk = max(opp_combat_ranks, key=lambda r: opp_piece_values[r])
            my_piece_values[Rank.ZHA] = opp_piece_values[opp_max_rk] * ratio
        else:
            my_piece_values[Rank.ZHA] = opp_piece_values.get(Rank.PAI, 30.0) * ratio

        my_combat_ranks = [rk for (clr, rk), cnt in my_counts.items()
                           if cnt > 0 and rk not in (Rank.LEI, Rank.QI, Rank.ZHA)]
        if my_combat_ranks:
            my_max_rk = max(my_combat_ranks, key=lambda r: my_piece_values[r])
            opp_piece_values[Rank.ZHA] = my_piece_values[my_max_rk] * ratio
        else:
            opp_piece_values[Rank.ZHA] = my_piece_values.get(Rank.PAI, 30.0) * ratio
    else:
        if opp_has_si or opp_has_jun:
            my_piece_values[Rank.ZHA] += 10.0
        if my_has_si or my_has_jun:
            opp_piece_values[Rank.ZHA] += 10.0

    # 计算存活物质总分 (棋盘明子 + 暗子池期望份额)
    my_material = sum(my_piece_values[rk] * cnt for (clr, rk), cnt in my_counts.items())
    opp_material = sum(opp_piece_values[rk] * cnt for (clr, rk), cnt in opp_counts.items())
    score = my_material - opp_material + (my_echelon - opp_echelon)

    # 3. 明子位置与阵型结构特征


    # 3.1 行营控制与营内围杀
    my_camps = 0
    opp_camps = 0
    for pos, pc in revealed_mine:
        if is_camp(pos):
            my_camps += 1
            score += w.camp_occ
            # 行营围杀压力：营内子对能击杀或兑掉的邻接敌明子施加围杀压力 (杜绝小子在营里对大子产生假围杀加分)
            siege = sum(1 for np in NEIGHBORS[pos]
                        if (e := state.board.get(np)) is not None and e.revealed and e.color == opp
                        and battle(pc.rank, e.rank) in (ATTACKER_WINS, BOTH_DIE))
            score += w.camp_siege * siege
        if is_hq(pos) and pc.rank != Rank.QI:
            score += w.hq_locked

    for pos, pc in revealed_opp:
        if is_camp(pos):
            opp_camps += 1
            score -= w.camp_occ
            siege = sum(1 for np in NEIGHBORS[pos]
                        if (e := state.board.get(np)) is not None and e.revealed and e.color == my
                        and battle(pc.rank, e.rank) in (ATTACKER_WINS, BOTH_DIE))
            score -= w.camp_siege * siege
        if is_hq(pos) and pc.rank != Rank.QI:
            score -= w.hq_locked

    # 3.1.1 占营比例非线性矩阵增益 (2026-09-06 实证：5:5 38.2% -> 6:4 52.1% -> 7:3 68.4% -> 8:2 83.3%)
    # 结构性矩阵优势要求净胜至少 2 营（对应 6:4 优势），避免单营出现阶跃失真
    net_camps = my_camps - opp_camps
    if abs(net_camps) >= 2:
        sign = 1.0 if net_camps > 0 else -1.0
        k = min(abs(net_camps), 6)
        camp_factor = {2: 1.0, 3: 1.5, 4: 2.2, 5: 2.6, 6: 3.0}.get(k, 3.0)
        score += sign * camp_factor * getattr(w, "camp_matrix_weight", 12.0)

    # 3.2 行营势力范围 (camp_zone)
    zone_net = 0
    for cp in CAMPS:
        if cp in state.board:
            continue
        for np_ in NEIGHBORS[cp]:
            e = state.board.get(np_)
            if e is not None and e.revealed and e.rank not in (Rank.LEI, Rank.QI):
                zone_net += 1 if e.color == my else -1
    score += w.camp_zone * zone_net

    # 4. 棋子机动力与死子/受阻惩罚 (Mobility & Blocked Penalties)
    my_mobility = 0
    opp_mobility = 0
    for pos, pc in revealed_mine:
        if pc.rank in (Rank.LEI, Rank.QI):
            continue
        moves = 0
        for np in NEIGHBORS[pos]:
            t = state.board.get(np)
            if t is None or (t.revealed and t.color == opp and not is_camp(np)):
                moves += 1
        if moves == 0 and not is_camp(pos):
            # 死子惩罚：大官动弹不得极度危险
            score -= 8.0 if pc.rank >= Rank.SHI else 4.0
        my_mobility += moves
        if is_rail(pos):
            my_mobility += 1  # 铁路畅通加分

    for pos, pc in revealed_opp:
        if pc.rank in (Rank.LEI, Rank.QI):
            continue
        moves = 0
        for np in NEIGHBORS[pos]:
            t = state.board.get(np)
            if t is None or (t.revealed and t.color == my and not is_camp(np)):
                moves += 1
        if moves == 0 and not is_camp(pos):
            score += 8.0 if pc.rank >= Rank.SHI else 4.0
        opp_mobility += moves
        if is_rail(pos):
            opp_mobility += 1

    score += 0.5 * (my_mobility - opp_mobility)

    # 5. 战术攻防网络 (Tactical Threats & Exposure)
    # 5.1 军旗暴露检查
    def can_take_flag(flag_color: str, attacker_rank: Rank) -> bool:
        if attacker_rank == Rank.LEI:
            return False
        if state.cfg.flag_gong_only and attacker_rank != Rank.GONG:
            return False
        if state.cfg.flag_needs_mines_cleared:
            mines_left = COMPOSITION[Rank.LEI] - sum(
                1 for d in state.dead
                if d.color == flag_color and d.rank == Rank.LEI)
            if mines_left > 0:
                return False
        return True

    my_flag = next(((p, pc) for p, pc in revealed_mine if pc.rank == Rank.QI), None)
    opp_flag = next(((p, pc) for p, pc in revealed_opp if pc.rank == Rank.QI), None)

    if my_flag:
        flag_pos = my_flag[0]
        if any(flag_pos in NEIGHBORS[ap] and can_take_flag(my, e.rank) for ap, e in revealed_opp):
            score -= w.flag_exposed * 1.5
    if opp_flag:
        flag_pos = opp_flag[0]
        if any(flag_pos in NEIGHBORS[ap] and can_take_flag(opp, m.rank) for ap, m in revealed_mine):
            score += w.flag_exposed * 1.5

    # 5.1.1 地雷护旗阵地加分 (对齐原版 APK 0x124094 地雷护旗额外 +80 分)
    guard_bonus = getattr(w, "mine_flag_guard_bonus", 0.0)
    if guard_bonus > 0.0:
        if my_flag:
            flag_pos = my_flag[0]
            my_guard_mines = sum(1 for pos, pc in revealed_mine if pc.rank == Rank.LEI and pos in NEIGHBORS[flag_pos])
            score += guard_bonus * my_guard_mines
        if opp_flag:
            flag_pos = opp_flag[0]
            opp_guard_mines = sum(1 for pos, pc in revealed_opp if pc.rank == Rank.LEI and pos in NEIGHBORS[flag_pos])
            score -= guard_bonus * opp_guard_mines

    # 5.2 相邻吃子威胁
    for pos, e in revealed_opp:
        if e.rank == Rank.QI:
            continue
        best_gain = 0.0
        for np in NEIGHBORS[pos]:
            if is_camp(np):
                continue
            m = state.board.get(np)
            if m is None or not m.revealed or m.color != my or m.rank == Rank.QI:
                continue
            res = battle(e.rank, m.rank)
            v = my_piece_values[m.rank]
            if res == ATTACKER_WINS:
                best_gain = max(best_gain, v)
            elif res == BOTH_DIE:
                best_gain = max(best_gain, v * 0.5)
        score -= (w.attack_camp if is_camp(pos) else w.threat) * best_gain

    for pos, m in revealed_mine:
        if m.rank == Rank.QI:
            continue
        best_gain = 0.0
        for np in NEIGHBORS[pos]:
            if is_camp(np):
                continue
            e = state.board.get(np)
            if e is None or not e.revealed or e.color != opp or e.rank == Rank.QI:
                continue
            res = battle(m.rank, e.rank)
            v = opp_piece_values[e.rank]
            if res == ATTACKER_WINS:
                best_gain = max(best_gain, v)
            elif res == BOTH_DIE:
                best_gain = max(best_gain, v * 0.5)
        score += (w.attack_camp if is_camp(pos) else w.attack) * best_gain

    # 6. 死区势能 (Fortress Score，仅当有明军旗暴露时计算死区)
    if w.fortress > 0 and (my_flag or opp_flag):
        fs_my = fortress_score(state, seat) if my_flag else 0.0
        fs_opp = fortress_score(state, 1 - seat) if opp_flag else 0.0
        score += w.fortress * (fs_my - fs_opp)

    # 6.1 残局工兵期权与和棋死锁折现 (2026-09-06 实证：65.4% 和棋局工兵残缺<=2颗)
    # 当全盘工兵残缺，且双方军旗均处于地雷保护下时，进攻期权消失，估值向和棋(0.0)折现收敛
    total_gong = my_counts.get((my, Rank.GONG), 0) + opp_counts.get((opp, Rank.GONG), 0)
    if total_gong <= 2 and state.ply > 60:
        my_mines = sum(1 for d in state.board.values() if d.revealed and d.color == my and d.rank == Rank.LEI)
        opp_mines = sum(1 for d in state.board.values() if d.revealed and d.color == opp and d.rank == Rank.LEI)
        if my_mines > 0 and opp_mines > 0:
            damping = 0.75 if total_gong <= 1 else 0.85
            score *= damping

    # 6.2 70 步无吃子限步时钟衰减
    # 当连续多手未吃子且逐步逼近判和时限（70步）时，非吃旗性物质优势随时间按二次方强力衰减归零
    limit_quiet = getattr(state.cfg, "no_capture_draw_plies", 70)
    if limit_quiet > 0 and state.quiet >= 20:
        progress = min(1.0, state.quiet / limit_quiet)
        score *= max(0.05, (1.0 - progress) ** 2)

    # 7. 暗子时差与节奏 (Hidden Tempo)
    my_active = sum(1 for p, pc in revealed_mine if pc.rank not in (Rank.LEI, Rank.QI))
    opp_active = sum(1 for p, pc in revealed_opp if pc.rank not in (Rank.LEI, Rank.QI))
    score += w.hidden_tempo * (my_active - opp_active)

    return score
