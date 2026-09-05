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

from .analysis import fortress_score
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


def _get_alive_counts(state: GameState) -> tuple[dict[tuple[str, Rank], int], dict[tuple[str, Rank], int]]:
    """统计双方当前存活总数（明子 + 暗子池期望份额）。

    返回: (my_alive, opp_alive) 映射 (color, rank) -> count
    """
    my = state.my_color()
    opp = other(my) if my else None

    # 从 remaining_types 获取暗子池
    rem = state.remaining_types()
    my_counts: dict[tuple[str, Rank], int] = {}
    opp_counts: dict[tuple[str, Rank], int] = {}

    for (clr, rk), cnt in rem.items():
        if clr == my:
            my_counts[(clr, rk)] = my_counts.get((clr, rk), 0) + cnt
        elif clr == opp:
            opp_counts[(clr, rk)] = opp_counts.get((clr, rk), 0) + cnt

    # 加上棋盘明子
    for pc in state.board.values():
        if pc.revealed:
            if pc.color == my:
                my_counts[(pc.color, pc.rank)] = my_counts.get((pc.color, pc.rank), 0) + 1
            elif pc.color == opp:
                opp_counts[(pc.color, pc.rank)] = opp_counts.get((pc.color, pc.rank), 0) + 1

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

    # 1. 存活子力统计（用于动态制霸与物质分）
    my_counts, opp_counts = _get_alive_counts(state)

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

    # 动态子力价值加成
    my_piece_values = dict(DEFAULT_PIECE_VALUES)
    opp_piece_values = dict(DEFAULT_PIECE_VALUES)

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

    # 敌方工兵全灭时：我方地雷与军旗安全系数飙升（敌方无法挖雷吃旗）
    if not opp_has_gong:
        my_piece_values[Rank.LEI] += 15.0
        my_piece_values[Rank.QI] += 25.0
    if not my_has_gong:
        opp_piece_values[Rank.LEI] += 15.0
        opp_piece_values[Rank.QI] += 25.0

    # 炸弹联动价值：若敌方有大官(司/军)，我方炸弹价值提升；反之降低
    if opp_has_si or opp_has_jun:
        my_piece_values[Rank.ZHA] += 10.0
    if my_has_si or my_has_jun:
        opp_piece_values[Rank.ZHA] += 10.0

    # 计算棋盘明子 + 暗子池物质总分
    my_material = 0.0
    opp_material = 0.0

    for pos, pc in state.board.items():
        if pc.revealed:
            if pc.color == my:
                my_material += my_piece_values[pc.rank]
            else:
                opp_material += opp_piece_values[pc.rank]

    # 暗子池精确期望分摊
    rem = state.remaining_types()
    for (clr, rk), n in rem.items():
        if clr == my:
            my_material += n * my_piece_values[rk]
        elif clr == opp:
            opp_material += n * opp_piece_values[rk]

    score = my_material - opp_material

    # 3. 明子位置与阵型结构特征
    revealed_mine: list[tuple[tuple[int, int], Piece]] = []
    revealed_opp: list[tuple[tuple[int, int], Piece]] = []

    for pos, pc in state.board.items():
        if pc.revealed:
            if pc.color == my:
                revealed_mine.append((pos, pc))
            else:
                revealed_opp.append((pos, pc))

    # 3.1 行营控制与营内围杀
    for pos, pc in revealed_mine:
        if is_camp(pos):
            score += w.camp_occ
            # 行营围杀压力：营内子邻接敌子
            siege = sum(1 for np in NEIGHBORS[pos]
                        if (e := state.board.get(np)) is not None and e.color == opp)
            score += w.camp_siege * siege
        if is_hq(pos) and pc.rank != Rank.QI:
            score += w.hq_locked

    for pos, pc in revealed_opp:
        if is_camp(pos):
            score -= w.camp_occ
            siege = sum(1 for np in NEIGHBORS[pos]
                        if (e := state.board.get(np)) is not None and e.color == my)
            score -= w.camp_siege * siege
        if is_hq(pos) and pc.rank != Rank.QI:
            score -= w.hq_locked

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

    # 5.2 相邻吃子威胁
    for pos, e in revealed_opp:
        if e.rank == Rank.QI or is_camp(pos):
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
        score -= w.threat * best_gain

    for pos, m in revealed_mine:
        if m.rank == Rank.QI or is_camp(pos):
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

    # 6. 死区势能 (Fortress Score)
    score += w.fortress * (fortress_score(state, seat) - fortress_score(state, 1 - seat))

    # 7. 暗子时差与节奏 (Hidden Tempo)
    my_active = sum(1 for p, pc in revealed_mine if pc.rank not in (Rank.LEI, Rank.QI))
    opp_active = sum(1 for p, pc in revealed_opp if pc.rank not in (Rank.LEI, Rank.QI))
    score += w.hidden_tempo * (my_active - opp_active)

    return score
