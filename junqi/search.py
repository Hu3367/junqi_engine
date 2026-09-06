"""专家级军棋翻棋搜索核心 (Expert Search Engine for Junqi Fanqi)。

吸收计算机暗棋 (Chinese Dark Chess) 与现代博弈搜索前沿技术：
1. 静态搜索 (Quiescence Search / Q-Search): 结合 MVV-LVA 消除地平线效应与吃子震荡
2. 期望极大极小 (Expectiminimax) 与 Star1 几率剪枝: 精确处理翻棋几率节点
3. 64 位 Zobrist 哈希与置换表 (Transposition Table): PV 走法迁移与深度剪枝
4. 多级走法排序启发式: TT Move -> MVV-LVA -> 杀手着法 -> 历史启发 -> 安全翻棋 -> 静步
5. 迭代加深 (Iterative Deepening Search / IDS): 毫秒级时间预算动态控制与防超时回退
6. 树内重复局面检测与防循环和棋
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from .analysis import fortress_score
from .config import EvalWeights, RuleConfig, SearchConfig
from .eval_expert import evaluate_expert
from .rules import (ATTACKER_WINS, BOTH_DIE, CAMPS, COMPOSITION, HQS,
                    NEIGHBORS, PLAY_POSITIONS, RANK_CN, Rank, battle,
                    is_camp, is_hq, is_rail, other)
from .state import WIN_SCORE, Action, GameState, Piece
from .tt import FLAG_EXACT, FLAG_LOWER_BOUND, FLAG_UPPER_BOUND, TranspositionTable
from .zobrist import compute_zobrist

# 杀手着法槽位数与历史启发上限
MAX_KILLERS = 2
MAX_HISTORY = 100_000
MAX_SEARCH_DEPTH = 64
DEFAULT_QSEARCH_DEPTH = 4


@dataclass
class SearchStats:
    nodes: int = 0
    qnodes: int = 0
    tt_hits: int = 0
    chance_nodes: int = 0
    star1_cutoffs: int = 0
    pvs_researches: int = 0
    max_depth: int = 0
    time_elapsed_ms: float = 0.0
    root_scores: list[tuple[Action, float]] = field(default_factory=list)


class ExpertSearchEngine:
    """专家级军棋翻棋搜索引擎。"""

    def __init__(self, weights: Optional[EvalWeights] = None,
                 tt_size_power: int = 18, seed: Optional[int] = None):
        self.w = weights or EvalWeights()
        self.tt = TranspositionTable(size_power=tt_size_power)
        self.rng = random.Random(seed)

        # 启发式表
        # killer_moves[depth] = list[Action]
        self.killers: list[list[Action]] = [[] for _ in range(MAX_SEARCH_DEPTH)]
        # history_table[(frm, to)] = score
        self.history: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

        # 搜索状态与统计
        self.stats = SearchStats()
        self.deadline: float = float("inf")
        self.stopped = False

    def clear_heuristics(self):
        """清空杀手着法与历史表（置换表选择性保留或清空）。"""
        self.killers = [[] for _ in range(MAX_SEARCH_DEPTH)]
        self.history.clear()

    # ------------------------------------------------------------- 走法排序

    def _score_action(self, act: Action, state: GameState, ply_depth: int,
                      tt_move: Optional[Action] = None) -> float:
        """为候选走法打分，实现高精度排序。"""
        # 1. 置换表最佳着法 (最高优先级)
        if tt_move is not None and act == tt_move:
            return 1_000_000.0

        my = state.my_color()

        if act.kind == "move":
            target = state.board.get(act.to)
            mover = state.board.get(act.frm)
            mover_rank = mover.rank if mover else Rank.PAI

            # 2. 吃明子: MVV-LVA 排序 (Most Valuable Victim - Least Valuable Attacker)
            if target is not None and target.revealed and target.color != my:
                res = battle(mover_rank, target.rank)
                victim_val = self.w.piece.get(target.rank, 30.0)
                attacker_val = self.w.piece.get(mover_rank, 30.0)

                # 2.1 行营单向打击特权 (2026-09-06 实证：前20手吃子整整 50.1% 源自行营扑杀)
                # 营内打营外绝对安全，营外打营内免疫；若在行营内发起吃子，赋予特权加分
                camp_outstrike_bonus = 0.0
                if is_camp(act.frm):
                    camp_outstrike_bonus = getattr(self.w, "camp_outstrike_bias", 400_000.0)

                # 2.2 小子贴身拆弹定式 (2026-09-06 实证：炸弹 47.5% 杀伤连排团工营)
                # 当目标是敌方炸弹，且攻击方是小子（连/排/工/营/团），主动撞弹消灭敌核武器，属于战略必争定式
                bomb_suicide_bonus = 0.0
                if target.rank == Rank.ZHA and mover_rank in (
                    Rank.LIAN, Rank.PAI, Rank.GONG, Rank.YING, Rank.TUAN
                ):
                    bomb_suicide_bonus = getattr(self.w, "bomb_suicide_exchange", 150_000.0)

                if target.rank == Rank.QI:
                    # 一步扛旗直接制胜，赋予最高战术排序优先级
                    return 900_000.0

                if res == ATTACKER_WINS:
                    # 稳赚吃子: 目标越值钱、攻击者越廉价越优先
                    return 500_000.0 + camp_outstrike_bonus + victim_val * 100.0 - attacker_val
                elif res == BOTH_DIE:
                    # 兑子: 炸弹或同级兑换 (含小子贴身拆弹战略加分)
                    return 300_000.0 + camp_outstrike_bonus + bomb_suicide_bonus + victim_val * 100.0 - attacker_val
                else:
                    # 亏损/送吃
                    return -100_000.0 + victim_val - attacker_val

            # 3. 进行营避险 / 占营进驻核心据点 (实证：前 20 手走子 82.7% 为进营)
            if is_camp(act.to):
                # 进驻空行营是绝对免死与据点化的战略特权，优先级高于普通翻棋
                camp_prio = 250_000.0 if act.to not in state.board else 150_000.0
                return camp_prio + self.w.piece.get(mover_rank, 20.0)

            # 4. 杀手着法 (Killer Moves)
            if ply_depth < len(self.killers) and act in self.killers[ply_depth]:
                return 50_000.0

            # 5. 历史启发 (History Heuristic)
            h_score = self.history.get((act.frm, act.to), 0)
            if h_score > 0:
                return 10_000.0 + min(h_score, 30_000.0)

            # 6. 普通移动 (Quiet Move)
            # 偏好向铁路或中心靠拢
            r, c = act.to
            center_bias = 4 - abs(r - 5.5) - abs(c - 2)
            return 1000.0 + center_bias * 10.0

        elif act.kind == "flip":
            # 翻棋排序：根据周围邻域攻防态势、行营据点辐射度与领地偏好计算综合指数
            pos = act.frm
            r, c = pos
            safety_score = 0.0
            opp = other(my) if my else None

            # 1. 依托行营辐射拓荒 (实证：96.2% 邻营翻棋，开局首翻即据点)
            # 若待翻暗子相邻有己方已占领的行营，享受高优先辐射翻棋
            camp_adjacent_bonus = 0.0
            if my is not None:
                for np in NEIGHBORS[pos]:
                    if is_camp(np):
                        cb = state.board.get(np)
                        if cb is not None and cb.revealed and cb.color == my:
                            camp_adjacent_bonus = getattr(self.w, "camp_adjacent_flip_bias", 50_000.0)
                            break

            # 若待翻暗子相邻为空行营，拥有翻出即进营的极高据点潜力 (如 2,2 / 4,2 / 3,1 / 3,3 各邻接 3 个行营!)
            empty_camps_adjacent = sum(1 for np in NEIGHBORS[pos] if is_camp(np) and np not in state.board)
            camp_potential_bonus = empty_camps_adjacent * 25_000.0

            # 2. 开局领地与中前场咽喉偏好 (避免盲目翻自家底线或敌方底线禁区)
            territory_bias = 0.0
            if my == "r":
                if 1 <= r <= 5:
                    territory_bias = 20_000.0
                elif r == 0:
                    territory_bias = -15_000.0
                elif 6 <= r <= 11:
                    # 敌方半场：开局己方无掩护时严禁盲目跨界替敌翻棋
                    territory_bias = -100_000.0
            elif my == "b":
                if 6 <= r <= 10:
                    territory_bias = 20_000.0
                elif r == 11:
                    territory_bias = -15_000.0
                elif 0 <= r <= 5:
                    # 敌方半场：开局己方无掩护时严禁盲目跨界替敌翻棋
                    territory_bias = -100_000.0
            else:
                # 首翻未定色时，偏好中央四角咽喉据点 (2..4 或 7..9 行)
                if 2 <= r <= 4 or 7 <= r <= 9:
                    territory_bias = 20_000.0
                elif r in (0, 11):
                    territory_bias = -15_000.0

            friendly_guards = 0
            enemy_threats = 0

            for np in NEIGHBORS[pos]:
                nb = state.board.get(np)
                if nb is not None and nb.revealed:
                    if nb.color == my:
                        if nb.rank >= Rank.SHI:
                            friendly_guards += 2
                        elif nb.rank not in (Rank.LEI, Rank.QI):
                            friendly_guards += 1
                    elif nb.color == opp:
                        if nb.rank >= Rank.SHI:
                            enemy_threats += 2
                        elif nb.rank not in (Rank.LEI, Rank.QI):
                            enemy_threats += 1

            # 若在敌方半场但有己方部队就位压境，恢复前线翻棋进攻权
            if territory_bias < -50_000.0 and friendly_guards > 0:
                territory_bias = 10_000.0 * friendly_guards

            if friendly_guards > enemy_threats:
                safety_score = 60_000.0 + (friendly_guards - enemy_threats) * 5000.0
            elif enemy_threats > friendly_guards:
                safety_score = 500.0 - enemy_threats * 1000.0
            else:
                safety_score = 20_000.0

            return safety_score + camp_adjacent_bonus + camp_potential_bonus + territory_bias

        return 0.0

    def _order_actions(self, acts: list[Action], state: GameState,
                       ply_depth: int, tt_move: Optional[Action] = None) -> list[Action]:
        """对合法动作列表进行启发式评分与排序，并对翻棋分支做专家级候选剪枝。"""
        if len(acts) <= 1:
            return acts

        # 区分移动动作与翻棋动作
        moves: list[Action] = []
        flips: list[Action] = []
        for a in acts:
            if a.kind == "move":
                moves.append(a)
            else:
                flips.append(a)

        # 翻棋候选剪枝 (暗棋经典优化：大量暗子时保留局部安全度与据点价值最高的前 K 个翻棋格)
        if len(flips) > 5:
            flips.sort(key=lambda a: self._score_action(a, state, ply_depth, tt_move), reverse=True)
            max_flips = 3 if moves else 8
            flips = flips[:max_flips]

        filtered_acts = moves + flips
        return sorted(filtered_acts,
                      key=lambda a: self._score_action(a, state, ply_depth, tt_move),
                      reverse=True)

    # ------------------------------------------------------------- 静态搜索 (QSearch)

    def _qsearch(self, state: GameState, alpha: float, beta: float,
                 depth_left: int = DEFAULT_QSEARCH_DEPTH) -> float:
        """静态搜索 (Quiescence Search)：专用于在叶子节点解决吃子与战术震荡。"""
        self.stats.qnodes += 1

        # 终局检查
        if state.is_terminal():
            if state.winner == -1:
                return 0.0
            win = WIN_SCORE - state.ply
            return win if state.winner == state.turn else -win

        # Stand-Pat 评估剪枝
        stand_pat = evaluate_expert(state, state.turn, self.w)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if depth_left <= 0:
            return stand_pat

        # 仅生成吃子动作和进行营避险动作
        acts = state.legal_actions()
        tactical_moves: list[Action] = []
        my = state.my_color()

        for a in acts:
            if a.kind != "move":
                continue
            target = state.board.get(a.to)
            # 吃明子
            if target is not None and target.revealed and target.color != my:
                tactical_moves.append(a)
            # 或者进营
            elif is_camp(a.to):
                tactical_moves.append(a)

        if not tactical_moves:
            return stand_pat

        # MVV-LVA 排序
        tactical_moves.sort(key=lambda a: self._score_action(a, state, 0, None), reverse=True)

        for a in tactical_moves:
            child = state.apply(a)
            score = -self._qsearch(child, -beta, -alpha, depth_left - 1)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ------------------------------------------------------------- 翻棋几率节点 (Chance Node / Star1)

    def _evaluate_chance_flip(self, state: GameState, flip_act: Action,
                             depth: int, ply_depth: int, alpha: float, beta: float,
                             path_history: set[int]) -> float:
        """对翻棋动作进行几率节点 (Chance Node) 期望计算与 Star1 剪枝。"""
        self.stats.chance_nodes += 1

        pos = flip_act.frm
        rem = state.remaining_types()
        total_hidden = sum(rem.values())
        if total_hidden <= 0:
            # 无暗子构成，安全回退
            child = state.apply(flip_act)
            return -self._negamax(child, depth - 1, ply_depth + 1, -beta, -alpha, path_history)

        # 几率前沿截断 (Chance-Node Depth Truncation):
        # 仅在根节点 (ply_depth == 0) 对候选动作展开全概率对抗子树；
        # 在博弈树深层 (depth <= 1 或 ply_depth >= 1)，直接由公共信念状态求精确解析期望，
        # 彻底杜绝深层连续几率节点引发的 (24)^d 组合指数爆炸，保障毫秒级下棋速度
        if depth <= 1 or ply_depth >= 1:
            expected = 0.0
            for (clr, rk), cnt in rem.items():
                prob = cnt / total_hidden
                b = dict(state.board)
                b[pos] = Piece(clr, rk, revealed=True)
                sc = dict(state.seat_color)
                ffd = state.first_flip_done
                if not ffd:
                    sc[state.turn] = clr
                    sc[1 - state.turn] = other(clr)
                    ffd = True
                child = GameState(
                    board=b, dead=state.dead, seat_color=sc, turn=1 - state.turn,
                    first_flip_done=ffd, ply=state.ply + 1, winner=state.winner,
                    win_reason=state.win_reason, cfg=state.cfg, quiet=state.quiet + 1
                )
                # 检查翻开暗子后是否触发终局（如无暗子且对手无棋可走困毙）
                if not child.hidden_positions() and not child._has_any_move():
                    child.winner, child.win_reason = state.turn, "immobilized"

                if child.is_terminal():
                    if child.winner == -1:
                        child_val = 0.0
                    else:
                        win = WIN_SCORE - child.ply
                        child_val = win if child.winner == child.turn else -win
                else:
                    child_val = evaluate_expert(child, child.turn, self.w)

                expected += prob * (-child_val)
            return expected

        # 构建概率分布 [((color, rank), prob)]
        outcomes: list[tuple[str, Rank, float]] = []
        for (clr, rk), cnt in rem.items():
            prob = cnt / total_hidden
            outcomes.append((clr, rk, prob))

        # 按概率降序排序以提升 Star1 剪枝效率
        outcomes.sort(key=lambda item: item[2], reverse=True)

        expected_value = 0.0
        remaining_prob = 1.0

        # Star1 战术边界：常规局面最大物质与结构估值上下界约 ±600 分
        v_max = 600.0
        v_min = -600.0

        for clr, rk, prob in outcomes:
            # Star1 几率剪枝检查
            # 1. Fail-Low 截断: 即使后续全取最好也无法达到 alpha
            if expected_value + remaining_prob * v_max <= alpha:
                self.stats.star1_cutoffs += 1
                return alpha

            # 2. Fail-High 截断: 即使后续全取最差也必定超过 beta
            if expected_value + remaining_prob * v_min >= beta:
                self.stats.star1_cutoffs += 1
                return beta

            # 实例化一个翻出 (clr, rk) 的具体状态
            b = dict(state.board)
            b[pos] = Piece(clr, rk, revealed=True)
            sc = dict(state.seat_color)
            ffd = state.first_flip_done
            if not ffd:
                sc[state.turn] = clr
                sc[1 - state.turn] = other(clr)
                ffd = True

            child = GameState(
                board=b,
                dead=state.dead,
                seat_color=sc,
                turn=1 - state.turn,
                first_flip_done=ffd,
                ply=state.ply + 1,
                winner=state.winner,
                win_reason=state.win_reason,
                cfg=state.cfg,
                quiet=state.quiet + 1
            )
            # 检查翻开暗子后是否触发终局
            if not child.hidden_positions() and not child._has_any_move():
                child.winner, child.win_reason = state.turn, "immobilized"

            # 递归搜索子节点：几率子节点使用全窗口搜索，杜绝父节点期望剪枝窗引起子节点提前截断失真
            v = -self._negamax(child, depth - 1, ply_depth + 1, -WIN_SCORE, WIN_SCORE, path_history)
            expected_value += prob * v
            remaining_prob -= prob

        return expected_value

    # ------------------------------------------------------------- 核心 Negamax 搜索

    def _negamax(self, state: GameState, depth: int, ply_depth: int,
                 alpha: float, beta: float, path_history: set[int]) -> float:
        """带置换表、静态搜索、杀手/历史启发与 Star1 几率剪枝的 Negamax 搜索。"""
        self.stats.nodes += 1

        # 超时检查 (每 512 节点检查一次系统时间)
        if (self.stats.nodes & 511) == 0:
            if time.perf_counter() >= self.deadline:
                self.stopped = True
                return alpha

        # 1. 终局判断
        if state.is_terminal():
            if state.winner == -1:
                return 0.0
            win = WIN_SCORE - state.ply
            return win if state.winner == state.turn else -win

        # 2. 树内重复局面检测 (Repetition Detection)
        zobrist_key = compute_zobrist(state)
        if zobrist_key in path_history:
            # 重复走子判和（0 分）
            return 0.0

        # 3. 置换表查询 (TT Lookup)
        orig_alpha = alpha
        tt_val, tt_move = self.tt.lookup(zobrist_key, depth, alpha, beta)
        if tt_val is not None and ply_depth > 0:
            return tt_val

        # 4. 叶子节点转入静态搜索 (QSearch)
        if depth <= 0:
            return self._qsearch(state, alpha, beta)

        acts = state.legal_actions()
        if not acts:
            # 无子可动，判负
            return -(WIN_SCORE - state.ply)

        # 5. 走法排序 (Move Ordering)
        ordered_acts = self._order_actions(acts, state, ply_depth, tt_move)

        best_score = -math.inf
        best_act = ordered_acts[0]
        flag = FLAG_UPPER_BOUND

        path_history.add(zobrist_key)

        for i, a in enumerate(ordered_acts):
            if self.stopped:
                break

            if a.kind == "flip":
                score = self._evaluate_chance_flip(state, a, depth, ply_depth, alpha, beta, path_history)
            else:
                child = state.apply(a)
                if i == 0:
                    # 主变例走法 (PV move): 全窗口搜索
                    score = -self._negamax(child, depth - 1, ply_depth + 1, -beta, -alpha, path_history)
                else:
                    # 非主变例走法 (Non-PV moves): 采用零窗口探测 (Null Window Search) 快速验证截断
                    score = -self._negamax(child, depth - 1, ply_depth + 1, -alpha - 1, -alpha, path_history)
                    if alpha < score < beta and not self.stopped:
                        # 探测击穿 (Fail-High): 该走法好于预期，触发全窗口重新搜索
                        self.stats.pvs_researches += 1
                        score = -self._negamax(child, depth - 1, ply_depth + 1, -beta, -score, path_history)

            if self.stopped:
                break

            if score > best_score:
                best_score = score
                best_act = a

            if score > alpha:
                alpha = score
                flag = FLAG_EXACT

            if alpha >= beta:
                # Alpha-Beta 剪枝 (Cut-node)
                flag = FLAG_LOWER_BOUND

                # 更新杀手着法与历史启发 (仅对确定性移动有效)
                if a.kind == "move":
                    if ply_depth < len(self.killers):
                        if a not in self.killers[ply_depth]:
                            self.killers[ply_depth].insert(0, a)
                            if len(self.killers[ply_depth]) > MAX_KILLERS:
                                self.killers[ply_depth].pop()

                    h_key = (a.frm, a.to)
                    self.history[h_key] = min(self.history.get(h_key, 0) + depth * depth, MAX_HISTORY)

                break

        path_history.discard(zobrist_key)

        # 6. 存入置换表 (TT Store)
        if not self.stopped:
            self.tt.store(zobrist_key, depth, best_score, flag, best_act)

        return best_score

    # ------------------------------------------------------------- 迭代加深与根决策 (IDS)

    def search(self, state: GameState, max_depth: int = 3,
               time_limit_ms: int = 0, avoid: Optional[set] = None
               ) -> tuple[Optional[Action], float, SearchStats]:
        """迭代加深搜索 (Iterative Deepening Search)。

        参数:
            state: 当前游戏局面
            max_depth: 最大搜索深度 (ply)
            time_limit_ms: 限时 (毫秒)，0 为不限时纯按深度
            avoid: 根节点需回避的可观察局面键集合 (防送循环)

        返回:
            (best_action, score, stats)
        """
        self.stats = SearchStats()
        self.stopped = False
        start_time = time.perf_counter()

        if time_limit_ms > 0:
            self.deadline = start_time + (time_limit_ms / 1000.0)
        else:
            self.deadline = float("inf")

        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return None, 0.0, self.stats
        if len(acts) == 1:
            return acts[0], 0.0, self.stats

        best_action: Optional[Action] = None
        best_score: float = -math.inf

        path_history: set[int] = set()

        for d in range(1, max_depth + 1):
            if self.stopped or (time_limit_ms > 0 and time.perf_counter() >= self.deadline):
                break

            # 根节点搜索
            zobrist_key = compute_zobrist(state)
            _, tt_move = self.tt.lookup(zobrist_key, d, -math.inf, math.inf)
            ordered_acts = self._order_actions(acts, state, 0, tt_move or best_action)

            current_d_best_act = ordered_acts[0]
            current_d_best_score = -math.inf
            alpha = -math.inf
            beta = math.inf
            d_scores: list[tuple[Action, float]] = []

            for a in ordered_acts:
                if self.stopped or (time_limit_ms > 0 and time.perf_counter() >= self.deadline):
                    break

                if a.kind == "flip":
                    score = self._evaluate_chance_flip(state, a, d, 0, alpha, beta, path_history)
                else:
                    child = state.apply(a)
                    score = -self._negamax(child, d - 1, 1, -beta, -alpha, path_history)

                # 避免命中 avoid 集合
                if avoid and a.kind == "move":
                    from .state import position_key
                    if position_key(state.apply(a)) in avoid:
                        score -= 150.0

                d_scores.append((a, score))

                if score > current_d_best_score:
                    current_d_best_score = score
                    current_d_best_act = a

                if score > alpha:
                    alpha = score

            if not self.stopped:
                best_action = current_d_best_act
                best_score = current_d_best_score
                self.stats.max_depth = d
                self.stats.root_scores = sorted(d_scores, key=lambda t: t[1], reverse=True)
                self.tt.store(zobrist_key, d, best_score, FLAG_EXACT, best_action)

        self.stats.time_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return best_action or acts[0], best_score, self.stats
