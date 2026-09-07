"""官方 APK (libjunqi.so) 1:1 极简纯净原生引擎模块。

本模块严格对齐手机原版 libjunqi.so 二进制逆向算法逻辑（P1 / P4 阶段）：
1. 静态估值核 (0x59f90 与 0x124094)：
   - 纯净物质价值差（司令 2560 至排长 30、工兵 80、地雷 70、军旗 50）+ 动态炸弹；
   - 60 格静态位置偏置（行营 +40/+50）；
   - 地雷守护军旗防御加分（+80）；
   - 绝不计算非原版专家复合特征（如死区势能、梯队火力网矩阵等）。
2. 博弈树拓扑结构 (0x5a50e - 0x5a536 与 0x59e80)：
   - 根节点分流决策：暗子翻棋走法由 0x5a3c0 启发式与开局偏好独立评分；
   - 树深层（ply >= 1）与静态搜索（QSearch）绝不递归展开暗子翻棋，纯明子 Alpha-Beta 搜索；
   - 彻底根除全概率几率树引起的奇偶深度地平线偏差。
3. 难度与 IDS 机制 (0x3404c / 0x34078 / 0x5ac3a)：
   - 初级：depth=2, 100ms, qdepth=8, 随机扰动 jitter=±30；
   - 中级：depth=3, 300ms, qdepth=12, 随机扰动 jitter=±10；
   - 高级：depth=4, 1000ms, qdepth=16, 最小 tie-break 扰动 jitter=±0.5；
   - IDS 耗时超过预算 25% 安全早停退出，保留上一层完整 PV。
4. 开局库偏好：
   - 首翻与开局强偏好中心 4 个行营周围的 6 个关键暗子位。
5. 判和与长捉规避：
   - 连续 70 步未吃子判和；3 次重复局面规避与长捉惩罚。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .config import RuleConfig
from .rules import (
    ALL_POSITIONS,
    CAMPS,
    COLORS,
    COLOR_CN,
    HQS,
    NEIGHBORS,
    PLAY_POSITIONS,
    Rank,
    battle,
    is_camp,
    is_hq,
    other,
)
from .state import Action, GameState, Piece, position_key
from .zobrist import compute_zobrist

Color = Optional[str]


# 官方 0x124094 权威 2 的幂次等比价值表
APK_PIECE_VALUES: Dict[Rank, float] = {
    Rank.SI: 2560.0,
    Rank.JUN: 1280.0,
    Rank.SHI: 640.0,
    Rank.LV: 320.0,
    Rank.TUAN: 160.0,
    Rank.YING: 80.0,
    Rank.LIAN: 40.0,
    Rank.PAI: 30.0,
    Rank.GONG: 80.0,
    Rank.ZHA: 426.0,  # 动态基准
    Rank.LEI: 70.0,
    Rank.QI: 50.0,
}

# 地雷护旗防御加成 (0x124094)
MINE_FLAG_GUARD_BONUS = 80.0

# 官方三档难度参数表 (0x3404c / 0x34078 / 0x34088)
APK_LEVEL_SPECS = {
    "beginner": {
        "depth": 2,
        "time_limit_ms": 100,
        "qsearch_depth": 8,
        "jitter": 30.0,
    },
    "intermediate": {
        "depth": 3,
        "time_limit_ms": 300,
        "qsearch_depth": 12,
        "jitter": 10.0,
    },
    "advanced": {
        "depth": 4,
        "time_limit_ms": 1000,
        "qsearch_depth": 16,
        "jitter": 0.5,
    },
}

# 开局中心行营核心辐射暗子位 (0x5a3c0 开局黄金格位)
CENTER_CAMP_FLIP_POSITIONS: Set[Tuple[int, int]] = {
    (3, 1), (3, 3), (4, 2),  # 上半场中心行营辐射位
    (7, 2), (8, 1), (8, 3),  # 下半场中心行营辐射位
}

# 行营位置偏好得分表 (对应 0x1226c8 / 0x5a000)
CAMP_POSITION_BONUS: Dict[Tuple[int, int], float] = {
    (2, 1): 40.0, (2, 3): 40.0, (3, 2): 50.0, (4, 1): 40.0, (4, 3): 40.0,
    (7, 1): 40.0, (7, 3): 40.0, (8, 2): 50.0, (9, 1): 40.0, (9, 3): 40.0,
}


def eval_apk_pure(state: GameState, my_color: Color) -> float:
    """1:1 复刻 libjunqi.so 0x59f90 的极简纯净静态估值函数。

    计算公式：
      score = (己方已知明子物质 - 敌方已知明子物质)
            + (己方行营占位加分 - 敌方行营占位加分)
            + (己方地雷护旗加分 - 敌方地雷护旗加分)
    """
    if state.is_terminal():
        w = state.winner
        if w is None or w == -1:
            return 0.0
        return 500_000.0 if state.seat_color.get(w) == my_color else -500_000.0

    if my_color is None:
        return 0.0

    opp_color = other(my_color)

    # 1. 动态炸弹定价：随存活敌方最高军衔缩放 (0x600ca 公式)
    my_max_rank_val = 0.0
    opp_max_rank_val = 0.0
    for p in state.board.values():
        if p.revealed:
            r_val = APK_PIECE_VALUES.get(p.rank, 0.0)
            if p.color == my_color and r_val > my_max_rank_val:
                my_max_rank_val = r_val
            elif p.color == opp_color and r_val > opp_max_rank_val:
                opp_max_rank_val = r_val

    # 炸弹价值取敌方存活最大军衔的 1/3，保底 160，最高 853
    my_bomb_val = max(160.0, min(853.33, opp_max_rank_val * (1.0 / 3.0)))
    opp_bomb_val = max(160.0, min(853.33, my_max_rank_val * (1.0 / 3.0)))

    my_score = 0.0
    opp_score = 0.0

    # 2. 遍历已知棋子计算物质与行营占位加分
    for pos, p in state.board.items():
        if not p.revealed:
            continue

        if p.rank == Rank.ZHA:
            val = my_bomb_val if p.color == my_color else opp_bomb_val
        else:
            val = APK_PIECE_VALUES.get(p.rank, 0.0)

        # 行营占位偏置加分 (0x1226c8)
        pos_bonus = CAMP_POSITION_BONUS.get(pos, 0.0)

        if p.color == my_color:
            my_score += (val + pos_bonus)
        else:
            opp_score += (val + pos_bonus)

    # 3. 地雷护旗防御阵地加成 (0x124094 +80)
    # 检测军旗周围是否有己方地雷守护
    for pos, p in state.board.items():
        if p.revealed and p.rank == Rank.QI:
            # 检查邻位地雷
            has_guard = False
            for nb in NEIGHBORS.get(pos, ()):
                guard_p = state.board.get(nb)
                if guard_p and guard_p.revealed and guard_p.color == p.color and guard_p.rank == Rank.LEI:
                    has_guard = True
                    break
            if has_guard:
                if p.color == my_color:
                    my_score += MINE_FLAG_GUARD_BONUS
                else:
                    opp_score += MINE_FLAG_GUARD_BONUS

    return my_score - opp_score


def eval_apk_flip_root(state: GameState, action: Action, my_color: Color) -> float:
    """1:1 对齐 libjunqi.so 0x5a3c0 的根节点翻棋启发式打分。

    纯净 APK 机制：在根节点，翻暗棋的基础期望继承当前全盘物质态势，
    并通过位置权重、邻营控制与局部攻防期权对翻棋候选走法直接进行战略评估。
    """
    pos = action.frm
    # 基础继承当前全盘态势估值
    score = eval_apk_pure(state, my_color) if my_color else 0.0

    revealed_friendly = [p for p in state.board.values() if p.revealed and p.color == my_color]
    friendly_outside_camp = [
        p_pos for p_pos, p in state.board.items()
        if p.revealed and p.color == my_color and not is_camp(p_pos)
    ]

    # 检查是否有未进营的己方明子可一步进空营
    has_camp_entrance_opportunity = False
    for p_pos in friendly_outside_camp:
        for nb in NEIGHBORS.get(p_pos, ()):
            if is_camp(nb) and state.board.get(nb) is None:
                has_camp_entrance_opportunity = True
                break
        if has_camp_entrance_opportunity:
            break

    # 1. 开局中心 4 个行营周围黄金位强偏好 (0x5a3c0)
    # 严格约束：仅在开局全盘无己方明子、处纯开局盲翻探索期时赋予 +150
    if not revealed_friendly and pos in CENTER_CAMP_FLIP_POSITIONS:
        score += 150.0

    # 2. 依托已占行营的单向扑杀与辐射拓荒特权 (行营免死且 8 向通达)
    has_friendly_camp = False
    has_opp_threat = False
    opp_color = other(my_color) if my_color else None

    for nb in NEIGHBORS.get(pos, ()):
        p = state.board.get(nb)
        if p and p.revealed:
            if my_color and p.color == my_color and is_camp(nb):
                # 己方行营驻扎明子，享有邻位绝对单向扑杀期权
                has_friendly_camp = True
            elif opp_color and p.color == opp_color and not is_camp(nb) and p.rank not in (Rank.LEI, Rank.QI):
                # 敌方明子在非行营格相邻，翻开有一定被敌方大子直接吃掉的风险
                has_opp_threat = True

    if has_friendly_camp:
        score += 120.0
    elif is_camp(pos):
        # 翻开行营内的棋子（虽开局行营无子，但若有变体）
        score += 60.0

    if has_opp_threat and not has_friendly_camp:
        score -= 40.0

    # 3. 战术纪律：若场上有未保护的己方明子且近邻有空营可进，严禁弃营盲目远端翻棋！
    if has_camp_entrance_opportunity and not has_friendly_camp:
        # 重度抑制远端盲翻，迫使 AI 优先占营建立据点
        score -= 200.0
    else:
        score += 20.0

    return score


@dataclass
class ApkSearchStats:
    """搜索统计指标。"""
    nodes: int = 0
    qnodes: int = 0
    depth_reached: int = 0
    time_spent_ms: float = 0.0
    tt_hits: int = 0
    root_scores: List[Tuple[Action, float]] = field(default_factory=list)


class ApkSearchEngine:
    """官方 APK (libjunqi.so) 1:1 原生搜索引擎实现。"""

    def __init__(self, tt_size_power: int = 18, seed: Optional[int] = None):
        self.tt_mask = (1 << tt_size_power) - 1
        # 置换表条目: key -> (depth, flag, score, best_move)
        # flag: 1=EXACT, 2=LOWER_BOUND, 3=UPPER_BOUND
        self.tt: Dict[int, Tuple[int, int, float, Optional[Action]]] = {}
        self.rng = random.Random(seed)
        self.stats = ApkSearchStats()

    def _get_hash(self, state: GameState) -> int:
        return compute_zobrist(state) & self.tt_mask

    def search(
        self,
        state: GameState,
        level: str = "advanced",
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None,
        qsearch_depth: Optional[int] = None,
        avoid: Optional[Set] = None,
    ) -> Tuple[Optional[Action], float, ApkSearchStats]:
        """执行符合原版 libjunqi.so 行为的 IDS 纯净搜索。"""
        spec = APK_LEVEL_SPECS.get(level, APK_LEVEL_SPECS["advanced"])
        max_depth = depth if depth is not None else spec["depth"]
        time_limit = time_limit_ms if time_limit_ms is not None else spec["time_limit_ms"]
        qdepth = qsearch_depth if qsearch_depth is not None else spec["qsearch_depth"]
        jitter_range = spec.get("jitter", 0.0)

        self.stats = ApkSearchStats()
        start_time = time.perf_counter()

        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return None, 0.0, self.stats
        if len(acts) == 1:
            return acts[0], 0.0, self.stats

        my_color = state.my_color()

        # 1. 根节点走法分类：走棋 (moves) 与 翻暗棋 (flips)
        moves = [a for a in acts if a.kind == "move"]
        flips = [a for a in acts if a.kind == "flip"]

        # 2. 翻暗棋在根节点由启发式打分 (0x5a3c0)
        flip_scores: List[Tuple[Action, float]] = []
        for f in flips:
            f_score = eval_apk_flip_root(state, f, my_color)
            if jitter_range > 0:
                f_score += self.rng.uniform(-jitter_range, jitter_range)
            flip_scores.append((f, f_score))

        # 3. 明子走法进入 Alpha-Beta + QSearch 树搜索
        best_move: Optional[Action] = None
        best_move_score = -999_999.0
        pv_move_scores: List[Tuple[Action, float]] = []

        if moves:
            # 迭代加深搜索 (IDS, 对齐 0x5ac3a)
            for cur_depth in range(1, max_depth + 1):
                cur_best_move = None
                cur_best_score = -999_999.0
                cur_scores = []
                alpha = -999_999.0
                beta = 999_999.0

                # 走法排序：优先上一轮 best_move，然后吃子
                ordered_moves = self._order_root_moves(moves, state, best_move)

                for m in ordered_moves:
                    nxt = state.apply(m)
                    self.stats.nodes += 1

                    # 重复局面与长捉避让惩罚
                    if avoid and (position_key(nxt) in avoid or compute_zobrist(nxt) in avoid):
                        score = -300_000.0
                    else:
                        # 递归 Alpha-Beta 搜索（注意：树深层绝不生成翻棋）
                        score = -self._alpha_beta(
                            nxt,
                            cur_depth - 1,
                            -beta,
                            -alpha,
                            qdepth,
                            start_time,
                            time_limit,
                            my_color,
                        )

                    cur_scores.append((m, score))

                    if score > cur_best_score:
                        cur_best_score = score
                        cur_best_move = m

                    if score > alpha:
                        alpha = score

                    # 耗时检查
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    if time_limit > 0 and elapsed_ms >= time_limit:
                        break

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                if cur_best_move is not None:
                    best_move = cur_best_move
                    best_move_score = cur_best_score
                    pv_move_scores = cur_scores
                    self.stats.depth_reached = cur_depth

                # IDS 25% 提前跳出机制 (对齐 0x5ac3a: r2 > r3 asr #2)
                if time_limit > 0 and elapsed_ms >= (time_limit * 0.25):
                    break

        # 4. 根节点走法仲裁：明子最优走法 vs 翻棋最优走法
        # 加入难度档位指定的随机扰动 (Jitter)
        all_candidate_scores: List[Tuple[Action, float]] = []

        for m, sc in pv_move_scores:
            adj_sc = sc
            if jitter_range > 0:
                adj_sc += self.rng.uniform(-jitter_range, jitter_range)
            all_candidate_scores.append((m, adj_sc))

        for f, sc in flip_scores:
            all_candidate_scores.append((f, sc))

        all_candidate_scores.sort(key=lambda t: t[1], reverse=True)
        self.stats.root_scores = all_candidate_scores
        self.stats.time_spent_ms = (time.perf_counter() - start_time) * 1000.0

        if all_candidate_scores:
            chosen_action, chosen_score = all_candidate_scores[0]
            return chosen_action, chosen_score, self.stats

        return acts[0], 0.0, self.stats

    def _order_root_moves(
        self, moves: List[Action], state: GameState, pv_move: Optional[Action]
    ) -> List[Action]:
        """根节点走法启发式排序。"""
        def move_priority(a: Action) -> float:
            if a == pv_move:
                return 1_000_000.0
            p = 0.0
            # 吃子优先 (MVV-LVA)
            target = state.board.get(a.to)
            if target and target.revealed:
                p += 100_000.0 + APK_PIECE_VALUES.get(target.rank, 0.0)
            # 进空营优先
            elif is_camp(a.to) and not is_camp(a.frm):
                p += 50_000.0
            return p

        return sorted(moves, key=move_priority, reverse=True)

    def _alpha_beta(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        qdepth: int,
        start_time: float,
        time_limit_ms: int,
        root_color: Color,
    ) -> float:
        """带 PVS 和 TT 的负极大值 Alpha-Beta 搜索（纯明子树，对齐 0x5a7d4）。"""
        # 1. 终局检查
        if state.is_terminal():
            w = state.winner
            if w == -1 or w is None:
                return 0.0
            my_c = state.my_color()
            base_win = 500_000.0 if state.win_reason == "flag" else 400_000.0
            return base_win + depth * 1000.0 if state.seat_color.get(w) == my_c else -base_win - depth * 1000.0

        # 2. 叶子节点转入静态搜索 (QSearch)
        if depth <= 0:
            return self._qsearch(state, alpha, beta, qdepth, root_color)

        # 3. 超时检查
        if time_limit_ms > 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms >= time_limit_ms:
                return eval_apk_pure(state, state.my_color())

        # 4. 置换表 (TT) 探测
        h = self._get_hash(state)
        tt_entry = self.tt.get(h)
        tt_move = None
        if tt_entry is not None:
            t_depth, t_flag, t_score, t_move = tt_entry
            tt_move = t_move
            if t_depth >= depth:
                self.stats.tt_hits += 1
                if t_flag == 1:  # EXACT
                    return t_score
                elif t_flag == 2 and t_score >= beta:  # LOWER_BOUND
                    return t_score
                elif t_flag == 3 and t_score <= alpha:  # UPPER_BOUND
                    return t_score

        # 5. 生成走法（严格遵循 libjunqi.so 0x59e80：树深层只搜明子，绝不生成暗棋翻棋！）
        legal_acts = state.legal_actions()
        moves = [a for a in legal_acts if a.kind == "move"]

        if not moves:
            # 无明子可走，直接静态估值返回
            return eval_apk_pure(state, state.my_color())

        # 6. 走法排序 (TT move > 吃子 > 进营)
        ordered_moves = self._order_moves(moves, state, tt_move)

        best_score = -999_999.0
        best_act = None
        orig_alpha = alpha

        # 7. PVS (零窗口探测) 循环
        for i, m in enumerate(ordered_moves):
            nxt = state.apply(m)
            self.stats.nodes += 1

            if i == 0:
                # 全窗口搜索
                score = -self._alpha_beta(
                    nxt, depth - 1, -beta, -alpha, qdepth, start_time, time_limit_ms, root_color
                )
            else:
                # 零窗口探测 (Null Window Probe)
                score = -self._alpha_beta(
                    nxt, depth - 1, -alpha - 1.0, -alpha, qdepth, start_time, time_limit_ms, root_color
                )
                # 探测失败重新全窗口搜索
                if alpha < score < beta:
                    score = -self._alpha_beta(
                        nxt, depth - 1, -beta, -score, qdepth, start_time, time_limit_ms, root_color
                    )

            if score > best_score:
                best_score = score
                best_act = m

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta 剪枝 (Cutoff)
                break

        # 行营驻守机制：翻棋对局中驻营子力无需强行出营送死，若全盘走法均劣于驻守则保持阵地
        my_c = state.my_color()
        if any(is_camp(pos) for pos, p in state.board.items() if p.revealed and p.color == my_c):
            stand_pat = eval_apk_pure(state, my_c)
            if best_score < stand_pat:
                best_score = stand_pat

        # 8. 存入置换表 (TT)
        if best_score <= orig_alpha:
            flag = 3  # UPPER_BOUND
        elif best_score >= beta:
            flag = 2  # LOWER_BOUND
        else:
            flag = 1  # EXACT
        self.tt[h] = (depth, flag, best_score, best_act)

        return best_score

    def _qsearch(
        self, state: GameState, alpha: float, beta: float, qdepth: int, root_color: Color
    ) -> float:
        """1:1 对齐 libjunqi.so 0x5a678 的纯吃子静态搜索。"""
        self.stats.qnodes += 1
        my_c = state.my_color()

        if state.is_terminal():
            w = state.winner
            if w == -1 or w is None:
                return 0.0
            return 300_000.0 if state.seat_color.get(w) == my_c else -300_000.0

        stand_pat = eval_apk_pure(state, my_c)

        if qdepth <= 0 or stand_pat >= beta:
            return stand_pat

        if stand_pat > alpha:
            alpha = stand_pat

        # Delta 剪枝：即使吃掉场上最高价值子力（司令 2560）仍低于 alpha，直接剪除
        if stand_pat + 2560.0 < alpha:
            return alpha

        # 严格只生成吃子动作 (Captures Only)
        legal_acts = state.legal_actions()
        captures = [
            a for a in legal_acts
            if a.kind == "move" and a.to in state.board and state.board[a.to].revealed
        ]

        if not captures:
            return stand_pat

        # 按 MVV-LVA 排序吃子
        captures.sort(
            key=lambda a: APK_PIECE_VALUES.get(state.board[a.to].rank, 0.0)
            - APK_PIECE_VALUES.get(state.board[a.frm].rank, 0.0),
            reverse=True,
        )

        for cap in captures:
            nxt = state.apply(cap)
            score = -self._qsearch(nxt, -beta, -alpha, qdepth - 1, root_color)

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def _order_moves(
        self, moves: List[Action], state: GameState, tt_move: Optional[Action]
    ) -> List[Action]:
        """内部节点走法启发式排序。"""
        def score(a: Action) -> float:
            if a == tt_move:
                return 500_000.0
            s = 0.0
            target = state.board.get(a.to)
            if target and target.revealed:
                # 吃子走法
                s += 100_000.0 + (
                    APK_PIECE_VALUES.get(target.rank, 0.0)
                    - APK_PIECE_VALUES.get(state.board[a.frm].rank, 0.0) * 0.1
                )
            elif is_camp(a.to) and not is_camp(a.frm):
                # 进驻空营
                s += 20_000.0
            return s

        return sorted(moves, key=score, reverse=True)
