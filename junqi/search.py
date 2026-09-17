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
from .core_bridge import (encode_state_blob, eval_expert_auto, make_cpp_qsearch,
                          make_cpp_search, make_cpp_weights, qsearch_auto)
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
DEFAULT_QSEARCH_DEPTH = 16

# 受限 Star1 展开的「战术重要性」分档。
#
# 为什么需要它（2026-09-16 实测）：受限展开原先取"剩余数量降序前 3"作为搜索子集，
# 而数量最多的身份恒为 连长/排长/工兵/地雷（每种 3 枚），司令/军长/炸弹（1~2 枚）
# 几乎总落入长尾只做静态估值 —— 与该增强要解决的"翻出敌大子被吃"盲区恰好相反。
# 改为按"战术重要性 → 出现概率 → 颜色 → 规范序"取前 3。
#
# ⚠ `_STAR1_KIND_ORDER` 与 `_STAR1_IMPORTANCE` 必须与
#   src_cpp/src/expert_search.cpp::kStar1Importance 完全一致
#   （C++ 侧有 expert_star1_importance() 自检入口 + 测试守卫）。
_STAR1_KIND_ORDER = (Rank.SI, Rank.JUN, Rank.SHI, Rank.LV, Rank.TUAN, Rank.YING,
                     Rank.LIAN, Rank.PAI, Rank.GONG, Rank.ZHA, Rank.LEI, Rank.QI)
_STAR1_IMPORTANCE = {Rank.SI: 5, Rank.JUN: 4, Rank.SHI: 3, Rank.LV: 2,
                     Rank.TUAN: 2, Rank.YING: 2, Rank.LIAN: 1, Rank.PAI: 1,
                     Rank.GONG: 1, Rank.ZHA: 6, Rank.LEI: 1, Rank.QI: 3}
_STAR1_KIND_INDEX = {r: i for i, r in enumerate(_STAR1_KIND_ORDER)}

# APK 开局库增强（2026-09-16）
# 官方 libjunqi.so 的中心行营黄金翻棋格 (对齐 0x5a3c0)
APK_CENTER_CAMP_FLIP_POSITIONS: Set[Tuple[int, int]] = frozenset({
    (3, 1), (3, 3), (4, 2),   # 上半场中心行营辐射位
    (7, 2), (8, 1), (8, 3),   # 下半场中心行营辐射位
})

# APK 开局优先翻棋加分（纯开局无子时生效）。
# ⚠ 量纲说明：APK 原生估价尺度是 SI=2560 / 连长=40，那里的 +150 是**强偏好**；
#   本引擎的翻棋排序尺度是 safety_score 20_000 + camp_expansion 最高 120_000，
#   故 +150 在这里只相当于 **同分并列时的 tie-break**（占 0.75%）。
#   实测效果：纯开局首翻 8/8 落在黄金格（改前落非黄金格），但 6 个黄金格同分，
#   实际恒定选到生成序第一的 (3,1) ⇒ 开局被确定化。要保持多样性需给格位分档或
#   允许随机 tie-break，属待决策项，本轮不改行为。
APK_FLIP_ROOT_BONUS = 150.0

# 切片 1（C++ 移植）：evaluate_expert 是否走 C++ 快路径。
# 默认值由等价性验收结果决定 —— 见
# docs/05-ExecutionPlans/CPP_EXPERT_ENGINE_PORT_PLAN.md 第五节"切片 1"。
# 验收判据：scratch/perf_baseline.py 的 280 状态估值 + 30 次 depth=2 搜索
# 逐位一致（阈值 1e-9）。未通过将本常量置回 False，C++ 侧即被完全旁路。
DEFAULT_USE_CPP_EVAL = True

# 切片 2：QSearch 是否走 C++ 快路径。验收判据同切片 1
# （perf_baseline 逐位一致 + 决策不变），未通过将本常量置回 False。
# C++ 侧未实现 QTT（纯缓存），故开启后 stats.qnodes 会**高于** Python 路径。
DEFAULT_USE_CPP_QSEARCH = True

# 切片 3：`_negamax` + 机会节点整棵子树是否走 C++。
# **根循环（IDS / degraded / avoid / root_scores）仍留在 Python** —— 它承载了
# 最易错的语义，留在原地可零风险复用既有实现与既有测试。
DEFAULT_USE_CPP_SEARCH = True


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
    # P0 修复（审查 C2/C3/C13，2026-09-15）
    degraded: bool = False             # 首层未完成 / 根循环层内被打断 / 兜底选招
    root_scores_bounded: list[bool] = field(default_factory=list)
    # True 表示该根节点分数只是**上界**（被 alpha 窗口截断），不是精确值。
    # 排序仍然有效（上界低于另一动作的精确分 ⇒ 确实更差），但不可当绝对分展示。
    #
    # degraded 为 True 时的两点含义（返回值仍是合法动作、分值仍然有限）：
    #   - `score` 来自最后一次有产出的层，是真实值的**下界**估计（未搜完的层，
    #     未搜索的动作可能更优）；
    #   - `root_scores` 的候选集合可能是**截断**的（不覆盖全部合法动作），
    #     逐个分值与排序依然有效，但覆盖面不完整——top-N 展示与蒸馏软分布
    #     应结合本标记判断。


# ----------------------------------------------------------------- 迭代加深时限判据

def ids_next_depth_estimate(elapsed_ms: float, last_depth_ms: float) -> float:
    """外推"下一层大约要跑多久"（毫秒）。

    采用迭代加深的经典假设——**累计耗时会随深度大约翻倍**，
    因此下一层耗时不小于：本层耗时，也不小于累计耗时的一半。
    取两者较大值，且**不做比值外推**。
    """
    return max(float(last_depth_ms), float(elapsed_ms) * 0.5)


def should_stop_ids(elapsed_ms: float, budget_ms: float,
                    last_depth_ms: float, prev_depth_ms: float = 0.0) -> bool:
    """是否应在开始下一层迭代加深前停止（纯函数，可单测）。

    判据：**已用时间 + 下一层预估耗时 > 预算**。

    为什么不是"总耗时 > 预算的 25%"（旧实现）：与"下一深度预计超时"的语义
    完全不符 —— 1000ms 预算下 depth-1 只要用掉 260ms 就退出，剩余 740ms 全浪费。

    为什么**也不做比值外推**（第一版修复的做法）：实测 endgame 局面
    d1/d2/d3/d4 = 2.6/26.9/173/598 ms，实测比值 6.1 会把 d4 估成 891ms，
    于是 1000ms 预算下在 d3 就停 —— 只用了 171ms（17% 预算），比旧实现还少搜一层。
    改用"翻倍"假设（估计值 = max(本层, 累计/2)）后：
    d3 后估计 173ms，173+173 < 1000 继续 → d4 跑完 598ms，
    之后估计 425ms，600+425 > 1000 停止 —— 拿到与旧实现相同的深度，且预算用满 60%。

    `prev_depth_ms` 仅为兼容旧签名保留，不参与计算。
    """
    if budget_ms <= 0:
        return False
    estimate = ids_next_depth_estimate(elapsed_ms, last_depth_ms)
    return elapsed_ms + estimate > budget_ms


class ExpertSearchEngine:
    """专家级军棋翻棋搜索引擎。"""

    def __init__(self, weights: Optional[EvalWeights] = None,
                 tt_size_power: int = 18, seed: Optional[int] = None,
                 qsearch_depth: int = DEFAULT_QSEARCH_DEPTH,
                 use_qtt: bool = True, qtt_size_power: int = 16,
                 use_cpp_eval: Optional[bool] = None,
                 use_cpp_qsearch: Optional[bool] = None,
                 use_cpp_search: Optional[bool] = None):
        self.w = weights or EvalWeights()
        self.tt = TranspositionTable(size_power=tt_size_power)
        # 注：专家搜索核心（Expectiminimax + alpha-beta）为完全确定性算法，
        # self.rng 保留仅用于兼容外部调用；搜索过程不依赖伪随机数。
        self.rng = random.Random(seed)
        self.qsearch_depth = qsearch_depth

        # 切片 1：C++ evaluate_expert 快路径开关（None = 取模块默认）。
        # 权重对象需转成 C++ 结构，构造成本数微秒，故按 self.w 身份缓存。
        self.use_cpp_eval = DEFAULT_USE_CPP_EVAL if use_cpp_eval is None else use_cpp_eval
        self._cpp_w = None
        self._cpp_w_src = None

        # 切片 2：C++ QSearch 快路径开关。
        self.use_cpp_qsearch = (DEFAULT_USE_CPP_QSEARCH if use_cpp_qsearch is None
                                else use_cpp_qsearch)
        self._cpp_qs = None

        # 切片 3：C++ 搜索子树（negamax + Star1 机会节点）开关。
        self.use_cpp_search = (DEFAULT_USE_CPP_SEARCH if use_cpp_search is None
                               else use_cpp_search)
        self._cpp_search = None

        # QSearch 专用置换表（P1，2026-09-15）。
        # **必须与主表分离**：主表 TTEntry.depth 是"剩余搜索深度"，qsearch 的
        # depth_left 是"剩余吃子链长度"，语义不同。共用一张表会让 qsearch 写入的
        # depth_left=10 条目被 _negamax 的 depth=2 查询命中（entry.depth >= depth
        # ⇒ 10 >= 2），直接返回错误分数。
        # 实测（残局 ply=90）：(zobrist, depth_left) 重复率 54.3%（35538 → 16253），
        # 值得付出每次 7.4us 的 zobrist 成本，换取 evaluate_expert（104.7us）的命中豁免。
        self.use_qtt = use_qtt
        self.qtt = TranspositionTable(size_power=qtt_size_power)

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
        """清空杀手着法与历史表（含 QSearch 置换表）。"""
        self.killers = [[] for _ in range(MAX_SEARCH_DEPTH)]
        self.history.clear()
        self.qtt.clear()
        if self._cpp_search is not None:
            self._cpp_search.clear_heuristics()

    # ------------------------------------------------------------- 估值派发

    def _cpp_weights(self):
        """返回 self.w 对应的 C++ 权重对象（按身份缓存）；不可用时关闭快路径。"""
        if self._cpp_w is None or self._cpp_w_src is not self.w:
            try:
                self._cpp_w = make_cpp_weights(self.w)
                self._cpp_w_src = self.w
            except Exception:  # noqa: BLE001 - 构建失败即永久退回 Python
                self.use_cpp_eval = False
                self._cpp_w = None
                self._cpp_w_src = None
        return self._cpp_w

    def _eval(self, state: GameState, seat: int) -> float:
        """叶子/机会节点估值：按 use_cpp_eval 在 C++ 与 Python 间派发。

        等价于 ``evaluate_expert(state, seat, self.w)``。C++ 侧不可用时
        ``eval_expert_auto`` 会自行透明降级，无需调用方处理。
        """
        if self.use_cpp_eval:
            return eval_expert_auto(state, seat, self.w,
                                    cpp_w=self._cpp_weights())
        return evaluate_expert(state, seat, self.w)

    # ------------------------------------------------------------- C++ 搜索子树（切片 3）

    def _cpp_search_engine(self):
        """返回复用的 C++ 搜索子树引擎；不可用时关闭开关。"""
        if self._cpp_search is None:
            try:
                self._cpp_search = make_cpp_search(self.w, self.qsearch_depth)
            except Exception:  # noqa: BLE001
                self.use_cpp_search = False
                self._cpp_search = None
        return self._cpp_search

    def _sync_cpp_search_stats(self):
        """把 C++ 子树的统计与 stopped 回写到 Python 侧 stats。"""
        cs = self._cpp_search
        if cs is None:
            return
        self.stats.nodes = cs.nodes
        self.stats.qnodes = cs.qnodes
        self.stats.chance_nodes = cs.chance_nodes
        self.stats.star1_cutoffs = cs.star1_cutoffs
        self.stats.pvs_researches = cs.pvs_researches
        self.stats.tt_hits = cs.tt_hits
        if cs.stopped:
            self.stopped = True

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
            attacker_val = self.w.piece.get(mover_rank, 30.0)

            # 2. 吃明子: MVV-LVA 排序 (Most Valuable Victim - Least Valuable Attacker)
            if target is not None and target.revealed and target.color != my:
                res = battle(mover_rank, target.rank)
                victim_val = self.w.piece.get(target.rank, 30.0)

                # 2.1 行营单向打击特权与安全出营判定
                # (用户核心准则：若出营击杀不会导致丢营，严禁扣分，吃子后下步能回营即属完全控制)
                camp_outstrike_bonus = 0.0
                camp_lose_risk = False
                if is_camp(act.frm):
                    enemy_can_enter = False
                    opp = other(my) if my else None
                    if opp is not None:
                        for np in NEIGHBORS[act.frm]:
                            if np == act.to:
                                continue
                            e = state.board.get(np)
                            if e is not None and e.revealed and e.color == opp and e.rank not in (Rank.LEI, Rank.QI):
                                enemy_can_enter = True
                                break
                        if not enemy_can_enter and is_rail(act.frm):
                            for rk_pos, rk_pc in state.board.items():
                                if rk_pos == act.to:
                                    continue
                                if rk_pc.revealed and rk_pc.color == opp and is_rail(rk_pos) and rk_pc.rank not in (Rank.LEI, Rank.QI):
                                    if rk_pos[0] == act.frm[0] or rk_pos[1] == act.frm[1]:
                                        enemy_can_enter = True
                                        break
                    if enemy_can_enter:
                        camp_lose_risk = True
                    else:
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
                    score = 500_000.0 + camp_outstrike_bonus + victim_val * 100.0 - attacker_val
                    if camp_lose_risk:
                        score -= 100_000.0
                    return score
                elif res == BOTH_DIE:
                    # 兑子: 炸弹或同级兑换 (含小子贴身拆弹战略加分)
                    score = 300_000.0 + camp_outstrike_bonus + bomb_suicide_bonus + victim_val * 100.0 - attacker_val
                    if camp_lose_risk:
                        score -= 100_000.0
                    return score
                else:
                    # 亏损/送吃
                    return -100_000.0 + victim_val - attacker_val

            # 3. 进行营避险 / 占营进驻核心据点 (实证：前 20 手走子 82.7% 为进营)
            if is_camp(act.to):
                if is_camp(act.frm):
                    # 3.0 营间互窜 (Camp-to-camp idle shuttle)：
                    # 在两营之间无吃子往复闲走，既不增加净占营数，又放弃既有据点与拓荒，赋予重度负优先级！
                    return -200_000.0
                # 进驻空行营是绝对免死与据点化的战略特权，优先级高于普通翻棋
                camp_prio = 250_000.0 if act.to not in state.board else 150_000.0
                # 若进营位置邻接敌方炸弹，具有"卡营逼弹"免死压制特权
                opp = other(my) if my else None
                pin_bomb_bonus = 0.0
                if opp is not None and act.to not in state.board:
                    for np in NEIGHBORS[act.to]:
                        e = state.board.get(np)
                        if e is not None and e.revealed and e.color == opp and e.rank == Rank.ZHA:
                            pin_bomb_bonus = 50_000.0
                            break
                return camp_prio + pin_bomb_bonus + self.w.piece.get(mover_rank, 20.0)

            # 3.1 离开行营走入空地 (非吃子出营：除逃营外，无故弃营属于严重失误)
            if is_camp(act.frm) and not is_camp(act.to):
                if mover_rank == Rank.ZHA:
                    return -350_000.0  # 严禁炸弹弃营乱窜
                elif mover_rank >= Rank.SHI:
                    return -250_000.0  # 严禁大子弃营乱走
                else:
                    opp = other(my) if my else None
                    if opp is not None:
                        for np in NEIGHBORS[act.frm]:
                            e = state.board.get(np)
                            if e is not None and e.revealed and e.color == opp and e.rank not in (Rank.LEI, Rank.QI):
                                if battle(e.rank, mover_rank) in (ATTACKER_WINS, BOTH_DIE):
                                    return -350_000.0  # 敌大子窥视下出营送死严惩
                    return -200_000.0  # 普通弱子无故弃营也予重罚，优先于营内拓荒或伏击

            # 3.2 大子向空行营安全中继推进 (2026-09-06 实战修复：截图19手师长安全挺进 (6,1)->(6,2)->中营)
            if not is_camp(act.frm) and not is_camp(act.to) and mover_rank >= Rank.SHI:
                # 检查落点 act.to 是否是通达空行营的安全中继站 (1步或2步可达空营)
                reaches_empty_camp = False
                for n1 in NEIGHBORS[act.to]:
                    if is_camp(n1) and n1 not in state.board:
                        reaches_empty_camp = True
                        break
                    if n1 not in state.board:
                        for cp in NEIGHBORS[n1]:
                            if is_camp(cp) and cp not in state.board:
                                reaches_empty_camp = True
                                break
                    if reaches_empty_camp:
                        break

                if reaches_empty_camp:
                    opp = other(my) if my else None
                    is_safe = True
                    if opp is not None:
                        for np in NEIGHBORS[act.to]:
                            e = state.board.get(np)
                            if e is not None and e.revealed and e.color == opp and e.rank not in (Rank.LEI, Rank.QI):
                                if battle(e.rank, mover_rank) in (ATTACKER_WINS, BOTH_DIE):
                                    is_safe = False
                                    break
                    if is_safe:
                        return 200_000.0 + attacker_val * 100.0

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
            
            # APK 开局库增强 (2026-09-16): 优先翻中心行营周围的黄金暗子格
            # 仅在纯开局阶段 (全盘无己方明子) 时生效，对齐官方 libjunqi.so 的 0x5a3c0 逻辑
            apk_flip_bonus = 0.0
            revealed_friendly_pieces = sum(1 for p in state.board.values() 
                                          if p.revealed and p.color == my)
            if revealed_friendly_pieces == 0 and pos in APK_CENTER_CAMP_FLIP_POSITIONS:
                apk_flip_bonus = APK_FLIP_ROOT_BONUS
            
            # 1. 依托行营辐射拓荒 (实证：96.2% 邻营翻棋，开局首翻即据点)
            # 用户核心战略：依托己方已控行营，向周围暗子辐射拓荒翻棋
            # 翻出自子可立即协同，翻出敌子被营内子单向就近扑杀无损失 (开局 50.1% 吃子源自行营扑杀)
            camp_expansion_bonus = 0.0
            has_friendly_camp = False
            friendly_camp_combat_rank = 0
            has_safe_empty_camp = False
            for np in NEIGHBORS[pos]:
                if is_camp(np):
                    cb = state.board.get(np)
                    if cb is not None and cb.revealed and cb.color == my:
                        has_friendly_camp = True
                        if cb.rank not in (Rank.LEI, Rank.QI):
                            friendly_camp_combat_rank = max(friendly_camp_combat_rank, cb.rank)
                    elif np not in state.board:
                        enemy_around_camp = any(
                            (e := state.board.get(enp)) is not None and e.revealed and e.color == opp
                            for enp in NEIGHBORS[np]
                        )
                        if not enemy_around_camp:
                            has_safe_empty_camp = True

            if has_friendly_camp:
                # 依托己方已控据点邻域拓荒：战略特权优先级 (介于空营挺进与普通翻棋之间)
                rank_boost = 15_000.0 if friendly_camp_combat_rank >= Rank.SHI else 5_000.0
                camp_expansion_bonus = 100_000.0 + rank_boost
                if has_safe_empty_camp:
                    camp_expansion_bonus += 20_000.0
            elif has_safe_empty_camp:
                camp_expansion_bonus = 40_000.0

            # 2. 开局领地与中前场咽喉偏好 (对称结构：避免盲目翻底线深处暗子)
            territory_bias = 0.0
            if 2 <= r <= 4 or 7 <= r <= 9:
                territory_bias = 25_000.0  # 行营核心辐射带
            elif r in (5, 6):
                territory_bias = 20_000.0  # 前线关隘与中路铁路
            elif r in (1, 10):
                territory_bias = 5_000.0   # 次底线
            elif r in (0, 11):
                territory_bias = -20_000.0 # 底线边角，开荒优先级较低

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

            if territory_bias < -50_000.0 and friendly_guards > 0:
                territory_bias = 10_000.0 * friendly_guards

            if friendly_guards > enemy_threats:
                safety_score = 60_000.0 + (friendly_guards - enemy_threats) * 5000.0
            elif enemy_threats > friendly_guards:
                safety_score = 500.0 - enemy_threats * 1000.0
            else:
                safety_score = 20_000.0

            # 合并 APK 开局库偏好 (纯开局阶段黄金格优先翻棋)
            total_score = safety_score + camp_expansion_bonus + territory_bias + apk_flip_bonus
            return total_score

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
        # 性能备注：内层翻棋候选的几率解析期望是单局 ~50 万次 evaluate_expert 的
        # 来源。2026-09-13 曾试验内层收紧（2/3 得 58s/局、3/5 得 70s/局），
        # 但 2/3 会使深度 3 下用户复盘验证的战术场景（test_camp_tactics_fix
        # 第 19 手）决策翻转，3/5 仅多 4% 且引入行为差异面，均予回退。
        # 教师决策质量优先于延迟；大幅降耗需走 C++ 移植（见 CHANGELOG 第七节）。
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
        """静态搜索 (Quiescence Search)：专用于在叶子节点解决吃子与战术震荡。

        对齐原版 APK 0x5a678 + 0x591d8，并叠加 P1.C 增强：
        1. 生成吃子/吃旗动作（Captures Only）。**P1.C 例外**：已明大子在营外被实质威胁时，
           额外生成"退入空行营避险"的静步候选（见下方生成段与 rank 阈值说明）；
        2. Delta Pruning 剪枝加速。局部 delta 只作用于"有被吃目标"的走法，避险静步不受它影响；
           全局 big_delta 上界仍成立 —— 避险的收益不超过被救子力自身价值；
        3. 延伸至更深交火线（默认 16 ply），彻底消除地平线反杀盲区；
        4. **终止性**不再只依赖"吃子必然减少子力"这一单调量，由 `depth_left <= 0` 守卫
           与"避险源必须非行营、目标必须为空行营"共同保证（避险次数有上限）。
        """
        # 切片 2：整棵静态搜索子树交给 C++（一次序列化，零每节点桥接开销）。
        # C++ 侧不实现 QTT（纯缓存，不改变返回值），故 qnodes 会比 Python 路径高。
        if self.use_cpp_qsearch:
            if self._cpp_qs is None:
                try:
                    self._cpp_qs = make_cpp_qsearch(self.w, self.qsearch_depth)
                except Exception:  # noqa: BLE001
                    self.use_cpp_qsearch = False
                    self._cpp_qs = None
            if self._cpp_qs is not None:
                val, qn = qsearch_auto(state, alpha, beta, depth_left,
                                       cpp_qs=self._cpp_qs)
                if val is not None:
                    self.stats.qnodes += qn
                    return val
                self.use_cpp_qsearch = False  # C++ 失败即永久退回 Python

        self.stats.qnodes += 1

        # 终局检查
        if state.is_terminal():
            if state.winner == -1:
                return 0.0
            win = WIN_SCORE - state.ply
            return win if state.winner == state.turn else -win

        # QSearch 置换表查询（与主表分离，理由见 __init__ 注释）。
        # 命中即省下一次 evaluate_expert（104.7us）与整棵吃子子树。
        orig_alpha = alpha
        zkey: Optional[int] = None
        if self.use_qtt:
            zkey = compute_zobrist(state)
            tt_val, _ = self.qtt.lookup(zkey, depth_left, alpha, beta)
            if tt_val is not None:
                return tt_val

        # Stand-Pat 评估剪枝
        stand_pat = self._eval(state, state.turn)
        if stand_pat >= beta:
            if zkey is not None:
                self.qtt.store(zkey, depth_left, beta, FLAG_LOWER_BOUND)
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if depth_left <= 0:
            # 静态近似值：既非上界也非下界，**不入表**（入了会污染真实分数）
            return stand_pat

        # Delta Pruning (大 Delta 剪枝)
        # 若即便吃掉全盘最贵子力（或军旗），加上安全裕量后依然无法超越 alpha，则提前剪枝
        max_piece_val = max(self.w.piece.values()) if (self.w and self.w.piece) else 100.0
        big_delta = max_piece_val + 200.0
        if stand_pat + big_delta < alpha:
            if zkey is not None:
                self.qtt.store(zkey, depth_left, alpha, FLAG_UPPER_BOUND)
            return alpha

        # 生成吃子动作（吃敌方明子或吃旗）与高危大子进营避难动作 (P1.C 增强)
        acts = state.legal_actions()
        tactical_moves: list[Action] = []
        my = state.my_color()

        for a in acts:
            if a.kind != "move":
                continue
            target = state.board.get(a.to)
            # 吃明子 (包含吃旗)
            if target is not None and target.revealed and target.color != my:
                tactical_moves.append(a)
            elif is_camp(a.to) and not is_camp(a.frm):
                # P1.C 增强：高危大子逃入行营避险。
                # rank 阈值说明：Rank 枚举里 ZHA=20、LEI=21、QI=22 都 ≥ SHI=11，
                # 所以 `>= Rank.SHI` 实际选中 {师长..司令, 炸弹, 地雷, 军旗}；
                # `or == ZHA` 因此是冗余的。实测地雷与军旗**不可移动**（legal_actions 为空），
                # 故实际生效集合 = {师长, 军长, 司令, 炸弹}。这里保留宽条件不影响行为，
                # 但若将来放开雷/旗移动规则，需重新确认语义。
                mover = state.board.get(a.frm)
                if mover is not None and mover.revealed and (mover.rank >= Rank.SHI or mover.rank == Rank.ZHA):
                    is_threatened = any(
                        (e := state.board.get(np)) is not None and e.revealed and e.color != my
                        and e.rank not in (Rank.QI, Rank.LEI)
                        and battle(e.rank, mover.rank) in ("attacker_wins", "both_die")
                        for np in NEIGHBORS[a.frm]
                    )
                    if is_threatened:
                        tactical_moves.append(a)

        if not tactical_moves:
            if zkey is not None:
                # stand_pat < beta 已在上面保证；高于原 alpha ⇒ 精确值，否则 fail-low 上界
                flag = FLAG_EXACT if stand_pat > orig_alpha else FLAG_UPPER_BOUND
                self.qtt.store(zkey, depth_left, stand_pat, flag)
            return stand_pat

        # MVV-LVA 排序
        tactical_moves.sort(key=lambda a: self._score_action(a, state, 0, None), reverse=True)

        for a in tactical_moves:
            # 局部 Delta 剪枝 (针对具体被吃子力价值)
            target = state.board.get(a.to)
            if target is not None and target.rank != Rank.QI:
                victim_val = self.w.piece.get(target.rank, 30.0)
                if stand_pat + victim_val + 50.0 < alpha:
                    continue

            child = state.apply(a)
            score = -self._qsearch(child, -beta, -alpha, depth_left - 1)
            if score >= beta:
                if zkey is not None:
                    self.qtt.store(zkey, depth_left, beta, FLAG_LOWER_BOUND)
                return beta
            if score > alpha:
                alpha = score

        if zkey is not None:
            flag = FLAG_EXACT if alpha > orig_alpha else FLAG_UPPER_BOUND
            self.qtt.store(zkey, depth_left, alpha, flag)
        return alpha

    # ------------------------------------------------------------- 翻棋几率节点 (Chance Node / Star1)

    def _evaluate_chance_flip(self, state: GameState, flip_act: Action,
                             depth: int, ply_depth: int, alpha: float, beta: float,
                             path_history: set[int]) -> float:
        """对翻棋动作进行几率节点 (Chance Node) 期望计算与 Star1 剪枝。"""
        self.stats.chance_nodes += 1

        pos = flip_act.frm

        if self.use_cpp_search:
            cs = self._cpp_search_engine()
            if cs is not None:
                try:
                    val = cs.chance_flip_blob(encode_state_blob(state),
                                              pos[0] * 5 + pos[1], depth,
                                              ply_depth, alpha, beta)
                    self._sync_cpp_search_stats()
                    return val
                except Exception:  # noqa: BLE001
                    self.use_cpp_search = False
        rem = state.remaining_types()
        total_hidden = sum(rem.values())
        if total_hidden <= 0:
            # 无暗子构成，安全回退
            child = state.apply(flip_act)
            return -self._negamax(child, depth - 1, ply_depth + 1, -beta, -alpha, path_history)

        # 几率前沿截断判断 (P1.A 增强):
        # 1. depth <= 1 或 ply_depth >= 2: 无足够搜索预算，直接由公共状态求精确解析期望；
        # 2. ply_depth == 1 (根节点的下一回合):
        #    - 若处于战术交火区且 depth >= 3: 允许受限 Star1 展开 (Top-3 概率身份以 depth - 2 递归，其余长尾做静态估值)；
        #    - 其余远离战场的翻棋保持极速解析期望，彻底杜绝深层 (24)^d 指数爆炸。
        def _is_tactical_flip_zone(flip_p: tuple[int, int]) -> bool:
            for np_ in NEIGHBORS[flip_p]:
                if is_camp(np_):
                    return True
                nb = state.board.get(np_)
                if nb is not None and nb.revealed and nb.rank not in (Rank.LEI, Rank.QI):
                    return True
            return False

        allow_restricted_star1 = (ply_depth == 1 and depth >= 3 and _is_tactical_flip_zone(pos))

        if (depth <= 1 or ply_depth >= 1) and not allow_restricted_star1:
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
                    child_val = self._eval(child, child.turn)

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

        max_search_outcomes = 3 if allow_restricted_star1 else len(outcomes)
        sub_depth = (depth - 2) if allow_restricted_star1 else (depth - 1)

        # 受限展开时的**搜索子集**（理由见文件顶部 _STAR1_IMPORTANCE）。
        # 只改变"哪 3 个身份被递归搜索"：累加顺序与 Star1 剪枝时机保持原样。
        search_slots: Optional[set[int]] = None
        if allow_restricted_star1:
            ranked = sorted(
                range(len(outcomes)),
                key=lambda i: (-_STAR1_IMPORTANCE[outcomes[i][1]],
                               -outcomes[i][2],
                               0 if outcomes[i][0] == "r" else 1,
                               _STAR1_KIND_INDEX[outcomes[i][1]]))
            search_slots = set(ranked[:max_search_outcomes])

        for idx, (clr, rk, prob) in enumerate(outcomes):
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

            # 递归搜索子节点或长尾静态估值
            if search_slots is None or idx in search_slots:
                v = -self._negamax(child, sub_depth, ply_depth + 1, -WIN_SCORE, WIN_SCORE, path_history)
            else:
                if child.is_terminal():
                    if child.winner == -1:
                        child_val = 0.0
                    else:
                        win = WIN_SCORE - child.ply
                        child_val = win if child.winner == child.turn else -win
                else:
                    child_val = self._eval(child, child.turn)
                v = -child_val

            expected_value += prob * v
            remaining_prob -= prob

        return expected_value

    # ------------------------------------------------------------- 核心 Negamax 搜索

    def _negamax(self, state: GameState, depth: int, ply_depth: int,
                 alpha: float, beta: float, path_history: set[int]) -> float:
        """带置换表、静态搜索、杀手/历史启发与 Star1 几率剪枝的 Negamax 搜索。

        切片 3：`use_cpp_search` 时整棵子树交由 C++（Python 只保留根循环）。
        """
        if self.use_cpp_search:
            cs = self._cpp_search_engine()
            if cs is not None:
                try:
                    val = cs.negamax_blob(encode_state_blob(state), depth,
                                          ply_depth, alpha, beta)
                    self._sync_cpp_search_stats()
                    return val
                except Exception:  # noqa: BLE001
                    self.use_cpp_search = False

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

        # 2. 叶子节点转入静态搜索 (QSearch)
        # P3 性能（审查 P5）：原实现在 depth<=0 判定**之前**无条件算 Zobrist，
        # 而叶子节点既不需要重复检测也不需要置换表，等于每个叶子白算一次
        # （compute_zobrist 内部遍历全盘 + 剩余子力池指纹）。
        if depth <= 0:
            return self._qsearch(state, alpha, beta, self.qsearch_depth)

        # 3. 树内重复局面检测 (Repetition Detection)
        zobrist_key = compute_zobrist(state)
        if zobrist_key in path_history:
            # 重复走子判和（0 分）
            return 0.0

        # 4. 置换表查询 (TT Lookup)
        orig_alpha = alpha
        tt_val, tt_move = self.tt.lookup(zobrist_key, depth, alpha, beta)
        if tt_val is not None and ply_depth > 0:
            return tt_val

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
               time_limit_ms: int = 0, avoid: Optional[set] = None,
               qsearch_depth: Optional[int] = None,
               as_evaluator: bool = False,
               exact_root_scores: bool = False
               ) -> tuple[Optional[Action], float, SearchStats]:
        """迭代加深搜索 (Iterative Deepening Search)。

        参数:
            state: 当前游戏局面
            max_depth: 最大搜索深度 (ply)
            time_limit_ms: 限时 (毫秒)，0 为不限时纯按深度
            avoid: 根节点需回避的可观察局面键集合 (防送循环)
            qsearch_depth: 自定义静态搜索深度上限 (None 则采用 self.qsearch_depth)
            as_evaluator: 估值器模式。作为子节点估值 oracle 调用时必须为 True：
                跳过"唯一合法走法直接返回 0 分"的决策捷径，返回该强制走法
                的真实搜索分（否则定化子局的必胜/必败线会被误估为 0）。
            exact_root_stats: 根节点每个动作都用**全窗口**搜索，得到可互相比较的
                精确分（C3 修复）。默认 False（PVS 收窄窗口，score 可能是上界，
                见 stats.root_scores_bounded）。需要 top-N 候选或做蒸馏教师
                软分布时必须设为 True，否则 fail-low 动作的上界会被误当精确分。

        返回:
            (best_action, score, stats)。**time_limit_ms > 0 时应先查 `stats.degraded`**：
            为 True 表示预算不足（根循环在层内被打断，或首层未完成），返回的
            (best_action, score) 是**最后一次有产出的层**的部分最优——它是真实值的
            有效下界估计，但该层的候选集合可能是**截断**的，即
            `stats.root_scores` 可能不覆盖全部合法动作（排序仍有效）。
            超时兜底时 score 为 0.0；**永不返回 ±inf**（C2 修复）。
        """
        self.stats = SearchStats()
        self.stopped = False
        start_time = time.perf_counter()

        old_qdepth = self.qsearch_depth
        if qsearch_depth is not None:
            self.qsearch_depth = qsearch_depth

        # 切片 3：C++ 子树每轮搜索重置统计与停止位（置换表跨轮保留，与 Python 一致）
        if self.use_cpp_search and self._cpp_search is not None:
            self._cpp_search.reset_stats()
            self._cpp_search.stopped = False
            self._cpp_search.qsearch_depth = self.qsearch_depth

        try:
            if time_limit_ms > 0:
                self.deadline = start_time + (time_limit_ms / 1000.0)
            else:
                self.deadline = float("inf")

            if self.use_cpp_search and self._cpp_search is not None:
                # C++ 用自己的单调钟；这里给的是"从现在起"的预算，与 self.deadline 同刻
                self._cpp_search.set_deadline_ms(float(time_limit_ms or 0))

            acts = state.legal_actions()
            if not acts or state.is_terminal():
                return None, 0.0, self.stats
            if len(acts) == 1 and not as_evaluator:
                return acts[0], 0.0, self.stats

            best_action: Optional[Action] = None
            best_score: float = -math.inf
            ordered_acts: list[Action] = list(acts)   # 兜底路径的可用引用

            path_history: set[int] = set()

            # 战术确定性优先准则 (用户核心原则：杜绝盲目翻暗棋赌概率)
            # 仅当移动走法属于【实质性吃子/战术制胜】或【进驻空行营】时，享有 0.5 分
            # 确定性优先特权；普通静步闲走严禁压制翻开邻营暗子开拓据点的行动！
            # P3 性能（审查 P5）：原本定义在动作循环体内，每动作每深度重建一次闭包。
            def _is_tactical(act: Action) -> bool:
                if act.kind != "move":
                    return False
                tgt = state.board.get(act.to)
                if tgt is not None and tgt.revealed:
                    return True
                if not is_camp(act.frm) and is_camp(act.to):
                    return True
                return False

            # 每层实测耗时（毫秒），用于外推下一层是否超预算
            depth_ms: list[float] = []

            for d in range(1, max_depth + 1):
                if self.stopped or (time_limit_ms > 0 and time.perf_counter() >= self.deadline):
                    break
                d_start = time.perf_counter()

                # 根节点搜索
                zobrist_key = compute_zobrist(state)
                _, tt_move = self.tt.lookup(zobrist_key, d, -math.inf, math.inf)
                ordered_acts = self._order_actions(acts, state, 0, tt_move or best_action)

                current_d_best_act = ordered_acts[0]
                current_d_best_score = -math.inf
                alpha = -math.inf
                beta = math.inf
                d_scores: list[tuple[Action, float]] = []
                d_bounded: list[bool] = []
                # C13：本层是否在动作循环内被时限打断（区别于"本层完整跑完"）。
                interrupted = False

                for a in ordered_acts:
                    if self.stopped or (time_limit_ms > 0 and time.perf_counter() >= self.deadline):
                        interrupted = True
                        break

                    if exact_root_scores:
                        # C3 修复：多候选展示/蒸馏需要**可互相比较的精确分**。
                        # 默认路径用收窄后的 [alpha, beta) 窗口，fail-low 时返回的
                        # 只是上界；这里对每个根动作都用全窗口搜索。
                        if a.kind == "flip":
                            score = self._evaluate_chance_flip(
                                state, a, d, 0, -math.inf, math.inf, path_history)
                        else:
                            child = state.apply(a)
                            score = -self._negamax(child, d - 1, 1, -math.inf,
                                                   math.inf, path_history)
                        bounded = False
                    else:
                        if a.kind == "flip":
                            score = self._evaluate_chance_flip(
                                state, a, d, 0, alpha, beta, path_history)
                            bounded = False   # 机会节点走 Star1 期望，本身即精确估值
                        else:
                            child = state.apply(a)
                            score = -self._negamax(child, d - 1, 1, -beta, -alpha,
                                                   path_history)
                            # fail-low（score <= alpha）时这个值只是上界
                            bounded = score <= alpha

                    # 避免命中 avoid 集合
                    if avoid and a.kind == "move":
                        from .state import position_key
                        if position_key(state.apply(a)) in avoid:
                            score -= 150.0

                    d_scores.append((a, score))
                    d_bounded.append(bounded)

                    is_better = False
                    if current_d_best_act is None:
                        is_better = True
                    elif a.kind == "flip" and current_d_best_act.kind == "move":
                        if _is_tactical(current_d_best_act):
                            if score > current_d_best_score + 0.5:
                                is_better = True
                        else:
                            if score >= current_d_best_score:
                                is_better = True
                    elif a.kind == "move" and current_d_best_act.kind == "flip":
                        if _is_tactical(a):
                            if score >= current_d_best_score - 0.5:
                                is_better = True
                        else:
                            if score > current_d_best_score:
                                is_better = True
                    elif score > current_d_best_score:
                        is_better = True

                    if is_better:
                        current_d_best_score = score
                        current_d_best_act = a
                        if score > alpha:
                            alpha = score

                depth_ms.append((time.perf_counter() - d_start) * 1000.0)

                if self.stopped:
                    # 子搜索内部已置 stopped：整层作废（既有语义），不更新任何统计
                    break

                # C13：根循环在**层内**被时限打断 ⇒ 本层未搜完。
                # 与 self.stopped 不同，这条路径此前完全无人处理：本层被当作
                # "已完成"，于是 degraded 不置位，并以 FLAG_EXACT 写入 depth-d
                # 根节点置换表条目（实际只是下界）。
                partial_layer = interrupted
                if partial_layer:
                    self.stats.degraded = True

                if d_scores:
                    # 决策仍沿用本层的（部分）最优——这是迭代加深的既定语义：
                    # 层内首个动作是 tt_move/上一层最优，走全窗口搜索，其余动作也都会
                    # 被验证，故"部分最优"是真实值的有效**下界**估计，比退回浅一层
                    # 更接近真相。A/B 实测（`scripts/ab_search_compare.py`）证明退回
                    # 浅一层会丢信息：处女局面 d=1 的全部翻棋同分 0.0，退回即等于弃权。
                    best_action = current_d_best_act
                    best_score = current_d_best_score
                    self.stats.max_depth = d
                    bounded_by_act = {a: b for (a, _s), b in zip(d_scores, d_bounded)}
                    order = sorted(range(len(d_scores)), key=lambda i: d_scores[i][1],
                                   reverse=True)
                    sorted_roots = [d_scores[i] for i in order]
                    if best_action is not None:
                        best_tuple = next((t for t in sorted_roots if t[0] == best_action), (best_action, best_score))
                        sorted_roots = [best_tuple] + [t for t in sorted_roots if t[0] != best_action]
                    self.stats.root_scores = sorted_roots
                    self.stats.root_scores_bounded = [
                        bounded_by_act.get(a, False) for a, _s in sorted_roots]
                    # 未搜完的层只能记**下界**：未搜索的动作可能更优。绝不能标 EXACT，
                    # 否则后续对同一局面（作为子树出现）的搜索会在 TT 查询处把它当精确值
                    # 直接返回，污染真实分数与 PV。
                    if math.isfinite(best_score) and best_action is not None:
                        self.tt.store(zobrist_key, d, best_score,
                                      FLAG_LOWER_BOUND if partial_layer else FLAG_EXACT,
                                      best_action)
                    # 注意：partial_layer 时 root_scores 是**截断**的候选集合——
                    # 集合内每个分值本身有效，但不完整。消费方（GUI top-N、
                    # train_search_distill 的 softmax）应结合 stats.degraded 判断。

                if partial_layer:
                    # 本层一个动作都没搜完时 d_scores 为空：无新信息，保留上一层结果
                    # （若连一层都没有，best_action 为 None，走下方 C2 兜底）。
                    break

                # 动态时间预算早停控制（对齐原版 APK 0x5ac3a 的意图：
                # 下一深度预计会超时就安全退出，保留当前深度完整的最优决策）。
                # C1 修复：旧判据是"总耗时 > 预算 25%"，与上述语义完全不符，
                # 会把大部分预算白白浪费掉（详见 should_stop_ids 注释）。
                elapsed_now = (time.perf_counter() - start_time) * 1000.0
                if should_stop_ids(elapsed_now, float(time_limit_ms or 0),
                                   depth_ms[-1],
                                   depth_ms[-2] if len(depth_ms) > 1 else 0.0):
                    break

            # C2 修复：首层未跑完就超时时，旧实现返回 (acts[0], -inf)；
            # 该 -inf 会在 hybrid_engine 取负成 +inf 参与排序，并被
            # train_search_distill 的全量 softmax 蒸馏，直接污染教师标签。
            if best_action is None or not math.isfinite(best_score):
                self.stats.degraded = True
                fallback = best_action or ordered_acts[0] or acts[0]
                self.stats.time_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return fallback, 0.0, self.stats

            self.stats.time_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return best_action, best_score, self.stats
        finally:
            self.qsearch_depth = old_qdepth
