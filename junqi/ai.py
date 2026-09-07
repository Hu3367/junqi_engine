"""决策核心：PIMC 完全信息蒙特卡洛 + negamax(alpha-beta) + 公开信息估值。

不完全信息处理：每个决策点采样 K 个与公开信息一致的完整世界（暗子
身份精确均匀采样），世界内按完全信息搜索，根动作取跨世界期望。
选择 PIMC 而非期望估值的原因：期望法会给"翻开自己的子"虚增物质分，
导致 AI 迷恋翻子、错失免费吃子；PIMC 叶子按世界真实子力计分，无此偏差。
"""
from __future__ import annotations

import math
import os
import random
import time

from .analysis import fortress_score
from .config import EvalWeights, SearchConfig
from .eval_expert import evaluate_expert
from .rules import (CAMPS, COMPOSITION, NEIGHBORS, Rank, battle, is_camp,
                    is_hq, other)
from .apk_agent import ApkNativeAgent
from .search import ExpertSearchEngine
from .state import WIN_SCORE, Action, GameState, position_key
from .tt import TranspositionTable
from .zobrist import compute_zobrist

REP_PENALTY = 150.0    # 走成"即将循环判和"局面的罚分（高于常规战术分，低于胜负分）


def evaluate(state: GameState, seat: int, w: EvalWeights) -> float:
    """公开信息估值（seat 视角）。暗子按剩余池精确构成摊派期望价值。
    注意：剩余池 = 总构成 − 明子 − 阵亡，与“存活数 − 明子数”不同口径，
    后者在公共局面（存在暗子）下会双减阵亡子造成系统性低估；
    PIMC 全翻开世界里两者重合，故该修正在世界内搜索中行为不变。"""
    my = state.seat_color[seat]
    opp = other(my)
    my_revealed = 0.0
    opp_revealed = 0.0
    for pc in state.board.values():
        if pc.revealed:
            v = w.piece[pc.rank]
            if pc.color == my:
                my_revealed += v
            else:
                opp_revealed += v
    hidden = any(not pc.revealed for pc in state.board.values())
    if hidden:
        # 暗子池精确构成（公开信息）：按颜色将池价值摊派给双方，
        # 位置与身份细节不可知，只计入物质期望，不产生虚假的翻子增值。
        my_pool = opp_pool = 0.0
        for (clr, rk), n in state.remaining_types().items():
            v = n * w.piece[rk]
            if clr == my:
                my_pool += v
            elif clr == opp:
                opp_pool += v
        score = (my_revealed + my_pool) - (opp_revealed + opp_pool)
    else:
        score = my_revealed - opp_revealed

    my_revealed_pieces = [(p, pc) for p, pc in state.board.items()
                          if pc.revealed]
    for pos, pc in my_revealed_pieces:                    # 位置项
        if is_camp(pos):
            score += w.camp_occ if pc.color == my else -w.camp_occ
            # 围杀压力：营内子每邻接一个敌子（含暗子）都是潜在围杀点
            siege = sum(1 for np in NEIGHBORS[pos]
                        if (e := state.board.get(np)) is not None
                        and e.color != pc.color)
            score += (w.camp_siege if pc.color == my else -w.camp_siege) * siege
        if is_hq(pos) and pc.rank != Rank.QI:
            score += w.hq_locked if pc.color == my else -w.hq_locked

    def pieces_of(color):
        return [(p, pc) for p, pc in my_revealed_pieces if pc.color == color]

    mine = pieces_of(my)
    theirs = pieces_of(opp)
    my_flag = next(((p, pc) for p, pc in mine if pc.rank == Rank.QI), None)
    opp_flag = next(((p, pc) for p, pc in theirs if pc.rank == Rank.QI), None)

    # 扛旗资格（与 state._attackable 同口径）：默认只有工兵能扛旗，且需先清光
    # 旗主一方的地雷——不满足条件的"威胁"不是真威胁，不计分。
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

    # 军旗暴露：相邻（含行营斜道）有能扛旗的敌明子
    def flag_threat(flag_pos, flag_color, attackers):
        return any(flag_pos in NEIGHBORS[ap] and can_take_flag(flag_color, e.rank)
                   for ap, e in attackers)

    if my_flag and flag_threat(my_flag[0], my, theirs):
        score -= w.flag_exposed
    if opp_flag and flag_threat(opp_flag[0], opp, mine):
        score += w.flag_exposed

    # 相邻明子的威胁/机会（正交+斜道一环）。行营内的子不可被攻击：
    # 作为攻击目标是假机会，作为被威胁对象是假风险，一律跳过。
    for pos, e in theirs:
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
            v = w.piece[m.rank]
            if res == "attacker_wins":
                best_gain = max(best_gain, v)
            elif res == "both_die":
                best_gain = max(best_gain, v * 0.5)
        score -= w.threat * best_gain
    for pos, m in mine:
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
            v = w.piece[e.rank]
            if res == "attacker_wins":
                best_gain = max(best_gain, v)
            elif res == "both_die":
                best_gain = max(best_gain, v * 0.5)
        # 行营内的子不可被反吃，其发起的威胁接近无风险，给高权重
        score += (w.attack_camp if is_camp(pos) else w.attack) * best_gain

    # ---- A2 增强项（基线 §4.2：行营势力 / 死区势能 / 暗子时差）----
    # 1) 行营势力范围：贴近空行营的活动子 = 占营/扩张潜力（已占营由 camp_occ 计）。
    #    只统计明子（公共信息），不计暗子。
    zone_net = 0
    for cp in CAMPS:
        if cp in state.board:
            continue
        for np_ in NEIGHBORS[cp]:
            e = state.board.get(np_)
            if e is not None and e.revealed and e.rank not in (Rank.LEI, Rank.QI):
                zone_net += 1 if e.color == my else -1
    score += w.camp_zone * zone_net

    # 2) 死区势能：fortress_score 差——劣势方封死死区≈锁定和棋，
    #    估值必须反映“和棋势能”，否则搜索会放弃筑垒而求战致败。
    score += w.fortress * (fortress_score(state, seat)
                           - fortress_score(state, 1 - seat))

    # 3) 暗子时差：暗子激活需“先翻后走”两回合，活动明子数差 = 行动节奏优势。
    #    物质项已计暗子期望价值，此处补其“不可立即行动”的折价。
    my_active = sum(1 for p, pc in mine
                    if pc.rank not in (Rank.LEI, Rank.QI)
                    and not (state.cfg.hq_locks_pieces and is_hq(p)))
    opp_active = sum(1 for p, pc in theirs
                     if pc.rank not in (Rank.LEI, Rank.QI)
                     and not (state.cfg.hq_locks_pieces and is_hq(p)))
    score += w.hidden_tempo * (my_active - opp_active)
    return score


class Agent:
    """PIMC（完全信息蒙特卡洛）搜索代理。

    每个决策点采样 K 个与公开信息一致的完整世界；世界内暗子全部按采样
    结果翻开，双方在完全信息下 negamax(alpha-beta) 对弈，根动作取跨世界
    期望。叶子用世界真实子力计分——翻开动作因此不产生虚假的物质增值
    （期望估值法会高估翻子，本方法无此偏差）。

    已知偏差（PIMC 固有）：世界内双方对暗子"全知"，会略微高估针对暗子的
    战术（如炸弹猎杀司令）。双方对称受影响，用于比较走法仍然可靠。
    """

    def __init__(self, search: SearchConfig | None = None,
                 weights: EvalWeights | None = None, seed=None):
        self.cfg = search or SearchConfig()
        self.w = weights or EvalWeights()
        self.rng = random.Random(seed)
        self.nodes = 0

    # ------------------------------------------------------------- 根决策

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: set | None = None):
        """返回 [(action, 跨世界平均分)] 按分降序，最多 topn 个。

        avoid：近期出现过的可观察局面键集合（对局层传入）。走完后命中这些
        键的动作视为"送对方循环判和/被捉长捉"，统一罚 REP_PENALTY——搜索
        本身看不到历史，重复规避只能在根节点做。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 0.0)]
        rep_hits = set()
        if avoid:
            for a in acts:
                if a.kind == "move" and \
                        position_key(state.apply(a)) in avoid:
                    rep_hits.add(a)
        seat = state.turn
        worlds = [state.instantiate(state.sample_world(self.rng, reveal=True))
                  for _ in range(self.cfg.samples)]
        acc = {a: 0.0 for a in acts}
        for wst in worlds:
            ordered = self._order(wst, acts)
            for a in ordered:
                child = wst.apply(a)
                if self.cfg.depth <= 0:
                    acc[a] += evaluate(child, seat, self.w)
                else:
                    acc[a] += -self._negamax(child, self.cfg.depth - 1,
                                             -math.inf, math.inf)
        result = sorted(((a, s / self.cfg.samples - (REP_PENALTY if a in rep_hits else 0.0))
                         for a, s in acc.items()),
                        key=lambda t: t[1], reverse=True)
        return result[:topn]

    def select_action(self, state: GameState, avoid: set | None = None) -> Optional[Action]:
        """返回最优单步决策动作 (统一 Agent 规范)。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else None

    # ------------------------------------------------------------- 搜索

    def _order(self, st: GameState, acts):
        """吃明子 > 普通走子/进营 > 翻子；吃子按被吃价值降序。"""

        def key(a: Action):
            if a.kind == "flip":
                return (2, 0.0)
            t = st.board.get(a.to)
            if t is not None and t.revealed and t.color != st.my_color():
                return (0, -self.w.piece[t.rank])
            return (1, 0.0)

        return sorted(acts, key=key)

    def _negamax(self, st: GameState, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if st.winner is not None:
            if st.winner == -1:
                return 0.0
            win = WIN_SCORE - st.ply
            return win if st.winner == st.turn else -win
        if depth <= 0:
            return evaluate(st, st.turn, self.w)
        acts = st.legal_actions()
        if not acts:
            return -(WIN_SCORE - st.ply)
        best = -math.inf
        for a in self._order(st, acts):
            v = -self._negamax(st.apply(a), depth - 1, -beta, -alpha)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                if alpha >= beta:
                    break
        return best


class ExpertAgent:
    """专家级传统搜索引擎代理 (Star1 Expectiminimax + QSearch + TT + IDS + Dynamic Dominance)。

    借鉴计算机暗棋 (Chinese Dark Chess) 领域专家算法体系：
    - Star1 期望极大极小 (Expectiminimax) 严格评估暗子翻棋几率节点，彻底杜绝 PIMC 策略融合与透视眼幻觉
    - 静态搜索 (Quiescence Search) 消除吃子地平线盲区与战术震荡
    - 64 位 Zobrist 置换表 (Transposition Table) 深度剪枝与 PV 走法迁移
    - 专家级动态制霸矩阵、棋子机动力、死子惩罚与局部翻棋安全度评估
    - 迭代加深搜索 (IDS) 支持毫秒级时间预算管理
    """

    def __init__(self, search: SearchConfig | None = None,
                 weights: EvalWeights | None = None,
                 tt_size_power: int = 18,
                 seed: int | None = None):
        self.cfg = search or SearchConfig()
        self.w = weights or EvalWeights()
        qdepth = getattr(self.cfg, "qsearch_depth", 16)
        self.engine = ExpertSearchEngine(weights=self.w, tt_size_power=tt_size_power, seed=seed,
                                         qsearch_depth=qdepth)
        self.rng = random.Random(seed)

    @property
    def nodes(self) -> int:
        return self.engine.stats.nodes

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: set | None = None) -> list[tuple[Action, float]]:
        """返回 [(action, score)] 列表，按分数降序。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 0.0)]

        best_act, best_score, stats = self.engine.search(
            state,
            max_depth=max(1, self.cfg.depth),
            time_limit_ms=self.cfg.time_limit_ms,
            avoid=avoid,
            qsearch_depth=getattr(self.cfg, "qsearch_depth", 16),
        )

        if topn <= 1 or best_act is None:
            return [(best_act or acts[0], best_score)]

        # 如果需要 topn 个候选走法，优先返回根节点真实搜索估值
        if stats.root_scores:
            return stats.root_scores[:topn]

        # 如果无根节点搜索分，回退到启发排序
        scored = []
        for a in acts:
            score = self.engine._score_action(a, state, 0, best_act)
            if a == best_act:
                score += 100_000.0
            scored.append((a, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:topn]

    def select_action(self, state: GameState, avoid: set | None = None) -> Optional[Action]:
        """返回最优单步决策动作 (统一 Agent 规范)。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else None


def win_probability(score: float, scale: float = 250.0) -> float:
    """把估值分粗略映射到胜率（仅用于展示，非严格校准）。"""
    return 1.0 / (1.0 + math.exp(-score / scale))


class NNAgent:
    """深度学习策略-价值网络代理（支持纯网络策略与 MCTS 搜索）。"""

    def __init__(self, model_path: str | None = None, net=None,
                 simulations: int = 100, c_puct: float = 0.6,
                 device: str | None = None, seed: int | None = None):
        import torch
        from .net import JunqiNet
        from .mcts import MCTS

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = random.Random(seed)

        if net is not None:
            self.net = net.to(device)
        elif model_path and os.path.exists(model_path):
            self.net = JunqiNet.load_from_file(model_path, device=device)
        else:
            self.net = JunqiNet().to(device)
            self.net.eval()

        self.mcts = MCTS(self.net, simulations=simulations, c_puct=c_puct, device=device)

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: set | None = None,
                       history_counts: dict | None = None):
        """返回 [(action, score)] 列表。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 1.0)]

        if self.simulations > 0:
            act, pi_vec, pi_dict, _ = self.mcts.search(state, temperature=1e-3, add_noise=False,
                                                        rng=self.rng, history_counts=history_counts,
                                                        avoid=avoid)
            scored = sorted(((a, pi_dict.get(a, 0.0)) for a in acts),
                            key=lambda t: t[1], reverse=True)
            return scored[:topn]
        else:
            policy_map, value = self.net.predict_state(state, seat=state.turn,
                                                       history_counts=history_counts,
                                                       device=self.device)
            from .state import position_key
            scored_list = []
            for a in acts:
                score = policy_map.get(a, 0.0)
                if avoid and position_key(state.apply(a)) in avoid:
                    score -= 100.0
                scored_list.append((a, score))
            scored_list.sort(key=lambda t: t[1], reverse=True)
            return scored_list[:topn]

    def select_action(self, state: GameState, avoid: set | None = None) -> Optional[Action]:
        """返回最优单步决策动作 (统一 Agent 规范)。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else None


class HybridAgent:
    """P2 混合引擎代理（神经网络全局大局观先验 + 专家搜索战术把关）。

    工作机制：
    1. 由深度神经网络 (JunqiNet/BC) 提供全盘大局观与翻棋/走子先验概率分布（挑选 Top-K 最优候选）；
    2. 由专家搜索引擎 (ExpertSearchEngine) 进行浅层战术搜索验证（QSearch + Star1 剪枝），
       识破敌方埋伏、炸弹陷阱、受困死子并锁定一步吃旗等致命战术；
    3. 融合网络先验与战术评估分，实现高水平真人翻棋节奏与零战术盲区。
    """

    def __init__(self, model_path: str = "models/bc_best.pt",
                 search_depth: int = 2, top_k: int = 6,
                 prior_weight: float = 0.25,
                 weights: EvalWeights | None = None,
                 device: str | None = None, seed: int | None = None):
        import torch
        from .net import JunqiNet

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.search_depth = max(1, search_depth)
        self.top_k = max(2, top_k)
        self.prior_weight = prior_weight
        self.weights = weights or EvalWeights()
        self.engine = ExpertSearchEngine(weights=self.weights, seed=seed)
        self.rng = random.Random(seed)

        if model_path and os.path.exists(model_path):
            self.net = JunqiNet.load_from_file(model_path, device=device)
        else:
            self.net = JunqiNet().to(device)
            self.net.eval()

    def choose_actions(self, state: GameState, topn: int = 1,
                       avoid: set | None = None) -> list[tuple[Action, float]]:
        """综合神经网络先验与专家搜索，返回 [(action, score)] 列表。"""
        acts = state.legal_actions()
        if not acts or state.is_terminal():
            return []
        if len(acts) == 1:
            return [(acts[0], 1.0)]

        # 1. 获取神经网络先验概率
        policy_map, nn_val = self.net.predict_state(state, seat=state.turn, device=self.device)

        # 2. 筛选 Top-K 网络候选走法
        sorted_by_nn = sorted(acts, key=lambda a: policy_map.get(a, 0.0), reverse=True)
        cand_set = set(sorted_by_nn[:self.top_k])

        # 3. 补充战术走法（吃旗、吃子、解救大子走法强制纳入候选）
        for a in acts:
            if a.kind == "move":
                tgt = state.board.get(a.to)
                if tgt and tgt.revealed:
                    cand_set.add(a)

        candidates = list(cand_set)

        # 4. 对候选走法进行战术评估与打分
        scored = []
        for a in candidates:
            if avoid and a in avoid:
                tactical_score = -WIN_SCORE + 100.0
            elif a.kind == "move":
                # 模拟执行移动
                nxt = state.apply(a)
                if nxt.is_terminal() and nxt.winner == state.turn:
                    # 一步制胜（直接吃旗赋予绝对最高斩杀优先级，避免因困毙平分被先验扰乱）
                    tgt = state.board.get(a.to)
                    if tgt and tgt.rank == Rank.QI:
                        tactical_score = WIN_SCORE + 5000.0
                    else:
                        tactical_score = WIN_SCORE
                elif nxt.is_terminal() and nxt.winner == -1:
                    tactical_score = 0.0
                elif self.search_depth <= 1:
                    tactical_score = -evaluate_expert(nxt, nxt.turn, self.weights)
                else:
                    _, opp_score, _ = self.engine.search(nxt, max_depth=self.search_depth - 1)
                    tactical_score = -opp_score
            else:
                # 翻棋几率节点
                tactical_score = self.engine._evaluate_chance_flip(
                    state, a, self.search_depth, 0, -WIN_SCORE, WIN_SCORE, set()
                )

            # 融合先验 log 似然与战术估值分
            p = max(policy_map.get(a, 1e-4), 1e-4)
            prior_term = math.log(p) * 20.0  # 约 -100 ~ 0 范围
            total_score = (1.0 - self.prior_weight) * tactical_score + self.prior_weight * prior_term
            scored.append((a, total_score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:topn]

    def select_action(self, state: GameState, avoid: set | None = None) -> Optional[Action]:
        """返回最优单步决策动作 (统一 Agent 规范)。"""
        scored = self.choose_actions(state, topn=1, avoid=avoid)
        if scored:
            return scored[0][0]
        acts = state.legal_actions()
        return acts[0] if acts else None

