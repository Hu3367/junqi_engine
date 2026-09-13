"""P1/P4 开局"快速占据行营"策略的结构性与统计学回归测试。

本文件把 `scripts/analyze_camp_opening.py`（唯一保留的分析+竞赛脚本）得出的
**已验证结论**固化为断言，防止后续改动无声破坏开局占营的几何前提、启发式
常量分解与竞赛统计规律。

覆盖的验证结论：
  1. 邻营度全表：度3=8位、度2=8位、度1=24位、度0=10位，邻接边共 64 条；
  2. 开局机动性定理：满盘暗子时明子的合法非吃子走法数 == 邻营度，且全部指向行营；
  3. `CENTER_CAMP_FLIP_POSITIONS` 是度3集合的真子集，几何完备化恰好补 {(2,2),(9,2)}；
  4. 营簇星形拓扑：中心营是簇内唯一营间枢纽，4 个黄金位完整覆盖该簇 5 营；
  5. 先手首翻打分 = 170（黄金位）/ 20（其它），分差 150 对 beginner jitter±30 稳健；
  6. 后手首翻 -40 威胁扣分只落在"敌可动明子相邻"的位，地雷/军旗不触发；
  7. 翻棋启发常量分解：辐射拓荒 = base+140，弃营盲翻 = base-200（可再 -40）；
  8. 已知边界缺陷：`has_camp_entrance_opportunity` 未排除地雷/军旗，会误罚 -200；
  9. 首占营闭式模型 E=(n+1)/(k+1)，结构性能差 Δ≈2.818 ply；
 10. 占营竞赛（10 营占满即停）：分配系统性偏向先手 ~5.34:4.66 而非 5:5；
 11. 中心营优先比角营优先快 3~4 倍，角营优先会把中心营入口锁死、ρ 塌缩到 1/4；
 12. 占营/拓荒交替慢于"有营就进"；首翻打底线死位少约 0.6 营（显著）；
 13. 补 (2,2)/(9,2) 无实质影响；后手各反制净收益 <=0.15 营（多数不显著）；
 14. 修正闭合模型（ρ 标定）预测占满手数误差 <8%、分配误差 <0.15 营。
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from junqi.apk_engine import (CENTER_CAMP_FLIP_POSITIONS, eval_apk_flip_root,
                              eval_apk_pure)
from junqi.config import RuleConfig
from junqi.rules import (CAMPS, COMPOSITION, NEIGHBORS, PLAY_POSITIONS, Rank,
                         is_camp)
from junqi.state import Action, GameState, Piece, deal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

IMMOVABLE = (Rank.LEI, Rank.QI)
DEG = {p: sum(1 for nb in NEIGHBORS[p] if is_camp(nb)) for p in PLAY_POSITIONS}
GOLDEN8 = {p for p in PLAY_POSITIONS if DEG[p] == 3}


# ------------------------------------------------------- 1. 邻营度全表
def test_camp_adjacency_degree_table():
    """邻营度分布与邻接边总数是开局占营能力的几何基线，不得漂移。"""
    buckets = {d: [p for p in PLAY_POSITIONS if DEG[p] == d] for d in (3, 2, 1, 0)}
    assert len(buckets[3]) == 8, buckets[3]
    assert len(buckets[2]) == 8, buckets[2]
    assert len(buckets[1]) == 24
    assert len(buckets[0]) == 10
    assert sum(len(v) for v in buckets.values()) == len(PLAY_POSITIONS) == 50
    # 邻接边总数：放子位侧 8*3+8*2+24*1 = 64，行营侧同样应为 64
    assert sum(DEG.values()) == 64
    camp_side = sum(1 for c in CAMPS for nb in NEIGHBORS[c] if not is_camp(nb))
    assert camp_side == 64
    # 度 0 位恰好是两条底线性（row0 / row11），翻出的子开局即冻结
    assert set(buckets[0]) == {(0, c) for c in range(5)} | {(11, c) for c in range(5)}
    assert GOLDEN8 == {(2, 2), (3, 1), (3, 3), (4, 2),
                       (7, 2), (8, 1), (8, 3), (9, 2)}


def test_center_camp_flip_positions_are_degree_three_subset():
    """代码黄金 6 位必须全部是度 3 位；几何完备化恰好补上 (2,2) 与 (9,2)。

    该断言不修改引擎常量，只固化"缺口在哪"的实证结论，供后续按二进制证据决策。
    """
    assert CENTER_CAMP_FLIP_POSITIONS <= GOLDEN8
    assert GOLDEN8 - CENTER_CAMP_FLIP_POSITIONS == {(2, 2), (9, 2)}
    # 缺口两位与已收录黄金位辐射同样多的行营，且同属各自营簇
    for p, camps in (((2, 2), {(3, 2), (2, 1), (2, 3)}),
                     ((9, 2), {(8, 2), (9, 1), (9, 3)})):
        assert {nb for nb in NEIGHBORS[p] if is_camp(nb)} == camps


# ------------------------------------------- 1b. 营簇星形拓扑与黄金位覆盖
CLUSTER_CAMPS = {"up": {(2, 1), (2, 3), (3, 2), (4, 1), (4, 3)},
                 "down": {(7, 1), (7, 3), (8, 2), (9, 1), (9, 3)}}
CLUSTER_GOLDEN = {"up": {(2, 2), (3, 1), (3, 3), (4, 2)},
                  "down": {(7, 2), (8, 1), (8, 3), (9, 2)}}
CENTER_CAMPS = {(3, 2), (8, 2)}


def test_camp_cluster_star_topology():
    """每个营簇是"中心营-4 角营"的星形：中心营与全部 4 角营相邻，角营互不相邻。

    这解释了 CAMP_POSITION_BONUS 为何给中心营 +50、角营 +40：中心营是簇内
    唯一的营间机动枢纽（行营斜人道只在行营之间连通）。
    """
    for k, camps in CLUSTER_CAMPS.items():
        center = CENTER_CAMPS & camps
        assert len(center) == 1
        center = next(iter(center))
        corners = camps - {center}
        assert len(corners) == 4
        assert corners <= {nb for nb in NEIGHBORS[center] if is_camp(nb)}
        for a in corners:                       # 角营互不相邻
            for b in corners - {a}:
                assert b not in NEIGHBORS[a], (a, b)
        for a in corners:                       # 角营只连中心营
            assert {nb for nb in NEIGHBORS[a] if is_camp(nb)} == {center}
    assert set(CLUSTER_CAMPS["up"]) | set(CLUSTER_CAMPS["down"]) == set(CAMPS)


def test_golden_positions_pairwise_disjoint_and_cover_cluster():
    """8 个度 3 黄金位两两不相邻、各只有 1 个非营邻居（恰为 8 个度 2 位），
    且每簇 4 个黄金位共同覆盖该簇全部 5 个行营，每位都辐射中心营。"""
    for i, a in enumerate(sorted(GOLDEN8)):
        for b in sorted(GOLDEN8)[i + 1:]:
            assert b not in NEIGHBORS[a], f"黄金位 {a} 与 {b} 相邻，违反互不侵犯"
    deg2 = {p for p in PLAY_POSITIONS if DEG[p] == 2}
    for p in GOLDEN8:
        non_camp_nb = [nb for nb in NEIGHBORS[p] if not is_camp(nb)]
        assert len(non_camp_nb) == 1 and non_camp_nb[0] in deg2, (p, non_camp_nb)
    for k, golds in CLUSTER_GOLDEN.items():
        camps = CLUSTER_CAMPS[k]
        covered = set()
        center = next(iter(CENTER_CAMPS & camps))
        for g in golds:
            adj = {nb for nb in NEIGHBORS[g] if is_camp(nb)}
            assert len(adj) == 3
            assert center in adj, f"{g} 未辐射中心营 {center}"
            covered |= adj
        assert covered == camps, (k, camps - covered)
        assert golds <= GOLDEN8


# ------------------------------------------- 2. 开局机动性定理
def test_opening_mobility_equals_camp_degree():
    """满盘暗子时，一枚刚翻开的明子的合法非吃子走法数 == 其邻营度。

    推论：邻营度就是开局阶段的机动性/占营能力；度 0 位翻出的子完全冻结。
    """
    cfg = RuleConfig()
    for pos in PLAY_POSITIONS:
        board = {q: (Piece("r", Rank.LIAN, True) if q == pos
                     else Piece("b", Rank.PAI, False)) for q in PLAY_POSITIONS}
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                       first_flip_done=True, ply=2, cfg=cfg)
        acts = st.legal_actions()
        moves = [a for a in acts if a.kind == "move"]
        assert len(moves) == DEG[pos], f"{pos}: 走法 {len(moves)} != 邻营度 {DEG[pos]}"
        assert all(a.to in CAMPS for a in moves), f"{pos}: 存在非进营的开局走法"
        # 其余 49 位仍为暗子，翻棋动作恒为 49
        assert sum(1 for a in acts if a.kind == "flip") == 49


# ------------------------------------------- 3. 先手首翻打分
def test_first_flip_score_golden_gap_and_jitter_robustness():
    """先手首翻：黄金位 170 / 其它 20，分差 150；对 beginner jitter±30 稳健。"""
    cfg = RuleConfig()
    st = deal(random.Random(2026), cfg)
    assert st.my_color() is None and len(st.hidden_positions()) == 50
    assert eval_apk_pure(st, None) == 0.0

    golden = {p: eval_apk_flip_root(st, Action("flip", p), None)
              for p in CENTER_CAMP_FLIP_POSITIONS}
    others = {p: eval_apk_flip_root(st, Action("flip", p), None)
              for p in PLAY_POSITIONS if p not in CENTER_CAMP_FLIP_POSITIONS}
    assert set(golden.values()) == {170.0}, golden
    assert set(others.values()) == {20.0}
    gap = 170.0 - 20.0
    assert gap == 150.0
    # beginner jitter ±30 => 最坏情形 170-30=140 仍 > 20+30=50
    assert 170.0 - 30.0 > 20.0 + 30.0
    # intermediate ±10 / advanced ±0.5 更稳健
    assert gap > 2 * 30.0


# ------------------------------------------- 4. 后手首翻威胁扣分
@pytest.mark.parametrize("rank,expect_threat", [
    (Rank.LIAN, True), (Rank.SI, True), (Rank.GONG, True),
    (Rank.LEI, False), (Rank.QI, False),
])
def test_second_flip_threat_penalty_exact(rank, expect_threat):
    """后手首翻：-40 只落在'敌可动明子相邻的非行营位'，地雷/军旗不触发。"""
    cfg = RuleConfig()
    base = deal(random.Random(2026), cfg)
    first_pos = (3, 1)
    board = dict(base.board)
    board[first_pos] = Piece("r", rank, True)
    st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=1,
                   first_flip_done=True, ply=1, cfg=cfg)
    me = st.my_color()
    assert me == "b"

    base_score = eval_apk_pure(st, me)
    # 黄金位彼此不相邻，故不受敌明子威胁影响：恒为 base + 150 + 20
    for gp in sorted(CENTER_CAMP_FLIP_POSITIONS):
        if gp == first_pos or st.board[gp].revealed:
            continue
        s = eval_apk_flip_root(st, Action("flip", gp), me)
        assert s == pytest.approx(base_score + 170.0), (gp, s)

    # 紧邻先手明子的非行营位 (3,0)：可动子 => base+20-40；雷/旗 => base+20
    adj = (3, 0)
    assert adj in NEIGHBORS[first_pos] and not is_camp(adj)
    s_adj = eval_apk_flip_root(st, Action("flip", adj), me)
    expected = base_score + 20.0 + (-40.0 if expect_threat else 0.0)
    assert s_adj == pytest.approx(expected), (rank, s_adj, expected)


# ------------------------------------------- 5. 启发常量分解
def test_flip_heuristic_constant_decomposition():
    """辐射拓荒 = base+140（+120 依托行营 +20）；弃营盲翻 = base-200（可再 -40）。

    这正是 scripts/diag_apk_blind_flip.py 在 400 个实战采样局面上测得的
    A 类 +140.0 / B 类 -200.0 常量分解的构造性复现。
    """
    cfg = RuleConfig()
    # 满盘暗子（蓝排长），再放入两枚红明子：(7,1) 行营内连长 + (3,1) 野外排长
    board = {q: Piece("b", Rank.PAI, False) for q in PLAY_POSITIONS}
    board[(3, 1)] = Piece("r", Rank.PAI, True)       # 紧邻空营 (2,1)/(3,2)/(4,1)
    board[(7, 1)] = Piece("r", Rank.LIAN, True)      # 已占下半场角营
    st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                   first_flip_done=True, ply=6, cfg=cfg)
    base = eval_apk_pure(st, "r")
    # 排长 30 + 连长 40 + (7,1) 角营位置偏置 40 = 110（中营 (3,2)/(8,2) 才是 +50）
    assert base == pytest.approx(30.0 + 40.0 + 40.0)

    # A 类：翻 (8,1)，紧邻己方已占行营 (7,1) => +120 依托行营 +20（-200 不适用）
    s_a = eval_apk_flip_root(st, Action("flip", (8, 1)), "r")
    assert s_a == pytest.approx(base + 140.0)

    # B 类：翻 (0,0) 底线性，不依托任何己方行营，而 (3,1) 有一步进空营机会 => -200
    s_b = eval_apk_flip_root(st, Action("flip", (0, 0)), "r")
    assert s_b == pytest.approx(base - 200.0)

    # B' 类：在 (1,0) 放一枚敌团长明子（(1,0) 邻 (0,0)）=> 再叠加 -40 威胁扣分
    board_c = dict(board)
    board_c[(1, 0)] = Piece("b", Rank.TUAN, True)
    st_c = GameState(board=board_c, seat_color={0: "r", 1: "b"}, turn=0,
                     first_flip_done=True, ply=6, cfg=cfg)
    base_c = eval_apk_pure(st_c, "r")
    assert base_c == pytest.approx(base - 160.0)
    s_c = eval_apk_flip_root(st_c, Action("flip", (0, 0)), "r")
    assert s_c == pytest.approx(base_c - 240.0)

    # 排序结论：A 类恒比 B 类高 340 分 => 引擎宁可辐射拓荒也不去占第 2 个营，
    # 而进营走法的搜索分实测中位仅 base+10，故 +140 的拓荒翻棋必然压过进营。
    assert (s_a - base) - (s_b - base) == pytest.approx(340.0)


# ------------------------------------------- 6. 已知边界缺陷
def test_camp_entrance_false_positive_with_immovable_piece():
    """边界缺陷固化：己方唯一明子是地雷/军旗且紧邻空营时，代码仍判 has_camp_entrance
    => 触发 -200 误罚，而实际上没有任何合法进营走法。

    实战 400 个采样局面中该误报出现 0 次（见 reports/diag_blind_flip_beginner.txt），
    故属低频边界问题；本测试固化现状，若将来修复需同步更新断言。
    """
    cfg = RuleConfig()
    board = {(3, 1): Piece("r", Rank.LEI, True), (11, 4): Piece("b", Rank.PAI, True)}
    for pos in PLAY_POSITIONS:
        board.setdefault(pos, Piece("r", Rank.PAI, False))
    st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                   first_flip_done=True, ply=2, cfg=cfg)

    # 真实合法进营走法为 0（地雷不可移动）
    empty_camps = [c for c in CAMPS if st.board.get(c) is None]
    assert len(empty_camps) == 10
    entries = [a for a in st.legal_actions()
               if a.kind == "move" and a.to in empty_camps]
    assert entries == []

    # 但代码判定"有进营机会" => 远端翻棋被误罚 -200
    base = eval_apk_pure(st, "r")          # 雷 70 - 排 30 = +40
    assert base == pytest.approx(40.0)
    s = eval_apk_flip_root(st, Action("flip", (0, 0)), "r")
    assert s == pytest.approx(base - 200.0)


# ------------------------------------------- 7. 占营竞赛闭式模型
def test_camp_race_closed_form_model():
    """无放回首次命中期望 E=(n+1)/(k+1) 与递推解一致，且结构性能差 Δ≈2.82 ply。"""
    movable_each = 25 - (COMPOSITION[Rank.LEI] + COMPOSITION[Rank.QI])
    assert movable_each == 21

    def rec(k, rest):
        if rest == 0:
            return Fraction(1)
        return Fraction(1) + Fraction(rest, k + rest) * rec(k, rest - 1)

    n, k = 49, movable_each
    assert rec(k, n - k) == Fraction(n + 1, k + 1) == Fraction(50, 22)
    ef = float(Fraction(50, 22))
    assert ef == pytest.approx(2.2727, abs=1e-4)

    p_first_movable = 42 / 50
    t_first = p_first_movable * 3 + (1 - p_first_movable) * (1 + 2 * ef + 2)
    t_second = 2 * ef + 2
    delta = t_second - t_first
    assert t_first == pytest.approx(3.727, abs=1e-3)
    assert t_second == pytest.approx(6.545, abs=1e-3)
    assert delta == pytest.approx(2.818, abs=1e-3)

    # 10 营耗尽时的闭式分配：先手 (C+Δr)/2, 后手 (C-Δr)/2, r = 1/T_second
    r = 1.0 / t_second
    pred_first = (10 + delta * r) / 2
    pred_second = (10 - delta * r) / 2
    assert pred_first - pred_second == pytest.approx(delta * r, abs=1e-9)
    assert pred_first == pytest.approx(5.2153, abs=1e-3)
    assert pred_second == pytest.approx(4.7847, abs=1e-3)


# ============================================================================
#  8. 占营竞赛（10 营占满即停）——固定评测证据
# ============================================================================
# 赛制：不打完整对局。第 10 个行营被占据的瞬间立即终止，统计双方占营数、
#       占满所用手数、占营顺序。样本 = RACES x 2（座位互换）。
# 依赖：scripts/analyze_camp_opening.py（唯一保留的分析/实验脚本）。

import analyze_camp_opening as aco

RACES = 200                # 每方向场次（总 2x = 400 场/组）；慢速配置另降采样
RACES_SLOW = 60            # 角营/交替族（85 ply 量级）每方向场次
SEED0 = 90000
SEED_SLOW = 90777          # 慢速配置用独立种子段，避免与快速组共用牌局
R_CENTER_CAMPS = {(3, 2), (8, 2)}
_CACHE: dict = {}


def _batch(a, b, n=RACES, seed0=SEED0):
    key = (a, b, n, seed0)
    if key not in _CACHE:
        _CACHE[key] = aco.race_batch(a, b, n, aco.race_config(), seed0=seed0)
    return _CACHE[key]


def _seat(a, b, n=RACES, seed0=SEED0):
    """先手(seat0) vs 后手(seat1) 视角。"""
    return aco.summarize_races(_batch(a, b, n, seed0), lambda r: 0)


def _policy(a, b, n=RACES, seed0=SEED0):
    """策略 A vs 策略 B 的配对视角（各半局执先/执后）。"""
    return aco.summarize_races(_batch(a, b, n, seed0),
                               lambda r: 0 if r["specs"][0] == a else 1)


def _rho(s):
    """由汇总结果反推稳态占营速率 ρ（营/ply/方）。"""
    tf, ts = s["first_camp_ply"]
    tfill = s["plies_to_fill"][0]
    return 0.5 * (s["camps"][0] / (tfill - tf) + s["camps"][1] / (tfill - ts))


def test_race_terminates_exactly_when_ten_camps_filled():
    """赛制正确性：每场都在第 10 个营被占的瞬间停止，双方占营数之和恒为 10。"""
    recs = _batch("BASE", "BASE")
    assert len(recs) == RACES * 2
    assert all(r["filled"] for r in recs), "存在未能占满 10 营的场次"
    for r in recs:
        assert sum(r["camps"]) == len(CAMPS) == 10, r["camps"]
        assert len(r["order"]) == 10
        assert r["plies"] <= 60, f"占满 10 营不应超过 60 手，实得 {r['plies']}"
        # 占营顺序的 ply 必须严格递增（行营一旦占据不可剥夺）
        plies = [p for p, _, _ in r["order"]]
        assert plies == sorted(plies) and len(set(plies)) == len(plies)
        assert {c for _, _, c in r["order"]} == set(CAMPS)


def test_race_split_systematically_favors_first_mover():
    """核心验证：10 营分配**不是** 5:5，而是系统性偏向先手（约 5.37 : 4.63）。

    注意区分两个易混指标（见报告【12】种子稳定性核查）：
      · 「第 10 营由谁拿下」均值 ≈ 50%，接近抛硬币，**不能**作为先手优势判据；
      · 「拿到 ≥6 营的概率」先手 ≈ 46% vs 后手 ≈ 26%，比值约 1.8:1 —— 这才是
        先手优势的正确表述。
    """
    s = _seat("BASE", "BASE")
    f, b = s["camps"]
    assert f + b == pytest.approx(10.0, abs=1e-9)
    assert 5.15 <= f <= 5.60, f"先手平均占营 {f} 超出 7 种子段实测区间"
    assert 4.40 <= b <= 4.85, f"后手平均占营 {b} 超出 7 种子段实测区间"
    # 配对差显著为正
    cm, cse, cz = s["camp_diff"]
    assert cm > 0.4 and cz >= 1.96, (cm, cse, cz)
    # 先手拿到 >=6 营的概率明显高于后手；5:5 只是少数情形
    assert s["win"][0] > s["win"][1] + 0.10, s["win"]
    assert s["win"][0] / max(1e-9, s["win"][1]) > 1.4, s["win"]
    assert s["win"][2] < 0.45, f"5:5 平局比例 {s['win'][2]} 过高"
    # 第 10 营归属接近抛硬币（7 种子段实测 49.0%~53.2%）
    assert 0.42 <= s["tenth_by_A"] <= 0.58, s["tenth_by_A"]


def test_race_first_camp_ply_matches_closed_form():
    """实测首占营手数与闭式模型（T_first=3.727, T_second=6.545, Δ=2.818）吻合。

    容差按 7 个独立种子段（各 400 场）的实测极差给定：
      T_first 3.305~3.460，T_second 6.150~6.725，Δ 2.710~3.320。
    Δ 的抽样波动约 ±0.21（n=400），故对理论值给 ±0.60 ply 容差。
    """
    s = _seat("BASE", "BASE")
    tf, ts = s["first_camp_ply"]
    assert tf == pytest.approx(3.727, abs=0.50), tf
    assert ts == pytest.approx(6.545, abs=0.60), ts
    assert (ts - tf) == pytest.approx(2.818, abs=0.60), (tf, ts)
    # 占满 10 营的手数极稳（7 种子段实测 22.59~22.70）
    assert s["plies_to_fill"][0] == pytest.approx(22.6, abs=1.2)


def test_center_camp_priority_fills_fastest():
    """占营优先级：中心营(+50) 优先远快于角营(+40) 优先，且完成率 100%。"""
    sc = _seat("P_center", "P_center")
    sk = _seat("P_corner", "P_corner", n=RACES_SLOW, seed0=SEED_SLOW)
    assert sc["completion"] == pytest.approx(1.0)
    assert sk["completion"] < 1.0, "角营优先应存在无法占满 10 营的场次"
    ratio = sk["plies_to_fill"][0] / sc["plies_to_fill"][0]
    assert ratio > 2.5, f"角营优先只应慢若干倍，实测倍率 {ratio:.2f}"
    z = aco.two_sample_z(sk["plies_to_fill"][0], sk["plies_se"],
                         sc["plies_to_fill"][0], sc["plies_se"])
    assert z >= 1.96, z


def test_corner_priority_collapses_steady_rate_and_locks_center_camp():
    """角营优先把稳态占营速率 ρ 打到 1/4，且中心营被迫最后才占（入口被锁死）。"""
    sc = _seat("P_center", "P_center")
    sk = _seat("P_corner", "P_corner", n=RACES_SLOW, seed0=SEED_SLOW)
    rc, rk = _rho(sc), _rho(sk)
    assert rc / rk > 3.0, f"ρ 塌缩倍率 {rc/rk:.2f} 不足 3"

    def center_rank(spec, n, seed0):
        ranks = []
        for r in _batch(spec, spec, n, seed0):
            seq = [c for _, _, c in r["order"]]
            hits = [i for i, c in enumerate(seq, 1) if c in R_CENTER_CAMPS]
            if len(hits) == 2:
                ranks.append(hits[-1])       # 第二个中心营的占营序号
        return sum(ranks) / len(ranks) if ranks else float("nan")

    fast = center_rank("P_center", RACES, SEED0)
    slow = center_rank("P_corner", RACES_SLOW, SEED_SLOW)
    assert fast < slow, (fast, slow)
    assert fast <= 3.0, f"中心营优先时第二个中心营应在前 3 个被占，实得 {fast:.2f}"


def test_alternate_flip_and_occupy_is_slower():
    """占营/拓荒交替策略（C）慢于'有营就进'（BASE），但仍远快于角营优先。"""
    sb = _seat("BASE", "BASE")
    sa = _seat("C_alternate", "C_alternate", n=RACES_SLOW, seed0=SEED_SLOW)
    assert sa["plies_to_fill"][0] > sb["plies_to_fill"][0] * 1.3
    z = aco.two_sample_z(sa["plies_to_fill"][0], sa["plies_se"],
                         sb["plies_to_fill"][0], sb["plies_se"])
    assert z >= 1.96, z
    sk = _seat("P_corner", "P_corner", n=RACES_SLOW, seed0=SEED_SLOW)
    assert sa["plies_to_fill"][0] < sk["plies_to_fill"][0]
    # 交替不改变最终分配的量级（仍是先手 ~5.3）
    assert sa["camps"][0] == pytest.approx(sb["camps"][0], abs=0.20)


def test_first_flip_on_frozen_row_costs_camps():
    """首翻打在度 0 底线死位：显著少占营，且首占营推迟约 2 ply。

    效应量随种子段波动（7 段实测 +0.340 ~ +0.675，池化 +0.505 ± 0.052），
    故此处只断言下界 0.25 与显著性方向，不断言点估计。
    """
    p = _policy("BASE", "X_firstedge")
    cm, cse, cz = p["camp_diff"]
    assert cm >= 0.25, f"BASE 应明显多占营，实测差 {cm:+.3f}"
    assert cz >= 1.96, (cm, cse, cz)
    assert p["first_camp_ply"][1] > p["first_camp_ply"][0] + 1.0
    assert p["win"][0] > p["win"][1], p["win"]


def test_golden8_vs_golden6_is_negligible():
    """补上缺失的 (2,2)/(9,2) 对占营竞赛无实质影响（差异不显著）。"""
    p = _policy("X_golden8", "BASE")
    cm, cse, cz = p["camp_diff"]
    assert abs(cm) < 0.20, cm
    assert abs(cz) < 1.96, (cm, cse, cz)


def test_counter_strategies_gain_only_within_noise():
    """后手四种反制相对 BASE 的净占营收益全部落在噪声内（|差|<0.25，不显著）。

    这是本赛制最重要的**负结果**：10 营总量固定且必然被填满，先手的
    Δ≈2.7 ply 时序优势决定了分配，后手的营簇/卡位选择只能微调 ±0.1 营。
    """
    for spec in ("R_follow", "R_split", "R_denial", "R_split_denial"):
        p = _policy("BASE", spec)
        cm, cse, cz = p["camp_diff"]
        assert abs(cm) < 0.25, (spec, cm, cse)
        assert abs(cz) < 1.96, (spec, cm, cse, cz)


def test_cluster_restricted_counters_slow_the_race():
    """限制翻棋营簇（follow/split）会把整场竞赛拖慢 ≥1.5 ply；denial 不拖慢。

    含义：营簇选择是"分配"手段而非"提速"手段。单纯为反制而自我限制翻棋位，
    代价是整体占营速度下降，自己并未因此多拿营 —— 故反制应优先用 denial
    （只改"进哪个营"，不改"翻哪个位"）。
    """
    base = _seat("BASE", "BASE")
    denial = _seat("BASE", "R_denial")
    for spec in ("R_follow", "R_split", "R_split_denial"):
        s = _seat("BASE", spec)
        assert s["plies_to_fill"][0] > base["plies_to_fill"][0] + 1.5, spec
        z = aco.two_sample_z(s["plies_to_fill"][0], s["plies_se"],
                             base["plies_to_fill"][0], base["plies_se"])
        assert z >= 1.96, (spec, z)
    # denial 不限制翻棋位 => 速度与 BASE 持平（是最"便宜"的反制）
    assert abs(denial["plies_to_fill"][0] - base["plies_to_fill"][0]) < 1.0


def test_split_counter_direction_on_first_mover_edge():
    """**更正后的结论**：三种后手反制都**无法**压缩先手优势。

    采用"同一副牌双座位"完美配对后，seat 视角的先手优势在四种反制下几乎相同
    （follow +0.820 / split +0.810 / denial +0.863 / combo +0.800）。
    上一轮报告的 "split 把先手优势从 +0.777 压到 +0.520" 已被证实是
    牌局级共同模噪声（旧设计用不同牌局交换座位，残差 SE≈0.10 营）。
    此处只断言：无论哪种反制，先手优势都显著为正且量级相近（0.3~1.0 营）。
    """
    advs = {}
    for spec in ("R_follow", "R_split", "R_denial", "R_split_denial"):
        s = _seat("BASE", spec)
        advs[spec] = s["camp_diff"]
        assert 0.30 < advs[spec][0] < 1.10, (spec, advs[spec])
        assert advs[spec][2] >= 1.96, (spec, advs[spec])
    spread = max(v[0] for v in advs.values()) - min(v[0] for v in advs.values())
    assert spread < 0.30, f"四种反制下的先手优势应量级相近，实测极差 {spread:.3f}"


def test_race_policies_keep_camp_discipline():
    """通畅制度下策略零违规：营间闲走 0 次、弃营 0 次（行营驻守纪律）。"""
    for spec in ("BASE", "P_center", "R_split", "R_denial"):
        s = _seat(spec, spec) if spec in ("BASE", "P_center") \
            else _policy("BASE", spec)
        assert s["shuttle"] == 0.0, (spec, s["shuttle"])
    s = _seat("BASE", "BASE")
    assert s["exit"] == 0.0, s["exit"]


def test_revised_model_predicts_fill_time_and_split():
    """修正闭合模型（ρ 标定 + 理论 T_first/T_second）预测占满手数与分配。

    T_fill = C/(2ρ) + (T_first + T_second)/2
    先手营 = ρ·(T_fill − T_first)，后手营 = ρ·(T_fill − T_second)
    """
    s = _seat("BASE", "BASE")
    rho = _rho(s)
    assert rho == pytest.approx(0.28, abs=0.05), rho
    t_first, t_second = 3.727, 6.545          # 闭式理论值（见 test_camp_race_closed_form_model）
    tf_pred = 10 / (2 * rho) + (t_first + t_second) / 2
    f_pred = rho * (tf_pred - t_first)
    b_pred = rho * (tf_pred - t_second)
    assert f_pred + b_pred == pytest.approx(10.0, abs=1e-6)
    # 与实测对照：手数误差 < 8%，分配误差 < 0.15 营
    assert abs(tf_pred - s["plies_to_fill"][0]) / s["plies_to_fill"][0] < 0.08
    assert abs(f_pred - s["camps"][0]) < 0.15
    assert abs(b_pred - s["camps"][1]) < 0.15


def test_occupation_curve_is_monotone_and_saturates():
    """占营曲线单调不减且在 ply~26 前饱和到 10（供文档绘图的数据契约）。"""
    recs = _batch("BASE", "BASE", n=40, seed0=SEED0 + 555)
    prev = (0.0, 0.0)
    for pl in range(2, 62, 2):
        c = [0, 0]
        for r in recs:
            for p, seat, _ in r["order"]:
                if p <= pl:
                    c[seat] += 1
        cur = (c[0] / len(recs), c[1] / len(recs))
        assert cur[0] >= prev[0] - 1e-9 and cur[1] >= prev[1] - 1e-9
        assert cur[0] + cur[1] <= 10.0 + 1e-9
        prev = cur
    assert prev[0] + prev[1] == pytest.approx(10.0, abs=1e-9)
    # ply 28 时应已饱和
    sat = [0, 0]
    for r in recs:
        for p, seat, _ in r["order"]:
            if p <= 28:
                sat[seat] += 1
    assert (sat[0] + sat[1]) / len(recs) >= 9.95


# ============================================================================
#  9. Δ 战术研究（核心目标函数 Δ = camps_A − camps_B）
# ============================================================================
DELTA_RACES = 150          # 每方向场次（总 2x = 300 场/组）
DELTA_SEED = 30000


def _dbatch(a, b, n=DELTA_RACES, seed0=DELTA_SEED):
    key = ("D", a, b, n, seed0)
    if key not in _CACHE:
        _CACHE[key] = aco.delta_batch(a, b, n, aco.race_config(), seed0=seed0)
    return _CACHE[key]


def _dpol(a, b, n=DELTA_RACES, seed0=DELTA_SEED):
    return aco.summarize_delta(_dbatch(a, b, n, seed0),
                               lambda r: 0 if r["specs"][0] == a else 1)


# ------------------------------------------- 9.1 规则可行性（决定哪些战术可测）
def test_suicide_attack_is_illegal_so_low_for_high_sacrifice_is_infeasible():
    """APK 规则 allow_suicide_attack=False ⇒ "送死低价值子换敌高价值子"不可行。

    合法的牺牲只有 both_die 一类：炸弹同尽、同衔相撞；以及"走入敌杀区"
    （普通走子，下一手被敌合法吃掉）。
    """
    from junqi.rules import battle
    cfg = RuleConfig()                       # 用项目默认规则，不用测量夹具
    assert cfg.allow_suicide_attack is False

    def legal_attack(mine, foe):
        board = {q: Piece("b", Rank.PAI, False) for q in PLAY_POSITIONS}
        board[(5, 2)] = Piece("r", mine, True)
        board[(5, 3)] = Piece("b", foe, True)
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                       first_flip_done=True, ply=4, cfg=cfg)
        return any(a.kind == "move" and a.to == (5, 3) for a in st.legal_actions())

    # defender_wins 的着法根本不进入 legal_actions
    assert battle(Rank.PAI, Rank.SI) == "defender_wins"
    assert legal_attack(Rank.PAI, Rank.SI) is False
    assert battle(Rank.GONG, Rank.JUN) == "defender_wins"
    assert legal_attack(Rank.GONG, Rank.JUN) is False
    assert battle(Rank.LIAN, Rank.SHI) == "defender_wins"
    assert legal_attack(Rank.LIAN, Rank.SHI) is False
    # both_die 的着法合法
    assert battle(Rank.ZHA, Rank.SI) == "both_die"
    assert legal_attack(Rank.ZHA, Rank.SI) is True
    assert battle(Rank.PAI, Rank.PAI) == "both_die"
    assert legal_attack(Rank.PAI, Rank.PAI) is True
    # attacker_wins 合法
    assert legal_attack(Rank.SI, Rank.PAI) is True


def test_camp_outstrike_is_one_directional():
    """行营单向扑杀特权：营内子可打营外相邻敌明子，反之不合法。

    这是 10 营竞赛窗口内**唯一**开放的吃子通道（见
    test_golden6_race_window_has_almost_no_combat）。
    """
    cfg = RuleConfig()
    board = {q: Piece("b", Rank.PAI, False) for q in PLAY_POSITIONS}
    board[(3, 2)] = Piece("r", Rank.LIAN, True)      # 己方连长驻中营
    board[(3, 1)] = Piece("b", Rank.PAI, True)       # 敌排长在营外邻位
    st_out = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                       first_flip_done=True, ply=4, cfg=cfg)
    assert any(a.kind == "move" and a.frm == (3, 2) and a.to == (3, 1)
               for a in st_out.legal_actions()), "营内子应能扑杀营外敌子"
    st_in = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=1,
                      first_flip_done=True, ply=5, cfg=cfg)
    assert not any(a.kind == "move" and a.frm == (3, 1) and a.to == (3, 2)
                   for a in st_in.legal_actions()), "营内子不可被攻击"


def test_camp_entry_priority_crowds_out_captures():
    """**吃子挤出效应**：合法吃子机会并不罕见，但"进空营"优先级把它完全挤出。

    实测（300 场/组，seed0=30000）：
      · D_BASE(golden6) 每场有 2.19 / 1.55 次合法吃子机会（占回合 18.9% / 13.9%）；
      · AG_anywhere("有吃子机会就吃") 每场同样有 ~2.0 次机会，
        但**实际吃子 0.00 次**（转化率 0%）—— 因为几乎每个回合都有空营可进，
        而进营是优先级 1；
      · 唯一能落地的吃子是**行营扑杀**（AG_any_out，0.49 次/场，100% 源自营内），
        因为它用的是另一枚子，不与"进空营"竞争同一枚子；
      · 强制接触组 CT_base(spread) 把机会提高到 ~16.9 次/场（约 8 倍），
        代价是占满 10 营的手数从 22.76 涨到 44.20（1.94 倍）。
    """
    base = _dpol("D_BASE", "D_BASE")
    ag = _dpol("AG_anywhere", "D_BASE")
    out = _dpol("AG_any_out", "D_BASE")
    contact = _dpol("CT_base", "CT_base")

    # 机会并不罕见
    assert base["capture_opp_per_race"][0] > 1.0, base["capture_opp_per_race"]
    assert ag["capture_opp_per_race"][0] > 1.0, ag["capture_opp_per_race"]
    # 但激进吃子策略一次也吃不到（被进营挤出）
    assert ag["captures_total"] < 0.05, ag["captures_total"]
    # 行营扑杀是唯一落地的吃子通道，且 100% 源自营内
    assert out["captures_total"] > 0.15, out["captures_total"]
    assert sum(out["camp_origin_caps"]) > 0.15
    assert sum(out["camp_origin_caps"]) == pytest.approx(out["captures_total"], abs=0.05)
    # 强制接触把机会提高 >=4 倍，代价是占满手数 >=1.6 倍
    assert contact["capture_opp_per_race"][0] > base["capture_opp_per_race"][0] * 4
    assert contact["plies_to_fill"][0] > base["plies_to_fill"][0] * 1.6
    # 所有组完成率都是 100%（吃子不会导致竞赛无法结束）
    for s in (base, ag, out, contact):
        assert s["completion"] == pytest.approx(1.0), s["completion"]


def test_lead_oscillates_but_final_delta_favors_first_mover():
    """过程拉锯、终局偏向先手：领先权平均易手 ~0.8 次/场，但终局 Δ 显著为正。

    跨 3 个种子段（各 240 场）实测 sign_flips = 0.771 / 0.846 / 0.887，
    单场分布约为 0 次:44% / 1 次:33% / 2 次:18% / ≥3 次:5%。

    注意 Δ 每次只变化 ±1，领先权易手必然经过 0；故 sign_flips 的定义是
    "与上一个非零 Δ 反号"，只比相邻两手会漏计（+1 -> 0 -> -1）。
    """
    s = _dpol("D_BASE", "D_BASE")
    assert 0.30 <= s["sign_flips"] <= 1.50, s["sign_flips"]
    assert s["delta"][0] > 0.30 and s["delta"][3] >= 1.96, s["delta"]
    recs = _dbatch("D_BASE", "D_BASE", n=30, seed0=DELTA_SEED + 91)
    assert any(r["sign_flips"] > 0 for r in recs), "领先权应确实发生易手"
    assert any(r["sign_flips"] == 0 for r in recs), "也应有全程未被反超的场次"


def test_delta_spread_is_driven_by_occupation_not_captures():
    """**关键转折点**：|Δ| 拉开到 2 与吃子完全无关。

    96~98% 的竞赛都会达到 |Δ|>=2，而达成前的平均吃子数为 **0.00**
    （跨 3 个种子段一致）。即 Δ 的拉开纯粹由占营节奏决定，
    不存在"第 N 次吃子导致 Δ 跳变"这样的转折点。
    """
    for sd in (DELTA_SEED, DELTA_SEED + 20000, DELTA_SEED + 40000):
        s = _dpol("D_BASE", "D_BASE", n=60, seed0=sd)
        assert s["spread2_rate"] > 0.90, (sd, s["spread2_rate"])
        assert s["captures_before_spread2"] < 0.10, (sd, s["captures_before_spread2"])
    # 强制接触族同理：Δ 拉开前平均吃子数仍 <=1，远低于其全场 3+ 次吃子
    c = _dpol("CT_anywhere", "CT_base")
    assert c["captures_before_spread2"] <= 1.5, c["captures_before_spread2"]
    assert c["captures_total"] > 2.0, c["captures_total"]


# ------------------------------------------- 9.2 用户要求的核心断言
def test_captures_do_not_necessarily_increase_delta():
    """**吃子只是手段而非目的：过度吃子反而降低 Δ。**

    三条独立证据：
      (1) 低接触窗口内"到处吃子但不出营"根本不触发（吃子≈0），Δ 不变；
      (2) 唯一有效的吃子是**行营扑杀**（驻营子出击），Δ 显著为正；
      (3) 强制接触下，"到处吃子 + 出营"比"到处吃子但守营"吃子更多，
          Δ 反而更低 —— 吃子数与 Δ 不成正比。
    """
    # (1) 低接触窗口：吃子无法触发，Δ ≈ 0
    ag = _dpol("AG_anywhere", "D_BASE")
    assert ag["captures_total"] < 0.20, ag["captures_total"]
    assert abs(ag["delta"][0]) < 0.30, ag["delta"]

    # (2) 行营扑杀是低接触窗口内唯一有效的吃子战术
    out = _dpol("AG_any_out", "D_BASE")
    assert out["captures_total"] > ag["captures_total"]
    assert out["delta"][0] > 0.10, out["delta"]
    assert out["delta"][0] > ag["delta"][0]

    # (3) 强制接触：吃子更多的变体 Δ 更低
    anywhere = _dpol("CT_anywhere", "CT_base")
    any_out = _dpol("CT_any_out", "CT_base")
    assert any_out["captures_total"] > anywhere["captures_total"], \
        (any_out["captures_total"], anywhere["captures_total"])
    assert any_out["delta"][0] < anywhere["delta"][0], \
        (any_out["delta"], anywhere["delta"])
    # 且过度吃子显著拖慢竞赛
    assert any_out["plies_to_fill"][0] > anywhere["plies_to_fill"][0]


def test_bait_exposure_has_no_effect_on_delta():
    """诱骗/暴露在 Δ 上无效：明子身份是公共信息，暴露本身不构成陷阱。

    四种暴露变体（高价值/低价值/随机，对手判断正确 or 贪交换）的 Δ
    全部落在噪声内，且吃子数与不暴露组一致（暴露并未诱发出任何交换）。
    """
    ref = _dpol("CT_base", "CT_base")
    for spec in ("CT_exhigh", "CT_exlow", "CT_exrandom"):
        s = _dpol(spec, "CT_base")
        assert abs(s["delta"][0]) < 0.35, (spec, s["delta"])
        assert abs(s["delta"][2]) < 1.96, (spec, s["delta"])
        # 暴露没有诱发出额外交换：吃子数与不暴露基准同级
        assert s["captures_total"] < 0.20, (spec, s["captures_total"])
    assert ref["captures_total"] < 0.20


def test_sacrifice_tactics_are_structurally_blocked():
    """牺牲换营在竞赛窗口内基本无法执行：炸弹同尽/走入敌杀区都极少触发。

    规则层已禁止"小子撞大子"（见 test_suicide_attack_is_illegal...），
    剩下的 both_die 与"走入敌杀区"在 ~23 ply 窗口内几乎没有目标。
    """
    for spec in ("SC_bomb", "SC_walk", "SC_walk_cap"):
        s = _dpol(spec, "D_BASE")
        assert s["captures_total"] < 0.20, (spec, s["captures_total"])
        assert abs(s["delta"][0]) < 0.30, (spec, s["delta"])
    # 强制接触下牺牲才开始触发，但 Δ 收益仍不显著
    w = _dpol("CT_walk", "CT_base")
    assert abs(w["delta"][2]) < 1.96, w["delta"]


def test_blocking_tactics_trade_speed_for_nothing():
    """阻挠策略：卡位(denial)零速度代价但 Δ≈0；限制营簇(follow/split)纯亏速度。"""
    deny = _dpol("BL_deny", "D_BASE")
    assert abs(deny["delta"][0]) < 0.30, deny["delta"]
    assert abs(deny["plies_to_fill"][0] - 22.8) < 1.5, deny["plies_to_fill"]
    for spec in ("BL_enemy", "BL_self"):
        s = _dpol(spec, "D_BASE")
        assert abs(s["delta"][0]) < 0.30, (spec, s["delta"])


def test_delta_race_conserves_ten_camps_and_records_events():
    """Δ 竞赛的记录完整性：占营守恒、事件字段齐全、纪律为零违规。"""
    recs = _dbatch("CT_anywhere", "CT_base", n=40, seed0=DELTA_SEED + 31)
    assert len(recs) == 80
    for r in recs:
        assert sum(r["camps"]) == 10, r["camps"]
        assert r["delta_seat"] == r["camps"][0] - r["camps"][1]
        assert r["captures_total"] == sum(r["captures"])
        assert len(r["delta_track"]) == r["plies"]
        assert r["shuttle"] == 0
        # 损失子力等级分布必须与吃子数自洽
        n_lost = sum(sum(r["lost_ranks"][i].values()) for i in (0, 1))
        assert n_lost >= r["captures_total"]
    # 强制接触组确实产生了战斗
    tot = sum(r["captures_total"] for r in recs) / len(recs)
    assert tot > 1.0, tot

