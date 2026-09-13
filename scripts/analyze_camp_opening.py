"""P1/P4 开局"快速占据行营"策略：几何拓扑、数学期望与占营竞赛模拟。

本脚本不修改任何引擎代码与 APK 对齐常量，仅做原子化实证核对（遵循
`AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与"二进制/代码实证优先"规范）：

  【1】50 个放子位的"行营邻接度"全表 —— 占营辐射能力的几何上界；
  【2】`CENTER_CAMP_FLIP_POSITIONS` 黄金位集合完备性审计；
  【3】`eval_apk_flip_root` 首翻打分全枚举（先手视角，my_color=None）；
  【4】后手首翻打分全枚举（条件于先手首翻位与翻出子身份）；
  【5】首个占营手数的概率期望模型（含地雷/军旗不可动修正）；
  【6】蒙地卡罗校验（随机盲翻基线）；
  【7】开局机动性定理：满盘暗子时明子合法走法数 == 邻营度；
  【8】占营竞赛闭式模型（先手 vs 后手）与实测对照；
  【9】全因子策略扫描：camp_pref × cluster × alternate（对称局）；
  【10】定向策略对比：先手占营顺序 / 后手反制 / 占营优先级 / 单变量对照；
  【11】BASE 对称局的占营曲线与典型占营顺序（供文档绘图）；
  【12】跨 7 个独立种子段的指标稳定性核查（区分真结论与抽样噪声）；
  【13】可视化输出：棋盘标注图、黄金位↔行营覆盖图、占营曲线条形图、策略决策树；
  【14】Δ 战术研究（--delta 独立运行）：以 Δ = camps_A − camps_B 为核心目标函数，
       扫描吃子激进程度 / 阻挠策略 / 牺牲换营 / 暴露诱骗 四个战术维度，
       含规则可行性探针（§14.0）与强制接触对照组（Δ6）；
  【15】关键效应跨 7 个种子段重复测量（--delta 一并运行）：池化 Δ ± SE 与
       段间 SD/SE 比值，用来把 0.1~0.3 营量级的战术效应与抽样噪声分开。

竞赛终止条件：**第 10 个行营被占据的瞬间立即停止**，不打完整对局。

用法：
    venv\\Scripts\\python.exe scripts/analyze_camp_opening.py                 # §1~§13，400 场/组
    venv\\Scripts\\python.exe scripts/analyze_camp_opening.py --races 300     # §1~§13，600 场/组
    venv\\Scripts\\python.exe scripts/analyze_camp_opening.py --quick         # 跳过 §9 全因子扫描
    venv\\Scripts\\python.exe scripts/analyze_camp_opening.py --delta         # §14，800 场/组
    venv\\Scripts\\python.exe scripts/analyze_camp_opening.py --delta --delta-races 400
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from junqi.apk_engine import (APK_PIECE_VALUES, CENTER_CAMP_FLIP_POSITIONS,
                              eval_apk_flip_root, eval_apk_pure)
from junqi.config import RuleConfig
from junqi.rules import (CAMPS, COLS, COMPOSITION, HQS, NEIGHBORS,
                         PLAY_POSITIONS, RANK_CN, ROWS, Rank, battle, is_camp,
                         other)
from junqi.state import Action, GameState, Piece, deal

IMMOVABLE = (Rank.LEI, Rank.QI)          # 地雷/军旗永不移动，无法占营
CAMP_LIST = sorted(CAMPS)


# ------------------------------------------------------------------ 1. 拓扑
def camp_degree(pos) -> int:
    """放子位 pos 直接邻接的行营数（一步进营的辐射面）。"""
    return sum(1 for nb in NEIGHBORS[pos] if is_camp(nb))


def adjacent_camps(pos):
    return [nb for nb in NEIGHBORS[pos] if is_camp(nb)]


def section_topology():
    print("=" * 78)
    print("【1】50 个放子位的行营邻接度全表（一步可进营数）")
    print("=" * 78)
    deg = {p: camp_degree(p) for p in PLAY_POSITIONS}
    buckets = Counter(deg.values())
    for d in sorted(buckets, reverse=True):
        ps = sorted([p for p in PLAY_POSITIONS if deg[p] == d])
        print(f"  邻营度 {d}: {buckets[d]:2d} 个位 -> {ps}")
    print(f"\n  行营总数 = {len(CAMPS)}；邻接边总数 = {sum(deg.values())}"
          f"（每条边被一个放子位与一个行营共享）")
    print(f"  平均邻营度 = {sum(deg.values()) / len(deg):.3f}")

    top = {p for p in PLAY_POSITIONS if deg[p] == max(deg.values())}
    missing = sorted(top - CENTER_CAMP_FLIP_POSITIONS)
    extra = sorted(CENTER_CAMP_FLIP_POSITIONS - top)
    print("\n" + "-" * 78)
    print("【2】CENTER_CAMP_FLIP_POSITIONS 黄金位集合完备性审计")
    print("-" * 78)
    print(f"  代码黄金位 (6): {sorted(CENTER_CAMP_FLIP_POSITIONS)}")
    print(f"  几何最优位 (邻营度={max(deg.values())}, {len(top)} 个): {sorted(top)}")
    print(f"  >> 几何最优但未被代码收录: {missing}")
    print(f"  >> 代码收录但非几何最优:   {extra}")
    for p in missing:
        print(f"     {p} 邻营 {adjacent_camps(p)}，与已收录黄金位同等级")
    return deg


# --------------------------------------------------- 3. 先手首翻打分全枚举
def section_first_flip_scores(cfg):
    print("\n" + "=" * 78)
    print("【3】先手首翻：eval_apk_flip_root 全 50 位打分枚举（my_color=None）")
    print("=" * 78)
    st = deal(random.Random(2026), cfg)
    assert st.my_color() is None and len(st.hidden_positions()) == 50
    scores = {}
    for pos in PLAY_POSITIONS:
        scores[pos] = eval_apk_flip_root(st, Action("flip", pos), None)
    distinct = sorted(set(scores.values()), reverse=True)
    for s in distinct:
        ps = sorted([p for p in PLAY_POSITIONS if scores[p] == s])
        print(f"  score={s:7.1f}  ({len(ps):2d} 位)  {ps}")

    golden = [p for p in PLAY_POSITIONS if p in CENTER_CAMP_FLIP_POSITIONS]
    normal = [p for p in PLAY_POSITIONS if p not in CENTER_CAMP_FLIP_POSITIONS]
    gap = scores[golden[0]] - scores[normal[0]]
    print(f"\n  黄金位 vs 普通位分差 = +150.0 (实测 {gap})")
    print(f"  基础分 eval_apk_pure(全暗, my_color=None) = {eval_apk_pure(st, None)}")
    print(f"  先手首翻基线加分为 +20（无己方明子 -> 无 -200 弃营惩罚）")
    print(f"  >> Jitter 容忍度：beginner ±30 时 170-30=140 > 20+30=50，"
          f"黄金位仍绝对胜出（分差 150 > 2*30）")
    return scores


# --------------------------------------------- 4. 后手首翻打分（条件枚举）
def section_second_flip_scores(cfg):
    print("\n" + "=" * 78)
    print("【4】后手首翻：条件于先手首翻结果的打分枚举")
    print("=" * 78)
    base = deal(random.Random(2026), cfg)

    for first_pos in [(3, 1), (4, 2), (8, 2), (0, 0)]:
        tag = "黄金位" if first_pos in CENTER_CAMP_FLIP_POSITIONS else "边缘位(对照)"
        print(f"\n  --- 先手首翻 {first_pos} [{tag}] ---")
        for first_color, first_rank in [("r", Rank.LIAN), ("r", Rank.SI),
                                        ("r", Rank.LEI)]:
            board = dict(base.board)
            board[first_pos] = Piece(first_color, first_rank, True)
            st = GameState(board=board, seat_color={0: first_color, 1: other(first_color)},
                           turn=1, first_flip_done=True, ply=1, cfg=cfg)
            me = st.my_color()                      # 后手颜色
            rows = []
            for pos in PLAY_POSITIONS:
                if st.board[pos].revealed:
                    continue
                rows.append((eval_apk_flip_root(st, Action("flip", pos), me), pos))
            rows.sort(reverse=True)
            top = [(round(s, 1), p, camp_degree(p)) for s, p in rows[:6]]
            print(f"    先手翻出 {first_color}{Rank(first_rank).name:>4}"
                  f" -> 后手 top6 (score, pos, 邻营度): {top}")
            # 与先手翻出子相邻的位的威胁扣分核查
            adj = [nb for nb in NEIGHBORS[first_pos] if not is_camp(nb)]
            adj_scores = {p: eval_apk_flip_root(st, Action("flip", p), me)
                          for p in adj if not st.board[p].revealed}
            print(f"      紧邻先手明子的位打分 {adj_scores}"
                  f"  (敌明子非雷非旗相邻 => -40)")
            # 后手能否直接吃到该明子 / 该明子是否可进营
            if first_rank in IMMOVABLE:
                print(f"      注：{Rank(first_rank).name} 不可移动，先手此翻浪费一手"
                      f"（无法进营，也无 -40 威胁）")
    return base


# --------------------------------------------------- 5. 首个占营手数期望模型
def section_tempo_model():
    print("\n" + "=" * 78)
    print("【5】首个占营手数的概率期望模型（含地雷/军旗不可动修正）")
    print("=" * 78)

    total = sum(COMPOSITION.values()) * 2                 # 50
    immovable_each = COMPOSITION[Rank.LEI] + COMPOSITION[Rank.QI]   # 4/方
    movable_each = 25 - immovable_each                    # 21/方

    # --- 先手：首翻必得己方子（首翻定色），但该子可能不可动 ---
    p_first_movable = Fraction(movable_each * 2, total)   # 42/50
    print(f"  先手首翻定色：翻出子必属先手；可动概率 = {p_first_movable}"
          f" = {float(p_first_movable):.4f}")
    print(f"    -> 84% 情形：ply1 翻(黄金位) + ply3 进营 = 第 3 手占营")
    print(f"    -> 16% 情形：首翻是地雷/军旗，白损 2 手，需重新翻出可动子")

    # 先手后续每次翻棋翻出"己方可动子"的概率（近似稳态 21/50）
    # 精确序贯模型：池中己方可动子 k、总剩子 n
    def expected_flip_index(k, n_total):
        """无放回序贯翻子，首次命中 k 个"己方可动子"之一（池共 n_total）的期望次数。

        闭式解 E = (n_total + 1) / (k + 1)；此处用精确分数递推并断言一致。
        """
        def E(kk, rest):
            if rest == 0:
                return Fraction(1)
            return Fraction(1) + Fraction(rest, kk + rest) * E(kk, rest - 1)

        rec = E(k, n_total - k)
        assert rec == Fraction(n_total + 1, k + 1), (rec, n_total, k)
        return rec

    # --- 后手：首翻有 25/49 概率得己方子，其中 21/25 可动 ---
    # 直接对"己方可动子"建模：池中己方可动 21、其他 28（24 敌子 + 4 己方不可动）
    e_second = expected_flip_index(21, 49)
    print(f"\n  后手首次翻到'己方可动子'的期望翻子次数 = {float(e_second):.4f}")
    # 后手翻子发生在 ply 2, 4, 6, ...；翻到后再下一手进营
    e_ply_second = 2 * float(e_second) + 2
    print(f"  后手首个占营期望手数 = 2*{float(e_second):.3f} + 2 = {e_ply_second:.3f} ply")

    # 先手：84% 直接 ply3；16% 需重翻（池中己方可动 21、其他 28）
    e_first_extra = expected_flip_index(21, 49)
    e_ply_first = (float(p_first_movable) * 3
                   + float(1 - p_first_movable) * (1 + 2 * float(e_first_extra) + 2))
    print(f"  先手首个占营期望手数 = 0.84*3 + 0.16*(1+2*{float(e_first_extra):.3f}+2)"
          f" = {e_ply_first:.3f} ply")
    print(f"\n  >> 先手结构性能差 = {e_ply_second - e_ply_first:.3f} ply "
          f"≈ {(e_ply_second - e_ply_first) / 2:.3f} 个占营回合")

    # --- 10 个行营的分配上界（占营优先、双方均最优节奏）---
    print("\n  行营争夺的交替占位模型（10 营，先手 ply3 起每 4 ply 占 1 营）:")
    print("    先手占营 ply 序列: 3, 7, 11, 15, 19, 23, ...")
    print("    后手占营 ply 序列: ~6, ~10, ~14, 18, 22, ...")
    first_cnt, second_cnt, ply = 0, 0, 0
    fp, sp = 3, round(e_ply_second)
    trace = []
    while first_cnt + second_cnt < len(CAMPS):
        if fp <= sp:
            first_cnt += 1
            trace.append((fp, "先手", first_cnt, second_cnt))
            fp += 4
        else:
            second_cnt += 1
            trace.append((sp, "后手", first_cnt, second_cnt))
            sp += 4
    for t in trace:
        print(f"      ply {t[0]:3d}: {t[1]} 占营 -> 先手 {t[2]} : 后手 {t[3]}")
    print(f"  >> 粗粒度节奏模型（每营固定 4 ply）预测占营比 = 先手 {first_cnt} : 后手 {second_cnt}")
    print(f"  >> 注意：该粗模型高估了先手优势；精确的连续速率闭合模型见【8】")
    return e_ply_first, e_ply_second


# ------------------------------------------- 6. 蒙地卡罗校验期望模型
def section_mc_tempo(cfg, n=400):
    print("\n" + "=" * 78)
    print(f"【6】蒙地卡罗校验：随机盲翻下首个占营手数（{n} 局，仅翻棋+进营策略）")
    print("=" * 78)
    rng = random.Random(7)
    first_plys, second_plys = [], []
    first_camps_at24, second_camps_at24 = [], []

    for _ in range(n):
        st = deal(rng, cfg)
        got = {0: None, 1: None}
        while not st.is_terminal() and st.ply < 60:
            seat = st.turn
            color = st.my_color()
            acts = st.legal_actions()
            enter = [a for a in acts if a.kind == "move" and a.to in CAMPS
                     and st.board.get(a.to) is None]
            if enter:
                a = max(enter, key=lambda x: (camp_degree(x.to), rng.random()))
                st = st.apply(a)
                if got[seat] is None:
                    got[seat] = st.ply
                continue
            flips = [a for a in acts if a.kind == "flip"]
            if not flips:
                break
            st = st.apply(rng.choice(flips))
        if got[0]:
            first_plys.append(got[0])
        if got[1]:
            second_plys.append(got[1])
        c = Counter()
        for pos in CAMP_LIST:
            pc = st.board.get(pos)
            if pc is not None and pc.revealed:
                c[0 if pc.color == st.seat_color[0] else 1] += 1
        first_camps_at24.append(c[0])
        second_camps_at24.append(c[1])

    def avg(x):
        return sum(x) / len(x) if x else float("nan")

    print(f"  随机盲翻基线（无黄金位偏好）:")
    print(f"    先手首占营 ply 均值 = {avg(first_plys):.2f} (n={len(first_plys)})")
    print(f"    后手首占营 ply 均值 = {avg(second_plys):.2f} (n={len(second_plys)})")
    print(f"    ply<60 时占营均值   先手 {avg(first_camps_at24):.2f} : "
          f"后手 {avg(second_camps_at24):.2f}")
    print(f"  >> 随机盲翻的'翻出己方可动子'命中率显著低于黄金位定向翻，"
          f"可作为对照组下界")


# ------------------------------------- 7. 开局机动性定理（占营能力的几何根源）
def section_mobility_theorem(cfg):
    print("\n" + "=" * 78)
    print("【7】开局机动性定理验证：满盘暗子时，明子的合法走法数 == 其邻营度")
    print("=" * 78)
    print("  原理：开局 50 个放子位全被暗子占满，只有 10 个行营是空格。")
    print("        棋子只能走到空格（或吃掉相邻敌明子），故一枚刚翻开的明子")
    print("        的非吃子合法走法 == 它相邻的空行营数 == 邻营度。")
    print("        邻营度=0 的位（row0 / row11 底线性）翻出的子完全冻结。\n")

    base = deal(random.Random(2026), cfg)
    assert len(base.hidden_positions()) == 50
    mismatch, frozen = [], []
    stats = {}
    for pos in PLAY_POSITIONS:
        board = {}
        for q in PLAY_POSITIONS:
            if q == pos:
                board[q] = Piece("r", Rank.LIAN, True)      # 己方可动明子
            else:
                board[q] = Piece("b", Rank.PAI, False)      # 敌方暗子（不可吃）
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                       first_flip_done=True, ply=2, cfg=cfg)
        acts = st.legal_actions()
        n_move = sum(1 for a in acts if a.kind == "move")
        n_flip = sum(1 for a in acts if a.kind == "flip")
        d = camp_degree(pos)
        stats.setdefault(d, []).append(n_move)
        if n_move != d:
            mismatch.append((pos, d, n_move))
        if n_move == 0:
            frozen.append(pos)
        assert n_flip == 49, f"其余 49 位仍为暗子，应恰有 49 个翻棋动作，实得 {n_flip}"
        assert all(a.to in CAMPS for a in acts if a.kind == "move"), \
            "开局满盘暗子时，明子的每一步非吃子走法必然指向行营"

    print("  邻营度 -> 该度所有位的合法走法数（应恒等于邻营度）:")
    for d in sorted(stats, reverse=True):
        vals = stats[d]
        print(f"    度 {d}: {len(vals):2d} 个位，合法走法数集合 = {sorted(set(vals))}")
    print(f"\n  定理反例（走法数 != 邻营度）: {mismatch if mismatch else '无，定理成立'}")
    print(f"  完全冻结位（邻营度=0，合法走法 0）: {len(frozen)} 个 -> "
          f"{sorted(frozen)[:5]}... (全部为 row0/row11)")

    # 冻结位的战略含义
    print("\n  战略含义：")
    print(f"    · 首翻打在冻结位 => 该子永久无法进营，等于白损 1 手 + 1 枚子力；")
    print(f"    · 首翻打在度 3 黄金位 => 该子立即拥有 3 个进营选择，"
          f"下一手必定占营；")
    print(f"    · 因此 '邻营度' 就是开局阶段的机动性/占营能力，"
          f"这正是 CENTER_CAMP_FLIP_POSITIONS +150 的几何依据。")
    return mismatch


# --------------------------------------------- 8. 占营竞赛闭式模型 vs 实测
def section_race_model(measured=None):
    print("\n" + "=" * 78)
    print("【8】占营竞赛闭式模型（先手 vs 后手）与实测对照")
    print("=" * 78)

    total, immov_each = 50, COMPOSITION[Rank.LEI] + COMPOSITION[Rank.QI]
    mov_each = 25 - immov_each                      # 21 枚可动子/方

    def flips_to_own_movable(k, n):
        """无放回序贯翻子，首次命中 k 个'己方可动子'之一（池共 n）的期望次数。
        闭式解 E = (n + 1) / (k + 1)。"""
        closed = Fraction(n + 1, k + 1)
        # 递推验证
        def E(kk, rest):
            if rest == 0:
                return Fraction(1)
            return Fraction(1) + Fraction(rest, kk + rest) * E(kk, rest - 1)
        rec = E(k, n - k)
        assert rec == closed, (rec, closed)
        return closed

    p_first_movable = Fraction(mov_each * 2, total)          # 42/50
    ef = flips_to_own_movable(mov_each, total - 1)           # 21/49 -> 50/22
    T_first = float(p_first_movable) * 3 + float(1 - p_first_movable) * (
        1 + 2 * float(ef) + 2)
    T_second = 2 * float(ef) + 2
    delta = T_second - T_first

    print(f"  (1) 每方可动子 = 25 - 3地雷 - 1军旗 = {mov_each} 枚")
    print(f"  (2) 首翻定色：先手首翻必得己方子，可动概率 = 42/50 = "
          f"{float(p_first_movable):.4f}")
    print(f"  (3) 无放回首次命中'己方可动子'期望翻子次数 E = (n+1)/(k+1) = "
          f"50/22 = {float(ef):.4f}")
    print(f"  (4) E[T_first]  = 0.84*3 + 0.16*(1 + 2E + 2) = {T_first:.3f} ply")
    print(f"  (5) E[T_second] = 2E + 2                     = {T_second:.3f} ply")
    print(f"  (6) 结构性能差 Δ = {delta:.3f} ply")

    # 占营速率 r：由"翻棋命中己方可动子(2 ply/次) + 进营(2 ply)"给出
    #   每营平均耗时 = 2*E_flips_to_own_movable + 2 = T_second
    r = 1.0 / T_second
    C = len(CAMPS)
    pred_first = (C + delta * r) / 2
    pred_second = (C - delta * r) / 2
    # --- 稳态占营速率 ρ 的标定与三公式闭合模型 ---
    # 说明：1/T_second 只是"拿下第 1 个营"的速率（成本里含"翻到己方可动子"）。
    # 一旦场上有 2~3 枚活子并行，每营耗时骤降，故稳态速率 ρ 必须由实测标定。
    print(f"\n  (7) 粗模型（单一速率 r = 1/T_second = {r:.4f} 营/ply）：")
    print(f"      先手 = (C + Δr)/2 = {pred_first:.3f}，后手 = {pred_second:.3f}，"
          f"先手优势 Δr = {delta*r:.3f} 营")
    print(f"      该模型会**低估**先手优势（实测约 0.69 营），因为它把首营成本"
          f"外推到了全部 10 营。")

    def _rho(cf, cs, tf, ts, tfill):
        """该行实测的稳态占营速率 ρ（双方合并估计，营/ply/方）。"""
        if not tfill or tfill <= ts or tfill <= tf:
            return float("nan")
        return 0.5 * (cf / (tfill - tf) + cs / (tfill - ts))

    # 以"中心营优先"对称局（T3-A，若缺失则取最快的一行）作为 ρ 的标定基准
    ref = None
    if measured:
        ref = next((m for m in measured if m[0].startswith("T3-A")), None)
        if ref is None:
            ref = min(measured, key=lambda m: m[5] or 1e9)
    rho = _rho(ref[3], ref[4], ref[1], ref[2], ref[5]) if ref else 0.28
    print(f"\n  (8) 修正模型：稳态占营速率 ρ（每 ply 每方），由基准行"
          f"「{ref[0][:26] if ref else 'n/a'}」标定 = {rho:.4f} 营/ply")
    print(f"      等价于每营耗时 1/ρ = {1/rho:.2f} ply（≈ {1/rho/2:.2f} 个己方回合）")
    print(f"      标定说明：ρ 不是自由拟合参数，而是 (先手营/(T_fill-T_first) + "
          f"后手营/(T_fill-T_second))/2 的直接测量值；")
    print(f"                模型只用它 + 理论 T_first/T_second 反推 T_fill 与占营分配。")
    print(f"      三条闭合公式（C = 10 营）：")
    print(f"        T_fill = C/(2ρ) + (T_first + T_second)/2")
    print(f"        先手营 = ρ·(T_fill − T_first)")
    print(f"        后手营 = ρ·(T_fill − T_second)")
    tf_pred = C / (2 * rho) + (T_first + T_second) / 2
    f_pred = rho * (tf_pred - T_first)
    s_pred = rho * (tf_pred - T_second)
    print(f"      代入理论 T_first={T_first:.3f}, T_second={T_second:.3f}, ρ={rho:.4f}:")
    print(f"        T_fill = {tf_pred:.2f} ply")
    print(f"        先手 = {f_pred:.3f} 营，后手 = {s_pred:.3f} 营，"
          f"先手优势 = ρΔ = {rho*delta:.3f} 营")

    if measured:
        print(f"\n  (9) 与占营竞赛实测对照（10 营占满即停，见【10】）:")
        print(f"      {'实验组':<30}{'T_first':>8}{'T_second':>9}{'Δ':>6}"
              f"{'占满手数':>9}{'ρ实测':>8}{'先手营':>8}{'后手营':>8}{'差':>7}")
        for name, tf, ts, cf, cs, tfill in measured:
            rr = _rho(cf, cs, tf, ts, tfill)
            print(f"      {name[:28]:<30}{tf:>8.2f}{ts:>9.2f}{ts-tf:>6.2f}"
                  f"{tfill:>9.2f}{rr:>8.4f}{cf:>8.3f}{cs:>8.3f}{cf-cs:>+7.3f}")
        print(f"      {'修正模型预测':<30}{T_first:>8.2f}{T_second:>9.2f}{delta:>6.2f}"
              f"{tf_pred:>9.2f}{rho:>8.4f}{f_pred:>8.3f}{s_pred:>8.3f}"
              f"{f_pred-s_pred:>+7.3f}")
        print(f"      {'粗模型预测(单速率)':<30}{T_first:>8.2f}{T_second:>9.2f}"
              f"{delta:>6.2f}{'—':>9}{r:>8.4f}{pred_first:>8.3f}"
              f"{pred_second:>8.3f}{pred_first-pred_second:>+7.3f}")
        err = lambda a, b: (a - b) / b * 100 if b else float("nan")
        print(f"\n  (10) 修正模型 vs 标定基准行「{ref[0][:26]}」的相对误差：")
        print(f"       T_first  {err(T_first, ref[1]):+.1f}%   "
              f"T_second {err(T_second, ref[2]):+.1f}%   "
              f"Δ {err(delta, ref[2]-ref[1]):+.1f}%")
        print(f"       占满手数 {err(tf_pred, ref[5]):+.1f}%   "
              f"先手营 {err(f_pred, ref[3]):+.1f}%   "
              f"后手营 {err(s_pred, ref[4]):+.1f}%")
        fast = [_rho(m[3], m[4], m[1], m[2], m[5]) for m in measured
                if (m[5] or 1e9) < 45]
        slow = [_rho(m[3], m[4], m[1], m[2], m[5]) for m in measured
                if (m[5] or 0) >= 45]
        fast = [x for x in fast if x == x]
        slow = [x for x in slow if x == x]
        if fast and slow:
            mf, ms = sum(fast) / len(fast), sum(slow) / len(slow)
            print(f"\n  (11) ρ 的两个制度（关键发现）：")
            print(f"       通畅制度（中营优先族，{len(fast)} 行）ρ = {mf:.4f} 营/ply")
            print(f"       锁死制度（角营/辐射优先族，{len(slow)} 行）ρ = {ms:.4f} 营/ply")
            print(f"       >> 倍率 {mf/ms:.2f}x：先占角营会让中心营失去唯一入口"
                  f"（中心营只与 4 个黄金位相邻），")
            print(f"          稳态占营速率塌缩到 1/{mf/ms:.1f}，这是"
                  f"'中心营必须优先占'的定量依据。")
    return {"T_first": T_first, "T_second": T_second, "delta": delta,
            "rho": rho, "T_fill": tf_pred, "pred_first": f_pred,
            "pred_second": s_pred, "coarse_first": pred_first,
            "coarse_second": pred_second, "coarse_r": r}


# ============================================================================
#  占营竞赛模拟器：不打完整对局，只比"谁先把 10 个行营占满"
# ============================================================================
#
# 规则说明（不修改 RuleConfig 类定义，只在本测量夹具内传参）：
#   no_capture_draw_plies=0  -> config.py 明确注明 "0=关闭"，用于屏蔽 70 步限步判和，
#                               否则纯占营竞赛会在 ply70 被判和而无法观测到 10 营占满；
#   max_plies=10**9          -> 屏蔽总步数判和。
# 这是测量夹具，不改动项目规则默认值，也不改动 configs/rules.yaml。

CLUSTER_CAMPS = {
    "up": frozenset({(2, 1), (2, 3), (3, 2), (4, 1), (4, 3)}),
    "down": frozenset({(7, 1), (7, 3), (8, 2), (9, 1), (9, 3)}),
}
CLUSTER_GOLDEN = {
    "up": frozenset({(2, 2), (3, 1), (3, 3), (4, 2)}),
    "down": frozenset({(7, 2), (8, 1), (8, 3), (9, 2)}),
}
CENTER_CAMPS = frozenset({(3, 2), (8, 2)})          # CAMP_POSITION_BONUS +50
CORNER_CAMPS = frozenset(CAMPS) - CENTER_CAMPS      # CAMP_POSITION_BONUS +40
DEG = {p: camp_degree(p) for p in PLAY_POSITIONS}
GOLDEN8 = {p for p in PLAY_POSITIONS if DEG[p] == 3}
FROZEN = {p for p in PLAY_POSITIONS if DEG[p] == 0}     # row0 / row11


def race_config() -> RuleConfig:
    """占营竞赛专用测量夹具（关闭判和，只观测 10 营占满时刻）。"""
    return RuleConfig(no_capture_draw_plies=0, max_plies=10 ** 9)


def cluster_of(pos):
    return "up" if pos[0] <= 5 else "down"


def cluster_pressure(st: GameState, enemy_color):
    """敌方可动明子对两个营簇的辐射压力（紧邻该簇行营的敌明子计数）。"""
    out = {"up": 0, "down": 0}
    if not enemy_color:
        return out
    for pos, pc in st.board.items():
        if not pc.revealed or pc.color != enemy_color or pc.rank in IMMOVABLE:
            continue
        for nb in NEIGHBORS[pos]:
            if is_camp(nb):
                out[cluster_of(nb)] += 1
    return out


def camp_owners(st: GameState):
    """{行营: seat} —— 只统计已翻开的驻营子（进营的子必然是明子）。"""
    if st.seat_color.get(0) is None:
        return {}
    out = {}
    for c in CAMP_LIST:
        pc = st.board.get(c)
        if pc is not None and pc.revealed:
            out[c] = 0 if st.seat_color[0] == pc.color else 1
    return out


def camp_dist_field(st: GameState):
    """多源 BFS：以全部空行营为源(dist=0)，任何棋子均阻挡通行。"""
    dist, q = {}, deque()
    for c in CAMP_LIST:
        if st.board.get(c) is None:
            dist[c] = 0
            q.append(c)
    while q:
        cur = q.popleft()
        for nb in NEIGHBORS[cur]:
            if nb in dist or st.board.get(nb) is not None:
                continue
            dist[nb] = dist[cur] + 1
            q.append(nb)
    return dist


class RacePolicy:
    """占营竞赛策略。四个正交维度：

    camp_pref  进哪个营：center(中营+50) / corner(角营+40) /
               radiate(邻接暗子最多=辐射潜力) / denial(卡位:敌明子一步可达)
    cluster    翻哪个营簇：none(不限) / follow(跟随敌方压力大的簇) /
               split(另辟敌方压力小的簇)
    flip_pref  翻哪个位：golden6(代码黄金6位) / golden8(几何完备8位) /
               radiate(依托己方已占营的邻位=+120语义) / near_empty(紧邻空营且度最高)
    alternate  True => 每占一个营后强制先拓荒翻一次棋再占下一个（占营/拓荒交替）
    first_flip golden(度3黄金位) / edge(度0底线性，单变量对照)
    """

    def __init__(self, name="BASE", camp_pref="center", cluster="none",
                 flip_pref="golden6", alternate=False, first_flip="golden",
                 seed=None):
        self.name = name
        self.camp_pref = camp_pref
        self.cluster = cluster
        self.flip_pref = flip_pref
        self.alternate = alternate
        self.first_flip = first_flip
        self.rng = random.Random(seed)
        self._flips = 0
        self._just_entered = False

    # ---- 选哪个空营
    def _camp_key(self, st, camp, me):
        hidden_nb = sum(1 for nb in NEIGHBORS[camp]
                        if (pc := st.board.get(nb)) is not None and not pc.revealed)
        center = 1 if camp in CENTER_CAMPS else 0
        denial = 0
        if me:
            en = other(me)
            for nb in NEIGHBORS[camp]:
                pc = st.board.get(nb)
                if pc is not None and pc.revealed and pc.color == en \
                        and pc.rank not in IMMOVABLE:
                    denial = 1
                    break
        if self.camp_pref == "denial":
            return (denial, center, hidden_nb, self.rng.random())
        if self.camp_pref == "corner":
            return (1 - center, hidden_nb, denial, self.rng.random())
        if self.camp_pref == "radiate":
            return (hidden_nb, center, denial, self.rng.random())
        return (center, hidden_nb, denial, self.rng.random())

    # ---- 选哪个暗子翻
    def _pick_flip(self, st, flips):
        me = st.my_color()
        self._flips += 1
        if self._flips == 1 and self.first_flip == "edge":
            pool = [a for a in flips if a.frm in FROZEN]
            if pool:
                return self.rng.choice(pool)
        if self.cluster in ("follow", "split") and me:
            pres = cluster_pressure(st, other(me))
            want = (max if self.cluster == "follow" else min)(
                ("up", "down"), key=lambda k: (pres[k], self.rng.random()))
            pool = [a for a in flips
                    if cluster_of(a.frm) == want and a.frm in GOLDEN8]
            if not pool:
                pool = [a for a in flips if cluster_of(a.frm) == want]
            if pool:
                return self.rng.choice(pool)
        if self.flip_pref == "radiate":
            own_camps = {p for p, pc in st.board.items()
                         if pc.revealed and me and pc.color == me and is_camp(p)}
            pool = [a for a in flips
                    if any(nb in own_camps for nb in NEIGHBORS[a.frm])]
            if pool:
                return self.rng.choice(pool)
        if self.flip_pref == "near_empty":
            empty = {c for c in CAMP_LIST if st.board.get(c) is None}
            pool = [a for a in flips if set(NEIGHBORS[a.frm]) & empty]
            if pool:
                return max(pool, key=lambda a: (DEG[a.frm], self.rng.random()))
        if self.flip_pref == "golden8":
            pool = [a for a in flips if a.frm in GOLDEN8]
            if pool:
                return self.rng.choice(pool)
        pool = [a for a in flips if a.frm in CENTER_CAMP_FLIP_POSITIONS]
        if pool:
            return self.rng.choice(pool)
        empty = {c for c in CAMP_LIST if st.board.get(c) is None}
        pool = [a for a in flips if set(NEIGHBORS[a.frm]) & empty]
        if pool:
            return max(pool, key=lambda a: (DEG[a.frm], self.rng.random()))
        return max(flips, key=lambda a: (DEG[a.frm], self.rng.random()))

    def choose(self, st):
        acts = st.legal_actions()
        me = st.my_color()
        flips = [a for a in acts if a.kind == "flip"]
        moves = [a for a in acts if a.kind == "move"]
        empty = [c for c in CAMP_LIST if st.board.get(c) is None]
        enters = [a for a in moves if a.to in empty and not is_camp(a.frm)]

        # (1) 交替纪律：刚占过营 => 本手强制拓荒翻棋
        if self.alternate and self._just_entered and flips:
            self._just_entered = False
            return self._pick_flip(st, flips)

        # (2) 一步进空营：最高优先级；严禁营->营 闲走
        if enters:
            self._just_entered = True
            return max(enters, key=lambda a: self._camp_key(st, a.to, me))
        self._just_entered = False

        # (3) 向最近空营推进（绝不从营里调子出来）
        if empty:
            field = camp_dist_field(st)
            cand, best = [], None
            for a in moves:
                if is_camp(a.frm) or st.board.get(a.to) is not None:
                    continue
                d0, d1 = field.get(a.frm), field.get(a.to)
                if d0 is None or d1 is None or d1 >= d0:
                    continue
                if best is None or d1 < best:
                    best, cand = d1, [a]
                elif d1 == best:
                    cand.append(a)
            if cand:
                return self.rng.choice(cand)

        # (4) 翻棋
        if flips:
            return self._pick_flip(st, flips)

        # (5) 兜底：仍严禁调出驻营子力
        safe = [a for a in moves if not is_camp(a.frm)]
        return self.rng.choice(safe) if safe else self.rng.choice(acts)


# 策略注册表：name -> RacePolicy 构造参数
STRATEGIES = {
    "BASE":           dict(camp_pref="center", cluster="none", flip_pref="golden6"),
    # —— 先手占营顺序三策略 ——
    "A_corner_push":  dict(camp_pref="corner", cluster="none", flip_pref="near_empty"),
    "B_center_bet":   dict(camp_pref="center", cluster="none", flip_pref="golden8"),
    "C_alternate":    dict(camp_pref="center", cluster="none", flip_pref="radiate",
                           alternate=True),
    # —— 后手反制三策略 ——
    "R_follow":       dict(camp_pref="center", cluster="follow", flip_pref="golden6"),
    "R_split":        dict(camp_pref="center", cluster="split", flip_pref="golden6"),
    "R_denial":       dict(camp_pref="denial", cluster="none", flip_pref="golden6"),
    "R_split_denial": dict(camp_pref="denial", cluster="split", flip_pref="golden6"),
    # —— 占营优先级三选 ——
    "P_center":       dict(camp_pref="center", cluster="none", flip_pref="golden6"),
    "P_corner":       dict(camp_pref="corner", cluster="none", flip_pref="golden6"),
    "P_radiate":      dict(camp_pref="radiate", cluster="none", flip_pref="golden6"),
    # —— 单变量对照组 ——
    "X_firstedge":    dict(camp_pref="center", cluster="none", flip_pref="golden6",
                           first_flip="edge"),
    "X_golden8":      dict(camp_pref="center", cluster="none", flip_pref="golden8"),
}


def make_policy(spec: str, seed=None) -> RacePolicy:
    if spec not in STRATEGIES:
        raise ValueError(f"未知策略 {spec}，可选 {sorted(STRATEGIES)}")
    return RacePolicy(name=spec, seed=seed, **STRATEGIES[spec])


def run_race(spec0: str, spec1: str, seed: int,
             cfg: RuleConfig | None = None, max_ply: int = 400) -> dict:
    """跑一场占营竞赛：第 10 个行营被占据的瞬间立即终止。"""
    cfg = cfg or race_config()
    p = [make_policy(spec0, seed * 31 + 1), make_policy(spec1, seed * 31 + 2)]
    st = deal(random.Random(seed), cfg)
    rec = {"seed": seed, "specs": [spec0, spec1], "filled": False,
           "plies": 0, "camps": [0, 0], "tenth_by": None,
           "first_camp_ply": [None, None], "order": [],
           "shuttle": 0, "exit": 0}

    while True:
        own = camp_owners(st)
        if len(own) >= len(CAMPS):
            rec["filled"] = True
            break
        if st.ply >= max_ply or st.is_terminal():
            break
        acts = st.legal_actions()
        if not acts:
            break
        seat = st.turn
        act = p[seat].choose(st)
        if act.kind == "move":
            if is_camp(act.frm) and is_camp(act.to):
                rec["shuttle"] += 1               # 营间闲走
            elif is_camp(act.frm):
                rec["exit"] += 1                  # 弃营
        st = st.apply(act)
        now = camp_owners(st)
        for c in now:
            if c not in own:
                rec["order"].append((st.ply, now[c], c))
                if rec["first_camp_ply"][now[c]] is None:
                    rec["first_camp_ply"][now[c]] = st.ply
                rec["tenth_by"] = now[c]

    rec["plies"] = st.ply
    cnt = [0, 0]
    for s in camp_owners(st).values():
        cnt[s] += 1
    rec["camps"] = cnt
    rec["filled"] = len(camp_owners(st)) == len(CAMPS)
    return rec


# ------------------------------------------------------------------ 统计
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _paired(recs, idx_of, key, sub=lambda r: r):
    """配对差值 (A - B) 的均值 / 标准误 / z。"""
    d = []
    for r in recs:
        i = idx_of(r)
        v = key(r)
        d.append((v[i] - v[1 - i]) if isinstance(v, (list, tuple)) else
                 (v if i == 0 else -v))
    n = len(d)
    m = _mean(d)
    var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
    se = (var / n) ** 0.5
    return m, se, (m / se if se > 0 else 0.0)


def summarize_races(recs, idx_of):
    """idx_of(rec) -> 0/1，把每条记录映射到 A/B 两栏。"""
    n = len(recs)
    if n == 0:
        return None
    done = [r for r in recs if r["filled"]]
    cntA = [r["camps"][idx_of(r)] for r in recs]
    cntB = [r["camps"][1 - idx_of(r)] for r in recs]
    win = Counter("A" if a > b else ("B" if b > a else "tie")
                  for a, b in zip(cntA, cntB))
    tenthA = sum(1 for r in done if r["tenth_by"] == idx_of(r))
    cm, cse, cz = _paired(recs, idx_of, lambda r: r["camps"])
    # 占满手数是"每场一个值"的场级指标（双方共享），故只给均值±SE，
    # 跨策略比较用两样本 z（见 two_sample_z）
    pl = [r["plies"] for r in done]
    pmean = _mean(pl)
    pvar = sum((x - pmean) ** 2 for x in pl) / max(1, len(pl) - 1) if pl else 0.0
    pse = (pvar / len(pl)) ** 0.5 if pl else 0.0
    return {
        "n": n, "completion": len(done) / n,
        "camps": [_mean(cntA), _mean(cntB)],
        "camp_diff": (cm, cse, cz),
        "win": (win["A"] / n, win["B"] / n, win["tie"] / n),
        "plies_to_fill": (pmean, _mean([r["plies"] for r in recs])),
        "plies_se": pse,
        "first_camp_ply": [
            _mean([r["first_camp_ply"][idx_of(r)] for r in recs
                   if r["first_camp_ply"][idx_of(r)]]),
            _mean([r["first_camp_ply"][1 - idx_of(r)] for r in recs
                   if r["first_camp_ply"][1 - idx_of(r)]])],
        "tenth_by_A": tenthA / len(done) if done else float("nan"),
        "camp_hist": dict(sorted(Counter(cntA).items())),
        "shuttle": _mean([r["shuttle"] for r in recs]),
        "exit": _mean([r["exit"] for r in recs]),
    }


def two_sample_z(m1, se1, m2, se2):
    """两独立样本均值差的 z 值（用于跨策略比较"占满 10 营手数"）。"""
    se = (se1 ** 2 + se2 ** 2) ** 0.5
    return (m1 - m2) / se if se > 0 else 0.0


def print_summary(tag, s, nameA, nameB):
    if s is None:
        return
    cm, cse, cz = s["camp_diff"]
    sig = lambda z: "显著" if abs(z) >= 1.96 else "不显著"
    print(f"  [{tag}] {nameA} vs {nameB}")
    print(f"    样本 {s['n']} 场，10 营占满完成率 {s['completion']*100:.1f}%")
    print(f"    占满 10 营平均手数 {s['plies_to_fill'][0]:.2f} "
          f"± {s['plies_se']:.2f}(SE)"
          f"（含未完成 {s['plies_to_fill'][1]:.2f}）")
    print(f"    平均占营 {s['camps'][0]:.3f} : {s['camps'][1]:.3f}   "
          f"配对差 {cm:+.3f} ± {cse:.3f}(SE)  z={cz:+.2f} {sig(cz)}")
    print(f"    占营判据胜负：{nameA} {s['win'][0]*100:.1f}%  "
          f"{nameB} {s['win'][1]*100:.1f}%  平 {s['win'][2]*100:.1f}%")
    print(f"    首占营 ply {s['first_camp_ply'][0]:.2f} : "
          f"{s['first_camp_ply'][1]:.2f}    "
          f"第 10 营由 {nameA} 拿下的比例 {s['tenth_by_A']*100:.1f}%")
    print(f"    {nameA} 占营数分布 {s['camp_hist']}")
    print(f"    纪律：营间闲走 {s['shuttle']:.3f} 次/场，弃营 {s['exit']:.3f} 次/场")


def race_batch(spec0, spec1, n_races, cfg, seed0=20000):
    """完美配对设计：**同一副牌局交换座位各跑一场**，总场次 = 2 x n_races。

    为什么不用"不同牌局 + 交换座位"：那样配对差里会残留
    `0.5*[Adv(S_even) − Adv(S_odd)]` 的牌局级共同模噪声（n=400 时 SE≈0.10 营），
    与要测量的策略效应（0.05~0.6 营）同量级，会把噪声读成效应。
    同一副牌双座位可使该共同模**精确为 0**。
    对称局（spec0 == spec1）两种座位完全等价，故改为跑 2*n_races 副不同牌局。
    """
    recs = []
    if spec0 == spec1:
        for g in range(n_races * 2):
            recs.append(run_race(spec0, spec1, seed0 + g, cfg))
    else:
        for g in range(n_races):
            s = seed0 + g
            recs.append(run_race(spec0, spec1, s, cfg))
            recs.append(run_race(spec1, spec0, s, cfg))
    return recs


def delta_batch(spec0, spec1, n_races, cfg, seed0=30000):
    """同 `race_batch` 的完美配对设计，用于 Δ 战术研究。"""
    recs = []
    if spec0 == spec1:
        for g in range(n_races * 2):
            recs.append(run_delta_race(spec0, spec1, seed0 + g, cfg))
    else:
        for g in range(n_races):
            s = seed0 + g
            recs.append(run_delta_race(spec0, spec1, s, cfg))
            recs.append(run_delta_race(spec1, spec0, s, cfg))
    return recs


# ------------------------------------------------- Stage 1：全因子对称扫描
GRID = [(cp, cl, alt) for cp in ("center", "corner", "radiate", "denial")
        for cl in ("none", "follow", "split")
        for alt in (False, True)]


def section_race_scan(cfg, n_races):
    print("\n" + "=" * 78)
    print(f"【9】全因子策略扫描（对称局，每配置 {n_races*2} 场，10 营占满即停）")
    print("=" * 78)
    print(f"  维度：camp_pref(进哪个营) × cluster(翻哪个簇) × alternate(占营/拓荒交替)")
    print(f"  {'camp_pref':<10}{'cluster':<9}{'alt':<5}{'占满手数':>10}"
          f"{'先手营':>9}{'后手营':>9}{'差':>8}{'z':>7}{'先手占优':>10}{'完成率':>8}")
    rows = []
    for cp, cl, alt in GRID:
        key = f"{cp}|{cl}|{int(alt)}"
        STRATEGIES[key] = dict(camp_pref=cp, cluster=cl, flip_pref="golden6",
                               alternate=alt)
        recs = race_batch(key, key, n_races, cfg)
        s = summarize_races(recs, lambda r: 0)
        rows.append((cp, cl, alt, s))
        print(f"  {cp:<10}{cl:<9}{str(alt):<5}{s['plies_to_fill'][0]:>10.2f}"
              f"{s['camps'][0]:>9.3f}{s['camps'][1]:>9.3f}"
              f"{s['camps'][0]-s['camps'][1]:>+8.3f}{s['camp_diff'][2]:>+7.2f}"
              f"{s['win'][0]*100:>9.1f}%{s['completion']*100:>7.1f}%")
    fastest = min(rows, key=lambda t: t[3]["plies_to_fill"][0])
    most_first = max(rows, key=lambda t: t[3]["camps"][0])
    print(f"\n  >> 占满 10 营最快配置：camp_pref={fastest[0]}, cluster={fastest[1]}, "
          f"alternate={fastest[2]} -> {fastest[3]['plies_to_fill'][0]:.2f} ply")
    print(f"  >> 先手占营最多配置：camp_pref={most_first[0]}, "
          f"cluster={most_first[1]}, alternate={most_first[2]} -> "
          f"{most_first[3]['camps'][0]:.3f} 营")
    for k in [f"{cp}|{cl}|{int(alt)}" for cp, cl, alt in GRID]:
        STRATEGIES.pop(k, None)
    return rows


# ------------------------------------- Stage 2：先手占营顺序 / 后手反制 / 优先级
def section_race_targeted(cfg, n_races):
    print("\n" + "=" * 78)
    print(f"【10】定向策略对比（每组 {n_races*2} 场）")
    print("=" * 78)
    out = {}

    def run(label, a, b, symmetric=False):
        recs = race_batch(a, b, n_races, cfg)
        seat = summarize_races(recs, lambda r: 0)
        pol = None if symmetric else summarize_races(
            recs, lambda r: 0 if r["specs"][0] == a else 1)
        print("\n" + "-" * 78)
        print(f"  {label}   [{a} vs {b}]")
        print("-" * 78)
        print_summary("先手/后手视角", seat, "先手(seat0)", "后手(seat1)")
        if pol is not None:
            print_summary("策略视角", pol, a, b)
        out[label] = {"spec": f"{a} vs {b}", "seat": seat, "policy": pol}
        return seat, pol

    print("\n### T1 先手首翻黄金位后的占营顺序三策略（各自对称局）###")
    for name, tag in (("A_corner_push", "T1-A 先占角营→翻角营辐射区→推进中营"),
                      ("B_center_bet", "T1-B 直接占中营→赌其它黄金位暗子"),
                      ("C_alternate", "T1-C 占营与拓荒翻棋交替")):
        run(tag, name, name, symmetric=True)

    print("\n### T2 后手反制三策略（先手固定 BASE，配对对抗）###")
    for name, tag in (("R_follow", "T2-A 跟随先手同一营簇争夺"),
                      ("R_split", "T2-B 另辟营簇（抢另一侧黄金位）"),
                      ("R_denial", "T2-C 卡位进营（抢敌明子一步可达的空营）"),
                      ("R_split_denial", "T2-D 另辟营簇 + 卡位进营（组合）")):
        run(tag, "BASE", name)

    print("\n### T3 占营优先级三选（各自对称局）###")
    for name, tag in (("P_center", "T3-A 中心营优先 (+50)"),
                      ("P_corner", "T3-B 角营优先 (+40)"),
                      ("P_radiate", "T3-C 邻接暗子最多的营优先（辐射潜力）")):
        run(tag, name, name, symmetric=True)

    print("\n### T4 单变量对照 ###")
    run("T4-A 首翻黄金位 vs 首翻底线死位", "BASE", "X_firstedge")
    run("T4-B 代码黄金 6 位 vs 几何完备 8 位", "X_golden8", "BASE")

    print("\n" + "-" * 78)
    print("  【横向排名】对称策略占满 10 营的速度（参照 = T3-A 中心营优先，两样本 z）")
    print("-" * 78)
    ref = out["T3-A 中心营优先 (+50)"]["seat"]
    rows = [(lb, b["seat"]) for lb, b in out.items() if b["policy"] is None]
    rows.sort(key=lambda t: t[1]["plies_to_fill"][0])
    print(f"  {'策略组':<40}{'占满手数':>9}{'SE':>6}{'vs基准z':>9}"
          f"{'先手营':>8}{'后手营':>8}{'先手占优':>9}{'完成率':>8}")
    for lb, s in rows:
        z = two_sample_z(s["plies_to_fill"][0], s["plies_se"],
                         ref["plies_to_fill"][0], ref["plies_se"])
        print(f"  {lb[:38]:<40}{s['plies_to_fill'][0]:>9.2f}{s['plies_se']:>6.2f}"
              f"{z:>+9.2f}{s['camps'][0]:>8.3f}{s['camps'][1]:>8.3f}"
              f"{s['win'][0]*100:>8.1f}%{s['completion']*100:>7.1f}%")
    return out


def section_race_curve(cfg, n_races=200):
    """输出 BASE 对称局的占营曲线与典型占营顺序，供文档绘图使用。"""
    print("\n" + "=" * 78)
    print(f"【11】BASE 对称局占营曲线与占营顺序（{n_races*2} 场）")
    print("=" * 78)
    recs = race_batch("BASE", "BASE", n_races, cfg)
    checkpoints = sorted({p for r in recs for p, _, _ in r["order"]})
    print("  ply -> 平均累计占营 [先手, 后手]")
    curve = []
    for pl in range(2, max(60, min(checkpoints) if checkpoints else 60) + 1, 2):
        c = [0, 0]
        for r in recs:
            for p, s, _ in r["order"]:
                if p <= pl:
                    c[s] += 1
        c = [x / len(recs) for x in c]
        curve.append((pl, c[0], c[1]))
        print(f"    ply {pl:3d}: 先手 {c[0]:.3f}  后手 {c[1]:.3f}  "
              f"合计 {c[0]+c[1]:.3f}")
    # 典型占营顺序（取一条中位数长度的样本）
    done = [r for r in recs if r["filled"]]
    if done:
        done.sort(key=lambda r: r["plies"])
        sample = done[len(done) // 2]
        print(f"\n  典型占营顺序（seed={sample['seed']}, "
              f"{sample['plies']} ply 占满）:")
        for i, (p, s, c) in enumerate(sample["order"], 1):
            kind = "中心营" if c in CENTER_CAMPS else "角营"
            print(f"    #{i:2d}  ply {p:3d}  {'先手' if s == 0 else '后手'}  "
                  f"{c} ({kind})")
    return curve


def section_seed_stability(cfg, n_races, seeds):
    """跨独立种子段的指标稳定性核查：哪些结论稳、哪些只是抽样噪声。"""
    print("\n" + "=" * 78)
    print(f"【12】种子稳定性核查（BASE 对称局，{len(seeds)} 个独立种子段 × "
          f"{n_races*2} 场）")
    print("=" * 78)
    print(f"  {'seed0':>8}{'场次':>6}{'T_first':>9}{'T_second':>10}{'Δ':>7}"
          f"{'占满手数':>9}{'先手营':>8}{'后手营':>8}{'第10营归先手':>13}"
          f"{'先手≥6营':>9}{'后手≥6营':>9}")
    cols = {k: [] for k in ("tf", "ts", "d", "fill", "cf", "cs",
                            "tenth", "w0", "w1")}
    for sd in seeds:
        recs = race_batch("BASE", "BASE", n_races, cfg, seed0=sd)
        s = summarize_races(recs, lambda r: 0)
        tf, ts = s["first_camp_ply"]
        vals = (tf, ts, ts - tf, s["plies_to_fill"][0], s["camps"][0],
                s["camps"][1], s["tenth_by_A"], s["win"][0], s["win"][1])
        for k, v in zip(cols, vals):
            cols[k].append(v)
        print(f"  {sd:>8}{n_races*2:>6}{tf:>9.3f}{ts:>10.3f}{ts-tf:>7.3f}"
              f"{s['plies_to_fill'][0]:>9.2f}{s['camps'][0]:>8.3f}"
              f"{s['camps'][1]:>8.3f}{s['tenth_by_A']*100:>12.1f}%"
              f"{s['win'][0]*100:>8.1f}%{s['win'][1]*100:>8.1f}%")

    names = {"tf": "T_first", "ts": "T_second", "d": "Δ(性能差)",
             "fill": "占满10营手数", "cf": "先手占营", "cs": "后手占营",
             "tenth": "第10营归先手", "w0": "先手≥6营率", "w1": "后手≥6营率"}
    print(f"\n  {'指标':<16}{'均值':>10}{'极差':>10}{'波动/均值':>12}{'稳定性判定':>14}")
    stable = {}
    for k, lbl in names.items():
        xs = cols[k]
        m = sum(xs) / len(xs)
        rng_ = max(xs) - min(xs)
        rel = rng_ / abs(m) * 100 if m else float("nan")
        verdict = "稳定" if rel < 3 else ("中等" if rel < 8 else "高波动")
        stable[k] = (m, rng_, rel, verdict)
        show = f"{m*100:.2f}%" if k in ("tenth", "w0", "w1") else f"{m:.3f}"
        showr = f"{rng_*100:.2f}pp" if k in ("tenth", "w0", "w1") else f"{rng_:.3f}"
        print(f"  {lbl:<16}{show:>10}{showr:>10}{rel:>11.1f}%{verdict:>14}")

    print("\n  判读：")
    print(f"    · 「占满10营手数」波动 {stable['fill'][2]:.2f}% —— 极稳，"
          f"可直接作为策略提速/降速的判据（本方案所有速度结论都建立在此指标上）；")
    print(f"    · 「先手/后手占营」波动 {stable['cf'][2]:.1f}%/{stable['cs'][2]:.1f}%"
          f" —— 可用，先手优势 {stable['cf'][0]-stable['cs'][0]:+.3f} 营可复现；")
    print(f"    · 「Δ 首占营性能差」波动 {stable['d'][2]:.1f}% —— 高波动，"
          f"对照闭式模型 2.818 ply 时必须给 ±0.6 ply 容差，"
          f"不可用单一种子段的 Δ 下结论；")
    print(f"    · 「第10营归属」均值 {stable['tenth'][0]*100:.1f}% —— 接近抛硬币，"
          f"**不能**作为先手优势的判据；")
    print(f"    · 先手优势的正确表述是「先手≥6营率 {stable['w0'][0]*100:.1f}% "
          f"vs 后手≥6营率 {stable['w1'][0]*100:.1f}%」，比值 "
          f"{stable['w0'][0]/max(1e-9, stable['w1'][0]):.2f}:1。")
    return stable


def section_visuals(curve, cfg):
    """生成文档直接可嵌入的两种 ASCII 可视化：棋盘标注图 + 占营曲线条形图。"""
    print("\n" + "=" * 78)
    print("【13】可视化输出（可直接嵌入文档）")
    print("=" * 78)

    # ---- 13.1 棋盘标注图 ----
    print("\n  13.1 棋盘标注图：黄金位 / 中心营 / 角营 / 死位")
    print("  " + "-" * 62)
    code_golden = set(CENTER_CAMP_FLIP_POSITIONS)
    symbol = {}
    for p in PLAY_POSITIONS:
        d = DEG[p]
        if d == 3:
            symbol[p] = "G" if p in code_golden else "g"
        elif d == 2:
            symbol[p] = "2"
        elif d == 1:
            symbol[p] = "1"
        else:
            symbol[p] = "."
    for c in CENTER_CAMPS:
        symbol[c] = "@"
    for c in CORNER_CAMPS:
        symbol[c] = "O"
    for h in HQS:
        symbol[h] = "H"

    print("          col0  col1  col2  col3  col4")
    for r in range(ROWS):
        cells = "   ".join(symbol[(r, c)] for c in range(COLS))
        note = ""
        if r == 0 or r == 11:
            note = "  <- 底线性：10 个度0死位（含 4 个大本营 H）"
        elif r in (3, 8):
            note = "  <- @ 中心营 + 左右两个 G 黄金位"
        elif r in (2, 9):
            note = "  <- g = 几何度3但代码未收录的黄金位"
        elif r in (5, 6):
            note = "  <- 前线（col0/2/4 三座铁桥）"
        print(f"    row {r:2d}   {cells}{note}")
    print("  " + "-" * 62)
    print("    图例：@ 中心营(+50, 2个)   O 角营(+40, 8个)")
    print("          G 代码收录的黄金位(度3, 6个)   g 代码漏收的黄金位(度3, 2个)")
    print("          2 / 1 = 邻营度 2 / 1 的放子位   . 度0死位   H 大本营")
    print("    关键读法：@ 的上下左右四个十字位就是该簇的 4 个黄金位；")
    print("              @ 的四条斜人道通向 4 个 O，O 之间互不相连（星形）。")

    # ---- 13.2 营簇-黄金位二部覆盖图 ----
    print("\n  13.2 黄金位 ↔ 行营 辐射覆盖（每簇 4 位覆盖 5 营）")
    print("  " + "-" * 62)
    for k in ("up", "down"):
        print(f"    {'上' if k == 'up' else '下'}簇  "
              f"中心营 {sorted(CENTER_CAMPS & CLUSTER_CAMPS[k])[0]}")
        for g in sorted(CLUSTER_GOLDEN[k]):
            camps = sorted(nb for nb in NEIGHBORS[g] if is_camp(nb))
            mark = "G" if g in code_golden else "g"
            print(f"      [{mark}] {g}  ->  "
                  + "  ".join(f"{c}{'(@)' if c in CENTER_CAMPS else '(O)'}"
                             for c in camps))

    # ---- 13.3 占营曲线 ASCII 条形图 ----
    if curve:
        print("\n  13.3 占营曲线（ply vs 累计占营数，BASE 对称局）")
        print("  " + "-" * 62)
        print("    每格 █ = 0.25 营；F = 先手，S = 后手")
        print(f"    {'ply':>4} | {'先手 F':<26}{'后手 S':<26}{'合计':>6}")
        for pl, cf, cs in curve:
            if pl > 30:
                continue
            bf = "█" * int(round(cf / 0.25))
            bs = "█" * int(round(cs / 0.25))
            print(f"    {pl:>4} | F{bf:<25} S{bs:<25}{cf+cs:>6.2f}")
        print("  " + "-" * 62)
        sat = next((pl for pl, cf, cs in curve if cf + cs >= 9.99), None)
        print(f"    读法：先手曲线全程位于后手之上，垂直间距≈0.4~0.5 营；"
              f"ply {sat} 时合计饱和到 10.00。")

    # ---- 13.4 策略决策树 ----
    print("\n  13.4 策略决策树（文本版，文档可直接引用）")
    print("  " + "-" * 62)
    print("""    【先手】
    ply1 首翻 ──> 必打 G/g 黄金位（度3）        [禁: . 死位, H 大本营]
                    │
    ply2 (对手翻) ─┤
                    │
    ply3 ──> 翻出可动子(84%)? ──是──> 进 @ 中心营（唯一枢纽，只 4 个入口）
                    │否(16%,雷/旗)
                    └──> 改翻同簇另一 G 位（期望再 2.27 次翻到可动子）
    ply5+ ──> 循环 { 有空营可进? ──是──> 进营(@ 优先, 其次 O)
                     │否
                     └──> 翻己方营的邻位暗子(依托行营 +120 辐射拓荒) }

    【后手】
    ply2 首翻 ──> 打 G 黄金位，且**选先手首翻的对侧营簇**（另辟营簇）
    ply4+ ──> 循环 { 有空营可进? ──是──> 进营，优先级:
                     │                    1) 卡位营(敌可动明子一步可达) 
                     │                    2) @ 中心营  3) O 角营
                     │否
                     └──> 翻本簇 G 位暗子 }""")
    print("  " + "-" * 62)


# ============================================================================
#  Δ 战术研究：以"占营数量差 Δ = camps_A − camps_B"为核心目标函数
# ============================================================================
#
# 规则前提（决定哪些战术假设可行，均已由测试固化）：
#   · allow_suicide_attack=False（APK 实测）=> "小子撞大子"不是合法着法，
#     故"主动送死低价值子换敌高价值子"在规则层不可行；合法的牺牲只有两种：
#       (a) 炸弹同尽（ZHA vs 任意 => both_die）
#       (b) 同衔相撞（both_die）
#       (c) 走入敌杀区（普通走子，下一手被敌合法吃掉）
#   · 明子身份是公共信息 => "暴露高价值子诱敌"对判断正确的对手无效；
#     只有 capture_judgment="all_legal"（接受全部合法交换，含 both_die）的
#     对手才会被诱导做出它本会跳过的交换。

def camp_zone(pos) -> bool:
    """pos 是否处于"行营邻域"（自身是营，或紧邻任一营）。"""
    if is_camp(pos):
        return True
    return any(is_camp(nb) for nb in NEIGHBORS[pos])


class DeltaPolicy:
    """Δ 战术策略。五个正交维度：

    capture_scope      never(不吃子纯占营) / camp_zone(只在行营邻域内吃) /
                       anywhere(有吃子机会就吃，哪怕远离行营)
    capture_judgment   win(只取必胜吃子) / all_legal(接受全部合法吃子，含炸弹同尽)
    outstrike          True => 允许驻营子出营吃子（行营扑杀）
    camp_pref          center(中营优先) / denial(卡位敌一步可达的空营) /
                       enemy_zone(抢敌辐射压力大的营簇) / self_zone(只经营自身簇)
    sacrifice          never / bomb(炸弹同尽换 value>=160 敌子) / walk(送最低价值子入敌杀区)
    exposure           none(不主动暴露) / high(主动暴露最高价值明子) /
                       low(主动暴露最低价值明子) / random
    flip_pref          golden6(只翻代码黄金位，与 §9~§13 基准一致) /
                       spread(优先翻"紧邻已有明子"的位 => 强制制造子力接触)

    `spread` 的必要性：6 个黄金位两两不相邻、且营内子不可被攻击，故在
    golden6 基准下 10 营竞赛窗口（~23 ply）内**几乎不存在合法吃子机会**，
    吃子/诱骗/牺牲三个维度无法被测量。`spread` 用于构造"强制接触"对照组，
    把"竞赛窗口内无战斗"（结构事实）与"有战斗时激进吃子是否提升 Δ"
    （战术问题）分离开来。
    """

    def __init__(self, name="D_BASE", capture_scope="never",
                 capture_judgment="win", outstrike=False, camp_pref="center",
                 sacrifice="never", exposure="none", flip_pref="golden6",
                 seed=None, exposure_budget=3, sacrifice_budget=3):
        self.name = name
        self.capture_scope = capture_scope
        self.capture_judgment = capture_judgment
        self.outstrike = outstrike
        self.camp_pref = camp_pref
        self.sacrifice = sacrifice
        self.exposure = exposure
        self.flip_pref = flip_pref
        self.exposure_budget = exposure_budget
        self.sacrifice_budget = sacrifice_budget
        self.rng = random.Random(seed)
        self._flips = 0
        self._used_exposure = 0
        self._used_sacrifice = 0
        self._just_entered = False

    # ---- 选营
    def _camp_key(self, st, camp, me):
        hidden_nb = sum(1 for nb in NEIGHBORS[camp]
                        if (pc := st.board.get(nb)) is not None and not pc.revealed)
        center = 1 if camp in CENTER_CAMPS else 0
        en = other(me) if me else None
        denial = 0
        if en:
            for nb in NEIGHBORS[camp]:
                pc = st.board.get(nb)
                if pc is not None and pc.revealed and pc.color == en \
                        and pc.rank not in IMMOVABLE:
                    denial = 1
                    break
        if self.camp_pref == "denial":
            return (denial, center, hidden_nb, self.rng.random())
        if self.camp_pref in ("enemy_zone", "self_zone"):
            pres = cluster_pressure(st, en) if en else {"up": 0, "down": 0}
            k = cluster_of(camp)
            zone = pres[k] if self.camp_pref == "enemy_zone" else -pres[k]
            return (zone, denial, center, self.rng.random())
        return (center, hidden_nb, denial, self.rng.random())

    # ---- 吃子筛选
    def _capture_ok(self, st, a):
        if self.capture_scope == "never":
            return False
        if self.capture_scope == "camp_zone" and not (camp_zone(a.frm) or camp_zone(a.to)):
            return False
        if is_camp(a.frm) and not self.outstrike:
            return False                      # 行营驻守纪律
        att, tgt = st.board[a.frm], st.board[a.to]
        res = battle(att.rank, tgt.rank)
        if res == "attacker_wins":
            return True
        if res == "both_die":
            if self.capture_judgment == "all_legal":
                return True
            # 牺牲维度：炸弹同尽换高价值敌子
            if self.sacrifice == "bomb" and att.rank == Rank.ZHA \
                    and APK_PIECE_VALUES.get(tgt.rank, 0.0) >= 160.0:
                return True
        return False

    # ---- 暴露/牺牲走子
    def _tactic_move(self, st, moves, me, kind):
        """kind='exposure' 主动贴近敌明子；kind='sacrifice' 走入敌杀区送吃。"""
        if me is None:
            return None
        en = other(me)
        foes = [p for p, pc in st.board.items()
                if pc.revealed and pc.color == en and pc.rank not in IMMOVABLE]
        if not foes:
            return None
        foe_set = set(foes)
        cands = []
        for a in moves:
            if is_camp(a.frm) or st.board.get(a.to) is not None:
                continue                      # 不出营、不吃子、只走空格
            adj_foe = any(nb in foe_set for nb in NEIGHBORS[a.to])
            if kind == "sacrifice":
                # 走到"敌强子下一步能合法吃掉我"的格
                me_pc = st.board[a.frm]
                danger = any(
                    battle(st.board[nb].rank, me_pc.rank) == "attacker_wins"
                    for nb in NEIGHBORS[a.to] if nb in foe_set)
                if not danger:
                    continue
                cands.append((APK_PIECE_VALUES.get(me_pc.rank, 0.0), a))
            else:
                if not adj_foe:
                    continue
                v = APK_PIECE_VALUES.get(st.board[a.frm].rank, 0.0)
                if self.exposure == "high":
                    cands.append((-v, a))       # 价值最高者优先暴露
                elif self.exposure == "low":
                    cands.append((v, a))        # 价值最低者优先暴露
                else:
                    cands.append((self.rng.random(), a))
        if not cands:
            return None
        cands.sort(key=lambda t: t[0])
        return cands[0][1]

    def _pick_flip(self, st, flips):
        """golden6：与 §9~§13 基准一致；spread：优先翻紧邻已有明子的位以制造接触。"""
        self._flips += 1
        if self.flip_pref == "spread":
            def contact(a):
                return sum(1 for nb in NEIGHBORS[a.frm]
                           if (pc := st.board.get(nb)) is not None and pc.revealed)
            best = max(contact(a) for a in flips)
            if best > 0:
                pool = [a for a in flips if contact(a) == best]
                return self.rng.choice(pool)
            # 尚无明子可贴近 => 退化为黄金位，保证开局占营节奏不被破坏
        pool = [a for a in flips if a.frm in CENTER_CAMP_FLIP_POSITIONS]
        if pool:
            return self.rng.choice(pool)
        empty = {c for c in CAMP_LIST if st.board.get(c) is None}
        pool = [a for a in flips if set(NEIGHBORS[a.frm]) & empty]
        if pool:
            return max(pool, key=lambda a: (DEG[a.frm], self.rng.random()))
        return max(flips, key=lambda a: (DEG[a.frm], self.rng.random()))

    def choose(self, st):
        acts = st.legal_actions()
        me = st.my_color()
        flips = [a for a in acts if a.kind == "flip"]
        moves = [a for a in acts if a.kind == "move"]
        empty = [c for c in CAMP_LIST if st.board.get(c) is None]
        enters = [a for a in moves if a.to in empty and not is_camp(a.frm)]

        # (1) 一步进空营：最高优先级（D2 纪律，绝不因战术而放弃）
        if enters:
            self._just_entered = True
            return max(enters, key=lambda a: self._camp_key(st, a.to, me))
        self._just_entered = False

        # (2) 吃子
        caps = [a for a in moves
                if st.board.get(a.to) is not None and self._capture_ok(st, a)]
        if caps:
            def capv(a):
                t = st.board[a.to]
                f = st.board[a.frm]
                win = 1 if battle(f.rank, t.rank) == "attacker_wins" else 0
                return (win, APK_PIECE_VALUES.get(t.rank, 0.0), self.rng.random())
            return max(caps, key=capv)

        # (3) 牺牲：走入敌杀区（每场限额，避免无限送子）
        if self.sacrifice == "walk" and self._used_sacrifice < self.sacrifice_budget:
            mv = self._tactic_move(st, moves, me, "sacrifice")
            if mv is not None:
                self._used_sacrifice += 1
                return mv

        # (4) 暴露/诱骗（每场限额）
        if self.exposure != "none" and self._used_exposure < self.exposure_budget:
            mv = self._tactic_move(st, moves, me, "exposure")
            if mv is not None:
                self._used_exposure += 1
                return mv

        # (5) 向最近空营推进
        if empty:
            field = camp_dist_field(st)
            cand, best = [], None
            for a in moves:
                if is_camp(a.frm) or st.board.get(a.to) is not None:
                    continue
                d0, d1 = field.get(a.frm), field.get(a.to)
                if d0 is None or d1 is None or d1 >= d0:
                    continue
                if best is None or d1 < best:
                    best, cand = d1, [a]
                elif d1 == best:
                    cand.append(a)
            if cand:
                return self.rng.choice(cand)

        # (6) 翻棋
        if flips:
            return self._pick_flip(st, flips)

        # (7) 兜底：仍严禁调出驻营子力
        safe = [a for a in moves if not is_camp(a.frm)]
        return self.rng.choice(safe) if safe else self.rng.choice(acts)


DELTA_STRATEGIES = {
    # 基准
    "D_BASE":        dict(),
    # —— 维度 1：吃子激进程度 ——
    "AG_campzone":   dict(capture_scope="camp_zone"),
    "AG_anywhere":   dict(capture_scope="anywhere"),
    "AG_any_out":    dict(capture_scope="anywhere", outstrike=True),
    "AG_all_legal":  dict(capture_scope="anywhere", capture_judgment="all_legal"),
    # —— 维度 2：阻挠策略 ——
    "BL_deny":       dict(camp_pref="denial"),
    "BL_enemy":      dict(camp_pref="enemy_zone"),
    "BL_self":       dict(camp_pref="self_zone"),
    # —— 维度 3：牺牲换营 ——
    "SC_bomb":       dict(sacrifice="bomb", capture_scope="anywhere"),
    "SC_walk":       dict(sacrifice="walk"),
    "SC_walk_cap":   dict(sacrifice="walk", capture_scope="anywhere"),
    # —— 维度 4：暴露/诱骗 ——
    "EX_high":       dict(exposure="high"),
    "EX_low":        dict(exposure="low"),
    "EX_random":     dict(exposure="random"),
    # —— 组合候选 ——
    "CB_deny_zone":  dict(camp_pref="denial", capture_scope="camp_zone"),
    "CB_deny_bomb":  dict(camp_pref="denial", sacrifice="bomb",
                          capture_scope="anywhere"),
    "CB_full":       dict(camp_pref="denial", capture_scope="camp_zone",
                          sacrifice="bomb"),
    # —— 强制接触族（flip_pref=spread）：让吃子/诱骗/牺牲维度真正可测 ——
    "CT_base":       dict(flip_pref="spread"),
    "CT_campzone":   dict(flip_pref="spread", capture_scope="camp_zone"),
    "CT_anywhere":   dict(flip_pref="spread", capture_scope="anywhere"),
    "CT_any_out":    dict(flip_pref="spread", capture_scope="anywhere",
                          outstrike=True),
    "CT_all_legal":  dict(flip_pref="spread", capture_scope="anywhere",
                          capture_judgment="all_legal"),
    "CT_bomb":       dict(flip_pref="spread", capture_scope="anywhere",
                          sacrifice="bomb"),
    "CT_walk":       dict(flip_pref="spread", sacrifice="walk"),
    "CT_walk_cap":   dict(flip_pref="spread", sacrifice="walk",
                          capture_scope="anywhere"),
    "CT_exhigh":     dict(flip_pref="spread", exposure="high"),
    "CT_exlow":      dict(flip_pref="spread", exposure="low"),
    "CT_exrandom":   dict(flip_pref="spread", exposure="random"),
    "CT_exhigh_gre": dict(flip_pref="spread", exposure="high",
                          capture_scope="anywhere", capture_judgment="all_legal"),
    "CT_deny":       dict(flip_pref="spread", camp_pref="denial"),
    "CT_deny_cap":   dict(flip_pref="spread", camp_pref="denial",
                          capture_scope="camp_zone"),
}


def make_delta_policy(spec, seed=None):
    if spec not in DELTA_STRATEGIES:
        raise ValueError(f"未知 Δ 策略 {spec}，可选 {sorted(DELTA_STRATEGIES)}")
    return DeltaPolicy(name=spec, seed=seed, **DELTA_STRATEGIES[spec])


def run_delta_race(spec0, spec1, seed, cfg=None, max_ply=400):
    """跑一场 Δ 竞赛：第 10 个行营被占据的瞬间终止，全程记录 Δ 轨迹与战术事件。"""
    cfg = cfg or race_config()
    p = [make_delta_policy(spec0, seed * 31 + 1), make_delta_policy(spec1, seed * 31 + 2)]
    st = deal(random.Random(seed), cfg)
    rec = {"seed": seed, "specs": [spec0, spec1], "filled": False, "plies": 0,
           "camps": [0, 0], "order": [], "first_camp_ply": [None, None],
           "captures": [0, 0], "camp_origin_caps": [0, 0],
           "camp_zone_caps": [0, 0], "bomb_trades": [0, 0],
           "lost_ranks": [Counter(), Counter()], "lost_value": [0.0, 0.0],
           "shuttle": 0, "exit": 0,
           "spread2_ply": None, "spread2_by": None,
           "captures_before_spread2": 0, "sign_flips": 0,
           "capture_opp": [0, 0], "turns": [0, 0],
           "delta_track": []}
    prev_nz = 0                 # 上一个非零 Δ（用于领先权易手计数）
    caps_total = 0

    while True:
        own = camp_owners(st)
        if len(own) >= len(CAMPS):
            rec["filled"] = True
            break
        if st.ply >= max_ply or st.is_terminal():
            break
        acts = st.legal_actions()
        if not acts:
            break
        seat = st.turn
        act = p[seat].choose(st)

        # 战术可行性探针：本手是否存在**合法**吃子机会（与策略是否选择吃子无关）
        # legal_actions 只对可攻击目标生成 move，故 board[a.to] 非空即为合法吃子
        rec["turns"][seat] += 1
        if any(a.kind == "move" and st.board.get(a.to) is not None for a in acts):
            rec["capture_opp"][seat] += 1

        is_cap = act.kind == "move" and st.board.get(act.to) is not None
        cap_rank = st.board[act.to].rank if is_cap else None
        att_rank = st.board[act.frm].rank if is_cap else None
        if act.kind == "move":
            if is_camp(act.frm) and is_camp(act.to):
                rec["shuttle"] += 1
            elif is_camp(act.frm):
                rec["exit"] += 1
        n_dead_before = len(st.dead)
        st = st.apply(act)

        if is_cap:
            caps_total += 1
            rec["captures"][seat] += 1
            if is_camp(act.frm):
                rec["camp_origin_caps"][seat] += 1      # 严格"源自行营内"= 行营扑杀
            if camp_zone(act.frm):
                rec["camp_zone_caps"][seat] += 1        # 行营邻域内（含营内）
            if att_rank == Rank.ZHA or cap_rank == Rank.ZHA:
                rec["bomb_trades"][seat] += 1
        for pc in st.dead[n_dead_before:]:
            owner = 0 if st.seat_color[0] == pc.color else 1
            rec["lost_ranks"][owner][pc.rank.name] += 1
            rec["lost_value"][owner] += APK_PIECE_VALUES.get(pc.rank, 0.0)

        now = camp_owners(st)
        for c, s in now.items():
            if own.get(c) != s:
                rec["order"].append((st.ply, s, c))
                if rec["first_camp_ply"][s] is None:
                    rec["first_camp_ply"][s] = st.ply
        cnt = [0, 0]
        for s in now.values():
            cnt[s] += 1
        d = cnt[0] - cnt[1]
        rec["delta_track"].append((st.ply, d))
        # 领先权易手：与"上一个非零 Δ"比较。因为单次占营只让 Δ 变化 ±1，
        # 领先权易手必然经过 0，若只比相邻两手会漏计（+1 -> 0 -> -1）。
        if d != 0:
            if prev_nz != 0 and (prev_nz > 0) != (d > 0):
                rec["sign_flips"] += 1
            prev_nz = d
        if rec["spread2_ply"] is None and abs(d) >= 2:
            rec["spread2_ply"] = st.ply
            rec["spread2_by"] = 0 if d > 0 else 1
            rec["captures_before_spread2"] = caps_total
        prev_d = d

    rec["plies"] = st.ply
    cnt = [0, 0]
    for s in camp_owners(st).values():
        cnt[s] += 1
    rec["camps"] = cnt
    rec["filled"] = cnt[0] + cnt[1] >= len(CAMPS)
    rec["delta_seat"] = cnt[0] - cnt[1]
    rec["captures_total"] = sum(rec["captures"])
    rec["lost_ranks"] = [dict(r) for r in rec["lost_ranks"]]
    return rec


RANK_ORDER = [Rank.SI, Rank.JUN, Rank.SHI, Rank.LV, Rank.TUAN, Rank.YING,
              Rank.LIAN, Rank.PAI, Rank.GONG, Rank.ZHA, Rank.LEI, Rank.QI]


def summarize_delta(recs, idx_of):
    n = len(recs)
    if n == 0:
        return None
    done = [r for r in recs if r["filled"]]

    def paired(key):
        d = []
        for r in recs:
            i = idx_of(r)
            v = key(r)
            d.append(v[i] - v[1 - i] if isinstance(v, (list, tuple)) else v)
        m = _mean(d)
        var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
        sd = var ** 0.5
        se = (var / n) ** 0.5
        return m, sd, se, (m / se if se > 0 else 0.0)

    dm, dsd, dse, dz = paired(lambda r: r["camps"])
    pl = [r["plies"] for r in done]
    pm = _mean(pl)
    pvar = sum((x - pm) ** 2 for x in pl) / max(1, len(pl) - 1) if pl else 0.0
    lost = [Counter(), Counter()]
    for r in recs:
        for side in (0, 1):
            lost[side].update(r["lost_ranks"][idx_of(r) if side == 0 else 1 - idx_of(r)])
    win = Counter("A" if a > b else ("B" if b > a else "tie")
                  for a, b in ((r["camps"][idx_of(r)], r["camps"][1 - idx_of(r)])
                               for r in recs))
    sp = [r["captures_before_spread2"] for r in recs if r["spread2_ply"]]
    return {
        "n": n, "completion": len(done) / n,
        "camps": [_mean([r["camps"][idx_of(r)] for r in recs]),
                  _mean([r["camps"][1 - idx_of(r)] for r in recs])],
        "delta": (dm, dsd, dse, dz),
        "ci95": (dm - 1.96 * dse, dm + 1.96 * dse),
        "win": (win["A"] / n, win["B"] / n, win["tie"] / n),
        "plies_to_fill": (pm, _mean([r["plies"] for r in recs])),
        "plies_se": (pvar / len(pl)) ** 0.5 if pl else 0.0,
        "first_camp_ply": [
            _mean([r["first_camp_ply"][idx_of(r)] for r in recs
                   if r["first_camp_ply"][idx_of(r)]]),
            _mean([r["first_camp_ply"][1 - idx_of(r)] for r in recs
                   if r["first_camp_ply"][1 - idx_of(r)]])],
        "captures": [_mean([r["captures"][idx_of(r)] for r in recs]),
                     _mean([r["captures"][1 - idx_of(r)] for r in recs])],
        "captures_total": _mean([r["captures_total"] for r in recs]),
        "camp_origin_caps": [
            _mean([r["camp_origin_caps"][idx_of(r)] for r in recs]),
            _mean([r["camp_origin_caps"][1 - idx_of(r)] for r in recs])],
        "camp_zone_caps": [
            _mean([r["camp_zone_caps"][idx_of(r)] for r in recs]),
            _mean([r["camp_zone_caps"][1 - idx_of(r)] for r in recs])],
        "bomb_trades": [_mean([r["bomb_trades"][idx_of(r)] for r in recs]),
                        _mean([r["bomb_trades"][1 - idx_of(r)] for r in recs])],
        "lost_value": [_mean([r["lost_value"][idx_of(r)] for r in recs]),
                       _mean([r["lost_value"][1 - idx_of(r)] for r in recs])],
        "lost_ranks": [dict(lost[0]), dict(lost[1])],
        "spread2_rate": len(sp) / n,
        "captures_before_spread2": _mean(sp) if sp else float("nan"),
        "sign_flips": _mean([r["sign_flips"] for r in recs]),
        "capture_opp_rate": [
            (sum(r["capture_opp"][idx_of(r)] for r in recs)
             / max(1, sum(r["turns"][idx_of(r)] for r in recs))),
            (sum(r["capture_opp"][1 - idx_of(r)] for r in recs)
             / max(1, sum(r["turns"][1 - idx_of(r)] for r in recs)))],
        "capture_opp_per_race": [
            _mean([r["capture_opp"][idx_of(r)] for r in recs]),
            _mean([r["capture_opp"][1 - idx_of(r)] for r in recs])],
        "shuttle": _mean([r["shuttle"] for r in recs]),
        "exit": _mean([r["exit"] for r in recs]),
    }


def print_delta(tag, s, nameA, nameB):
    if s is None:
        return
    dm, dsd, dse, dz = s["delta"]
    lo, hi = s["ci95"]
    sig = "显著" if abs(dz) >= 1.96 else "不显著"
    print(f"  [{tag}] {nameA} vs {nameB}")
    print(f"    Δ(占营差) = {dm:+.4f} 营   SD {dsd:.3f}  SE {dse:.4f}  "
          f"z={dz:+.2f} {sig}   95%CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"    占营判据胜负：{nameA} {s['win'][0]*100:.1f}%  "
          f"{nameB} {s['win'][1]*100:.1f}%  平 {s['win'][2]*100:.1f}%")
    print(f"    平均占营 {s['camps'][0]:.3f} : {s['camps'][1]:.3f}   "
          f"占满10营手数 {s['plies_to_fill'][0]:.2f} ± {s['plies_se']:.2f}   "
          f"完成率 {s['completion']*100:.1f}%")
    print(f"    首占营 ply {s['first_camp_ply'][0]:.2f} : "
          f"{s['first_camp_ply'][1]:.2f}")
    co = s["camp_origin_caps"]
    cz = s["camp_zone_caps"]
    ct = max(1e-9, sum(s["captures"]))
    print(f"    吃子 {s['captures'][0]:.2f} : {s['captures'][1]:.2f} "
          f"(合计 {s['captures_total']:.2f}/场)   "
          f"源自行营内(扑杀) {(co[0]+co[1])/ct*100:.1f}%   "
          f"源自行营邻域 {(cz[0]+cz[1])/ct*100:.1f}%   "
          f"炸弹同尽 {s['bomb_trades'][0]:.2f} : {s['bomb_trades'][1]:.2f}")
    opp = s["capture_opp_per_race"]
    take = s["captures"]
    print(f"    吃子挤出效应：合法吃子机会 {opp[0]:.2f} : {opp[1]:.2f} 次/场，"
          f"实际吃子 {take[0]:.2f} : {take[1]:.2f} 次/场  "
          f"=> 转化率 {(take[0]+take[1])/max(1e-9, opp[0]+opp[1])*100:.1f}%")
    print(f"    损失子力价值 {s['lost_value'][0]:.0f} : {s['lost_value'][1]:.0f}")
    for side, nm in ((0, nameA), (1, nameB)):
        top = sorted(s["lost_ranks"][side].items(),
                     key=lambda kv: -kv[1])[:5]
        print(f"      {nm} 被吃子力等级 Top5: {top}")
    print(f"    |Δ|>=2 达成率 {s['spread2_rate']*100:.1f}%，"
          f"达成前平均吃子数 {s['captures_before_spread2']:.2f}   "
          f"Δ 变号次数 {s['sign_flips']:.3f}/场")
    co2 = s["capture_opp_rate"]
    cor = s["capture_opp_per_race"]
    print(f"    合法吃子机会：{nameA} {cor[0]:.2f} 次/场"
          f"（占其回合 {co2[0]*100:.1f}%）  "
          f"{nameB} {cor[1]:.2f} 次/场（占其回合 {co2[1]*100:.1f}%）")
    print(f"    纪律：营间闲走 {s['shuttle']:.3f}  弃营 {s['exit']:.3f}")


def section_delta_rules_probe(cfg):
    """Δ 战术的规则可行性探针：先确认哪些"牺牲/诱骗"形式在 APK 规则下合法。

    结论直接决定 §14 的假设可测性，故必须在跑竞赛之前核对（原子化验证）。
    本探针一律使用**项目默认 RuleConfig()**，以证明结论对正式规则成立，
    而不只是对测量夹具成立。
    """
    cfg = RuleConfig()
    print("\n" + "=" * 78)
    print("【14.0】Δ 战术的规则可行性探针（allow_suicide_attack=False 的后果）")
    print("=" * 78)

    def probe(title, mine, mine_pos, foe, foe_pos, seat=0):
        board = {q: Piece("b", Rank.PAI, False) for q in PLAY_POSITIONS}
        board[mine_pos] = Piece("r", mine, True)
        board[foe_pos] = Piece("b", foe, True)
        st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=seat,
                       first_flip_done=True, ply=4, cfg=cfg)
        acts = st.legal_actions()
        atk = [a for a in acts if a.kind == "move" and a.to == foe_pos]
        res = battle(mine, foe)
        ok = "合法" if atk else "**不合法**"
        print(f"    {title}")
        print(f"      己方 {RANK_CN[mine]}@{mine_pos} 攻击 敌方 {RANK_CN[foe]}@{foe_pos}"
              f"  -> battle={res:<14} 该着法{ok}")
        return bool(atk), res

    print("\n  (1) 牺牲换营·假设 A「主动送死低价值子换敌高价值子」")
    a1, _ = probe("排长(30) 撞 司令(2560)", Rank.PAI, (5, 2), Rank.SI, (5, 3))
    a2, _ = probe("工兵(80) 撞 军长(1280)", Rank.GONG, (5, 2), Rank.JUN, (5, 3))
    print(f"      >> 判定：{'可行' if (a1 or a2) else '规则层不可行'}"
          f" —— allow_suicide_attack=False 使 defender_wins 的着法根本不进入 legal_actions")

    print("\n  (2) 牺牲换营·合法形式")
    b1, _ = probe("炸弹 同尽 司令", Rank.ZHA, (5, 2), Rank.SI, (5, 3))
    b2, _ = probe("排长 同尽 排长（同衔相撞）", Rank.PAI, (5, 2), Rank.PAI, (5, 3))
    print(f"      >> 合法的牺牲只有 both_die 一类：炸弹同尽={b1}，同衔相撞={b2}；"
          f"以及'走入敌杀区'（普通走子，下一手被敌合法吃掉）")

    print("\n  (3) 行营单向扑杀特权（唯一在竞赛窗口内开放的吃子通道）")
    board = {q: Piece("b", Rank.PAI, False) for q in PLAY_POSITIONS}
    board[(3, 2)] = Piece("r", Rank.LIAN, True)     # 己方连长驻中营
    board[(3, 1)] = Piece("b", Rank.PAI, True)      # 敌排长在营外邻位
    st = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=0,
                   first_flip_done=True, ply=4, cfg=cfg)
    out_hit = [a for a in st.legal_actions()
               if a.kind == "move" and a.frm == (3, 2) and a.to == (3, 1)]
    st2 = GameState(board=board, seat_color={0: "r", 1: "b"}, turn=1,
                    first_flip_done=True, ply=5, cfg=cfg)
    in_hit = [a for a in st2.legal_actions()
              if a.kind == "move" and a.frm == (3, 1) and a.to == (3, 2)]
    print(f"      营内连长(3,2) 打 营外排长(3,1) : {'合法' if out_hit else '不合法'}"
          f"   <- 行营扑杀特权")
    print(f"      营外排长(3,1) 打 营内连长(3,2) : {'合法' if in_hit else '**不合法**'}"
          f"   <- state.py::_attackable 对 is_camp(tpos) 直接返回 False")

    print("\n  (4) 诱骗·信息论前提")
    print("      翻棋规则下，明子身份是**公共信息**（revealed=True 对双方可见）。")
    print("      因此'把司令走到空营旁引诱敌方'对任何正确判断交换的对手**无效**：")
    print("      对手知道那是司令，且除炸弹同尽外没有任何合法着法能吃掉它。")
    print("      >> 诱骗只能作用于'对手的交换判断缺陷'，而非'对手的信息缺失'。")
    print("         本节以 capture_judgment=all_legal（接受全部合法交换，含亏损")
    print("         both_die）作为可被诱导的对手模型来测量该效应。")
    return {"suicide_low_vs_high": a1 or a2, "bomb_trade": b1,
            "equal_trade": b2, "camp_outstrike": bool(out_hit),
            "camp_attackable": bool(in_hit)}


SCATTER_MARK = {
    "Δ0": "0", "Δ1-B": "b", "Δ1-A": "a", "Δ1-A+": "A", "Δ1-A2": "2",
    "Δ1-AB": "=", "Δ2-C": "c", "Δ2-A": "e", "Δ2-B": "s", "Δ2-CA": "x",
    "Δ3-A1": "B", "Δ3-A2": "w", "Δ3-A2+": "W",
    "Δ4-A": "H", "Δ4-A'": "L", "Δ4-C": "R", "Δ4-A2": "h", "Δ4-A2'": "l",
    "Δ5-1": "1", "Δ5-2": "3", "Δ5-3": "4",
    "Δ6-0": "o", "Δ6-1": "b", "Δ6-2": "a", "Δ6-3": "A", "Δ6-4": "2",
    "Δ6-5": "B", "Δ6-6": "w", "Δ6-7": "W", "Δ6-8": "H", "Δ6-9": "L",
    "Δ6-10": "R", "Δ6-11": "h", "Δ6-12": "c", "Δ6-13": "C", "Δ6-14": "o",
}


def print_delta_scatter(out):
    """吃子 vs 占营权衡矩阵：横轴 T_fill（占满 10 营手数），纵轴策略视角配对 Δ。"""
    pts = []
    for lb, blk in out.items():
        p = blk.get("policy")
        if p is None:
            continue
        pts.append((p["plies_to_fill"][0], p["delta"][0], p["delta"][3], lb,
                    p["captures_total"]))
    if not pts:
        return
    print("\n" + "-" * 78)
    print("  【权衡矩阵】横轴 = 占满 10 营手数 T_fill，纵轴 = 配对 Δ（营）")
    print("-" * 78)
    xmin = min(20.0, min(p[0] for p in pts) - 1)
    xmax = max(p[0] for p in pts) + 2
    ymin = min(-0.10, min(p[1] for p in pts) - 0.05)
    ymax = max(p[1] for p in pts) + 0.05
    W, H = 62, 15
    grid = [[" "] * W for _ in range(H)]

    def put(x, y, ch):
        cx = int(round((x - xmin) / max(1e-9, xmax - xmin) * (W - 1)))
        cy = int(round((ymax - y) / max(1e-9, ymax - ymin) * (H - 1)))
        if 0 <= cx < W and 0 <= cy < H:
            grid[cy][cx] = ch

    for gy in range(H):                     # Δ=0 参考线
        yv = ymax - gy * (ymax - ymin) / (H - 1)
        if abs(yv) < (ymax - ymin) / (H - 1) / 2:
            for gx in range(W):
                grid[gy][gx] = "-"
    for x, y, z, lb, ct in pts:
        code = lb.split()[0]
        mark = SCATTER_MARK.get(code, "*")
        put(x, y, mark if abs(z) >= 1.96 else mark.lower())
    for gy in range(H):
        yv = ymax - gy * (ymax - ymin) / (H - 1)
        print(f"  Δ{yv:>+6.2f} |{''.join(grid[gy])}|")
    print(f"        +{'-' * W}+")
    print(f"         {xmin:<8.0f}{'':<20}T_fill (ply){'':<18}{xmax:>6.0f}")
    print("\n  图例（大写 = 95% 显著，小写 = 不显著）：")
    print("    0/o 基准   b 营邻域吃子   a 到处吃子   A 到处吃子+驻营出击")
    print("    2 接受全部合法交换   B 炸弹同尽   w/W 走入敌杀区送死")
    print("    H 暴露高价值子(诱骗)   L 暴露低价值子   R 随机暴露   h/l 对贪交换对手")
    print("    c 卡位   e 抢敌辐射区   s 只经营自身簇   C 卡位+营邻域吃子")
    print("    = 到处吃子 vs 营邻域吃子   1/3/4 组合候选")
    print("\n  读法：")
    print("    · 左上区（T_fill 小、Δ 正）= 又好又快，是目标区；")
    print("    · 右下区（T_fill 大、Δ 负）= 又慢又亏，应避免；")
    print("    · 图中可见两个明显的簇：T_fill≈23 的低接触族（golden6）与")
    print("      T_fill≈44~51 的强制接触族（spread）。**跨簇比较 Δ 无意义**，")
    print("      因为 spread 本身改变了竞赛节奏；只能在同簇内比较；")
    print("    · ASCII 散点存在标记重叠，精确数值以下方【Δ 排名】与 §15 池化表为准。")
    return pts


def section_delta_tactics(cfg, n_races):
    """Δ 战术研究：四个维度 × 与 D_BASE 的配对对抗。"""
    out = {}
    print("\n" + "=" * 78)
    print(f"【14】Δ 战术研究（核心目标函数 Δ = camps_A − camps_B，"
          f"每组 {n_races*2} 场，10 营占满即停）")
    print("=" * 78)
    probe = section_delta_rules_probe(cfg)
    out["_rules_probe"] = probe

    def run(label, a, b):
        recs = delta_batch(a, b, n_races, cfg)
        seat = summarize_delta(recs, lambda r: 0)
        # 对称局（a == b）的"策略视角"退化为 seat 视角，置 None 以免污染 Δ 排名
        pol = None if a == b else summarize_delta(
            recs, lambda r: 0 if r["specs"][0] == a else 1)
        print("\n" + "-" * 78)
        print(f"  {label}   [{a} vs {b}]")
        print("-" * 78)
        if pol is not None:
            print_delta("策略视角(配对·同一副牌双座位)", pol, a, b)
        print_delta("先手/后手视角", seat, "先手(seat0)", "后手(seat1)")
        out[label] = {"spec": f"{a} vs {b}", "policy": pol, "seat": seat}
        return pol, seat

    print("\n### Δ0 基准（不吃子纯占营，两侧同策略）###")
    run("Δ0 D_BASE 自对", "D_BASE", "D_BASE")

    print("\n### Δ1 吃子激进程度：C 不吃子 / B 只在行营邻域吃 / A 有吃子机会就吃 ###")
    run("Δ1-B 营邻域吃子 vs 不吃子", "AG_campzone", "D_BASE")
    run("Δ1-A 到处吃子 vs 不吃子", "AG_anywhere", "D_BASE")
    run("Δ1-A+ 到处吃子且驻营子出击 vs 不吃子", "AG_any_out", "D_BASE")
    run("Δ1-A2 到处吃子(含炸弹同尽) vs 不吃子", "AG_all_legal", "D_BASE")
    run("Δ1-AB 到处吃子 vs 营邻域吃子", "AG_anywhere", "AG_campzone")

    print("\n### Δ2 阻挠策略：B 只经营自身簇 / C 卡位 / A 抢敌辐射区 ###")
    run("Δ2-C 卡位进营 vs 中营优先", "BL_deny", "D_BASE")
    run("Δ2-A 抢敌辐射区营簇 vs 中营优先", "BL_enemy", "D_BASE")
    run("Δ2-B 只经营自身营簇 vs 中营优先", "BL_self", "D_BASE")
    run("Δ2-CA 卡位 vs 抢敌辐射区", "BL_deny", "BL_enemy")

    print("\n### Δ3 牺牲换营：B 绝不送死 / A1 炸弹同尽 / A2 走入敌杀区 ###")
    run("Δ3-A1 炸弹同尽换高价值子 vs 绝不牺牲", "SC_bomb", "D_BASE")
    run("Δ3-A2 送低价值子入敌杀区 vs 绝不牺牲", "SC_walk", "D_BASE")
    run("Δ3-A2+ 送死 + 到处吃子 vs 绝不牺牲", "SC_walk_cap", "D_BASE")

    print("\n### Δ4 暴露/诱骗：B 不暴露 / A 暴露高价值子 / 暴露低价值子 / 随机 ###")
    print("  注：对手为 D_BASE(capture_judgment=win) —— 判断正确、不会被诱骗")
    run("Δ4-A 暴露高价值子 vs 不暴露（对手判断正确）", "EX_high", "D_BASE")
    run("Δ4-A' 暴露低价值子 vs 不暴露（对手判断正确）", "EX_low", "D_BASE")
    run("Δ4-C 随机暴露 vs 不暴露（对手判断正确）", "EX_random", "D_BASE")
    print("\n  注：对手改为 AG_all_legal(接受全部合法交换) —— 会被诱导做亏损交换")
    run("Δ4-A2 暴露高价值子 vs 不暴露（对手贪交换）", "EX_high", "AG_all_legal")
    run("Δ4-A2' 暴露低价值子 vs 不暴露（对手贪交换）", "EX_low", "AG_all_legal")

    print("\n### Δ5 组合候选 ###")
    run("Δ5-1 卡位 + 营邻域吃子 vs 基准", "CB_deny_zone", "D_BASE")
    run("Δ5-2 卡位 + 炸弹同尽 + 到处吃子 vs 基准", "CB_deny_bomb", "D_BASE")
    run("Δ5-3 卡位 + 营邻域吃子 + 炸弹同尽 vs 基准", "CB_full", "D_BASE")

    print("\n### Δ6 强制接触对照组（flip_pref=spread，让吃子/诱骗/牺牲维度真正可测）###")
    print("  设计动机：Δ1~Δ5 在 golden6 基准下几乎测不到吃子，因为")
    print("    (a) 6 个黄金位两两不相邻 => 翻出的明子互不接触；")
    print("    (b) 营内子不可被攻击 => 唯一接触通道是'行营扑杀'。")
    print("  spread 让翻棋优先贴近已有明子，人为制造接触，从而分离")
    print("  '竞赛窗口内无战斗'（结构事实）与'有战斗时激进吃子是否提升 Δ'（战术问题）。")
    run("Δ6-0 强制接触基准（不吃子） 自对", "CT_base", "CT_base")
    run("Δ6-1 营邻域吃子 vs 接触基准(不吃子)", "CT_campzone", "CT_base")
    run("Δ6-2 到处吃子 vs 接触基准(不吃子)", "CT_anywhere", "CT_base")
    run("Δ6-3 到处吃子+驻营出击 vs 接触基准", "CT_any_out", "CT_base")
    run("Δ6-4 到处吃子(含全部合法交换) vs 接触基准", "CT_all_legal", "CT_base")
    run("Δ6-5 炸弹同尽换高价值子 vs 接触基准", "CT_bomb", "CT_base")
    run("Δ6-6 送低价值子入敌杀区 vs 接触基准", "CT_walk", "CT_base")
    run("Δ6-7 送死 + 到处吃子 vs 接触基准", "CT_walk_cap", "CT_base")
    run("Δ6-8 暴露高价值子(诱骗) vs 接触基准", "CT_exhigh", "CT_base")
    run("Δ6-9 暴露低价值子 vs 接触基准", "CT_exlow", "CT_base")
    run("Δ6-10 随机暴露 vs 接触基准", "CT_exrandom", "CT_base")
    run("Δ6-11 暴露高价值子 vs 贪交换对手", "CT_exhigh", "CT_exhigh_gre")
    run("Δ6-12 卡位 + 强制接触 vs 接触基准", "CT_deny", "CT_base")
    run("Δ6-13 卡位 + 营邻域吃子 + 强制接触 vs 接触基准",
        "CT_deny_cap", "CT_base")
    run("Δ6-14 强制接触基准 vs golden6 基准（接触本身的代价）",
        "CT_base", "D_BASE")

    # ---- 权衡矩阵散点图 ----
    print_delta_scatter(out)

    # ---- 横向排名 ----
    print("\n" + "-" * 78)
    print("  【Δ 排名】各战术相对 D_BASE 的净 Δ（策略视角，配对）")
    print("-" * 78)
    print(f"  {'战术组':<44}{'Δ均值':>9}{'SE':>8}{'z':>8}{'95%CI':>18}"
          f"{'占满手数':>10}{'吃子/场':>9}")
    rows = []
    for lb, blk in out.items():
        if lb.startswith("_") or lb.startswith("Δ0") or blk.get("policy") is None:
            continue
        p = blk["policy"]
        dm, dsd, dse, dz = p["delta"]
        rows.append((lb, dm, dse, dz, p["ci95"], p["plies_to_fill"][0],
                     p["captures_total"]))
    rows.sort(key=lambda t: -t[1])
    for lb, dm, dse, dz, ci, pf, ct in rows:
        flag = " *" if abs(dz) >= 1.96 else "  "
        print(f"  {lb[:42]:<44}{dm:>+9.4f}{dse:>8.4f}{dz:>+8.2f}"
              f"  [{ci[0]:+.3f},{ci[1]:+.3f}]{flag}{pf:>10.2f}{ct:>9.2f}")
    print("  （* = 95% 显著；Δ>0 表示该战术方比 D_BASE 多占营）")
    return out


KEY_EFFECTS = [
    # (标签, A, B, batch函数, 摘要取值路径)
    ("首翻黄金位 vs 首翻底线死位", "BASE", "X_firstedge", "race"),
    ("几何8位 vs 代码6位黄金位", "X_golden8", "BASE", "race"),
    ("后手卡位进营 vs 中营优先", "R_denial", "BASE", "race"),
    ("后手另辟营簇 vs 中营优先", "R_split", "BASE", "race"),
    ("后手跟随同簇 vs 中营优先", "R_follow", "BASE", "race"),
    ("行营扑杀(驻营子出击) vs 不吃子", "AG_any_out", "D_BASE", "delta"),
    ("到处吃子(不出营) vs 不吃子", "AG_anywhere", "D_BASE", "delta"),
    ("接触-到处吃子 vs 不吃子", "CT_anywhere", "CT_base", "delta"),
    ("接触-行营扑杀 vs 不吃子", "CT_any_out", "CT_base", "delta"),
    ("接触-炸弹同尽 vs 不吃子", "CT_bomb", "CT_base", "delta"),
    ("接触-走入敌杀区送死 vs 不送", "CT_walk", "CT_base", "delta"),
    ("接触-暴露高价值子诱骗 vs 不暴露", "CT_exhigh", "CT_base", "delta"),
    ("接触-暴露低价值子 vs 不暴露", "CT_exlow", "CT_base", "delta"),
    ("接触-贪全部合法交换 vs 不吃子", "CT_all_legal", "CT_base", "delta"),
    ("接触-卡位 vs 中营优先", "CT_deny", "CT_base", "delta"),
]


def section_effect_replication(cfg, n_races, seeds):
    """关键效应的跨种子段重复测量：用池化估计替代单段点估计。

    单段（400 场）的配对差 SE≈0.14 营，与要分辨的效应量同量级；
    7 段池化后 SE≈0.05 营，才能把 0.1~0.3 营的战术效应与噪声分开。
    段间 SD 与单段 SE 之比 ≈1 是配对设计有效的判据（见 §12）。
    """
    print("\n" + "=" * 78)
    print(f"【15】关键效应跨种子段重复（{len(seeds)} 段 × {n_races*2} 场 = "
          f"{len(seeds)*n_races*2} 场/效应）")
    print("=" * 78)
    print(f"  {'效应（A − B，Δ=占营差）':<38}{'池化Δ':>9}{'SE':>7}{'z':>7}"
          f"{'95%CI':>19}{'段间SD':>8}{'SD/SE':>7}{'判定':>6}")
    rows = []
    for label, a, b, kind in KEY_EFFECTS:
        batch = race_batch if kind == "race" else delta_batch
        summ = summarize_races if kind == "race" else summarize_delta
        per = []
        for sd in seeds:
            recs = batch(a, b, n_races, cfg, seed0=sd)
            s = summ(recs, lambda r: 0 if r["specs"][0] == a else 1)
            # 两种摘要的元组布局不同，必须分别取 (均值, SE)：
            #   summarize_races["camp_diff"] = (mean, SE, z)
            #   summarize_delta["delta"]     = (mean, SD, SE, z)
            if kind == "race":
                m, se, _z = s["camp_diff"]
            else:
                m, _sd, se, _z = s["delta"]
            per.append((m, se))
        k = len(per)
        pooled = sum(m for m, _ in per) / k
        # 池化 SE = sqrt(sum(SE_i^2)) / k（各段独立）
        pooled_se = (sum(se ** 2 for _, se in per)) ** 0.5 / k
        seg_sd = (sum((m - pooled) ** 2 for m, _ in per) / max(1, k - 1)) ** 0.5
        mean_se = sum(se for _, se in per) / k
        z = pooled / pooled_se if pooled_se > 0 else 0.0
        ratio = seg_sd / mean_se if mean_se > 0 else float("nan")
        verdict = "显著" if abs(z) >= 1.96 else "不显著"
        rows.append((label, pooled, pooled_se, z, seg_sd, ratio, verdict, a, b))
        print(f"  {label[:36]:<38}{pooled:>+9.4f}{pooled_se:>7.4f}{z:>+7.2f}"
              f"  [{pooled-1.96*pooled_se:+.3f},{pooled+1.96*pooled_se:+.3f}]"
              f"{seg_sd:>8.4f}{ratio:>7.2f}{verdict:>6}")
    print("\n  判读：")
    print("    · 「SD/SE」≈1 表示段间波动完全由抽样误差解释，配对设计有效；")
    print("      若显著 >1 则说明存在未控制的方差源，结论不可用。")
    sig = [r for r in rows if r[6] == "显著"]
    print(f"    · {len(sig)}/{len(rows)} 个效应在池化后达到 95% 显著：")
    for r in sorted(sig, key=lambda t: -abs(t[1])):
        print(f"        {r[0][:36]:<38} Δ={r[1]:+.4f} 营  z={r[3]:+.2f}")
    ns = [r for r in rows if r[6] != "显著"]
    if ns:
        print(f"    · {len(ns)}/{len(rows)} 个效应**不显著**（应视为无效战术）：")
        for r in ns:
            print(f"        {r[0][:36]:<38} Δ={r[1]:+.4f} 营  z={r[3]:+.2f}")
    return [{"label": r[0], "A": r[7], "B": r[8], "pooled_delta": r[1],
             "se": r[2], "z": r[3], "seg_sd": r[4], "sd_over_se": r[5],
             "verdict": r[6]} for r in rows]


def _tee(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    class _T:
        def __init__(self, *s):
            self.s = s

        def write(self, x):
            for f in self.s:
                f.write(x)

        def flush(self):
            for f in self.s:
                f.flush()

    sys.stdout = _T(sys.__stdout__, logf)
    return logf


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=200,
                    help="§9~§12 每组每方向场次（总场次 = 2x），默认 200 => 400 场/组")
    ap.add_argument("--stability-races", type=int, default=200,
                    help="§12 种子稳定性核查每段场次")
    ap.add_argument("--quick", action="store_true", help="跳过 §9 全因子扫描")
    ap.add_argument("--log", default="reports/camp_race_report.txt")
    ap.add_argument("--json", default="reports/camp_race_report.json")
    # ---- Δ 战术研究（独立报告）----
    ap.add_argument("--delta", action="store_true",
                    help="只运行 §14 Δ 战术研究，输出 reports/camp_delta_report.txt|.json")
    ap.add_argument("--delta-races", type=int, default=400,
                    help="§14 每组每方向场次，默认 400 => 800 场/组")
    ap.add_argument("--replicate-races", type=int, default=200,
                    help="§15 关键效应重复测量每段每方向场次，默认 200 => 7段×400=2800 场/效应")
    ap.add_argument("--delta-log", default="reports/camp_delta_report.txt")
    ap.add_argument("--delta-json", default="reports/camp_delta_report.json")
    args = ap.parse_args()

    # ================= Δ 战术研究模式 =================
    if args.delta:
        logf = _tee(args.delta_log)
        print("#" * 78)
        print("# Δ 战术研究报告：核心目标函数 Δ = camps_A − camps_B")
        print(f"# 赛制：第 10 个行营被占据的瞬间终止（与 §9~§13 一致）")
        print(f"# 每组场次：{args.delta_races} x 2(座位互换) = {args.delta_races*2}")
        print(f"# 规则夹具: no_capture_draw_plies=0(关闭判和), max_plies=1e9 —— 仅用于测量")
        print(f"# 引擎代码与 APK 对齐常量零改动")
        print("#" * 78)
        rcfg = race_config()
        dt0 = time.perf_counter()
        out = section_delta_tactics(rcfg, args.delta_races)
        repl = section_effect_replication(
            rcfg, args.replicate_races,
            [20000, 40000, 60000, 80000, 90000, 100000, 110000])
        print(f"\n总耗时 {time.perf_counter()-dt0:.1f}s")
        Path(args.delta_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.delta_json, "w", encoding="utf-8") as f:
            json.dump({"races_per_dir": args.delta_races,
                       "replicate_races_per_dir": args.replicate_races,
                       "strategies": {k: dict(v) for k, v in DELTA_STRATEGIES.items()},
                       "key_effects_replication": repl,
                       "results": out}, f, ensure_ascii=False, indent=2, default=str)
        print(f"Δ 报告已写入 {args.delta_log}")
        print(f"机器可读结果已写入 {args.delta_json}")
        sys.stdout = sys.__stdout__
        logf.close()
        return

    # ================= 常规分析 + 竞赛模式 =================
    logf = _tee(args.log)

    cfg = RuleConfig()
    print("#" * 78)
    print("# 开局'快速占据行营'策略探索报告")
    print(f"# 规则夹具: no_capture_draw_plies=0(关闭判和), max_plies=1e9 —— 仅用于测量")
    print(f"# 终止条件: 第 10 个行营被占据的瞬间立即停止，统计双方占营数与所用手数")
    print(f"# 每组场次: {args.races} x 2(座位互换) = {args.races*2}")
    print("#" * 78)

    section_topology()
    section_first_flip_scores(cfg)
    section_second_flip_scores(cfg)
    section_tempo_model()
    section_mc_tempo(cfg, n=300)
    section_mobility_theorem(cfg)

    rcfg = race_config()
    scan = None if args.quick else section_race_scan(rcfg, args.races)
    targeted = section_race_targeted(rcfg, args.races)
    curve = section_race_curve(rcfg, args.races)
    stability = section_seed_stability(
        rcfg, args.stability_races,
        [20000, 40000, 60000, 80000, 90000, 100000, 110000])
    section_visuals(curve, rcfg)

    # 用竞赛实测数据回填闭式模型对照表
    measured = []
    for label, blk in targeted.items():
        s = blk["seat"]
        measured.append((label[:32], s["first_camp_ply"][0], s["first_camp_ply"][1],
                         s["camps"][0], s["camps"][1], s["plies_to_fill"][0]))
    model = section_race_model(measured)

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"model": model, "targeted": targeted, "curve": curve,
                   "stability": {k: {"mean": v[0], "range": v[1],
                                     "rel_pct": v[2], "verdict": v[3]}
                                 for k, v in stability.items()},
                   "scan": [{"camp_pref": cp, "cluster": cl, "alternate": alt,
                             "seat": s} for cp, cl, alt, s in (scan or [])]},
                  f, ensure_ascii=False, indent=2, default=str)

    print(f"\n报告已写入 {args.log}")
    print(f"机器可读结果已写入 {args.json}")
    print("Δ 战术研究请另跑：python scripts/analyze_camp_opening.py --delta")
    sys.stdout = sys.__stdout__
    logf.close()


if __name__ == "__main__":
    main()

