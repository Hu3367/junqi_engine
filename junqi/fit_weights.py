"""从真实复盘数据行为克隆拟合估值权重（纯标准库，条件 logistic 回归）。

  python -m junqi fit ../军旗复盘 --verify 60

原理：复盘含完整暗子身份表，回放得到每个决策点的真实全知盘面。对每个
决策点取人类实际动作与若干候选动作，提取与 evaluate() 同构的线性特征，
用条件 logit（softmax over 候选）极大化"人类选中动作"的对似然——
等价于学一套让 PIMC 估值最贴近人类决策的 EvalWeights。

特征口径 = 全知（PIMC 世界内 evaluate 的实际调用口径），学到的权重可直接
替换 EvalWeights。BC 只用过程标签（人类走法），940 局无胜负结果的局全部可用。
"""
from __future__ import annotations

import json
import math
import os
import random

from .config import EvalWeights, RuleConfig
from .replay import SPECIAL_EVENT, board_from_table, cell_rc, load_dir
from .rules import CAMPS, COMPOSITION, NEIGHBORS, RANK_CN, Rank, battle, is_camp, is_hq, other
from .state import Action, GameState

RANKS = [Rank.SI, Rank.JUN, Rank.SHI, Rank.LV, Rank.TUAN, Rank.YING,
          Rank.LIAN, Rank.PAI, Rank.GONG, Rank.ZHA, Rank.LEI, Rank.QI]

# ⚠ 不变式：本列表必须与 EvalWeights 的字段**一一对应**（`to_eval_weights` 按名字写回），
# 且 `default_vector()` 必须落在 `_bounds()` 内、是 `_project()` 的**不动点**。
# 三项都由 `tests/test_p1_advanced_enhancements.py` 守卫。
#
# `mobility` 曾被列在此处，但 `EvalWeights` 没有对应字段 ⇒ 拟合出的系数会被静默丢弃；
# 更关键的是 evaluate_expert 的机动力项是**另一套定义**（含铁路 +1、排除行营内的敌子，
# 且把"死子惩罚 −8/−4"直接并入 score），系数无法迁移。
# 因此这里不列 mobility；要真正可调需先把两侧定义统一（见 docs/CHANGELOG 第十五批）。
FEATURE_NAMES = [f"p_{r.name}" for r in RANKS] + [
    "camp_occ", "camp_zone", "camp_siege", "hq_locked", "flag_exposed",
    "mine_guard", "fortress", "hidden_tempo",
    "threat", "attack", "attack_camp", "flip_bias"
]

# 特征量纲缩放字典
FEATURE_SCALE = {
    **{f"p_{r.name}": 1.0 for r in RANKS},
    "camp_occ": 1.0, "camp_zone": 1.0, "camp_siege": 1.0, "hq_locked": 1.0,
    "flag_exposed": 1.0, "mine_guard": 1.0,
    "fortress": 10.0, "hidden_tempo": 1.0,
    "threat": 100.0, "attack": 100.0, "attack_camp": 100.0, "flip_bias": 1.0
}

CAPTURE_UNIT = 50.0     # 威胁/机会特征的单位折算


def _capture_pressure(state: GameState, attackers: list[tuple[tuple[int, int], Piece]], target_color: str) -> float:
    """attackers 侧对相邻敌明子（非旗）的可得价值总量（同尽记半价）。"""
    total = 0.0
    for pos, m in attackers:
        if m.rank == Rank.QI or m.rank == Rank.LEI:
            continue
        for np_ in NEIGHBORS[pos]:
            if is_camp(np_):
                continue
            e = state.board.get(np_)
            if e is None or not e.revealed or e.color != target_color or e.rank == Rank.QI:
                continue
            res = battle(m.rank, e.rank)
            if res == "attacker_wins":
                total += CAPTURE_UNIT
            elif res == "both_die":
                total += CAPTURE_UNIT * 0.5
    return total


def features(state: GameState, seat: int) -> dict[str, float]:
    """严格遵循公共信息边界的专家特征（seat 视角，正=有利）。

    符合 AGENTS.md 与基线方案硬约束：
    1. 首翻定色前完全对称（特征严格全为 0.0）；
    2. 绝不遍历或窥探暗子的真实身份与颜色；
    3. 暗子仅以公共剩余池（remaining_types）的期望份额计入物质分；
    4. 对齐 evaluate_expert 的核心几何与战术特征。
    """
    f = {name: 0.0 for name in FEATURE_NAMES}
    my = state.seat_color.get(seat)
    if my is None or not state.first_flip_done:
        return f

    opp = other(my)

    # 1. 物质项（明子 + 公共暗子池期望份额）
    # 明子贡献
    revealed_mine: list[tuple[tuple[int, int], Piece]] = []
    revealed_opp: list[tuple[tuple[int, int], Piece]] = []
    for pos, pc in state.board.items():
        if pc.revealed:
            side = 1.0 if pc.color == my else -1.0
            f[f"p_{pc.rank.name}"] += side
            if pc.color == my:
                revealed_mine.append((pos, pc))
            else:
                revealed_opp.append((pos, pc))

    # 公共暗子池期望份额贡献 (精确摊派，无透视)
    rem = state.remaining_types()
    for (clr, rk), cnt in rem.items():
        side = 1.0 if clr == my else -1.0
        f[f"p_{rk.name}"] += side * cnt

    # 2. 行营控制与营内围杀
    for pos, pc in revealed_mine:
        if is_camp(pos):
            f["camp_occ"] += 1.0
            for np in NEIGHBORS[pos]:
                e = state.board.get(np)
                if e is not None and e.revealed and e.color == opp and battle(pc.rank, e.rank) in ("attacker_wins", "both_die"):
                    f["camp_siege"] += 1.0
        if is_hq(pos) and pc.rank != Rank.QI and state.cfg.hq_locks_pieces:
            f["hq_locked"] += 1.0

    for pos, pc in revealed_opp:
        if is_camp(pos):
            f["camp_occ"] -= 1.0
            for np in NEIGHBORS[pos]:
                m = state.board.get(np)
                if m is not None and m.revealed and m.color == my and battle(pc.rank, m.rank) in ("attacker_wins", "both_die"):
                    f["camp_siege"] -= 1.0
        if is_hq(pos) and pc.rank != Rank.QI and state.cfg.hq_locks_pieces:
            f["hq_locked"] -= 1.0

    # 3. 行营势力范围 (贴近空行营的活动明子净差)
    zone_net = 0.0
    for cp in CAMPS:
        if cp in state.board:
            continue
        for np_ in NEIGHBORS[cp]:
            e = state.board.get(np_)
            if e is not None and e.revealed and e.rank not in (Rank.LEI, Rank.QI):
                zone_net += 1.0 if e.color == my else -1.0
    f["camp_zone"] = zone_net

    # 4. 军旗暴露与地雷守护
    my_flag = next(((p, pc) for p, pc in revealed_mine if pc.rank == Rank.QI), None)
    opp_flag = next(((p, pc) for p, pc in revealed_opp if pc.rank == Rank.QI), None)

    def can_take(flag_color: str, atk: Rank) -> bool:
        if atk == Rank.LEI:
            return False
        if state.cfg.flag_gong_only and atk != Rank.GONG:
            return False
        if state.cfg.flag_needs_mines_cleared:
            mines_left = COMPOSITION[Rank.LEI] - sum(1 for d in state.dead if d.color == flag_color and d.rank == Rank.LEI)
            if mines_left > 0:
                return False
        return True

    if my_flag and any(my_flag[0] in NEIGHBORS[ap] and can_take(my, e.rank) for ap, e in revealed_opp):
        f["flag_exposed"] -= 1.0
    if opp_flag and any(opp_flag[0] in NEIGHBORS[ap] and can_take(opp, m.rank) for ap, m in revealed_mine):
        f["flag_exposed"] += 1.0

    if my_flag:
        f["mine_guard"] += sum(1.0 for pos, pc in revealed_mine if pc.rank == Rank.LEI and pos in NEIGHBORS[my_flag[0]])
    if opp_flag:
        f["mine_guard"] -= sum(1.0 for pos, pc in revealed_opp if pc.rank == Rank.LEI and pos in NEIGHBORS[opp_flag[0]])

    # 5. 死区势能差 (若有军旗暴露)
    if my_flag or opp_flag:
        from .analysis import fortress_score
        fs_my = fortress_score(state, seat) if my_flag else 0.0
        fs_opp = fortress_score(state, 1 - seat) if opp_flag else 0.0
        f["fortress"] = fs_my - fs_opp

    # 6. 暗子时差 (活动明子数净差)
    my_active = sum(1 for p, pc in revealed_mine if pc.rank not in (Rank.LEI, Rank.QI))
    opp_active = sum(1 for p, pc in revealed_opp if pc.rank not in (Rank.LEI, Rank.QI))
    f["hidden_tempo"] = float(my_active - opp_active)

    # 7. 战术威胁与机会
    f["threat"] = -_capture_pressure(state, revealed_opp, my)
    f["attack"] = _capture_pressure(state, [(p, pc) for p, pc in revealed_mine if not is_camp(p)], opp)
    f["attack_camp"] = _capture_pressure(state, [(p, pc) for p, pc in revealed_mine if is_camp(p)], opp)

    return f


def default_vector() -> list[float]:
    """现有 EvalWeights 默认值作为 warm start（按**特征名**取值，不用下标）。

    不变式（有测试守卫）：
    1. `default_vector()` 必须落在 `_bounds()` 内 ⇒ 是 `_project()` 的**不动点**；
    2. `to_eval_weights(default_vector())` 必须逐字段等于 `EvalWeights()`。
    """
    w = EvalWeights()
    f: dict[str, float] = {f"p_{r.name}": float(w.piece[r]) for r in RANKS}
    f.update({
        "camp_occ": w.camp_occ,
        "camp_zone": w.camp_zone,
        "camp_siege": w.camp_siege,
        "hq_locked": w.hq_locked,
        "flag_exposed": w.flag_exposed,
        "mine_guard": w.mine_flag_guard_bonus / 10.0,
        "fortress": w.fortress / 10.0,
        "hidden_tempo": w.hidden_tempo,
        "threat": w.threat * 100.0,
        "attack": w.attack * 100.0,
        "attack_camp": w.attack_camp * 100.0,
        "flip_bias": 0.0,   # 翻子整体倾向：仅分析用，不写回 EvalWeights
    })
    return [f[name] for name in FEATURE_NAMES]


def to_eval_weights(vec: list[float]) -> EvalWeights:
    """把优化参数向量映射回 EvalWeights 对象（按特征名查表，杜绝下标错位）。"""
    v = dict(zip(FEATURE_NAMES, vec))
    w = EvalWeights()
    for r in RANKS:
        w.piece[r] = max(1.0, round(v[f"p_{r.name}"], 1))
    w.camp_occ = round(v["camp_occ"], 2)
    w.camp_zone = round(v["camp_zone"], 2)
    w.camp_siege = round(v["camp_siege"], 2)
    w.hq_locked = round(v["hq_locked"], 2)
    w.flag_exposed = round(v["flag_exposed"], 2)
    w.mine_flag_guard_bonus = round(v["mine_guard"] * 10.0, 2)
    w.fortress = round(v["fortress"] * 10.0, 2)
    w.hidden_tempo = round(v["hidden_tempo"], 2)
    w.threat = round(v["threat"] / 100.0, 4)
    w.attack = round(v["attack"] / 100.0, 4)
    w.attack_camp = round(v["attack_camp"] / 100.0, 4)
    return w


# ---------------------------------------------------------------- 数据采集

def collect_points(games, points_per_game: int, max_cands: int, rng):
    """从回放对局采样决策点：{"cands": [[f...]], "chosen": int, ...}。"""
    points = []
    for g in games:
        if not g.replay_ok or not g.moves:
            continue
        st = GameState(board=board_from_table(g.table))
        idxs = sorted(rng.sample(range(len(g.moves)),
                                 min(points_per_game, len(g.moves))))
        want = set(idxs)
        for i, (a, b, c) in enumerate(g.moves):
            if i > idxs[-1] or (a, b, c) == SPECIAL_EVENT:
                break
            act = Action("flip", cell_rc(a)) if (a == b and c == 1) \
                else Action("move", cell_rc(a), cell_rc(b))
            if i in want:
                acts = st.legal_actions()
                if act not in acts or len(acts) < 2:
                    continue
                chosen = acts.index(act)
                if len(acts) > max_cands:
                    others = [x for j, x in enumerate(acts) if j != chosen]
                    rng.shuffle(others)
                    acts = [act] + others[:max_cands - 1]
                    chosen = 0
                seat = st.turn
                cands = []
                for cand in acts:
                    fv = features(st.apply(cand), seat)
                    fv["flip_bias"] = 1.0 if cand.kind == "flip" else 0.0
                    cands.append([fv[name] / FEATURE_SCALE[name]
                                  for name in FEATURE_NAMES])
                points.append({"cands": cands, "chosen": chosen})
            st = st.apply(act)
    return points


# ---------------------------------------------------------------- 训练

def score_of(w, cands):
    return [sum(wv * fv for wv, fv in zip(w, cand)) for cand in cands]


# 语义符号约束：按**特征名**给出 (下界, 上界)，区间表达在**向量单位**
# （= 特征原始值 / FEATURE_SCALE，即 train()/_project() 实际作用的单位）。
#
# ⚠ 曾用写死下标表达这些约束，`FEATURE_NAMES` 扩容后约束被施加到**错误的特征**上：
#   flag_exposed 被夹到 2（应为 0~200）、camp_siege / hq_locked 符号反转、
#   mine_guard 被夹到 2，而 threat / attack / attack_camp 变成完全无界。
#   实测一次 `_project()` 就把 warm start 毁掉 ⇒ 一律按名字索引。
_BOUND_BY_NAME: dict[str, tuple[float, float]] = {
    "camp_occ": (0.0, 60.0),        # 占营为正贡献
    "camp_zone": (-20.0, 40.0),     # 行营势力范围净差，可正可负
    "camp_siege": (0.0, 40.0),      # ≥0（与 attack_camp 部分共线，防符号翻转）
    "hq_locked": (-60.0, 0.0),      # ≤0（大子被困大本营是坏形）
    "flag_exposed": (0.0, 200.0),   # ≥0（特征自身已带符号，权重为正）
    "mine_guard": (0.0, 40.0),      # ≥0（地雷护旗为正贡献）
    "fortress": (0.0, 30.0),        # ≥0（死区势能差，越大越好）
    "hidden_tempo": (0.0, 30.0),    # ≥0（活动明子多者占优）
    "threat": (0.0, 60.0),          # ≥0（特征为负向量）
    "attack": (0.0, 200.0),         # ≥0
    "attack_camp": (0.0, 200.0),    # ≥0
    "flip_bias": (-20.0, 20.0),     # 自由（翻子整体倾向）
}


def _bounds() -> dict[int, tuple[float, float]]:
    b: dict[int, tuple[float, float]] = {i: (1.0, 300.0) for i in range(len(RANKS))}
    for name, rng in _BOUND_BY_NAME.items():
        b[FEATURE_NAMES.index(name)] = rng
    return b


def _project(w):
    all_bounds = _bounds()
    for j, (lo, hi) in all_bounds.items():
        w[j] = min(hi, max(lo, w[j]))
    return w


def train(points, val_pts, init, epochs: int, lr: float, l2: float = 1e-5):
    """条件 logit 全批梯度下降。

    决策点内对特征做均值中心化：softmax 只依赖候选间差值，中心化消除
    物质分等大常数项导致的饱和（初始权重下多数点 score 差几百，梯度
    全部堵在饱和区），中心化后梯度恢复区分度。"""
    d = len(FEATURE_NAMES)
    w = list(init)
    best_w, best_val = list(init), hit_rate(val_pts, init)
    n = len(points)
    for ep in range(epochs):
        grad = [0.0] * d
        loss = 0.0
        for p in points:
            cands = p["cands"]
            m = len(cands)
            mean = [sum(c[j] for c in cands) / m for j in range(d)]
            centered = [[c[j] - mean[j] for j in range(d)] for c in cands]
            scores = score_of(w, centered)
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            z = sum(exps)
            loss += -(scores[p["chosen"]] - mx - math.log(z))
            for k in range(m):
                coef = exps[k] / z - (1.0 if k == p["chosen"] else 0.0)
                if coef:
                    fv = centered[k]
                    for j in range(d):
                        grad[j] += coef * fv[j]
        for j in range(d):
            w[j] -= lr * (grad[j] / n + l2 * w[j])
        _project(w)
        if (ep + 1) % 5 == 0 or ep == 0:
            va = hit_rate(val_pts, w)
            if va > best_val:
                best_val, best_w = va, list(w)
            print(f"  epoch {ep+1}/{epochs}  loss={loss/n:.4f}  val_top1={va:.3f}")
    print(f"  最优验证 top1={best_val:.3f}")
    return best_w


def hit_rate(points, w, topk=1):
    hits = tot = 0
    for p in points:
        scores = score_of(w, p["cands"])
        order = sorted(range(len(scores)), key=lambda k: -scores[k])
        tot += 1
        if p["chosen"] in order[:topk]:
            hits += 1
    return hits / max(tot, 1)


def avg_cands(points):
    return sum(len(p["cands"]) for p in points) / max(len(points), 1)


# ---------------------------------------------------------------- 主入口

def run_fit(path, points_per_game=20, max_cands=24, epochs=60, lr=0.5,
            seed=0, out="reports/fit_weights.json", verify=0, workers=8):
    rng = random.Random(seed)
    print(f"解析+回放 {path} …")
    games = load_dir(path, check=False)
    ok = [g for g in games if g.replay_ok and g.moves]
    print(f"可用对局 {len(ok)}/{len(games)}")
    print(f"采样决策点（每局≤{points_per_game}，候选≤{max_cands}）…")
    points = collect_points(ok, points_per_game, max_cands, rng)
    rng.shuffle(points)
    k = int(len(points) * 0.85)
    train_pts, val_pts = points[:k], points[k:]
    print(f"决策点：训练 {len(train_pts)} / 验证 {len(val_pts)}，"
          f"平均候选 {avg_cands(points):.1f}（均匀基线 top1={1/avg_cands(points):.3f}）")

    init = default_vector()
    base_acc = hit_rate(val_pts, init)
    print(f"初始(现有默认)权重 验证 top1={base_acc:.3f}")
    w = train(train_pts, val_pts, init, epochs, lr)
    acc1 = hit_rate(val_pts, w)
    acc3 = hit_rate(val_pts, w, topk=3)
    print(f"拟合后 验证 top1={acc1:.3f} top3={acc3:.3f}")

    ew = to_eval_weights(w)
    old = EvalWeights()
    result = {
        "feature_names": FEATURE_NAMES,
        "vector": [round(v, 4) for v in w],
        "eval_weights": ew.to_dict(),
        "metrics": {"top1": round(acc1, 4), "top3": round(acc3, 4),
                    "top1_init": round(base_acc, 4),
                    "n_train": len(train_pts), "n_val": len(val_pts)},
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n拟合权重已存 {out}\n")
    print("EvalWeights 对照（旧 → 新）：")
    for i, r in enumerate(RANKS):
        print(f"  {RANK_CN[r]:<3} {old.piece[r]:>6.1f} → {ew.piece[r]:>6.1f}")
    for name, oldv, newv in (("camp_occ", old.camp_occ, ew.camp_occ),
                             ("camp_siege", old.camp_siege, ew.camp_siege),
                             ("hq_locked", old.hq_locked, ew.hq_locked),
                             ("flag_exposed", old.flag_exposed, ew.flag_exposed),
                             ("threat", old.threat, ew.threat),
                             ("attack", old.attack, ew.attack),
                             ("attack_camp", old.attack_camp, ew.attack_camp)):
        print(f"  {name:<13} {oldv:>6.2f} → {newv:>6.2f}")
    print(f"  {'flip_bias':<13} {'-':>6} → {w[-1]:>6.2f}  (翻子整体倾向，分析用)")

    if verify:
        from .tune import contest
        print(f"\n镜像验证：新权重 vs 旧权重 greedy {verify} 局 …")
        fit = contest(ew, old, "greedy", verify, workers, 12345, None)
        print(f"新权重镜像分 {fit:.3f}（>0.5 为优于旧权重）")
