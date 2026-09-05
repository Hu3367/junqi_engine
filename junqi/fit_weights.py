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
from .rules import COMPOSITION, NEIGHBORS, RANK_CN, Rank, battle, is_camp, is_hq
from .state import Action, GameState

RANKS = [Rank.SI, Rank.JUN, Rank.SHI, Rank.LV, Rank.TUAN, Rank.YING,
         Rank.LIAN, Rank.PAI, Rank.GONG, Rank.ZHA, Rank.LEI, Rank.QI]

FEATURE_NAMES = [f"p_{r.name}" for r in RANKS] + [
    "camp_occ", "camp_siege", "hq_locked", "flag_exposed",
    "threat", "attack", "attack_camp", "flip_bias"]

# gain 类特征量纲大，预缩放保持梯度均衡
FEATURE_SCALE = {**{f"p_{r.name}": 1.0 for r in RANKS},
                 "camp_occ": 1.0, "camp_siege": 1.0, "hq_locked": 1.0,
                 "flag_exposed": 1.0, "threat": 100.0, "attack": 100.0,
                 "attack_camp": 100.0, "flip_bias": 1.0}

CAPTURE_UNIT = 50.0     # 威胁/机会特征的单位折算


def _capture_pressure(state, attackers, target_color) -> float:
    """attackers 侧对相邻敌明子（非旗）的可得价值总量（同尽记半价）。"""
    total = 0.0
    for pos, m in attackers:
        if m.rank == Rank.QI:
            continue
        for np_ in NEIGHBORS[pos]:
            if is_camp(np_):
                continue
            e = state.board.get(np_)
            if e is None or e.color != target_color or e.rank == Rank.QI:
                continue
            res = battle(m.rank, e.rank)
            if res == "attacker_wins":
                total += CAPTURE_UNIT
            elif res == "both_die":
                total += CAPTURE_UNIT * 0.5
    return total


def features(state: GameState, seat: int) -> dict:
    """全知口径线性特征（seat 视角，正=有利）。与 evaluate() 项一一对应。"""
    my = state.seat_color[seat]
    opp = "b" if my == "r" else "r"
    f = {name: 0.0 for name in FEATURE_NAMES}
    my_flag = opp_flag = None
    mine, theirs = [], []
    for pos, pc in state.board.items():
        side = 1.0 if pc.color == my else -1.0
        f[f"p_{pc.rank.name}"] += side
        (mine if side > 0 else theirs).append((pos, pc))
        if pc.rank == Rank.QI:
            if side > 0:
                my_flag = pos
            else:
                opp_flag = pos
        if is_camp(pos):
            f["camp_occ"] += side
            for np_ in NEIGHBORS[pos]:
                e = state.board.get(np_)
                if e is not None and e.color != pc.color:
                    f["camp_siege"] += side
        if is_hq(pos) and pc.rank != Rank.QI:
            f["hq_locked"] += side

    def can_take(flag_color, atk):
        if atk == Rank.LEI:
            return False
        if state.cfg.flag_gong_only and atk != Rank.GONG:
            return False
        if state.cfg.flag_needs_mines_cleared:
            mines_left = COMPOSITION[Rank.LEI] - sum(
                1 for p_, pc_ in state.board.items()
                if pc_.color == flag_color and pc_.rank == Rank.LEI)
            if mines_left > 0:
                return False
        return True

    def threatened(flag_pos, flag_color, attackers):
        return any(flag_pos in NEIGHBORS[ap] and can_take(flag_color, e.rank)
                   for ap, e in attackers)

    if my_flag and threatened(my_flag, my, theirs):
        f["flag_exposed"] -= 1.0
    if opp_flag and threatened(opp_flag, opp, mine):
        f["flag_exposed"] += 1.0

    # 威胁（敌子可吃我方，负向）与机会（我方可吃敌子，分营内/营外发起）
    f["threat"] = -_capture_pressure(state, theirs, my)
    f["attack"] = _capture_pressure(
        state, [(p, pc) for p, pc in mine if not is_camp(p)], opp)
    f["attack_camp"] = _capture_pressure(
        state, [(p, pc) for p, pc in mine if is_camp(p)], opp)
    return f


def default_vector() -> list:
    """现有 EvalWeights 默认值作为 warm start（与特征顺序对齐）。"""
    w = EvalWeights()
    vec = [float(w.piece[r]) for r in RANKS]
    vec += [w.camp_occ, w.camp_siege, w.hq_locked, w.flag_exposed,
            w.threat * 100.0, w.attack * 100.0, w.attack_camp * 100.0, 0.0]
    return vec


def to_eval_weights(vec: list) -> EvalWeights:
    w = EvalWeights()
    for i, r in enumerate(RANKS):
        w.piece[r] = max(1.0, round(vec[i], 1))
    w.camp_occ = round(vec[12], 2)
    w.camp_siege = round(vec[13], 2)
    w.hq_locked = round(vec[14], 2)
    w.flag_exposed = round(vec[15], 2)
    w.threat = round(vec[16] / 100.0, 4)
    w.attack = round(vec[17] / 100.0, 4)
    w.attack_camp = round(vec[18] / 100.0, 4)
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


# 语义符号约束：(特征索引, 下界, 上界)——学到的权重必须落在合理区域
def _bounds():
    b = {}
    for i, r in enumerate(RANKS):
        b[i] = (1.0, 300.0)
    b[12] = (0.0, 60.0)      # camp_occ ≥0
    b[13] = (0.0, 40.0)      # camp_siege ≥0（与 attack_camp 部分共线，防符号翻转）
    b[14] = (-60.0, 0.0)     # hq_locked ≤0
    b[15] = (0.0, 200.0)     # flag_exposed ≥0
    b[16] = (0.0, 2.0)       # threat ≥0
    b[17] = (0.0, 2.0)       # attack ≥0
    b[18] = (0.0, 5.0)       # attack_camp ≥0
    b[19] = (-20.0, 20.0)    # flip_bias 自由
    return b


def _project(w):
    for j, (lo, hi) in _bounds().items():
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
