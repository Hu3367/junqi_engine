"""估值权重自动调优：镜像自对弈 + 随机爬山。

  python -m junqi tune --rounds 3 --candidates 6 --games 40 --workers 8

原理：挑战者权重 vs 基线权重打镜像对局（先后手各半，greedy 策略、
官方规则终局），得分 = (胜 + 0.5*和) / 局数；每轮随机扰动基线生成若干
候选，谁镜像胜率显著过半（> 0.5 + margin）谁成为新基线。贪心(depth 0)
单步期望策略走法完全由权重决定，是最便宜且对权重最敏感的训练信号。

产出：reports/tune.json（全部轮次记录）+ 控制台最优权重（可直接粘进
EvalWeights / from_dict 加载）。
"""
from __future__ import annotations

import json
import os
import random
from multiprocessing import Pool

from .config import EvalWeights, RuleConfig
from .selfplay import play_game

# 参与扰动的权重键（piece 用 Rank 名，其余为标量项）
TUNE_KEYS = [
    ("piece", "GONG"), ("piece", "ZHA"), ("piece", "LEI"), ("piece", "QI"),
    ("piece", "SI"), ("piece", "SHI"), ("piece", "PAI"),
    ("scalar", "flag_exposed"), ("scalar", "camp_occ"),
    ("scalar", "camp_siege"), ("scalar", "attack"), ("scalar", "attack_camp"),
    ("scalar", "threat"),
]
PIECE_MIN = {"GONG": 20, "ZHA": 25, "LEI": 10, "QI": 20, "SI": 60,
             "SHI": 40, "PAI": 8}
PIECE_MAX = {"GONG": 80, "ZHA": 90, "LEI": 60, "QI": 90, "SI": 160,
             "SHI": 120, "PAI": 40}
SCALAR_RANGE = {"flag_exposed": (10.0, 90.0), "camp_occ": (0.0, 25.0),
                "camp_siege": (0.0, 14.0), "attack": (0.05, 0.6),
                "attack_camp": (0.2, 1.6), "threat": (0.1, 0.6)}
MAT_WEIGHT = 0.0005     # 适应度中平均歼敌子力差的系数（胜率 1 分制下）


def perturb(base: EvalWeights, rng: random.Random,
            n_keys: int = 3) -> EvalWeights:
    """随机挑 n_keys 个键按 [0.75, 1.35] 扰动（piece 限合理区间）。"""
    d = base.to_dict()
    for kind, key in rng.sample(TUNE_KEYS, n_keys):
        if kind == "piece":
            lo, hi = PIECE_MIN[key], PIECE_MAX[key]
            d["piece"][key] = max(lo, min(hi, round(
                d["piece"][key] * rng.uniform(0.75, 1.35))))
        else:
            lo, hi = SCALAR_RANGE[key]
            d[key] = round(max(lo, min(hi, d[key] * rng.uniform(0.75, 1.35))), 3)
    return EvalWeights.from_dict(d)


def _contest_chunk(job):
    """子进程：挑战者 vs 基线打一段镜像对局，返回 (胜, 和, 局数, 子力差和)。"""
    cand_d, base_d, spec, base_seed, n, max_plies = job
    cand = EvalWeights.from_dict(cand_d)
    base = EvalWeights.from_dict(base_d)
    cfg = RuleConfig(max_plies=max_plies) if max_plies else RuleConfig()
    wins = draws = 0
    matsum = 0.0
    for i in range(n):
        seed = base_seed + i
        # 一半先手、一半后手（镜像），消除先手优势偏差
        challenger_seat = 0 if i % 2 == 0 else 1
        w_ch, w_b = (cand, base) if challenger_seat == 0 else (base, cand)
        rec = play_game(spec, spec, seed, cfg, weights0=w_ch, weights1=w_b)
        if rec["winner"] == challenger_seat:
            wins += 1
        elif rec["winner"] == -1 or rec["winner"] is None:
            draws += 1
        # 歼敌子力差：官方规则下 AI 对局多和棋，胜负信号稀疏，
        # 用连续的子力差给权重变化提供梯度（挑权重直接作用在吃子选择上）
        matsum += (rec[f"captured{challenger_seat}"]
                   - rec[f"captured{1 - challenger_seat}"])
    return wins, draws, n, matsum


def contest(cand: EvalWeights, base: EvalWeights, spec: str, games: int,
            workers: int, base_seed: int, max_plies: int | None) -> float:
    """挑战者对基线的镜像得分 = 胜率 + 0.0005×平均歼敌子力差。
    胜率部分 (胜 + 0.5×和)/局数，0.5 为持平；子力差为和棋局提供梯度。"""
    jobs = []
    chunk = max(1, games // max(workers, 1))
    for w in range(workers):
        n = chunk if w < workers - 1 else games - chunk * (workers - 1)
        if n > 0:
            jobs.append((cand.to_dict(), base.to_dict(), spec,
                         base_seed + w * 50_000, n, max_plies))
    if workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(_contest_chunk, jobs)
    else:
        parts = [_contest_chunk(j) for j in jobs]
    wins = sum(p[0] for p in parts)
    draws = sum(p[1] for p in parts)
    n = sum(p[2] for p in parts)
    matsum = sum(p[3] for p in parts)
    wr = (wins + 0.5 * draws) / max(n, 1)
    return wr + MAT_WEIGHT * matsum / max(n, 1)


def tune(rounds: int = 2, candidates: int = 6, games: int = 40,
         workers: int = 1, spec: str = "greedy", margin: float = 0.06,
         seed: int = 0, out: str = "reports/tune.json",
         max_plies: int | None = None) -> dict:
    """爬山主循环。返回 {"baseline": dict, "history": [...]} 并落盘。"""
    rng = random.Random(seed)
    base = EvalWeights()
    base_fit = 0.5                      # 基线对自身镜像恒为 0.5
    history = [{"round": 0, "baseline": base.to_dict(), "fit": base_fit,
                "note": "default weights"}]
    print(f"基线 = 默认权重，镜像基准分 0.5；每轮 {candidates} 个候选 × "
          f"{games} 局（{spec}），显著阈值 {0.5 + margin:.2f}")
    for r in range(1, rounds + 1):
        best = None
        for ci in range(candidates):
            cand = perturb(base, rng)
            fit = contest(cand, base, spec, games, workers,
                          seed * 1_000_000 + r * 10_000 + ci * 100, max_plies)
            diff = cand.to_dict()
            print(f"  轮{r} 候选{ci}: 镜像分 {fit:.3f} "
                  f"{'*' if fit > 0.5 + margin else ''} {diff}")
            if best is None or fit > best[1]:
                best = (cand, fit)
        cand, fit = best
        if fit > 0.5 + margin:
            print(f"  轮{r}: 接受新基线（{fit:.3f} > {0.5 + margin:.2f}）")
            base, base_fit = cand, fit
            history.append({"round": r, "baseline": base.to_dict(),
                            "fit": fit, "note": "accepted"})
        else:
            print(f"  轮{r}: 无显著改进（最好 {fit:.3f}），保留原基线")
            history.append({"round": r, "baseline": base.to_dict(),
                            "fit": fit, "note": "kept"})
    result = {"baseline": base.to_dict(), "fit": base_fit, "history": history,
              "spec": spec, "games": games, "rounds": rounds}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n最优权重已存 {out}\n基准镜像分 {base_fit:.3f}\n"
          f"baseline = EvalWeights.from_dict({json.dumps(base.to_dict(), ensure_ascii=False)})")
    return result
