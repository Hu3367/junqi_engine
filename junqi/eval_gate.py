"""P0 评测门控：统计严谨的模型晋级评测协议 (Statistical Promotion Gate)。

针对和棋密集规则（历史门控 0胜/0负/40和 无法证明改进）设计的评测协议，
方法学对齐 reports/camp_race_report.txt 的战术研究基准：

1. 配对同牌 (Paired by Deal)：同一 seed 产生同一副牌，候选/基准先后手各半，
   消除先手结构优势（实测先手 +0.7 营）与发牌运气；
2. 固定种子 (Deterministic)：所有随机源由实验种子派生，结果可复现；
3. 统计检验：配对 z 检验 + 得分率 Wilson 区间 + 三元 SPRT 序贯检验；
4. 终局原因拆分：flag / no_capture / repetition / immobilized 分列，
   防止"循环和棋减少"被误读为"棋力提升"；
5. 晋级判据（AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §7）：
   整体得分率显著优于基准，且 SPRT 不接受 H0，且各场景无严重退化。

用法：
    python -m junqi gate --a hybrid2 --b expert2 --games 200 --seed 2026
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import RuleConfig
from .state import GameState


# ----------------------------------------------------------------- 统计基元

def wilson_ci(successes: float, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson 得分区间。successes 可为加权得分（如含 0.5 和棋），n 为样本数。

    返回 (lower, upper)。n=0 时返回 (0.0, 1.0)。
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_z_test(per_seed_scores: List[float], null_mean: float = 1.0
                  ) -> Dict[str, float]:
    """配对检验：每 seed 一组（先后手各一局）合计得分 s_i ∈ [0, 2]。

    H0：两引擎等强 => E[s_i] = 1.0。返回均值/标准差/标准误/z 值/双侧 p 近似。
    """
    n = len(per_seed_scores)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "z": 0.0, "p_two_sided": 1.0}
    mean = sum(per_seed_scores) / n
    if n > 1:
        var = sum((s - mean) ** 2 for s in per_seed_scores) / (n - 1)
        sd = math.sqrt(max(var, 0.0))
        se = sd / math.sqrt(n)
    else:
        sd, se = 0.0, 0.0
    z = (mean - null_mean) / se if se > 1e-12 else (
        0.0 if abs(mean - null_mean) < 1e-12 else math.copysign(999.0, mean - null_mean))
    # 双侧 p 值（正态近似）
    p_two = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": n, "mean": mean, "sd": sd, "se": se, "z": z, "p_two_sided": p_two}


def _elo_to_score(elo: float) -> float:
    """Elo 差 -> 期望得分率（胜1 和0.5 负0 口径）。"""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_trinomial(wins: int, draws: int, losses: int,
                   elo0: float = 0.0, elo1: float = 65.0,
                   alpha: float = 0.05, beta: float = 0.05,
                   empirical_draw_rate: Optional[float] = None) -> Dict[str, object]:
    """三元 SPRT（W/D/L 序贯概率比检验，fishtest 同款思路）。

    H0：期望得分率 = _elo_to_score(elo0)；H1：期望得分率 = _elo_to_score(elo1)。
    和棋率用经验值（默认取当前样本），胜负率由得分率反解：
        w = s - d/2, l = 1 - s - d/2（截断到非负）。
    返回 {"llr", "lower", "upper", "decision"}，decision ∈
    {"accept_h1", "accept_h0", "continue"}。
    """
    n = wins + draws + losses
    d_hat = (draws / n) if empirical_draw_rate is None else empirical_draw_rate
    d_hat = min(max(d_hat, 0.0), 0.98)

    def _trinomial(elo: float) -> Tuple[float, float, float]:
        s = _elo_to_score(elo)
        w = max(s - d_hat / 2.0, 1e-6)
        l = max(1.0 - s - d_hat / 2.0, 1e-6)
        d = max(1.0 - w - l, 1e-6)
        # 归一化防漂移
        tot = w + d + l
        return w / tot, d / tot, l / tot

    p0w, p0d, p0l = _trinomial(elo0)
    p1w, p1d, p1l = _trinomial(elo1)
    llr = (wins * math.log(p1w / p0w)
           + draws * math.log(p1d / p0d)
           + losses * math.log(p1l / p0l))
    lower = math.log(alpha / (1.0 - beta))
    upper = math.log((1.0 - alpha) / beta)
    if llr >= upper:
        decision = "accept_h1"
    elif llr <= lower:
        decision = "accept_h0"
    else:
        decision = "continue"
    return {"llr": llr, "lower": lower, "upper": upper, "decision": decision,
            "elo0": elo0, "elo1": elo1}


# ----------------------------------------------------------------- 对局执行

def _score_from_perspective(rec: dict, spec: str) -> Tuple[float, str]:
    """把一条对局记录换算成 spec 引擎视角的 (得分, 终局原因)。"""
    if rec["a"] == spec:
        seat = 0
    elif rec["b"] == spec:
        seat = 1
    else:
        raise ValueError(f"记录中找不到策略 {spec}: a={rec['a']} b={rec['b']}")
    w = rec.get("winner")
    if w is None or w == -1:
        return 0.5, rec.get("reason") or "unknown"
    return (1.0 if w == seat else 0.0), rec.get("reason") or "unknown"


def _run_pair(job) -> List[dict]:
    """子进程任务：同一 seed 跑一配对（A 先手 + B 先手），返回两条记录。"""
    spec_a, spec_b, seed, max_plies, model_a, model_b = job
    from .selfplay import play_game
    cfg = RuleConfig(max_plies=max_plies) if max_plies else RuleConfig()
    r0 = play_game(spec_a, spec_b, seed, cfg,
                   model_path0=model_a, model_path1=model_b)
    r1 = play_game(spec_b, spec_a, seed, cfg,
                   model_path0=model_b, model_path1=model_a)
    return [r0, r1]


def run_gate(spec_a: str, spec_b: str, seeds: List[int],
             workers: int = 1, max_plies: Optional[int] = None,
             model_a: Optional[str] = None, model_b: Optional[str] = None,
             elo0: float = 0.0, elo1: float = 65.0,
             out_dir: Optional[str] = None) -> dict:
    """执行配对门控评测并返回完整报告 dict。

    每个种子跑两局（候选 A 先手 / 后手各一局，同一副牌），
    报告 A 相对 B 的得分率、配对检验、Wilson 区间、SPRT 与终局原因拆分。
    """
    from .selfplay import play_game

    jobs = [(spec_a, spec_b, s, max_plies, model_a, model_b) for s in seeds]
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            chunks = pool.map(_run_pair, jobs)
        records = [r for c in chunks for r in c]
    else:
        records = [r for j in jobs for r in _run_pair(j)]

    per_seed: List[float] = []
    totals = {"wins": 0, "draws": 0, "losses": 0}
    reasons: Dict[str, Dict[str, int]] = {}
    seat_split = {"as_first": [0.0, 0], "as_second": [0.0, 0]}

    for s in seeds:
        pair = [r for r in records if r["seed"] == s]
        if len(pair) != 2:
            continue
        s0, reason0 = _score_from_perspective(pair[0], spec_a)
        s1, reason1 = _score_from_perspective(pair[1], spec_a)
        per_seed.append(s0 + s1)
        for sc, rec, seat_idx in ((s0, pair[0], 0), (s1, pair[1], 1)):
            key = "as_first" if rec["a"] == spec_a else "as_second"
            seat_split[key][0] += sc
            seat_split[key][1] += 1
            bucket = "wins" if sc == 1.0 else ("draws" if sc == 0.5 else "losses")
            totals[bucket] += 1
            reasons.setdefault(rec.get("reason") or "unknown",
                               {"wins": 0, "draws": 0, "losses": 0})
            reasons[rec.get("reason") or "unknown"][bucket] += 1

    n_games = totals["wins"] + totals["draws"] + totals["losses"]
    score_sum = totals["wins"] + 0.5 * totals["draws"]
    score_rate = score_sum / n_games if n_games else 0.0
    wil = wilson_ci(score_sum, n_games)

    decisive = totals["wins"] + totals["losses"]
    dec_rate = totals["wins"] / decisive if decisive else 0.0
    dec_wil = wilson_ci(totals["wins"], decisive)

    paired = paired_z_test(per_seed)
    sprt = sprt_trinomial(totals["wins"], totals["draws"], totals["losses"],
                          elo0=elo0, elo1=elo1)

    # 晋级判据（基线 §7：整体显著改善 + SPRT 不接受 H0）
    promote = bool(n_games > 0 and wil[0] > 0.5 and sprt["decision"] != "accept_h0")

    report = {
        "spec_a": spec_a, "spec_b": spec_b,
        "model_a": model_a, "model_b": model_b,
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "n_seeds": len(per_seed), "n_games": n_games,
        "totals": totals,
        "score_rate": score_rate,
        "score_rate_wilson": wil,
        "decisive_only_win_rate": dec_rate,
        "decisive_only_wilson": dec_wil,
        "paired": paired,
        "sprt": sprt,
        "reason_breakdown": reasons,
        "seat_split": {k: {"score_rate": (v[0] / v[1] if v[1] else 0.0),
                           "games": v[1]} for k, v in seat_split.items()},
        "promote": promote,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"gate_{spec_a}_vs_{spec_b}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return report


def format_gate_report(rep: dict) -> str:
    """渲染人读文本报告（对齐 camp 系列报告风格）。"""
    t = rep["totals"]
    wil = rep["score_rate_wilson"]
    paired = rep["paired"]
    sprt = rep["sprt"]
    lines = [
        "=" * 78,
        f"评测门控：{rep['spec_a']} (候选) vs {rep['spec_b']} (基准)",
        "=" * 78,
        f"  配对场次：{rep['n_seeds']} 组种子 x 先后手互换 = {rep['n_games']} 局",
        f"  总战绩：  胜 {t['wins']} / 和 {t['draws']} / 负 {t['losses']}",
        f"  得分率 (胜1/和0.5/负0)：{rep['score_rate']:.4f}  "
        f"Wilson 95%CI [{wil[0]:.4f}, {wil[1]:.4f}]",
        f"  仅计胜负局胜率：{rep['decisive_only_win_rate']:.4f}  "
        f"Wilson 95%CI [{rep['decisive_only_wilson'][0]:.4f}, "
        f"{rep['decisive_only_wilson'][1]:.4f}]  (n={t['wins'] + t['losses']})",
        "",
        f"  配对检验（每种子先后手合计得分 vs 期望 1.0）：",
        f"    mean={paired['mean']:.4f}  sd={paired['sd']:.4f}  "
        f"se={paired['se']:.4f}  z={paired['z']:+.3f}  p={paired['p_two_sided']:.4f}",
        "",
        f"  SPRT 三元序贯检验 (elo0={sprt.get('elo0', 0.0)}):",
        f"    LLR={sprt['llr']:+.3f}  界 [{sprt['lower']:.3f}, {sprt['upper']:.3f}]"
        f"  判定: {sprt['decision']}",
        "",
        "  终局原因拆分：",
        f"    {'原因':<14}{'胜':>6}{'和':>6}{'负':>6}",
    ]
    for reason, cnt in sorted(rep["reason_breakdown"].items(),
                              key=lambda kv: -sum(kv[1].values())):
        lines.append(f"    {reason:<14}{cnt['wins']:>6}{cnt['draws']:>6}{cnt['losses']:>6}")
    for k, v in rep["seat_split"].items():
        lines.append(f"  座位拆分 {k}: 得分率 {v['score_rate']:.4f} ({v['games']} 局)")
    lines.append("")
    lines.append(f"  晋级判定 promote = {rep['promote']} "
                 f"(判据: 得分率 Wilson 下界 > 0.5 且 SPRT 不接受 H0)")
    lines.append("=" * 78)
    return "\n".join(lines)
