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


def paired_delta_test(per_seed_deltas: List[float]) -> Dict[str, object]:
    """配对 Δ 检验：H0 两引擎等强 ⇒ E[d_i] = 0。

    ``d_i = Δ(r0) - Δ(r1)``，其中 ``Δ = 座位0视角估值 - 座位1视角估值``。
    在 r0 中候选执先，其优势就是 ``Δ(r0)``；在 r1 中候选执后，其优势是 ``-Δ(r1)``，
    故 ``d_i`` 即「候选在本组配对里的净估值优势」，>0 表示候选更强。

    为什么需要它（2026-09-15 实测）：官方口径在高和棋率下几乎无信息
    （200 局仅 16-56 局胜负）；裁决式判分把 Δ 压成 0/0.5/1 只保留**符号**，
    丢掉了幅度。本检验直接使用 Δ 的**幅度与符号**，
    t 统计量对幅度敏感、符号检验对离群值稳健，两者互为补充。

    且因为是**配对差分**，任何「座位 0 的固有优势」被自动消掉
    —— 实测镜像对局（同一模型）下 Δ 的座位偏好很强（先手优势 +90 量级），
    但配对差分后均值 ≈ 0。

    注意：`t` 的正态近似在小样本（n<20）下偏乐观，报告同时给出符号检验 p 值。
    """
    n = len(per_seed_deltas)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "t": 0.0,
                "p_two_sided": 1.0, "pos": 0, "neg": 0, "sign_p": 1.0}
    mean = sum(per_seed_deltas) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in per_seed_deltas) / (n - 1)
        sd = math.sqrt(max(var, 0.0))
        se = sd / math.sqrt(n)
    else:
        sd, se = 0.0, 0.0
    t = (mean / se) if se > 1e-12 else (
        0.0 if abs(mean) < 1e-12 else math.copysign(999.0, mean))
    p_two = math.erfc(abs(t) / math.sqrt(2.0))

    pos = sum(1 for d in per_seed_deltas if d > 0)
    neg = sum(1 for d in per_seed_deltas if d < 0)
    m = pos + neg
    if m == 0:
        sign_p = 1.0
    else:
        # 正态近似 + 连续性校正（0.5）
        z = max((abs(pos - m / 2.0) - 0.5) / math.sqrt(m / 4.0), 0.0)
        sign_p = math.erfc(z / math.sqrt(2.0))
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "p_two_sided": p_two, "pos": pos, "neg": neg, "sign_p": sign_p}


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


def _score_at_seat(rec: dict, seat: int) -> Tuple[float, str]:
    """按**显式座位**把对局记录换算成候选视角的 (得分, 终局原因)。

    P0 审计修复（2026-09-14）：原实现靠 spec 名推断候选座位，当两侧 spec 同名
    （如都用 `nn_mcts_20`、仅权重不同——这是所有正式门控的用法）时，
    `rec["a"] == spec_a` 恒真 → 每对局中"候选执后手"的那一局被**反向计分**，
    把真实差异系统性压回 0.5（历史报告 `as_second: 0 局` 即此征兆）。
    座位由 `_run_pair` 的构造决定，必须显式传入。
    """
    w = rec.get("winner")
    if w is None or w == -1:
        return 0.5, rec.get("reason") or "unknown"
    return (1.0 if w == seat else 0.0), rec.get("reason") or "unknown"


def _run_pair(job) -> List[dict]:
    """子进程任务：同一 seed 跑一配对（A 先手 + B 先手），返回两条记录。

    P0 修复（审查 R3，2026-09-15）：新增 init_json / device 两个字段。
    init_json 非空时**两局共用同一副初始局面**（配对同牌），否则每局各自随机发牌；
    device 由调用方透传，不再写死 "cpu"。
    """
    spec_a, spec_b, seed, max_plies, model_a, model_b, init_json, device = job
    if isinstance(init_json, (list, tuple)):
        raise ValueError("init_json 必须是单个初始局面（JSON 字符串），"
                         "每个 seed 的局面由 run_gate 的 init_states 按位给出")

    from .selfplay import play_game
    cfg = RuleConfig(max_plies=max_plies) if max_plies else RuleConfig()
    init = GameState.from_json(init_json) if init_json else None
    kw = {"device": device}
    r0 = play_game(spec_a, spec_b, seed, cfg,
                   model_path0=model_a, model_path1=model_b,
                   init_state=init, **kw)
    # 同一副牌重来：从同一初始局面重建，避免上一局就地改动了它的状态
    init2 = GameState.from_json(init_json) if init_json else None
    r1 = play_game(spec_b, spec_a, seed, cfg,
                   model_path0=model_b, model_path1=model_a,
                   init_state=init2, **kw)
    return [r0, r1]


def run_gate(spec_a: str, spec_b: str, seeds: List[int],
             workers: int = 1, max_plies: Optional[int] = None,
             model_a: Optional[str] = None, model_b: Optional[str] = None,
             elo0: float = 0.0, elo1: float = 65.0,
             out_dir: Optional[str] = None,
             out_name: Optional[str] = None,
             adjudicate_margin: float = 0.0,
             init_states: Optional[List[Optional[str]]] = None,
             device: str = "cpu") -> dict:
    """执行配对门控评测并返回完整报告 dict。

    每个种子跑两局（候选 A 先手 / 后手各一局，同一副牌），
    报告 A 相对 B 的得分率、配对检验、Wilson 区间、SPRT 与终局原因拆分。

    init_states（P0 修复 R3）：与 seeds **等长**，给出每个种子要用的初始局面
    （`GameState.to_json()` 字符串）。非空时两局共用同一副初始局面，实现真正的
    "配对同牌"；该项为 None 时按 seed 各自随机发牌（原行为）。
    device：透传给对局执行层，默认 "cpu"；训练主循环可传 CUDA 设备。
    """
    from .selfplay import play_game

    init_states = list(init_states) if init_states is not None else [None] * len(seeds)
    if len(init_states) != len(seeds):
        raise ValueError(f"init_states 长度 {len(init_states)} 与 seeds "
                         f"{len(seeds)} 不一致（配对同牌要求一一对应）")

    jobs = [(spec_a, spec_b, s, max_plies, model_a, model_b, js, device)
            for s, js in zip(seeds, init_states)]
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            chunks = pool.map(_run_pair, jobs)
        records = [r for c in chunks for r in c]
    else:
        records = [r for j in jobs for r in _run_pair(j)]

    per_seed: List[float] = []
    adj_per_seed: List[float] = []
    adj_totals = {"wins": 0, "draws": 0, "losses": 0}
    adj_margins: List[float] = []
    delta_per_seed: List[float] = []      # d_i = Δ(r0) - Δ(r1)，配对 Δ 检验用
    totals = {"wins": 0, "draws": 0, "losses": 0}
    reasons: Dict[str, Dict[str, int]] = {}
    seat_split = {"as_first": [0.0, 0], "as_second": [0.0, 0]}

    for s in seeds:
        pair = [r for r in records if r["seed"] == s]
        if len(pair) != 2:
            continue
        # _run_pair 保证 pair[0] 为"候选执先"，pair[1] 为"候选执后"（同牌先后手互换）
        s0, reason0 = _score_at_seat(pair[0], 0)
        s1, reason1 = _score_at_seat(pair[1], 1)
        per_seed.append(s0 + s1)
        a0 = adjudicate_record(pair[0], spec_a, margin=adjudicate_margin, seat_a=0)
        a1 = adjudicate_record(pair[1], spec_a, margin=adjudicate_margin, seat_a=1)
        adj_per_seed.append(a0 + a1)
        if pair[0].get("final_eval_sym0") is not None and pair[0].get("final_eval_sym1") is not None:
            adj_margins.append(abs(pair[0]["final_eval_sym0"] - pair[0]["final_eval_sym1"]))
        elif pair[0].get("final_eval0") is not None and pair[0].get("final_eval1") is not None:
            adj_margins.append(abs(pair[0]["final_eval0"] - pair[0]["final_eval1"]))
        d0 = record_eval_delta(pair[0])
        d1 = record_eval_delta(pair[1])
        if d0 is not None and d1 is not None:
            delta_per_seed.append(d0 - d1)
        for sc_a in (a0, a1):
            adj_totals["wins" if sc_a == 1.0 else ("draws" if sc_a == 0.5 else "losses")] += 1
        for sc, rec, seat_idx in ((s0, pair[0], 0), (s1, pair[1], 1)):
            key = "as_first" if seat_idx == 0 else "as_second"
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

    # 裁决式判分（测量用，不参与 promote 判定）：把和棋局按终局专家估值判出胜负，
    # 使有效样本从"决胜负局"扩大到全部对局（高和棋率下唯一有分辨力的口径）。
    adj_n = sum(adj_totals.values())
    adj_sum = adj_totals["wins"] + 0.5 * adj_totals["draws"]
    adj_rate = (adj_sum / adj_n) if adj_n else 0.0
    adj_wil = wilson_ci(adj_sum, adj_n)
    adj_paired = paired_z_test(adj_per_seed) if adj_per_seed else None
    # 估值来源标注：新记录用镜像对称化口径（消座位标签偏差），旧报告回退原口径。
    adj_source = "symmetric" if any(
        r.get("final_eval_sym0") is not None and r.get("final_eval_sym1") is not None
        for r in records) else "raw"
    adjudicated = {
        "totals": dict(adj_totals), "n_games": adj_n,
        "score_rate": adj_rate, "score_rate_wilson": adj_wil,
        "paired": adj_paired,
        "eval_source": adj_source,
        "median_abs_eval_gap": (sorted(adj_margins)[len(adj_margins) // 2]
                                if adj_margins else None),
        "margin": adjudicate_margin,
    }

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
        "adjudicated": adjudicated,
        "paired_delta": paired_delta_test(delta_per_seed),
        "reason_breakdown": reasons,
        "seat_split": {k: {"score_rate": (v[0] / v[1] if v[1] else 0.0),
                           "games": v[1]} for k, v in seat_split.items()},
        "promote": promote,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # out_name：避免同 spec 的多次门控互相覆盖（例如 nn_mcts_20 对 nn_mcts_20）
        fname = out_name or f"gate_{spec_a}_vs_{spec_b}"
        with open(os.path.join(out_dir, f"{fname}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return report


def adjudicate_record(rec: dict, spec_a: str, margin: float = 0.0,
                      seat_a: Optional[int] = None,
                      use_symmetric: bool = True) -> float:
    """A 视角的裁决得分（1/0.5/0）：有胜负按胜负；和棋按终局专家估值裁决。

    背景（2026-09-14 诊断）：同源模型间 72-92% 对局判和，专家对专家也 88% 和棋；
    官方得分率判据在如此高和棋率下分辨力极低（200 局仅 16-56 局胜负样本）。
    裁决式判分把全部对局变为有信息的样本（n=200），用于**测量**强度差异。

    估值来源（2026-09-15 修复）：默认 `use_symmetric=True`，优先读
    `final_eval_sym0 / final_eval_sym1`（镜像对称化口径，见
    `eval_expert.evaluate_expert_dual`）。原因：原口径 `final_eval0/1` 含
    **按座位标签的加性偏差**，在零阈值下被放大成系统性符号偏置 —— 镜像自对局
    （应 ≈0.5）实测裁决得分率 0.600，反而不低于已知更强方的 0.575，使该口径
    无法分辨真实强度差。对称化后该偏差被严格抵消。
    记录中缺少对称化字段（旧报告）时自动回退到原口径。

    margin：估值差小于该值算和棋（默认 0 = 任意差即判，估值为子力分数量纲）。
    仅评测口径，不改变规则定义与训练奖励。**不参与 promote 判定。**
    """
    if seat_a is None:                    # 未显式给出时才回退到 spec 名推断（同名会歧义）
        seat_a = 0 if rec.get("a") == spec_a else 1
    w = rec.get("winner")
    if w is not None and w != -1:
        return 1.0 if w == seat_a else 0.0
    ev0, ev1 = rec.get("final_eval0"), rec.get("final_eval1")
    if use_symmetric:
        s0, s1 = rec.get("final_eval_sym0"), rec.get("final_eval_sym1")
        if s0 is not None and s1 is not None:
            ev0, ev1 = s0, s1
    if ev0 is None or ev1 is None:
        return 0.5
    ev_a, ev_b = (ev0, ev1) if seat_a == 0 else (ev1, ev0)
    if abs(ev_a - ev_b) <= margin:
        return 0.5
    return 1.0 if ev_a > ev_b else 0.0


def record_eval_delta(rec: dict) -> Optional[float]:
    """从对局记录取「座位 0 视角 − 座位 1 视角」的估值差 Δ。

    优先用镜像对称化字段（final_eval_sym0/1），缺失时回退原口径。
    两者都缺（估值失败）返回 None。供 `paired_delta_test` 使用。
    """
    e0, e1 = rec.get("final_eval_sym0"), rec.get("final_eval_sym1")
    if e0 is None or e1 is None:
        e0, e1 = rec.get("final_eval0"), rec.get("final_eval1")
    if e0 is None or e1 is None:
        return None
    return e0 - e1


def promote_candidate(report: dict, model_a: Optional[str],
                      best_path: str = os.path.join("models", "best.pt"),
                      backup_dir: Optional[str] = None) -> Tuple[bool, str]:
    """正式晋级协议：仅当门控报告 promote=True 时把候选写入 best_path。

    这是**唯一**允许改动发布模型的入口（AGENTS.md：不得只因训练 loss 下降覆盖 best.pt）。
    旧模型先带时间戳备份；未通过则完全不触碰 best.pt。
    返回 (是否已更新, 说明文本)。
    """
    import shutil
    import time as _time
    if not report.get("promote") or not model_a:
        return False, (f"未晋级（promote={report.get('promote')}，"
                       f"得分率={report.get('score_rate', 0.0):.4f}），"
                       f"best.pt 未变更")
    if not os.path.exists(model_a):
        return False, f"候选权重不存在: {model_a}"
    backup_dir = backup_dir or os.path.dirname(best_path) or "."
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists(best_path):
        bak = os.path.join(backup_dir,
                           f"best_legacy_{_time.strftime('%Y%m%d_%H%M%S')}.pt")
        shutil.copyfile(best_path, bak)
    else:
        bak = None
    shutil.copyfile(model_a, best_path)
    msg = (f"✅ 正式门控通过（得分率={report.get('score_rate', 0.0):.4f}，"
           f"SPRT={report.get('sprt', {}).get('decision')}）→ 已更新 {best_path}")
    if bak:
        msg += f"；旧模型备份: {bak}"
    return True, msg


def format_gate_report(rep: dict) -> str:
    """渲染人读文本报告（对齐 camp 系列报告风格）。"""
    t = rep["totals"]
    adj = rep.get("adjudicated") or {}
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
    if adj:
        aw = adj["score_rate_wilson"]
        lines.append(f"  裁决式判分（和棋局按终局专家估值判，测量用/不参与 promote；"
                     f"估值来源={adj.get('eval_source', 'raw')}）：")
        lines.append(f"    战绩 胜 {adj['totals']['wins']} / 和 {adj['totals']['draws']} / "
                     f"负 {adj['totals']['losses']}  （n={adj['n_games']}）")
        lines.append(f"    裁决得分率 {adj['score_rate']:.4f}  Wilson 95%CI "
                     f"[{aw[0]:.4f}, {aw[1]:.4f}]  半宽 ±{(aw[1]-aw[0])/2:.4f}")
        if adj.get("median_abs_eval_gap") is not None:
            lines.append(f"    终局估值差中位 {adj['median_abs_eval_gap']:.1f}"
                         f"（margin={adj['margin']}）")
    lines.append(f"  晋级判定 promote = {rep['promote']} "
                 f"(判据: 得分率 Wilson 下界 > 0.5 且 SPRT 不接受 H0)")
    pd = rep.get("paired_delta")
    if pd and pd.get("n"):
        lines.append("")
        lines.append("  配对 Δ 检验（保留估值差幅度；配对差分自动消掉座位固有优势，"
                     "测量用/不参与 promote）：")
        lines.append(f"    d = Δ(r0) - Δ(r1)   n={pd['n']}  mean={pd['mean']:+.3f}  "
                     f"sd={pd['sd']:.3f}  se={pd['se']:.3f}")
        lines.append(f"    t={pd['t']:+.3f}  p(双侧)={pd['p_two_sided']:.4f}   "
                     f"符号 正{pd['pos']}/负{pd['neg']}  符号检验 p={pd['sign_p']:.4f}")
    lines.append("=" * 78)
    return "\n".join(lines)
