"""P2 验收第 4 条审计：非法 / 送旗 / 无意义循环比例（全部带分母）。

背景
----
`AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §5 P2 验收第 4 条要求
"GUI 中出现明显非法、送旗、无意义循环的比例为零或接近零"。
此前工程里**只有** `eval_bc.py` 输出的一句 "非法动作预测率 0.00%"，
而且是硬编码基线表里的一行、没有分母、也没覆盖送旗与无意义循环。
本脚本补上这条验收的可复算证据。

口径定义（每条都写清分子与分母，避免事后解释）
--------------------------------------------
- 分母 `decisions`：双方**实际做出选择**的次数（= 各局 plies 之和）；
  另有局级分母 `games`。所有比率同时给 per-100-decisions 与 per-game 两个版本。

1) 非法 (illegal)
   - `illegal_action`：策略返回的动作不在当前 `legal_actions()` 中（或返回 None）。
     命中时用首个合法动作兜底以便对局继续，但该次**仍然计数**。
   - `illegal_transition`：`apply()` 之后状态不变量被破坏 ——
     `ply` 未恰好 +1、或 `len(board)+len(dead)` 不守恒（棋子凭空产生/消失）、
     或出现越界坐标。这两项按"零容忍"判定（必须恰好为 0）。

2) 送旗 (flag_gift)
   军旗 `Rank.QI` 不可移动（`GameState.legal_actions` 明确排除 LEI/QI），
   因此"送旗"只能表现为**自己走出一手，制造出对方一步可吃己方军旗的窗口**：
       pre  = 走子前，对方已经能一步吃己方军旗
       post = 走子后，对方能一步吃己方军旗
   仅当 `(not pre) and post` 时计一次 —— 即这个窗口是**本方这一手新造出来的**。
   与"最后是否真的被吃"解耦，避免幸存者偏差（没被惩罚就不算送）。
   门控判定复用 `legal_actions()`（唯一真源），不在本脚本里重写
   `flag_gong_only` / `flag_needs_mines_cleared` / 明子不可攻击等规则。
   本规则集 `flag_gong_only=True` ⇒ 制造出的窗口必然是对方工兵。
   - `flag_loss_games`：终局 `reason == "flag"`（军旗被吃）的局数。
   - `flag_loss_by_gift_games`：其中**输方最后一手正是送旗**的局数。

3) 无意义循环 (meaningless_loop)
   - `end_repetition`：相同局面重复达 `repetition_draw_count` 判和。
   - `end_max_plies` / `end_no_capture`：触顶 1000 手 / 连续 70 手无吃子判和。
   - `ping_pong`：同座位 A->B->A->B 往复踱步（复用 `scripts/mine_blunders.py` 探针 4）。
   - `turtling`：行营龟缩连续 6 手拒翻（复用探针 5）。

附带（同一分母下给出，作为"明显战术错误"的旁证）
------------------------------------------------
复用 `mine_blunders` 探针 1-3：`camp_abandonment`（弃营送死/丢营）、
`top_piece_suicide`（大子白送/炸弹贱卖）、`sapper_suicide`（工兵白送）。

判定
----
`AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 没有量化"接近零"，本脚本给出**显式操作化定义**，
并与项目既有传统强引擎（默认 `expert2`）在同一起始局面上做**对照**：

- `illegal_*` 必须恰好为 0（硬性，任何 >0 直接 FAIL）；
- 送旗 / 循环类软指标与对照做 **Wilson 非劣性判定**（与 `junqi.eval_gate` 同一统计真源）：
  判 FAIL 需**同时**"候选 Wilson 下界 > 对照 Wilson 上界"（统计上确实更差）
  与"点估计 > 对照 × `--tolerance`"（幅度上确实更差），
  避免 60 局量级的噪声被判成失败；
- 每 100 决策口径另加绝对上限 `--abs-cap`（"接近零"的实操门槛），
  局占比口径不加（判和率由开局集合与规则决定，绝对阈值只会造成假失败）；
- 失旗终局（`flag_loss_games`）本身不是错误（是正常胜负方式），
  只用 **`flag_loss_by_gift_games / games`**（输方最后一手正是自己造的窗口）作为局级证据。

用法
----
    python scripts/audit_p2_game_quality.py --games 60 \
        --a hybrid2 --b search2 --model-a models/bc_best.pt \
        --compare-a expert2 --init-set eval_sets/endgame.jsonl

`--init-set none`（或 `--random-deal`）改用每局随机发牌，对应 GUI 从开局下起的情形。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from junqi.config import RuleConfig                       # noqa: E402
from junqi.eval_gate import wilson_ci                     # noqa: E402  (统计真源)
from junqi.rules import COLS, ROWS, Rank                  # noqa: E402
from junqi.selfplay import make_strategy                  # noqa: E402
from junqi.state import GameState, deal, position_key     # noqa: E402
from mine_blunders import BlunderDetector                 # noqa: E402  (复用已固化探针)

# 软指标：(显示名, 分子在 metrics 中的键, 分母在 metrics 中的键, 是否适用绝对上限)
# 每 100 决策口径适用绝对上限（"接近零"）；局占比口径不适用（见 judge 文档）。
_SOFT_METRICS = (
    ("送旗（自造一步可吃旗窗口，每100决策）", "flag_gift", "decisions", True),
    ("往复踱步事件段（探针，每100决策）", "probe_ping_pong", "decisions", True),
    ("龟缩拒翻事件段（探针，每100决策）", "probe_turtling", "decisions", True),
    ("送旗致失旗（局占比）", "flag_loss_by_gift_games", "games", False),
    ("无意义循环：重复判和+触顶（局占比）", "loop_soft_games", "games", False),
)


# ----------------------------------------------------------------- 送旗探针

def seat_can_take_flag(st: GameState, seat: int, flag_color: str) -> bool:
    """在局面 `st` 下，`seat` 一方是否存在"一步吃掉 flag_color 军旗"的合法动作。

    `legal_actions()` 只依赖 `turn`（经 `my_color()`）、`board`、`dead`、
    `first_flip_done` 与 `cfg`，与其历史无关；因此当 `st.turn != seat` 时，
    把 `turn` 换到 `seat` 后再调用 `legal_actions()`，得到的正是"对方在这一局面下的
    合法动作集"。这样做是为了**复用 `legal_actions` 作为规则唯一真源**，而不是在本脚本里
    重写 `flag_gong_only` / `flag_needs_mines_cleared` / 暗子不可攻击 / 行营免疫等门控。
    """
    if st.turn == seat:
        s = st
    else:
        s = st.copy()
        s.turn = seat
        s._rem_cache = None
    for a in s.legal_actions():
        if a.kind != "move":
            continue
        pc = s.board.get(a.to)
        if pc is not None and pc.rank == Rank.QI and pc.color == flag_color:
            return True
    return False


def check_invariants(before: GameState, after: GameState) -> Optional[str]:
    """状态转移不变量；违反时返回原因字符串。"""
    if after.ply != before.ply + 1:
        return f"ply {before.ply} -> {after.ply}（应恰好 +1）"
    n_before = len(before.board) + len(before.dead)
    n_after = len(after.board) + len(after.dead)
    if n_after != n_before:
        return f"子力总数不守恒 {n_before} -> {n_after}"
    for (r, c), pc in after.board.items():
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return f"越界坐标 {(r, c)}"
        if pc.color not in ("r", "b"):
            return f"非法颜色 {pc.color!r} @ {(r, c)}"
    for pc in after.dead:
        if pc.color not in ("r", "b"):
            return f"阵亡子非法颜色 {pc.color!r}"
    return None


# ----------------------------------------------------------------- 单局执行

def _episodes(plies: List[int]) -> int:
    """把命中 ply 列表折叠成「事件段」数：连续的 ply 视为同一次事件。

    为什么必须折叠：往复踱步等探针在其持续期间**每一手**都会命中，
    直接按 ply 计数会把一个长循环放大成几十次"失误"（冒烟实测原始命中
    是事件段数的数倍）。折叠后才是"发生了多少次"。
    """
    if not plies:
        return 0
    ps = sorted(plies)
    n = 1
    for prev, cur in zip(ps, ps[1:]):
        if cur - prev > 1:
            n += 1
    return n


def run_one_game(cfg: RuleConfig, detector: BlunderDetector,
                 spec_a: str, spec_b: str,
                 model_a: Optional[str], model_b: Optional[str],
                 seed: int, init: Optional[GameState],
                 device: str) -> Dict[str, Any]:
    """跑一局并返回 (局级计数, 旁证探针计数, 明细)。"""
    rng = random.Random(seed * 7919 + 13)
    s0 = make_strategy(spec_a, seed=seed * 31 + 1, model_path=model_a, device=device)
    s1 = make_strategy(spec_b, seed=seed * 31 + 2, model_path=model_b, device=device)

    st = init.copy() if init is not None else deal(random.Random(seed), cfg)
    seen = Counter()
    history: List[dict] = []

    g = Counter()
    probe = Counter()
    details: List[dict] = []

    illegal_samples: List[dict] = []
    transition_samples: List[dict] = []
    gift_samples: List[dict] = []

    last_move_ply = {0: None, 1: None}
    last_gift_ply = {0: None, 1: None}

    while not st.is_terminal():
        pk = position_key(st)
        seen[pk] += 1
        if seen[pk] >= cfg.repetition_draw_count:
            st.winner, st.win_reason = -1, "repetition"
            g["end_repetition"] = 1
            break

        acts = st.legal_actions()
        if not acts:
            st.winner, st.win_reason = 1 - st.turn, "immobilized"
            g["end_immobilized"] = 1
            break

        avoid = {k for k, n in seen.items() if n >= cfg.repetition_draw_count - 1}
        seat = st.turn
        strategy = s0 if seat == 0 else s1
        act = strategy.choose(st, rng, avoid=avoid, history_counts=seen)

        g["decisions"] += 1

        # ---- 指标 1a：非法动作 ----
        if act is None or act not in acts:
            g["illegal_action"] += 1
            if len(illegal_samples) < 20:
                illegal_samples.append({"seed": seed, "ply": st.ply, "seat": seat,
                                        "action": str(act)})
            act = acts[0]

        # ---- 送旗判定（需要走子前后两个局面）----
        my_flag_color = st.my_color(seat)
        pre_gift = (my_flag_color is not None
                    and seat_can_take_flag(st, 1 - seat, my_flag_color))

        mover = st.board.get(act.frm) if act.kind == "move" else None
        target = st.board.get(act.to) if act.kind == "move" else None
        history.append({
            "ply": st.ply, "seat": seat, "action": act, "action_str": str(act),
            "mover": mover, "target": target, "state": st.copy(),
        })

        nxt = st.apply(act)
        last_move_ply[seat] = st.ply

        # ---- 指标 1b：状态转移不变量 ----
        bad = check_invariants(st, nxt)
        if bad is not None:
            g["illegal_transition"] += 1
            if len(transition_samples) < 20:
                transition_samples.append({"seed": seed, "ply": st.ply, "seat": seat,
                                           "action": str(act), "reason": bad})

        # ---- 指标 2：送旗 ----
        if not nxt.is_terminal() and my_flag_color is not None:
            post_gift = seat_can_take_flag(nxt, 1 - seat, my_flag_color)
            if post_gift:
                g["flag_window_post"] += 1
            if pre_gift:
                g["flag_window_pre"] += 1
            if (not pre_gift) and post_gift:
                g["flag_gift"] += 1
                last_gift_ply[seat] = st.ply
                if len(gift_samples) < 20:
                    gift_samples.append({"seed": seed, "ply": st.ply, "seat": seat,
                                         "action": str(act)})

        st = nxt

    g["plies"] = st.ply
    g["winner"] = -1 if st.winner is None else st.winner
    reason = st.win_reason or "unknown"
    g[f"end_{reason}"] = 1

    # ---- 指标 2 的局级证据：旗被吃，且输方最后一手正是自己制造的窗口 ----
    if reason == "flag" and st.winner in (0, 1):
        loser = 1 - st.winner
        g["flag_loss"] = 1
        if last_gift_ply[loser] is not None and last_gift_ply[loser] == last_move_ply[loser]:
            g["flag_loss_by_gift"] = 1

    # ---- 旁证探针（复用 mine_blunders 的 5 个探针）----
    probe_hits: Dict[str, List[int]] = {}
    for p_idx in range(len(history)):
        for probe_fn, key in ((detector.check_camp_abandonment, "camp_abandonment"),
                              (detector.check_top_piece_suicide, "top_piece_suicide"),
                              (detector.check_sapper_suicide, "sapper_suicide"),
                              (detector.check_ping_pong_loop, "ping_pong"),
                              (detector.check_camp_turtling, "turtling")):
            try:
                if probe_fn(p_idx, history):
                    probe_hits.setdefault(key, []).append(history[p_idx]["ply"])
            except Exception:  # noqa: BLE001  探针异常不得中断审计
                probe["probe_error"] += 1

    probe_episodes = Counter({k: _episodes(v) for k, v in probe_hits.items()})
    probe_hits_counter = Counter({k: len(v) for k, v in probe_hits.items()})

    return {"counts": g, "probe": probe_episodes, "probe_hits": probe_hits_counter,
            "reason": reason,
            "illegal_samples": illegal_samples,
            "transition_samples": transition_samples,
            "gift_samples": gift_samples,
            "plies": st.ply}


# ----------------------------------------------------------------- 汇总

def _per100(x: int, decisions: int) -> float:
    return (100.0 * x / decisions) if decisions else 0.0


def _ratio(x: int, n: int) -> float:
    return (x / n) if n else 0.0


def summarize(label: str, runs: List[dict], tolerance: float,
              abs_cap: float) -> Dict[str, Any]:
    keys = ("decisions", "illegal_action", "illegal_transition", "flag_gift",
            "flag_window_pre", "flag_window_post",
            "flag_loss", "flag_loss_by_gift", "end_repetition", "end_no_capture",
            "end_max_plies", "end_immobilized", "end_flag")
    tot = Counter()
    probe_tot = Counter()
    probe_hits_tot = Counter()
    for r in runs:
        for k in keys:
            tot[k] += r["counts"].get(k, 0)
        for k, v in r["probe"].items():
            probe_tot[k] += v
        for k, v in r.get("probe_hits", {}).items():
            probe_hits_tot[k] += v

    games = len(runs)
    decisions = tot["decisions"]
    gifts = tot["flag_gift"]

    metrics = {
        "games": games,
        "decisions": decisions,
        "avg_plies": (statistics.fmean([r["plies"] for r in runs]) if runs else 0.0),
        "illegal_action": tot["illegal_action"],
        "illegal_action_per100": _per100(tot["illegal_action"], decisions),
        "illegal_transition": tot["illegal_transition"],
        "illegal_transition_per100": _per100(tot["illegal_transition"], decisions),
        "flag_gift": gifts,
        "flag_gift_per100": _per100(gifts, decisions),
        "flag_window_post": tot["flag_window_post"],
        "flag_window_post_per100": _per100(tot["flag_window_post"], decisions),
        "flag_window_pre": tot["flag_window_pre"],
        "flag_window_pre_per100": _per100(tot["flag_window_pre"], decisions),
        "flag_gift_games": sum(1 for r in runs if r["counts"].get("flag_gift", 0) > 0),
        "flag_gift_game_rate": _ratio(
            sum(1 for r in runs if r["counts"].get("flag_gift", 0) > 0), games),
        "flag_loss_games": tot["flag_loss"],
        "flag_loss_by_gift_games": tot["flag_loss_by_gift"],
        "flag_loss_by_gift_game_rate": _ratio(tot["flag_loss_by_gift"], games),
        "end_repetition_games": tot["end_repetition"],
        "end_repetition_game_rate": _ratio(tot["end_repetition"], games),
        "end_no_capture_games": tot["end_no_capture"],
        "end_max_plies_games": tot["end_max_plies"],
        "end_immobilized_games": tot["end_immobilized"],
        "end_flag_games": tot["end_flag"],
        "loop_soft_games": tot["end_repetition"] + tot["end_max_plies"],
        "loop_soft_game_rate": _ratio(
            tot["end_repetition"] + tot["end_max_plies"], games),
        "probe_ping_pong": probe_tot["ping_pong"],
        "probe_ping_pong_per100": _per100(probe_tot["ping_pong"], decisions),
        "probe_turtling": probe_tot["turtling"],
        "probe_turtling_per100": _per100(probe_tot["turtling"], decisions),
        "probe_camp_abandonment": probe_tot["camp_abandonment"],
        "probe_camp_abandonment_per100": _per100(probe_tot["camp_abandonment"], decisions),
        "probe_top_piece_suicide": probe_tot["top_piece_suicide"],
        "probe_top_piece_suicide_per100": _per100(probe_tot["top_piece_suicide"], decisions),
        "probe_sapper_suicide": probe_tot["sapper_suicide"],
        "probe_sapper_suicide_per100": _per100(probe_tot["sapper_suicide"], decisions),
        "probe_raw_hits": dict(probe_hits_tot),
        "probe_error": probe_tot["probe_error"],
    }
    return {"label": label, "metrics": metrics,
            "illegal_samples": [s for r in runs for s in r["illegal_samples"]][:20],
            "transition_samples": [s for r in runs for s in r["transition_samples"]][:20],
            "gift_samples": [s for r in runs for s in r["gift_samples"]][:20]}


def judge(cand: Dict[str, Any], base: Optional[Dict[str, Any]],
          tolerance: float, abs_cap: float) -> Dict[str, Any]:
    """按显式操作化定义给出 PASS/FAIL。

    判据分两档，避免把不同量纲混用同一条阈值：

    - **硬性**：非法动作 / 非法状态转移必须恰好为 0（`abs_cap` 与统计检验都不参与）。
    - **送旗 / 循环类（软指标）**：与对照配置做 **Wilson 非劣性判定**，
      与 `junqi.eval_gate` 同一真源、同一统计纪律。判为 FAIL 需同时满足：
        1. `候选 Wilson 下界 > 对照 Wilson 上界`（95% 水平上确实更差），且
        2. 点估计 `> 对照点估计 × --tolerance`（幅度上也确实更差）。
      条件 1 保证不被 60 局量级的噪声判假失败（长期记忆：分辨 0.05 得分率差需 ~1500 局）；
      条件 2 保证"两边都很脏"时不会因为不显著而放过。
      对照点估计为 0 时跳过条件 2（`0 × 任何倍数还是 0`，会造成"出现一次即失败"的假失败），
      此时只靠条件 1 与绝对上限。
    - 每 100 决策口径的软指标另加**绝对上限** `--abs-cap`
      （"接近零"的实操门槛）；局占比口径不加绝对上限 ——
      判和局占比由开局集合、`max_plies` 与规则决定，在残局靶场上
      expert 级引擎自身就有两位数的判和率，绝对阈值只会制造假失败。
    """
    cm = cand["metrics"]
    items: List[Dict[str, Any]] = []

    items.append({
        "criterion": "非法动作率 = 0（硬性）",
        "value": f"{cm['illegal_action_per100']:.4f}/100 决策"
                 f"（{cm['illegal_action']} 次）",
        "pass": cm["illegal_action"] == 0,
        "hard": True,
    })
    items.append({
        "criterion": "非法状态转移率 = 0（硬性）",
        "value": f"{cm['illegal_transition_per100']:.4f}/100 决策"
                 f"（{cm['illegal_transition']} 次）",
        "pass": cm["illegal_transition"] == 0,
        "hard": True,
    })

    for name, k_key, n_key, use_abs_cap in _SOFT_METRICS:
        kc, nc = cm[k_key], cm[n_key]
        scale = 100.0 if n_key == "decisions" else 1.0
        unit = "/100 决策" if n_key == "decisions" else "（局占比）"
        pc = (kc / nc) if nc else 0.0
        cand_lo, cand_hi = wilson_ci(kc, nc)
        value = f"{scale*pc:.4f}{unit}（{kc}/{nc}）"

        if base is None:
            if not use_abs_cap:
                items.append({"criterion": name, "value": value,
                              "reference": "（无对照配置，仅报告不判定）",
                              "pass": True, "hard": False, "informational": True})
                continue
            ok = scale * pc <= abs_cap
            ref = (f"绝对上限 ≤ {abs_cap:g}/100 决策；"
                   f"候选 Wilson 95%CI [{scale*cand_lo:.3f}, {scale*cand_hi:.3f}]")
            items.append({"criterion": name, "value": value, "reference": ref,
                          "pass": bool(ok), "hard": False})
            continue

        bm = base["metrics"]
        kb, nb = bm[k_key], bm[n_key]
        pb = (kb / nb) if nb else 0.0
        base_lo, base_hi = wilson_ci(kb, nb)
        sig_worse = cand_lo > base_hi
        magnitude_worse = (pb > 0.0) and (pc > pb * tolerance)
        excess = use_abs_cap and (scale * pc > abs_cap)
        ok = not (sig_worse and magnitude_worse) and not excess

        ref = (f"对照 {scale*pb:.4f}、Wilson [{scale*base_lo:.4f}, {scale*base_hi:.4f}]；"
               f"候选 Wilson [{scale*cand_lo:.4f}, {scale*cand_hi:.4f}]；"
               f"统计更差={'是' if sig_worse else '否'}")
        if pb > 0.0:
            ref += f"、幅度更差({tolerance:g}x)={'是' if magnitude_worse else '否'}"
        if use_abs_cap:
            ref += f"；绝对上限 ≤ {abs_cap:g}/100{'（超限）' if excess else ''}"
        items.append({"criterion": name, "value": value, "reference": ref,
                      "pass": bool(ok), "hard": False})

    return {"items": items,
            "passed": all(i["pass"] for i in items),
            "hard_failed": any(not i["pass"] and i["hard"] for i in items)}


# ----------------------------------------------------------------- 报告渲染

def render_md(rep: Dict[str, Any]) -> str:
    ts = rep["generated_at"]
    lines = [
        "# P2 验收第 4 条审计：非法 / 送旗 / 无意义循环比例",
        "",
        f"- 生成时间：{ts}",
        f"- 起始局面：{rep['init_desc']}",
        f"- 判定容差：tolerance={rep['tolerance']}，绝对上限 {rep['abs_cap']}/100 决策",
        "",
        "> 口径说明：分母 `decisions` = 双方实际做选择的次数（各局 plies 之和）。",
        "> 送旗定义为「本方这一手**新造出**对方一步可吃己方军旗的窗口」，与最后是否真被吃解耦。",
        "> 复现：`" + rep["repro"] + "`",
        "",
        "## 一、逐项判定",
        "",
        "| 判据 | 候选值 | 参照 | 结论 |",
        "|---|---|---|---|",
    ]
    for item in rep["verdict"]["items"]:
        mark = ("ℹ️ 仅报告" if item.get("informational")
                else ("✅ PASS" if item["pass"] else "❌ FAIL"))
        lines.append(f"| {item['criterion']} | {item['value']} | "
                     f"{item.get('reference', '必须为 0')} | {mark} |")
    lines += ["", f"**总判定：{'✅ 通过' if rep['verdict']['passed'] else '❌ 未通过'}**", ""]

    lines += ["## 二、配置对照", "",
              "| 配置 | 局数 | 平均手数 | 非法动作 | 非法转移 | 送旗/100 | 送旗致失旗(局) | "
              "重复判和(局) | 触顶(局) | 往复踱步/100 | 龟缩/100 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in rep["configs"]:
        m = cfg["metrics"]
        lines.append(
            f"| {cfg['label']} | {m['games']} | {m['avg_plies']:.0f} | "
            f"{m['illegal_action']} | {m['illegal_transition']} | "
            f"{m['flag_gift_per100']:.3f} | {m['flag_loss_by_gift_games']} | "
            f"{m['end_repetition_games']} | {m['end_max_plies_games']} | "
            f"{m['probe_ping_pong_per100']:.3f} | {m['probe_turtling_per100']:.3f} |")
    lines.append("")

    lines += ["### 送旗指标的有效性检查", "",
              "「送旗」只有在**对方本来就存在一步吃旗的可能**时才有意义，",
              "而本规则集 `flag_needs_mines_cleared=True`、`flag_gong_only=True`",
              "（须先挖掉对方 3 颗地雷，且只有工兵能吃旗），窗口可能整局不出现。",
              "因此必须同时报告窗口出现次数，否则 0 次送旗只是「指标失效」而非「棋风干净」。",
              "",
              "| 配置 | 走子前已有窗口(每100决策) | 走子后存在窗口(每100决策) | 其中由本方新造(=送旗) |",
              "|---|---|---|---|"]
    for cfg in rep["configs"]:
        m = cfg["metrics"]
        lines.append(f"| {cfg['label']} | {m['flag_window_pre_per100']:.3f} | "
                     f"{m['flag_window_post_per100']:.3f} | {m['flag_gift']} |")
    lines.append("")
    if all(c["metrics"]["flag_window_post"] == 0 for c in rep["configs"]):
        lines += ["> ⚠️ 所有配置的可吃旗窗口都是 **0**：本靶场上「送旗」的分子与分母同时为零，",
                  "> 该指标**无分辨力**，只能作为「未见明显送旗」的弱证据，",
                  "> 不能据此宣称「已证明不送旗」。要真正验证需专门构造**对方地雷已挖完**的残局局面。",
                  ""]

    lines += ["## 三、旁证：明显战术错误（复用 mine_blunders 探针，事件段/每100决策）", "",
              "| 配置 | 弃营送死/丢营(每100) | 大子白送(每100) | 工兵白送(每100) |",
              "|---|---|---|---|"]
    for cfg in rep["configs"]:
        m = cfg["metrics"]
        lines.append(f"| {cfg['label']} | {m['probe_camp_abandonment_per100']:.3f} | "
                     f"{m['probe_top_piece_suicide_per100']:.3f} | "
                     f"{m['probe_sapper_suicide_per100']:.3f} |")
    lines.append("")

    lines += ["## 四、终局原因分布（局数）", "",
              "| 配置 | flag | immobilized | no_capture | repetition | max_plies |",
              "|---|---|---|---|---|---|"]
    for cfg in rep["configs"]:
        m = cfg["metrics"]
        lines.append(f"| {cfg['label']} | {m['end_flag_games']} | "
                     f"{m['end_immobilized_games']} | {m['end_no_capture_games']} | "
                     f"{m['end_repetition_games']} | {m['end_max_plies_games']} |")
    lines.append("")

    for cfg in rep["configs"]:
        if cfg["gift_samples"]:
            lines += [f"### 送旗样本（{cfg['label']}，最多 20 条）", "",
                      "| seed | ply | 座位 | 动作 |", "|---|---|---|---|"]
            for s in cfg["gift_samples"]:
                lines.append(f"| {s['seed']} | {s['ply']} | {s['seat']} | `{s['action']}` |")
            lines.append("")
        if cfg["illegal_samples"] or cfg["transition_samples"]:
            lines += [f"### ⚠️ 非法样本（{cfg['label']}）", ""]
            lines += [f"- {json.dumps(s, ensure_ascii=False)}"
                      for s in cfg["illegal_samples"] + cfg["transition_samples"]]
            lines.append("")

    lines += ["## 五、局限", "",
              "- 本审计在**无头自对弈**下运行，走的是与 GUI 相同的策略层与规则层",
              "  （`junqi.selfplay.make_strategy` + `GameState.legal_actions/apply`），",
              "  但不经过 GUI 的事件循环与坐标映射；GUI 侧的点击合法性由 P0 的",
              "  `legal_actions` 门控保证，二者共享同一真源。",
              "- 探针 1-3（弃营/大子/工兵）是 2026-09-06 固化的启发式断言，",
              "  单条命中不等于真实恶手，只用同分母做**配置间相对比较**。",
              ""]
    return "\n".join(lines)


# ----------------------------------------------------------------- 主流程

def load_init_states(path: Optional[str], games: int) -> Tuple[List[Optional[GameState]], str]:
    if not path or path.lower() == "none":
        return [None] * games, "每局随机发牌（deal(seed)）"
    with open(path, "r", encoding="utf-8") as f:
        pool = [ln.strip() for ln in f if ln.strip()]
    if not pool:
        raise SystemExit(f"起始局面文件为空: {path}")
    states = [GameState.from_json(pool[i % len(pool)]) for i in range(games)]
    return states, f"{path}（{len(pool)} 个局面，按位取 {games} 个，配对同牌）"


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 验收第 4 条：非法/送旗/无意义循环比例审计")
    ap.add_argument("--games", type=int, default=60, help="每配置对局数（默认 60）")
    ap.add_argument("--a", default="hybrid2", help="候选引擎（默认 hybrid2）")
    ap.add_argument("--b", default="search2", help="对手引擎（默认 search2）")
    ap.add_argument("--model-a", default="models/bc_best.pt", help="候选权重")
    ap.add_argument("--model-b", default=None, help="对手权重")
    ap.add_argument("--compare-a", default="expert2",
                    help="对照配置的候选引擎（项目既有传统强引擎，默认 expert2；none 关闭）")
    ap.add_argument("--compare-model-a", default=None, help="对照配置权重")
    ap.add_argument("--seed-base", type=int, default=900000, help="基础种子")
    ap.add_argument("--init-set", default="eval_sets/endgame.jsonl",
                    help="起始局面 jsonl（none = 每局随机发牌）")
    ap.add_argument("--max-plies", type=int, default=None, help="手数上限（默认引擎 1000）")
    ap.add_argument("--tolerance", type=float, default=1.5,
                    help="候选率相对对照引擎率的容许倍数（默认 1.5）")
    ap.add_argument("--abs-cap", type=float, default=2.0,
                    help="送旗/循环类每 100 决策的绝对上限（默认 2.0）")
    ap.add_argument("--device", default="cpu", help="计算设备")
    ap.add_argument("--out-dir", default="reports", help="报告输出目录")
    ap.add_argument("--out-name", default=None, help="输出文件名前缀")
    args = ap.parse_args()

    cfg = RuleConfig(max_plies=args.max_plies) if args.max_plies else RuleConfig()
    detector = BlunderDetector()
    starts, init_desc = load_init_states(args.init_set, args.games)
    seeds = list(range(args.seed_base, args.seed_base + args.games))

    setups = [(f"{args.a} vs {args.b}", args.a, args.b, args.model_a, args.model_b)]
    if args.compare_a and args.compare_a.lower() != "none":
        setups.append((f"{args.compare_a} vs {args.b}（对照）",
                       args.compare_a, args.b, args.compare_model_a, args.model_b))

    runs_by_label: Dict[str, List[dict]] = {}
    for label, sa, sb, ma, mb in setups:
        t0 = time.time()
        print(f"[audit] 开始 {label}：{args.games} 局，起始局面 {init_desc}", flush=True)
        runs = []
        for i, seed in enumerate(seeds):
            r = run_one_game(cfg, detector, sa, sb, ma, mb, seed, starts[i], args.device)
            runs.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{args.games}] seed={seed} plies={r['plies']} "
                      f"reason={r['reason']} 累计送旗={sum(x['counts'].get('flag_gift', 0) for x in runs)}"
                      f" 非法={sum(x['counts'].get('illegal_action', 0) for x in runs)}"
                      f" | {time.time()-t0:.0f}s", flush=True)
        runs_by_label[label] = runs
        print(f"[audit] {label} 完成，耗时 {time.time()-t0:.0f}s", flush=True)

    configs = [summarize(label, runs_by_label[label], args.tolerance, args.abs_cap)
               for label, *_ in setups]
    cand = configs[0]
    base = configs[1] if len(configs) > 1 else None
    verdict = judge(cand, base, args.tolerance, args.abs_cap)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    repro = (f"python scripts/audit_p2_game_quality.py --games {args.games} "
             f"--a {args.a} --b {args.b} --model-a {args.model_a} "
             f"--compare-a {args.compare_a} --init-set {args.init_set} "
             f"--seed-base {args.seed_base}")
    report = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "init_desc": init_desc,
        "seed_base": args.seed_base,
        "tolerance": args.tolerance,
        "abs_cap": args.abs_cap,
        "repro": repro,
        "verdict": verdict,
        "configs": configs,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.out_name or f"p2_game_quality_{args.a}_vs_{args.b}_{ts}"
    json_path = os.path.join(args.out_dir, f"{prefix}.json")
    md_path = os.path.join(args.out_dir, f"{prefix}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_md(report))

    print("\n" + "=" * 70)
    for item in verdict["items"]:
        print(f"  {'✅' if item['pass'] else '❌'} {item['criterion']}: {item['value']}"
              f"    [{item.get('reference', '必须为 0')}]")
    print(f"  总判定: {'通过' if verdict['passed'] else '未通过'}")
    print(f"  JSON: {json_path}\n  MD:   {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
