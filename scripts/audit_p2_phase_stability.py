"""P2 验收第 2 条：固定开局 / 中盘 / 尾盘集合的「无明显崩溃」压测。

验收原文（`AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §P2）：
    固定开局、中盘、尾盘集合均无明显崩溃。

口径
----
与第 4 条（对局级质量审计，抽样 60 局）互补，本脚本走**决策级全量覆盖**：
对每个阶段的**全部**快照，让候选与对照引擎各做一次单步决策，逐条检查

  1. `choose_actions` 是否抛异常（真正的崩溃）；
  2. 存在合法动作时是否返回**空**（内层决策崩溃；策略包装层会用随机兜底把它藏起来）；
  3. 首选动作是否落在 `legal_actions()` 中（非法动作）；
  4. 单步耗时分布（p50 / p95 / max），用来发现「没崩但慢到不可用」。
  5. 终局 / 无合法动作的快照，正确行为是返回空 —— 单独计数，不计入崩溃。

判定：`n_exceptions == 0` 且 `n_empty == 0` 且 `n_illegal == 0` ⇒ 该阶段该引擎 PASS。
描述性指标（legal_actions 规模、首选动作 kind 分布）一并输出，用于说明集合非退化。

为什么不用 `Strategy.choose()`
------------------------------
`HybridStrategy.choose` / `ExpertStrategy.choose` 在拿到空结果时会 `rng.choice(legal)`
兜底 —— 那会把「内层决策崩溃」掩盖成一次随机走子。本脚本因此优先直接调用
`strat.agent.choose_actions`（两个默认引擎都暴露 `.agent`），拿不到时才退回 `choose`。

复现
----
    python scripts/audit_p2_phase_stability.py --out-dir reports \
        --out-name p2_phase_stability
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from junqi.selfplay import make_strategy          # noqa: E402
from junqi.state import GameState                 # noqa: E402

PHASES = ("opening", "midgame", "endgame")
_RNG = random.Random(20260917)


def percentile(values: List[float], q: float) -> float:
    """线性插值分位数（不依赖 numpy，保持脚本可独立运行）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def load_snapshots(path: str) -> List[GameState]:
    """读入某阶段的全部快照（不抽样）。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise SystemExit("快照文件为空: " + path)
    return [GameState.from_json(ln) for ln in lines]


def _ask(strat, agent, st: GameState):
    """向内层 agent 或策略包装层要一次决策，返回 [(action, score)] / [action]。"""
    if agent is not None:
        return agent.choose_actions(st, topn=1)
    return [strat.choose(st, _RNG)]


def probe_one(strat, st: GameState) -> Dict[str, object]:
    """对一个快照做一次单步决策压测，返回一条记录。

    `rec["empty"]` 只在「存在合法动作却返回空」时为 True（violation），
    终局 / 无合法动作快照返回空是正确行为，不算 violation。
    """
    rec: Dict[str, object] = {
        "terminal": bool(st.is_terminal()),
        "n_legal": 0,
        "empty": False,
        "illegal": False,
        "exception": None,
        "elapsed_ms": 0.0,
        "top_kind": None,
    }
    agent = getattr(strat, "agent", None)

    if st.is_terminal():
        # 终局快照正确行为 = 返回空；只验证不抛异常
        try:
            _ask(strat, agent, st)
        except Exception:
            rec["exception"] = traceback.format_exc(limit=3)
        return rec

    acts = st.legal_actions()
    rec["n_legal"] = len(acts)
    if not acts:
        # 无合法动作且未被判定终局 —— 引擎应返回空且不抛异常
        try:
            _ask(strat, agent, st)
        except Exception:
            rec["exception"] = traceback.format_exc(limit=3)
        return rec

    t0 = time.perf_counter()
    try:
        out = _ask(strat, agent, st)
    except Exception:
        rec["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        rec["exception"] = traceback.format_exc(limit=3)
        return rec
    rec["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0

    if not out:
        rec["empty"] = True
        return rec

    act = out[0][0] if isinstance(out[0], tuple) else out[0]
    rec["top_kind"] = getattr(act, "kind", None)
    if act not in acts:
        rec["illegal"] = True

    return rec


def summarize(records: List[Dict[str, object]]) -> Dict[str, object]:
    lat = [float(r["elapsed_ms"]) for r in records if int(r["n_legal"]) > 0]
    kinds: Dict[str, int] = {}
    for r in records:
        k = r["top_kind"]
        if k:
            kinds[str(k)] = kinds.get(str(k), 0) + 1
    legal_sizes = [int(r["n_legal"]) for r in records if int(r["n_legal"]) > 0]
    n_exc = sum(1 for r in records if r["exception"])
    n_empty = sum(1 for r in records if r["empty"])
    n_illegal = sum(1 for r in records if r["illegal"])
    return {
        "n_snapshots": len(records),
        "n_terminal": sum(1 for r in records if r["terminal"]),
        "n_no_legal_actions": sum(
            1 for r in records if not r["terminal"] and int(r["n_legal"]) == 0),
        "n_exceptions": n_exc,
        "n_empty": n_empty,
        "n_illegal": n_illegal,
        "latency_ms_p50": percentile(lat, 0.50),
        "latency_ms_p95": percentile(lat, 0.95),
        "latency_ms_max": max(lat) if lat else 0.0,
        "mean_n_legal": (sum(legal_sizes) / len(legal_sizes)) if legal_sizes else 0.0,
        "top_kind_hist": kinds,
        "pass": (n_exc == 0 and n_empty == 0 and n_illegal == 0),
    }


def render_md(rep: Dict[str, object]) -> str:
    L: List[str] = []
    title = "P2 验收第 2 条：固定开局 / 中盘 / 尾盘集合「无明显崩溃」压测"
    L += ["# " + title, ""]
    L += [f"- 生成时间：{rep['generated_at']}",
          "- 口径：**决策级全量覆盖**（每阶段全部快照，不抽样），"
          "每快照对每个引擎做一次单步决策",
          f"- 引擎：{', '.join(rep['agents'])}（候选 {rep['agents'][0]}，"
          f"对照 {rep['agents'][-1]}）",
          "- 复现：`python scripts/audit_p2_phase_stability.py --out-dir reports "
          "--out-name p2_phase_stability`", ""]
    L += ["> 崩溃定义：① 抛异常；② 存在合法动作却返回空；③ 首选动作不在 "
          "`legal_actions()` 中。",
          "> 终局 / 无合法动作的快照单独计数（其正确行为就是返回空），不计入崩溃。", ""]

    L += ["## 一、逐项判定", ""]
    L += ["| 阶段 | 引擎 | 快照 | 终局 | 无合法动作 | 异常 | 返回空 | 非法 | 判定 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for phase in rep["phases"]:
        for ag in rep["agents"]:
            s = rep["results"][phase][ag]
            verdict = "✅ PASS" if s["pass"] else "❌ FAIL"
            L.append(f"| {phase} | {ag} | {s['n_snapshots']} | {s['n_terminal']} | "
                     f"{s['n_no_legal_actions']} | {s['n_exceptions']} | {s['n_empty']} | "
                     f"{s['n_illegal']} | {verdict} |")
    L.append("")

    L += ["## 二、单步耗时（有合法动作的快照，毫秒）", "",
          "| 阶段 | 引擎 | p50 | p95 | max | 平均合法动作数 |",
          "|---|---|---|---|---|---|"]
    for phase in rep["phases"]:
        for ag in rep["agents"]:
            s = rep["results"][phase][ag]
            L.append(f"| {phase} | {ag} | {s['latency_ms_p50']:.1f} | "
                     f"{s['latency_ms_p95']:.1f} | {s['latency_ms_max']:.1f} | "
                     f"{s['mean_n_legal']:.2f} |")
    L.append("")

    L += ["## 三、首选动作 kind 分布（旁证：集合非退化）", "",
          "| 阶段 | 引擎 | flip | move | 合计 |", "|---|---|---|---|---|"]
    for phase in rep["phases"]:
        for ag in rep["agents"]:
            h = rep["results"][phase][ag]["top_kind_hist"]
            nf = int(h.get("flip", 0))
            nm = int(h.get("move", 0))
            L.append(f"| {phase} | {ag} | {nf} | {nm} | {nf + nm} |")
    L.append("")

    if rep["exception_samples"]:
        L += ["## 四、⚠️ 异常样本", ""]
        for s in rep["exception_samples"][:10]:
            L += [f"- **{s['phase']} / {s['agent']} / 第 {s['index']} 条**", "",
                  "```", str(s["traceback"]).rstrip(), "```", ""]
    if rep["illegal_samples"]:
        L += ["## 五、⚠️ 非法动作样本", ""]
        for s in rep["illegal_samples"][:10]:
            L.append("- " + json.dumps(s, ensure_ascii=False))
        L.append("")

    L += ["## 六、局限", "",
          "- 本压测只验证**单步决策**不崩溃，不验证连续对局中的行为质量；后者由 "
          "`scripts/audit_p2_game_quality.py`（对局级）覆盖，两者互补。",
          "- 每个快照只测一次、`seed` 固定；两个默认引擎的搜索是确定性的"
          "（`ExpertSearchEngine.rng` 全仓只赋值不读取），因此结果可复现。",
          "- 快照来自真实复盘，只覆盖「实战出现过的局面分布」。", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 验收第 2 条：三阶段集合无明显崩溃压测")
    ap.add_argument("--eval-dir", default="eval_sets", help="阶段集合目录")
    ap.add_argument("--sets", nargs="*", default=list(PHASES),
                    help="阶段名（对应 <eval-dir>/<name>.jsonl），默认三段全测")
    ap.add_argument("--agents", nargs="*", default=["hybrid2", "expert2"],
                    help="被测引擎（第一个为候选，最后一个为对照）")
    ap.add_argument("--model-a", default="models/bc_best.pt", help="候选引擎权重")
    ap.add_argument("--device", default="cpu", help="计算设备")
    ap.add_argument("--out-dir", default="reports", help="报告输出目录")
    ap.add_argument("--out-name", default="p2_phase_stability", help="输出文件名前缀")
    args = ap.parse_args()

    results: Dict[str, Dict[str, object]] = {}
    exception_samples: List[dict] = []
    illegal_samples: List[dict] = []

    for phase in args.sets:
        path = os.path.join(args.eval_dir, phase + ".jsonl")
        if not os.path.exists(path):
            raise SystemExit("找不到阶段集合: " + path)
        snaps = load_snapshots(path)
        print(f"[phase-stability] {phase}: {len(snaps)} 个快照 <- {path}", flush=True)
        results[phase] = {}

        for agent_spec in args.agents:
            model_path = args.model_a if agent_spec == args.agents[0] else None
            strat = make_strategy(agent_spec, seed=20260917, model_path=model_path,
                                  device=args.device)
            inner = "agent.choose_actions" if hasattr(strat, "agent") else "strategy.choose"
            print(f"  [{phase}] {agent_spec}: 单步压测开始（经 {inner}）", flush=True)

            t0 = time.time()
            records: List[Dict[str, object]] = []
            for i, st in enumerate(snaps):
                rec = probe_one(strat, st)
                records.append(rec)
                if rec["exception"] and len(exception_samples) < 10:
                    exception_samples.append({
                        "phase": phase, "agent": agent_spec, "index": i,
                        "traceback": str(rec["exception"]),
                    })
                if rec["illegal"] and len(illegal_samples) < 10:
                    illegal_samples.append({
                        "phase": phase, "agent": agent_spec, "index": i})

            s = summarize(records)
            results[phase][agent_spec] = s
            print(f"  [{phase}] {agent_spec}: 异常={s['n_exceptions']} "
                  f"空={s['n_empty']} 非法={s['n_illegal']} "
                  f"p50={s['latency_ms_p50']:.1f}ms p95={s['latency_ms_p95']:.1f}ms "
                  f"耗时={time.time() - t0:.0f}s "
                  f"=> {'PASS' if s['pass'] else 'FAIL'}", flush=True)

    overall = all(results[p][a]["pass"] for p in results for a in results[p])
    rep = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_dir": args.eval_dir,
        "phases": list(args.sets),
        "agents": list(args.agents),
        "results": results,
        "exception_samples": exception_samples,
        "illegal_samples": illegal_samples,
        "overall_pass": overall,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, args.out_name + ".json")
    mp = os.path.join(args.out_dir, args.out_name + ".md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(render_md(rep))

    print("=" * 68)
    for phase in rep["phases"]:
        for agent_spec in rep["agents"]:
            s = results[phase][agent_spec]
            print(f"  {'OK  ' if s['pass'] else 'FAIL'} {phase:8s} {agent_spec:9s} "
                  f"异常={s['n_exceptions']} 空={s['n_empty']} 非法={s['n_illegal']} "
                  f"p95={s['latency_ms_p95']:.1f}ms")
    print(f"  总判定: {'通过' if overall else '未通过'}")
    print("  JSON:", jp)
    print("  MD:  ", mp)
    print("=" * 68)


if __name__ == "__main__":
    main()
