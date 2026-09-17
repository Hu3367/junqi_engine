"""自对弈研究管线：策略池对抗、对局记录、统计聚合、多进程批量、Markdown 报告。"""
from __future__ import annotations

import csv
import math
import os
import random
from collections import Counter
from dataclasses import dataclass

from .ai import Agent, evaluate
from .config import EvalWeights, RuleConfig, SearchConfig
from .rules import CAMPS, COMPOSITION, CORNER_JUNCTIONS, HQS, RANK_CN, Rank, battle
from .state import Action, GameState, deal, position_key

VALUE_DEFAULT = EvalWeights().piece


# ---------------------------------------------------------------- 策略

class Strategy:
    name = "base"

    def choose(self, state: GameState, rng: random.Random,
               avoid: set | None = None,
               history_counts: dict | None = None) -> Action:
        raise NotImplementedError


class RandomStrategy(Strategy):
    name = "random"

    def choose(self, state, rng, avoid=None, history_counts=None):
        return rng.choice(state.legal_actions())


class AgentStrategy(Strategy):
    """包装 Agent：depth=0 为贪心(单步期望)，depth>=1 为搜索。
    temperature>0 时在前 top_k 个候选中按 softmax 随机，增加对局多样性。"""

    def __init__(self, depth=0, samples=6, temperature=0.0, seed=None,
                 top_k=3, weights: EvalWeights | None = None):
        self.agent = Agent(SearchConfig(depth=depth, samples=samples),
                           weights=weights, seed=seed)
        self.temperature = temperature
        self.top_k = top_k
        self.name = (f"greedy_k{samples}" if depth == 0
                     else f"search_d{depth}_k{samples}")

    def choose(self, state, rng, avoid=None, history_counts=None):
        scored = self.agent.choose_actions(state, topn=self.top_k, avoid=avoid)
        if not scored:
            return rng.choice(state.legal_actions())
        if self.temperature <= 0 or len(scored) == 1:
            return scored[0][0]
        top = max(s for _, s in scored)          # 防必胜大分溢出 exp
        exps = [math.exp((s - top) / self.temperature) for _, s in scored]
        total = sum(exps)
        r = rng.random() * total
        acc = 0.0
        for (a, _), e in zip(scored, exps):
            acc += e
            if r <= acc:
                return a
        return scored[-1][0]


class NNStrategy(Strategy):
    """包装 NNAgent：支持纯网络策略（nn）与 MCTS 搜索策略（nn_mcts）。"""

    def __init__(self, model_path: str | None = None, sims: int = 60,
                 device: str | None = None, seed: int | None = None):
        from .ai import NNAgent
        self.agent = NNAgent(model_path=model_path, simulations=sims,
                             device=device, seed=seed)
        self.name = f"nn_sims{sims}" if sims > 0 else "nn_policy"

    def choose(self, state, rng, avoid=None, history_counts=None):
        scored = self.agent.choose_actions(state, topn=1, avoid=avoid, history_counts=history_counts)
        if not scored:
            return rng.choice(state.legal_actions())
        return scored[0][0]


class ExpertStrategy(Strategy):
    """包装 ExpertAgent：基于 Star1 期望极大极小、QSearch、置换表与专家评估的搜索引擎。"""

    def __init__(self, depth: int = 2, time_limit_ms: int = 0,
                 weights: EvalWeights | None = None, seed: int | None = None):
        from .ai import ExpertAgent
        self.agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms),
                                 weights=weights, seed=seed)
        self.name = f"expert_d{depth}" if time_limit_ms <= 0 else f"expert_t{time_limit_ms}ms"

    def choose(self, state: GameState, rng: random.Random,
               avoid: set | None = None, history_counts: dict | None = None) -> Action:
        scored = self.agent.choose_actions(state, topn=1, avoid=avoid)
        if not scored:
            return rng.choice(state.legal_actions())
        return scored[0][0]


class HybridStrategy(Strategy):
    """包装 HybridAgent：神经网络全局大局观先验 + 专家搜索战术过滤。"""

    def __init__(self, depth: int = 2, model_path: str | None = None,
                 top_k: int = 6, weights: EvalWeights | None = None,
                 device: str | None = None, seed: int | None = None):
        from .ai import HybridAgent
        mp = model_path or ("models/bc_best.pt" if os.path.exists("models/bc_best.pt") else "models/best.pt")
        self.agent = HybridAgent(model_path=mp, search_depth=depth, top_k=top_k,
                                 weights=weights, device=device, seed=seed)
        self.name = f"hybrid_d{depth}"

    def choose(self, state: GameState, rng: random.Random,
               avoid: set | None = None, history_counts: dict | None = None) -> Action:
        scored = self.agent.choose_actions(state, topn=1, avoid=avoid)
        if not scored:
            return rng.choice(state.legal_actions())
        return scored[0][0]


class P4HybridStrategy(Strategy):
    """包装 HybridDecisionEngine：P4.2 多世界采样 + NN 先验 + QSearch 战术定价。"""

    def __init__(self, model_path: str | None = None, k_worlds: int = 4,
                 device: str = "cpu", seed: int | None = None):
        from .hybrid_engine import HybridDecisionEngine
        mp = model_path or ("models/best.pt" if os.path.exists("models/best.pt") else None)
        self.agent = HybridDecisionEngine(model_path=mp, k_worlds=k_worlds,
                                          device=device, seed=seed)
        self.name = "p4hybrid"

    def choose(self, state: GameState, rng: random.Random,
               avoid: set | None = None, history_counts: dict | None = None) -> Action:
        scored = self.agent.choose_actions(state, topn=1, avoid=avoid,
                                           history_counts=history_counts)
        if not scored:
            return rng.choice(state.legal_actions())
        return scored[0][0]


def make_strategy(spec: str, seed=None,
                  weights: EvalWeights | None = None,
                  model_path: str | None = None,
                  device: str = "cpu") -> Strategy:
    """'random' | 'greedy' | 'search2' | 'expert2' | 'hybrid2' | 'p4hybrid' | 'nn' | 'nn_mcts' -> 策略实例。"""
    if spec == "random":
        return RandomStrategy()
    if spec.startswith("p4hybrid"):
        return P4HybridStrategy(model_path=model_path, device=device, seed=seed)
    if spec == "greedy":
        return AgentStrategy(depth=0, samples=6, temperature=25.0, seed=seed,
                             weights=weights)
    if spec.startswith("hybrid"):
        depth = int(spec.replace("hybrid", "") or 3)
        return HybridStrategy(depth=depth, model_path=model_path, weights=weights, device=device, seed=seed)
    if spec.startswith("expert"):
        depth = int(spec.replace("expert", "") or 2)
        return ExpertStrategy(depth=depth, weights=weights, seed=seed)
    if spec.startswith("search"):
        depth = int(spec.replace("search", "") or 2)
        return AgentStrategy(depth=depth, samples=4, temperature=15.0, seed=seed,
                             weights=weights)
    if spec == "nn":
        return NNStrategy(model_path=model_path, sims=0, device=device, seed=seed)
    if spec.startswith("nn_mcts") or spec == "nn_rl":
        sims = 60
        if "_" in spec and spec.split("_")[-1].isdigit():
            sims = int(spec.split("_")[-1])
        return NNStrategy(model_path=model_path, sims=sims, device=device, seed=seed)
    raise ValueError(f"未知策略: {spec}")


# ---------------------------------------------------------------- 对局

def flip_region(pos) -> str:
    r, c = pos
    if pos in CORNER_JUNCTIONS:
        return "corner"
    if r in (0, 11):
        return "hq_row"
    if r in (5, 6):
        return "front"
    return "inner"


# 评测裁决用：对局期间采样专家估值的间隔（手）。理由见文件末 play_game 的估值记录说明。
EVAL_SAMPLE_EVERY = 10


def play_game(spec0: str, spec1: str, seed: int, cfg: RuleConfig | None = None,
              weights0: EvalWeights | None = None,
              weights1: EvalWeights | None = None,
              model_path0: str | None = None,
              model_path1: str | None = None,
              init_state: GameState | None = None,
              device: str = "cpu") -> dict:
    """完整对局一局。策略0 执座位0（先手）。返回统计记录 dict。

    和棋判定与引擎 state.apply 一致（70 步无吃子 / 1000 手），另加对局层的
    相同局面循环判和（APK 规则：可观察局面重复 repetition_draw_count 次判和，
    搜索看不到历史，由本循环维护并经 avoid 传给策略规避）。"""
    cfg = cfg or RuleConfig()
    rng = random.Random(seed * 7919 + 13)
    s0 = make_strategy(spec0, seed=seed * 31 + 1, weights=weights0, model_path=model_path0, device=device)
    s1 = make_strategy(spec1, seed=seed * 31 + 2, weights=weights1, model_path=model_path1, device=device)
    st = init_state.copy() if init_state is not None else deal(random.Random(seed), cfg)

    rec = {"seed": seed, "a": spec0, "b": spec1, "winner": None, "reason": None,
           "plies": 0, "first_flip_pos": None, "first_flip_region": None,
           "first_flip_rank": None,
           "flips0": 0, "flips1": 0, "early_flips0": 0, "early_flips1": 0,
           "captured0": 0.0, "captured1": 0.0,
           "camp_enter0": 0, "camp_enter1": 0, "camp_kill0": 0, "camp_kill1": 0,
           "flag_taken": 0}

    def seat_of_color(color: str) -> int:
        return 0 if st.seat_color[0] == color else 1

    seen = Counter()
    first_flip = True
    last_live_eval = None                  # 对局期间最后一次有效估值（评测裁决用）
    # 镜像对称化口径（P1）：同一次采样顺带记录，抵消 evaluate_expert 的座位标签偏差
    last_live_eval_sym = None

    eval_failures = 0                      # C8：估值失败次数（进对局记录，可观测）

    while not st.is_terminal():
        if st.ply % EVAL_SAMPLE_EVERY == 0:
            try:
                from .eval_expert import evaluate_expert_dual
                e0, e1, sy0, sy1 = evaluate_expert_dual(st, ignore_rule_draw=True)
                if abs(e0) + abs(e1) > 1e-9:
                    last_live_eval = (e0, e1)
                if abs(sy0) + abs(sy1) > 1e-9:
                    last_live_eval_sym = (sy0, sy1)
            except Exception as exc:      # noqa: BLE001
                # C8 修复：原为 `pass`。估值缺失会让门控的裁决式判分静默退化为
                # 0.5（等于把该局当平局），却完全不出现在报告里。
                eval_failures += 1
                if eval_failures == 1:
                    print(f"⚠️ 对局 seed={seed}: 专家估值采样失败 "
                          f"({type(exc).__name__}: {exc})，该局裁决判分可能退化为 0.5",
                          flush=True)
        seen[position_key(st)] += 1
        if seen[position_key(st)] >= cfg.repetition_draw_count:
            rec.update(winner=-1, reason="repetition")
            break
        avoid = {k for k, n in seen.items() if n >= cfg.repetition_draw_count - 1}
        acts = st.legal_actions()
        if not acts:
            rec.update(winner=1 - st.turn, reason="immobilized")
            break
        mover_seat = st.turn
        act = (s0 if mover_seat == 0 else s1).choose(st, rng, avoid=avoid, history_counts=seen)

        if act.kind == "flip":
            pc = st.board[act.frm]
            if first_flip:
                rec.update(first_flip_pos=str(act.frm),
                           first_flip_region=flip_region(act.frm),
                           first_flip_rank=pc.rank.name)
                first_flip = False
            rec[f"flips{mover_seat}"] += 1
            if st.ply < 12:
                rec[f"early_flips{mover_seat}"] += 1
            st = st.apply(act)
            continue

        mover = st.board[act.frm]
        target = st.board.get(act.to)
        st = st.apply(act)

        if target is None and act.to in CAMPS:
            rec[f"camp_enter{mover_seat}"] += 1
        if act.frm in CAMPS and target is not None:
            rec[f"camp_kill{mover_seat}"] += 1

        if target is not None:
            res = battle(mover.rank, target.rank)
            if res == "attacker_wins":
                rec[f"captured{mover_seat}"] += VALUE_DEFAULT[target.rank]
                if target.rank == Rank.QI:
                    rec["flag_taken"] = 1
            elif res == "both_die":
                rec[f"captured{mover_seat}"] += VALUE_DEFAULT[target.rank]
                rec[f"captured{1 - mover_seat}"] += VALUE_DEFAULT[mover.rank]
            else:
                rec[f"captured{1 - mover_seat}"] += VALUE_DEFAULT[mover.rank]

    # 工兵存活：0=全灭 1=有明工兵存活 None=藏而未亮
    if st.seat_color[0] is not None:
        for seat in (0, 1):
            color = st.seat_color[seat]
            dead_eng = sum(1 for pc in st.dead if pc.color == color
                           and pc.rank == Rank.GONG)
            alive_revealed = any(pc.revealed and pc.color == color
                                 and pc.rank == Rank.GONG
                                 for pc in st.board.values())
            rec[f"eng{seat}"] = 0 if dead_eng == COMPOSITION[Rank.GONG] \
                else (1 if alive_revealed else None)

    rec["plies"] = st.ply
    if st.winner is not None and rec["winner"] is None:
        rec["winner"] = st.winner
        rec["reason"] = st.win_reason

    # 终局裁决估值（仅评测用，公开信息口径，不参与训练奖励）
    #
    # 动机（2026-09-14 诊断）：同源模型间 72-92% 对局判和，专家对专家也 88% 和棋，
    # 门控的"得分率"判据在如此高和棋率下几乎没有分辨力（200 局只剩 16-56 局胜负样本）。
    # 记录终局专家估值（evaluate_expert，仅用公共信息），使评测可在和棋局上做
    # **裁决式判分**，把有效样本从"决胜负局"扩大到全部对局。
    # 这属于评测口径扩展，不改变任何规则定义与训练奖励（z 仍为纯终局结果）。
    try:
        from .eval_expert import evaluate_expert_dual
        e0, e1, sy0, sy1 = evaluate_expert_dual(st, ignore_rule_draw=True)
        is_dead = abs(e0) + abs(e1) <= 1e-9
        if is_dead and last_live_eval is not None:
            # 终局为结构性死锁（估值恒 0）时，回退到对局中最后一次有效估值：
            # 否则"和棋局按终局估值裁决"在恰好需要它的场合全部退化为 0.5
            # （实测 40 组种子镜像 80 局中有 74 局如此）。
            e0, e1 = last_live_eval
        # 镜像对称化口径同理回退（sym 恒 0 同样发生在结构性死锁终局）
        is_dead_sym = abs(sy0) + abs(sy1) <= 1e-9
        if is_dead_sym and last_live_eval_sym is not None:
            sy0, sy1 = last_live_eval_sym
        rec["final_eval0"] = round(e0, 3)
        rec["final_eval1"] = round(e1, 3)
        # P1（2026-09-15）：镜像对称化终局估值。裁决判分优先用它，
        # 因为原口径含按座位标签的加性偏差，会在零阈值下放大成符号偏置。
        rec["final_eval_sym0"] = round(sy0, 3)
        rec["final_eval_sym1"] = round(sy1, 3)
        # C8 附带修复：原实现为算这个布尔值**又调了两次** evaluate_expert
        # （共 4 次），而 is_dead 已经算出来了。这里直接复用。
        rec["last_live_eval_used"] = bool(is_dead)
    except Exception as exc:                              # noqa: BLE001
        # C8 修复：估值不可用时不影响对局记录，但必须可观测
        eval_failures += 1
        if eval_failures == 1:
            print(f"⚠️ 对局 seed={seed}: 终局专家估值失败 "
                  f"({type(exc).__name__}: {exc})，final_eval 置 None", flush=True)
        rec["final_eval0"] = None
        rec["final_eval1"] = None
        rec["final_eval_sym0"] = None
        rec["final_eval_sym1"] = None
        rec["last_live_eval_used"] = None
    rec["eval_failures"] = eval_failures
    return rec


# ---------------------------------------------------------------- 批量执行

def _run_chunk(job):
    """子进程任务：跑一段对局并返回记录列表（Windows spawn 安全：模块级函数）。"""
    spec0, spec1, base_seed, n, max_plies, weights0, weights1, mpath0, mpath1, init_states_json = job
    cfg = RuleConfig(max_plies=max_plies) if max_plies else RuleConfig()
    init_states = [GameState.from_json(s, cfg) for s in init_states_json] if init_states_json else [None] * n
    return [play_game(spec0, spec1, base_seed + i, cfg,
                      weights0=weights0, weights1=weights1,
                      model_path0=mpath0, model_path1=mpath1,
                      init_state=init_states[i % len(init_states)] if init_states else None)
            for i in range(n)]


def run_selfplay(spec_a: str, spec_b: str, games: int, workers: int = 1,
                 base_seed: int = 0, out_dir: str = "reports",
                 max_plies: int | None = None, also_swap: bool = True,
                 model_path_a: str | None = None,
                 model_path_b: str | None = None,
                 eval_set: str | None = None) -> dict:
    """批量对局 + 汇总报告。also_swap=True 时再跑一组先后手互换，消除先手偏差。
    eval_set: 可选 jsonl 评测集文件路径。"""
    os.makedirs(out_dir, exist_ok=True)
    init_states_json = []
    if eval_set and os.path.exists(eval_set):
        with open(eval_set, "r", encoding="utf-8") as f:
            init_states_json = [line.strip() for line in f if line.strip()]
        if init_states_json:
            games = min(games, len(init_states_json))

    pairs = [(spec_a, spec_b, model_path_a, model_path_b)]
    if also_swap:
        pairs.append((spec_b, spec_a, model_path_b, model_path_a))
    jobs = []
    games_per_pair = games if also_swap else games
    for pi, (s0, s1, mp0, mp1) in enumerate(pairs):
        chunk = max(1, games_per_pair // max(workers, 1))
        seed_base = base_seed + pi * 1_000_000
        for w in range(workers):
            start_idx = w * chunk
            n = chunk if w < workers - 1 else games_per_pair - chunk * (workers - 1)
            chunk_jsons = init_states_json[start_idx:start_idx + n] if init_states_json else []
            if n > 0:
                jobs.append((s0, s1, seed_base + w * 50_000, n, max_plies,
                             None, None, mp0, mp1, chunk_jsons))

    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            chunks = pool.map(_run_chunk, jobs)
    else:
        chunks = [_run_chunk(j) for j in jobs]
    records = [r for c in chunks for r in c]

    csv_path = os.path.join(out_dir, f"games_{spec_a}_vs_{spec_b}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    report = aggregate(records, spec_a, spec_b)
    md_path = os.path.join(out_dir, f"report_{spec_a}_vs_{spec_b}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    return {"records": records, "csv": csv_path, "report": md_path,
            "report_text": report}


# ---------------------------------------------------------------- 统计聚合

def _pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "-"


def aggregate(records, spec_a, spec_b) -> str:
    n = len(records)
    lines = [f"# 自对弈报告：{spec_a} vs {spec_b}", "",
             f"- 对局数：{n}",
             f"- 平均局长：{sum(r['plies'] for r in records) / max(n, 1):.0f} 手", ""]

    # 对局结果：按 (先手, 后手) 策略组合分组，消除互换组混淆
    pairs = {}
    for r in records:
        pairs.setdefault((r["a"], r["b"]), []).append(r)
    lines += ["## 对局结果（按策略组合）", "",
              "| 先手 vs 后手 | 局数 | 先手胜 | 后手胜 | 和棋 |", "|---|---|---|---|---|"]
    for (sa, sb), rs in sorted(pairs.items()):
        w0 = sum(1 for r in rs if r["winner"] == 0)
        w1 = sum(1 for r in rs if r["winner"] == 1)
        d = len(rs) - w0 - w1
        lines.append(f"| {sa} vs {sb} | {len(rs)} | {_pct(w0, len(rs))} | "
                     f"{_pct(w1, len(rs))} | {_pct(d, len(rs))} |")
    # 总先手优势（跨策略组）
    a_win = sum(1 for r in records if r["winner"] == 0)
    b_win = sum(1 for r in records if r["winner"] == 1)
    draws = sum(1 for r in records if r["winner"] == -1 or r["winner"] is None)
    lines += ["", f"**总体先手优势：先手胜 {_pct(a_win, n)}，后手胜 {_pct(b_win, n)}，"
              f"和棋 {_pct(draws, n)}**", ""]

    # 首翻区域与先后手胜率
    by_region = {}
    for r in records:
        reg = r.get("first_flip_region")
        if reg:
            by_region.setdefault(reg, []).append(r)
    if by_region:
        lines += ["## 首翻位置区域 → 先手胜率", "",
                  "| 区域 | 局数 | 先手胜率 | 后手胜率 | 和棋 |", "|---|---|---|---|---|"]
        for reg, rs in sorted(by_region.items()):
            w0 = sum(1 for r in rs if r["winner"] == 0)
            w1 = sum(1 for r in rs if r["winner"] == 1)
            d = len(rs) - w0 - w1
            lines.append(f"| {reg} | {len(rs)} | {_pct(w0, len(rs))} | "
                         f"{_pct(w1, len(rs))} | {_pct(d, len(rs))} |")
        lines.append("")

    # 首翻棋子价值与胜率
    by_rank = {}
    for r in records:
        rk = r.get("first_flip_rank")
        if rk:
            by_rank.setdefault(rk, []).append(r)
    if by_rank:
        lines += ["## 首翻棋子 → 先手胜率", "",
                  "| 首翻棋子 | 局数 | 先手胜率 |", "|---|---|---|"]
        for rk, rs in sorted(by_rank.items(),
                             key=lambda kv: -len(kv[1])):
            w0 = sum(1 for r in rs if r["winner"] == 0)
            lines.append(f"| {RANK_CN[Rank[rk]]} | {len(rs)} | {_pct(w0, len(rs))} |")
        lines.append("")

    # 前 12 手翻子数 vs 胜率
    lines += ["## 前 12 手翻子节奏（先手方）→ 先手胜率", "",
              "| 前12手翻子数 | 局数 | 先手胜率 |", "|---|---|---|"]
    buckets = {"0-2": lambda x: x <= 2, "3-5": lambda x: 3 <= x <= 5,
               "6+": lambda x: x >= 6}
    for label, cond in buckets.items():
        rs = [r for r in records if cond(r["early_flips0"])]
        if rs:
            w0 = sum(1 for r in rs if r["winner"] == 0)
            lines.append(f"| {label} | {len(rs)} | {_pct(w0, len(rs))} |")
    lines.append("")

    # 工兵存活 vs 胜率
    eng_rows = []
    for seat, label in ((0, "先手"), (1, "后手")):
        alive = [r for r in records if r.get(f"eng{seat}") == 1]
        dead = [r for r in records if r.get(f"eng{seat}") == 0]
        if alive and dead:
            wa = sum(1 for r in alive if r["winner"] == seat)
            wd = sum(1 for r in dead if r["winner"] == seat)
            eng_rows.append(f"| {label}：工兵存活 | {len(alive)} | {_pct(wa, len(alive))} |")
            eng_rows.append(f"| {label}：工兵全灭 | {len(dead)} | {_pct(wd, len(dead))} |")
    if eng_rows:
        lines += ["## 工兵存活与胜负", "", "| 状态 | 局数 | 该方胜率 |", "|---|---|---|"]
        lines += eng_rows + [""]

    # 胜负原因分布
    reasons = Counter(r.get("reason") or "unknown" for r in records)
    lines += ["## 终局原因", "", "| 原因 | 局数 |", "|---|---|"]
    for k, v in reasons.most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # 平均吃子价值
    c0 = sum(r["captured0"] for r in records) / max(n, 1)
    c1 = sum(r["captured1"] for r in records) / max(n, 1)
    lines += ["## 平均每局歼敌子力价值", "",
              f"- 先手方：{c0:.0f} 分　后手方：{c1:.0f} 分", ""]
    return "\n".join(lines)
