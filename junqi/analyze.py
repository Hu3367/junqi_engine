"""人类对战记录分析：读取 GUI 落盘的 games/*.json，重演对局并统计人类战法。

  python -m junqi analyze                # 分析 games/ 全部记录
  python -m junqi analyze --dir games --out reports/human_analysis.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

from .ai import EvalWeights
from .config import RuleConfig
from .rules import CAMPS, COLOR_CN, RANK_CN, Rank
from .state import Action, deal


def load_records(games_dir: str):
    records = []
    for path in sorted(glob.glob(os.path.join(games_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            rec["_file"] = os.path.basename(path)
            records.append(rec)
        except (json.JSONDecodeError, KeyError):
            print(f"跳过损坏记录: {path}")
    return records


def replay(rec, cfg: RuleConfig | None = None):
    """按种子+着法序列精确重演，返回 (状态列表, 终局状态)。"""
    cfg = cfg or RuleConfig()
    st = deal(random.Random(rec["seed"]), cfg)
    states = [st]
    for m in rec["moves"]:
        act = Action(m["kind"], tuple(m["frm"]),
                     tuple(m["to"]) if m.get("to") else None)
        st = st.apply(act)
        states.append(st)
    return states, st


def analyze(records, out_path: str | None = None) -> str:
    n = len(records)
    if n == 0:
        return "没有可分析的记录（先在 GUI 里完成几局人机对战）"

    human_seat_win = Counter()
    seat_games = Counter()
    lengths = []
    early_flips = {"human": [], "ai": []}
    camp_enter = {"human": 0, "ai": 0}
    camp_final = {"human": 0, "ai": 0}
    camp_kills = {"human": 0, "ai": 0}
    captures_value = {"human": 0.0, "ai": 0.0}
    dead_by_rank = {"human": Counter(), "ai": Counter()}
    first_flip = {"human": [], "ai": []}
    opened_with_flip = Counter()
    per_game = []

    from .rules import battle
    W = EvalWeights().piece

    for rec in records:
        try:
            states, final = replay(rec)
        except Exception as e:  # 记录与引擎版本不兼容等
            print(f"重演失败 {rec.get('_file')}: {e}")
            continue
        human = rec["human_seat"]
        seat_games["games"] += 1
        lengths.append(rec["plies"])
        if rec["winner"] == human:
            human_seat_win["win"] += 1
        elif rec["winner"] == 1 - human:
            human_seat_win["loss"] += 1
        else:
            human_seat_win["draw"] += 1

        e_flips = {"human": 0, "ai": 0}
        camp_kill_local = {"human": 0, "ai": 0}
        camp_enter_local = {"human": 0, "ai": 0}
        seat_of_color = {}

        for i, m in enumerate(rec["moves"]):
            st = states[i]
            seat = m["seat"]
            side = "human" if seat == human else "ai"
            if st.seat_color[0] is not None:
                seat_of_color = {st.seat_color[0]: 0, st.seat_color[1]: 1}
            if m["kind"] == "flip":
                if st.ply < 12:
                    e_flips[side] += 1
                pc = st.board.get(tuple(m["frm"]))
                if pc is not None:
                    first_flip[side].append(
                        (f"{pc.color}{RANK_CN[pc.rank]}", tuple(m["frm"])))
                opened_with_flip[side] += 1
            else:
                frm, to = tuple(m["frm"]), tuple(m["to"])
                mover = st.board.get(frm)
                target = st.board.get(to)
                if mover is None:
                    continue
                if to in CAMPS and target is None:
                    camp_enter_local[side] += 1
                if frm in CAMPS and target is not None:
                    camp_kill_local[side] += 1
                if target is not None and st.seat_color.get(0):
                    # 记吃子价值（按走子方）
                    res = battle(mover.rank, target.rank)
                    if res == "attacker_wins":
                        captures_value[side] += W[target.rank]
                    elif res == "both_die":
                        captures_value[side] += W[target.rank]

        camp_enter["human"] += camp_enter_local["human"]
        camp_enter["ai"] += camp_enter_local["ai"]
        camp_kills["human"] += camp_kill_local["human"]
        camp_kills["ai"] += camp_kill_local["ai"]
        early_flips["human"].append(e_flips["human"])
        early_flips["ai"].append(e_flips["ai"])

        for pos, pc in final.board.items():
            if pos in CAMPS and final.seat_color.get(0):
                side = "human" if seat_of_color.get(pc.color) == human else "ai"
                camp_final[side] += 1
        for pc in final.dead:
            if final.seat_color.get(0):
                side = ("human" if seat_of_color.get(pc.color) == human else "ai")
                dead_by_rank[side][f"{COLOR_CN[pc.color]}{RANK_CN[pc.rank]}"] += 1

        per_game.append((rec.get("_file", "?"), rec["plies"],
                         "胜" if rec["winner"] == human
                         else ("负" if rec["winner"] == 1 - human else "和"),
                         camp_enter_local["human"], camp_kill_local["human"]))

    w, l, dr = human_seat_win["win"], human_seat_win["loss"], human_seat_win["draw"]
    lines = ["# 人类对战记录分析", "",
             f"- 对局数：{n}　你的战绩：**{w}胜 {l}负 {dr}和**"
             f"（胜率 {w / max(n, 1):.0%}）",
             f"- 平均局长：{sum(lengths) / max(len(lengths), 1):.0f} 手", ""]

    lines += ["## 翻子节奏（前 12 手翻子数）", "",
              "| 一方 | 平均 | 你赢的局 | 你输的局 |", "|---|---|---|---|"]
    win_idx = [i for i, r in enumerate(records)
               if r["winner"] == r["human_seat"]][:len(early_flips["human"])]
    # 简化：按胜负分组统计人类翻子数
    hf = early_flips["human"]
    hw = [f for i, f in enumerate(hf) if records[i]["winner"] == records[i]["human_seat"]]
    hl = [f for i, f in enumerate(hf) if records[i]["winner"] == 1 - records[i]["human_seat"]]
    hd = [f for i, f in enumerate(hf)
          if records[i]["winner"] not in (0, 1)]
    lines.append(f"| 你 | {sum(hf) / max(len(hf), 1):.1f} | "
                 f"均{sum(hw) / max(len(hw), 1):.1f} | 均{sum(hl) / max(len(hl), 1):.1f} |")
    af = early_flips["ai"]
    lines.append(f"| AI | {sum(af) / max(len(af), 1):.1f} | - | - |")
    lines.append("")

    lines += ["## 行营运用（围杀指标）", "",
              "| 指标 | 你 | AI |", "|---|---|---|",
              f"| 进营次数/局 | {camp_enter['human'] / max(n, 1):.1f} | "
              f"{camp_enter['ai'] / max(n, 1):.1f} |",
              f"| 营内发起吃子/局 | {camp_kills['human'] / max(n, 1):.1f} | "
              f"{camp_kills['ai'] / max(n, 1):.1f} |",
              f"| 终局占营数 | {camp_final['human']} | {camp_final['ai']} |"]
    lines.append("")

    lines += ["## 歼敌价值（每局平均）", "",
              f"- 你：{captures_value['human'] / max(n, 1):.0f} 分　"
              f"AI：{captures_value['ai'] / max(n, 1):.0f} 分", ""]

    lines += ["## 你的阵亡棋子榜（前 10）", "", "| 棋子 | 次数 |", "|---|---|"]
    for name, cnt in dead_by_rank["human"].most_common(10):
        lines.append(f"| {name} | {cnt} |")
    lines.append("")

    if first_flip["human"]:
        from collections import Counter as C
        ff = C(f"{r}@{p}" for r, p in first_flip["human"])
        lines += ["## 你的首翻选择（前 8）", "", "| 颜色棋子@位置 | 次数 |", "|---|---|"]
        for k, cnt in ff.most_common(8):
            lines.append(f"| {k} | {cnt} |")
        lines.append("")

    lines += ["## 最近对局明细", "", "| 文件 | 局长 | 结果 | 你进营 | 营内杀 |", "|---|---|---|---|---|"]
    for row in per_game[-10:]:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    text = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(prog="junqi analyze")
    parser.add_argument("--dir", default="games")
    parser.add_argument("--out", default="reports/human_analysis.md")
    args = parser.parse_args(argv)
    records = load_records(args.dir)
    text = analyze(records, args.out)
    print(text)
    print(f"\n报告已保存: {args.out}")


if __name__ == "__main__":
    main()
