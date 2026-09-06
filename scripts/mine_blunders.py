"""军棋 AI 战术失误自动挖掘管道 (Junqi Tactical Blunder Mining Pipeline)。

脱离 GUI 进行高速自动化无头自博弈，基于 6 大战术断言探针（Invariants）实时监控对局，
自动截获并归档 AI 走出的病态着法（如弃营送死、大子白送、工兵自杀、无谓循环踱步、龟缩拒翻等）。
自动输出结构化 JSON 错题库与 Markdown 诊断复盘报告。

用法示例:
    python scripts/mine_blunders.py --games 10 --engine-white hybrid2 --engine-black expert2
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from junqi.ai import evaluate_expert
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.rules import (ATTACKER_WINS, BOTH_DIE, CAMPS, DEFENDER_WINS, HQS,
                         NEIGHBORS, RANK_CN, Rank, battle, is_camp, is_hq, other)
from junqi.selfplay import make_strategy
from junqi.state import Action, GameState, Piece, deal, position_key


# ---------------------------------------------------------------- 失误探针定义

class BlunderDetector:
    """战术病态走法与规则违背探针聚合器。"""

    def __init__(self):
        self.weights = EvalWeights()

    def check_camp_abandonment(self, ply_idx: int, history: List[dict]) -> Optional[dict]:
        """探针 1: 弃营丢营/弃营送死。
        AI 棋子离开行营，次手在目标格被敌方击杀，或原行营次手即被敌方入驻。"""
        curr = history[ply_idx]
        act = curr["action"]
        if act.kind != "move" or not is_camp(act.frm) or is_camp(act.to):
            return None

        mover = curr["mover"]
        target = curr["target"]
        if mover is None:
            return None

        # 检查是否为了吃大子/军旗而正常出营
        if target is not None and target.revealed and target.rank in (Rank.QI, Rank.SI, Rank.JUN):
            return None

        # 检查下一手 (对手走子)
        if ply_idx + 1 < len(history):
            nxt = history[ply_idx + 1]
            opp_act = nxt["action"]
            if opp_act.kind == "move":
                # 次手在出营目标格被敌军反杀
                if opp_act.to == act.to:
                    killer = nxt["mover"]
                    if killer is not None and killer.revealed:
                        b_res = battle(killer.rank, mover.rank)
                        if b_res in (ATTACKER_WINS, BOTH_DIE):
                            return {
                                "type": "camp_abandonment",
                                "sub_type": "counter_killed",
                                "severity": "CRITICAL",
                                "description": (
                                    f"座位 {curr['seat']} 的 {mover.color}{RANK_CN[mover.rank]} "
                                    f"放弃行营 {act.frm} 移动到 {act.to} "
                                    f"（目标: {target.color + RANK_CN[target.rank] if target else '空地'}），"
                                    f"次手立即在 {act.to} 被敌方 {killer.color}{RANK_CN[killer.rank]} 击杀！"
                                )
                            }
                # 次手原行营被敌军直接侵入占领 (丢营)
                if opp_act.to == act.frm:
                    invader = nxt["mover"]
                    return {
                        "type": "camp_abandonment",
                        "sub_type": "camp_lost",
                        "severity": "HIGH",
                        "description": (
                            f"座位 {curr['seat']} 的 {mover.color}{RANK_CN[mover.rank]} "
                            f"离开行营 {act.frm}，次手原行营立即被敌方 "
                            f"{invader.color + RANK_CN[invader.rank] if invader else '棋子'} 占领！"
                        )
                    }
        return None

    def check_top_piece_suicide(self, ply_idx: int, history: List[dict]) -> Optional[dict]:
        """探针 2: 大子白送 / 炸弹撞廉价小子。"""
        curr = history[ply_idx]
        act = curr["action"]
        if act.kind != "move":
            return None

        mover = curr["mover"]
        target = curr["target"]
        if mover is None or not mover.revealed:
            return None

        # Case A: 炸弹主动撞廉价小子 (排长/连长/营长/工兵) 自爆
        if mover.rank == Rank.ZHA and target is not None and target.revealed:
            if target.rank in (Rank.PAI, Rank.LIAN, Rank.YING, Rank.GONG):
                return {
                    "type": "top_piece_suicide",
                    "sub_type": "bomb_wasted_on_minor",
                    "severity": "CRITICAL",
                    "description": (
                        f"座位 {curr['seat']} 的炸弹主动攻击敌方廉价小子 "
                        f"{target.color}{RANK_CN[target.rank]}，战略重器被严重贱卖！"
                    )
                }

        # Case B: 司令/军长/师长出击后次手被反杀
        if mover.rank in (Rank.SI, Rank.JUN, Rank.SHI):
            if ply_idx + 1 < len(history):
                nxt = history[ply_idx + 1]
                opp_act = nxt["action"]
                if opp_act.kind == "move" and opp_act.to == act.to:
                    killer = nxt["mover"]
                    if killer is not None and killer.revealed:
                        b_res = battle(killer.rank, mover.rank)
                        if b_res in (ATTACKER_WINS, BOTH_DIE):
                            # 若没有换掉等值大子，判定为大子白送
                            killed_val = self.weights.piece.get(target.rank, 0.0) if (target and target.revealed) else 0.0
                            lost_val = self.weights.piece.get(mover.rank, 100.0)
                            if lost_val - killed_val > 25.0:
                                return {
                                    "type": "top_piece_suicide",
                                    "sub_type": "big_piece_trapped",
                                    "severity": "CRITICAL",
                                    "description": (
                                        f"座位 {curr['seat']} 的高价值大子 {mover.color}{RANK_CN[mover.rank]} "
                                        f"移动至 {act.to}，次手被敌方 {killer.color}{RANK_CN[killer.rank]} 击杀，"
                                        f"净损失战力 {lost_val - killed_val:.1f} 分！"
                                    )
                                }
        return None

    def check_sapper_suicide(self, ply_idx: int, history: List[dict]) -> Optional[dict]:
        """探针 3: 工兵自杀 (撞大子或出击被立即击杀)。"""
        curr = history[ply_idx]
        act = curr["action"]
        if act.kind != "move":
            return None

        mover = curr["mover"]
        target = curr["target"]
        if mover is None or mover.rank != Rank.GONG:
            return None

        # 撞非地雷的大子自杀
        if target is not None and target.revealed and target.rank != Rank.LEI:
            b_res = battle(mover.rank, target.rank)
            if b_res == DEFENDER_WINS:
                return {
                    "type": "sapper_suicide",
                    "sub_type": "charge_into_big",
                    "severity": "HIGH",
                    "description": (
                        f"座位 {curr['seat']} 工兵在 {act.frm}->{act.to} "
                        f"主动撞击敌方 {target.color}{RANK_CN[target.rank]} 撞死！"
                    )
                }

        # 挖雷或走步后次手立即被杀
        if ply_idx + 1 < len(history):
            nxt = history[ply_idx + 1]
            opp_act = nxt["action"]
            if opp_act.kind == "move" and opp_act.to == act.to:
                killer = nxt["mover"]
                if killer is not None and killer.revealed:
                    b_res = battle(killer.rank, mover.rank)
                    if b_res in (ATTACKER_WINS, BOTH_DIE):
                        return {
                            "type": "sapper_suicide",
                            "sub_type": "sapper_counter_killed",
                            "severity": "MEDIUM",
                            "description": (
                                f"座位 {curr['seat']} 工兵移动至 {act.to}，"
                                f"次手立即被敌方 {killer.color}{RANK_CN[killer.rank]} 反杀！"
                            )
                        }
        return None

    def check_ping_pong_loop(self, ply_idx: int, history: List[dict]) -> Optional[dict]:
        """探针 4: 无意义往复踱步 (A->B->A->B) 且盘面存在未翻暗子。"""
        curr = history[ply_idx]
        seat = curr["seat"]
        # 提取该座位的最近 4 次走步
        seat_moves = [h for h in history[:ply_idx + 1] if h["seat"] == seat]
        if len(seat_moves) < 4:
            return None

        m1, m2, m3, m4 = seat_moves[-4:]
        if (m1["action"].kind == "move" and m2["action"].kind == "move" and
            m3["action"].kind == "move" and m4["action"].kind == "move"):
            a1, a2, a3, a4 = m1["action"], m2["action"], m3["action"], m4["action"]
            # 检查是否为 A->B, B->A, A->B, B->A 循环
            if (a1.frm == a2.to and a1.to == a2.frm and
                a3.frm == a1.frm and a3.to == a1.to and
                a4.frm == a2.frm and a4.to == a2.to):
                # 检查盘面是否仍有可翻暗子
                st = curr["state"]
                if len(st.hidden_positions()) > 0:
                    return {
                        "type": "ping_pong_loop",
                        "sub_type": "two_station_loop",
                        "severity": "MEDIUM",
                        "description": (
                            f"座位 {seat} 在 {a1.frm} 与 {a1.to} 之间连续 4 次无意义往复踱步，"
                            f"盘面仍存 {len(st.hidden_positions())} 颗暗子未翻！"
                        )
                    }
        return None

    def check_camp_turtling(self, ply_idx: int, history: List[dict]) -> Optional[dict]:
        """探针 5: 龟缩拒翻。处于安全行营中连续 6 手以上拒不翻开邻近暗子。"""
        curr = history[ply_idx]
        seat = curr["seat"]
        seat_moves = [h for h in history[:ply_idx + 1] if h["seat"] == seat]
        if len(seat_moves) < 6:
            return None

        recent_6 = seat_moves[-6:]
        # 若最近 6 手全为 move (0 次 flip)
        if all(h["action"].kind == "move" for h in recent_6):
            st = curr["state"]
            hidden = st.hidden_positions()
            if len(hidden) >= 6:
                # 检查是否有暗子直接邻接行营
                camp_adj_hidden = any(
                    any(n in hidden for n in NEIGHBORS[c])
                    for c in CAMPS if c in st.board and st.board[c].color == st.my_color(seat)
                )
                if camp_adj_hidden:
                    return {
                        "type": "camp_turtling",
                        "sub_type": "refuse_to_flip",
                        "severity": "LOW",
                        "description": (
                            f"座位 {seat} 占据行营连续 6 手拒绝翻棋，"
                            f"丧失利用行营据点向邻近暗子辐射拓荒的战略主动权。"
                        )
                    }
        return None


# ---------------------------------------------------------------- 挖掘执行器

def serialize_state_simple(st: GameState) -> dict:
    """序列化盘面关键信息。"""
    pieces = []
    for (r, c), pc in st.board.items():
        pieces.append({
            "pos": [r, c],
            "color": pc.color,
            "rank": int(pc.rank),
            "rank_cn": RANK_CN[pc.rank],
            "revealed": pc.revealed,
        })
    return {
        "ply": st.ply,
        "turn": st.turn,
        "first_flip_done": st.first_flip_done,
        "seat_color": {str(k): v for k, v in st.seat_color.items()},
        "pieces": pieces,
        "dead_count": len(st.dead),
    }


def run_mining(
    games: int = 10,
    engine_white: str = "hybrid2",
    engine_black: str = "expert2",
    seed_base: int = 42,
    output_dir: str = "reports/blunders",
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[str, str]:
    """运行自博弈对局，挖掘战术失误，并生成结构化诊断报告。"""
    os.makedirs(output_dir, exist_ok=True)
    cfg = RuleConfig()
    detector = BlunderDetector()

    all_blunders: List[dict] = []
    blunder_counter: Counter = Counter()

    start_time = time.time()
    print(f"============================================================")
    print(f" [START] 启动军棋战术漏洞自动化挖掘管道 (Junqi Blunder Miner)")
    print(f" 对局局数: {games} | 白方: {engine_white} | 黑方: {engine_black} | 基础种子: {seed_base}")
    print(f" 输出目录: {output_dir}")
    print(f"============================================================\n")

    for g_idx in range(games):
        seed = seed_base + g_idx * 997
        rng = random.Random(seed * 37 + 11)

        s0 = make_strategy(engine_white, seed=seed * 31 + 1, model_path=model_path, device=device)
        s1 = make_strategy(engine_black, seed=seed * 31 + 2, model_path=model_path, device=device)

        st = deal(random.Random(seed), cfg)
        history: List[dict] = []
        seen = Counter()

        game_start = time.time()
        while not st.is_terminal():
            pk = position_key(st)
            seen[pk] += 1
            if seen[pk] >= cfg.repetition_draw_count:
                st.winner = -1
                st.win_reason = "repetition"
                break

            acts = st.legal_actions()
            if not acts:
                st.winner = 1 - st.turn
                st.win_reason = "immobilized"
                break

            avoid = {k for k, n in seen.items() if n >= cfg.repetition_draw_count - 1}
            mover_seat = st.turn
            strategy = s0 if mover_seat == 0 else s1

            act = strategy.choose(st, rng, avoid=avoid, history_counts=seen)

            mover = st.board.get(act.frm) if act.kind == "move" else None
            target = st.board.get(act.to) if act.kind == "move" else None

            step_record = {
                "ply": st.ply,
                "seat": mover_seat,
                "action": act,
                "action_str": str(act),
                "mover": mover,
                "target": target,
                "state": st.copy(),
            }
            history.append(step_record)

            st = st.apply(act)

        game_elapsed = time.time() - game_start

        # ---------------- 对该局完整轨迹进行全探针扫描 ----------------
        game_blunders: List[dict] = []
        for p_idx in range(len(history)):
            # 探针 1: 弃营丢营
            res1 = detector.check_camp_abandonment(p_idx, history)
            if res1:
                game_blunders.append((p_idx, res1))

            # 探针 2: 大子白送
            res2 = detector.check_top_piece_suicide(p_idx, history)
            if res2:
                game_blunders.append((p_idx, res2))

            # 探针 3: 工兵自杀
            res3 = detector.check_sapper_suicide(p_idx, history)
            if res3:
                game_blunders.append((p_idx, res3))

            # 探针 4: 循环踱步
            res4 = detector.check_ping_pong_loop(p_idx, history)
            if res4:
                game_blunders.append((p_idx, res4))

            # 探针 5: 龟缩拒翻
            res5 = detector.check_camp_turtling(p_idx, history)
            if res5:
                game_blunders.append((p_idx, res5))

        for p_idx, b_info in game_blunders:
            step = history[p_idx]
            b_type = b_info["type"]
            blunder_counter[b_type] += 1

            # 上下文切片
            ctx_start = max(0, p_idx - 2)
            ctx_end = min(len(history), p_idx + 3)
            context_moves = [
                f"Ply {h['ply']} [Seat {h['seat']}]: {h['action_str']}"
                for h in history[ctx_start:ctx_end]
            ]

            all_blunders.append({
                "game_id": g_idx + 1,
                "seed": seed,
                "ply": step["ply"],
                "seat": step["seat"],
                "engine": engine_white if step["seat"] == 0 else engine_black,
                "action": step["action_str"],
                "blunder_type": b_type,
                "sub_type": b_info.get("sub_type"),
                "severity": b_info.get("severity", "MEDIUM"),
                "description": b_info["description"],
                "context_moves": context_moves,
                "board_state": serialize_state_simple(step["state"]),
            })

        print(f"[{g_idx+1:02d}/{games:02d}] 种子: {seed} | 手数: {st.ply} | 结果: 胜者={st.winner} ({st.win_reason}) "
              f"| 捕获失误: {len(game_blunders)} 个 | 耗时: {game_elapsed:.2f}s")

    total_elapsed = time.time() - start_time
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- 导出 JSON 错题库 ----------------
    json_path = os.path.join(output_dir, f"blunders_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": ts,
            "total_games": games,
            "engine_white": engine_white,
            "engine_black": engine_black,
            "total_blunders": len(all_blunders),
            "blunder_counts": dict(blunder_counter),
            "blunders": all_blunders,
        }, f, ensure_ascii=False, indent=2)

    # ---------------- 导出 Markdown 报告 ----------------
    md_path = os.path.join(output_dir, f"blunder_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 军棋战术失误自动挖掘诊断报告\n\n")
        f.write(f"- **生成时间**：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **自博弈局数**：{games} 局\n")
        f.write(f"- **对阵配置**：座位 0 (红) `{engine_white}` vs 座位 1 (蓝) `{engine_black}`\n")
        f.write(f"- **总耗时**：{total_elapsed:.2f} 秒（平均 {total_elapsed/games:.2f} 秒/局）\n")
        f.write(f"- **共捕获失误总数**：`{len(all_blunders)}` 处\n\n")

        f.write(f"## 一、 失误类型分布统计\n\n")
        f.write(f"| 失误类别 | 中文说明 | 捕获次数 | 严重程度 |\n")
        f.write(f"| :--- | :--- | :--- | :--- |\n")
        type_desc = {
            "camp_abandonment": ("弃营丢营 / 弃营送死", "CRITICAL"),
            "top_piece_suicide": ("大子白送 / 炸弹贱卖", "CRITICAL"),
            "sapper_suicide": ("工兵非排雷白送", "HIGH"),
            "ping_pong_loop": ("无意义循环往复踱步", "MEDIUM"),
            "camp_turtling": ("行营龟缩拒不翻棋", "LOW"),
        }
        for b_type, count in blunder_counter.most_common():
            desc, sev = type_desc.get(b_type, ("未知失误", "MEDIUM"))
            f.write(f"| `{b_type}` | {desc} | **{count}** | `{sev}` |\n")

        f.write(f"\n## 二、 典型致命失误案例复盘 (Top Blunder Cases)\n\n")
        # 按严重程度优先展示 (CRITICAL > HIGH > MEDIUM > LOW)
        sev_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_blunders = sorted(all_blunders, key=lambda x: sev_rank.get(x.get("severity", "MEDIUM"), 2))
        sample_cases = sorted_blunders[:10]
        if not sample_cases:
            f.write(f"> 🎉 本批次自博弈未捕获到任何战术探针失误，引擎战术硬约束运行平稳！\n\n")
        else:
            for idx, item in enumerate(sample_cases, 1):
                f.write(f"### 案例 {idx}: {item['description']}\n\n")
                f.write(f"- **对局/种子**：Game #{item['game_id']} (Seed: `{item['seed']}`)\n")
                f.write(f"- **发生手数**：第 {item['ply']} 手 | 执行引擎: `{item['engine']}`\n")
                f.write(f"- **执行动作**：`{item['action']}`\n")
                f.write(f"- **上下文走法**：\n")
                for c_move in item['context_moves']:
                    f.write(f"  - {c_move}\n")
                f.write(f"\n")

        f.write(f"## 三、 结构化测试集导出\n\n")
        f.write(f"所有包含完整棋盘与棋子布局的失误局面已同步导出至结构化数据集：\n")
        f.write(f"- 错题库文件：`{json_path}`\n")
        f.write(f"- 可直接调用 `tests/test_camp_topology_fuzz.py` 或编写回归用例加载该 JSON 持续测试。\n")

    print(f"\n============================================================")
    print(f" [DONE] 挖掘完成！共捕获 {len(all_blunders)} 个战术失误")
    for b_type, count in blunder_counter.most_common():
        print(f"  - {b_type}: {count} 次")
    print(f" [OUTPUT] 结构化错题库已保存: {json_path}")
    print(f" [REPORT] 诊断报告已生成: {md_path}")
    print(f"============================================================\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="军棋战术失误自动挖掘管道")
    parser.add_argument("--games", type=int, default=10, help="自博弈对局数 (默认 10)")
    parser.add_argument("--engine-white", type=str, default="hybrid2", help="座位 0 引擎 (默认 hybrid2)")
    parser.add_argument("--engine-black", type=str, default="expert2", help="座位 1 引擎 (默认 expert2)")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--output-dir", type=str, default="reports/blunders", help="输出报告目录")
    parser.add_argument("--model-path", type=str, default=None, help="模型路径")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")

    args = parser.parse_args()
    run_mining(
        games=args.games,
        engine_white=args.engine_white,
        engine_black=args.engine_black,
        seed_base=args.seed,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
