"""ExpertAgent vs 官方 APK 原生 AI (难度3 / advanced) 200 手 (100 回合) 对弈评测。

评测依据与规范：
- 严格按照 200 手 (100 回合) 截断 (max_plies=200)。
- 军棋翻棋高水平对抗中防御和棋概率较高，因此以终局场面分 (物质分/态势分) 作为最终裁决标准。
- 支持多核 CPU 并发并行计算。
- 输出详细数据：包括物质分差、大子存活、炸弹利用、工兵留存、行营占领数与综合场面分。
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from junqi.ai import ApkNativeAgent, ExpertAgent
from junqi.config import EvalWeights, RuleConfig, SearchConfig
from junqi.eval_expert import DEFAULT_PIECE_VALUES, _get_alive_counts, evaluate_expert
from junqi.rules import CAMPS, RANK_CN, Rank, other
from junqi.state import GameState, deal, position_key


APK_PIECE_VALUES: dict[Rank, float] = {
    Rank.SI: 2560.0,
    Rank.JUN: 1280.0,
    Rank.SHI: 640.0,
    Rank.LV: 320.0,
    Rank.TUAN: 160.0,
    Rank.YING: 80.0,
    Rank.LIAN: 40.0,
    Rank.PAI: 30.0,
    Rank.GONG: 80.0,
    Rank.ZHA: 426.0,
    Rank.LEI: 70.0,
    Rank.QI: 50.0,
}


def _evaluate_game_state(st: GameState, expert_seat: int) -> dict[str, Any]:
    """计算终局时的多维度场面分与子力留存统计。"""
    expert_color = st.seat_color.get(expert_seat)
    apk_color = other(expert_color) if expert_color else None

    # 1. 存活统计 (明子 + 暗子池期望份额 = 初始编制 - 阵亡)
    my_counts, opp_counts = _get_alive_counts(st, my=expert_color)

    # 2. 棋盘实际存活子统计
    expert_board_pieces: dict[Rank, int] = {rk: 0 for rk in Rank}
    apk_board_pieces: dict[Rank, int] = {rk: 0 for rk in Rank}
    expert_camps = 0
    apk_camps = 0
    unrevealed_count = 0

    for pos, pc in st.board.items():
        if not pc.revealed:
            unrevealed_count += 1
            continue
        if expert_color and pc.color == expert_color:
            expert_board_pieces[pc.rank] += 1
            if pos in CAMPS:
                expert_camps += 1
        elif apk_color and pc.color == apk_color:
            apk_board_pieces[pc.rank] += 1
            if pos in CAMPS:
                apk_camps += 1

    # 3. 物质分计算 (标准线性分 & APK 等比分)
    std_exp_mat = sum(DEFAULT_PIECE_VALUES[rk] * my_counts.get((expert_color, rk), 0) for rk in DEFAULT_PIECE_VALUES) if expert_color else 0.0
    std_apk_mat = sum(DEFAULT_PIECE_VALUES[rk] * opp_counts.get((apk_color, rk), 0) for rk in DEFAULT_PIECE_VALUES) if apk_color else 0.0
    std_mat_delta = std_exp_mat - std_apk_mat

    apk_exp_mat = sum(APK_PIECE_VALUES[rk] * my_counts.get((expert_color, rk), 0) for rk in APK_PIECE_VALUES) if expert_color else 0.0
    apk_apk_mat = sum(APK_PIECE_VALUES[rk] * opp_counts.get((apk_color, rk), 0) for rk in APK_PIECE_VALUES) if apk_color else 0.0
    apk_mat_delta = apk_exp_mat - apk_apk_mat

    # 4. 专家级综合态势估值 (evaluate_expert)
    expert_board_score = evaluate_expert(st, seat=expert_seat)

    # 5. 核心子力存活情况
    exp_si = my_counts.get((expert_color, Rank.SI), 0) if expert_color else 0
    apk_si = opp_counts.get((apk_color, Rank.SI), 0) if apk_color else 0
    exp_jun = my_counts.get((expert_color, Rank.JUN), 0) if expert_color else 0
    apk_jun = opp_counts.get((apk_color, Rank.JUN), 0) if apk_color else 0
    exp_shi = my_counts.get((expert_color, Rank.SHI), 0) if expert_color else 0
    apk_shi = opp_counts.get((apk_color, Rank.SHI), 0) if apk_color else 0
    exp_lv = my_counts.get((expert_color, Rank.LV), 0) if expert_color else 0
    apk_lv = opp_counts.get((apk_color, Rank.LV), 0) if apk_color else 0
    exp_zha = my_counts.get((expert_color, Rank.ZHA), 0) if expert_color else 0
    apk_zha = opp_counts.get((apk_color, Rank.ZHA), 0) if apk_color else 0
    exp_gong = my_counts.get((expert_color, Rank.GONG), 0) if expert_color else 0
    apk_gong = opp_counts.get((apk_color, Rank.GONG), 0) if apk_color else 0
    exp_lei = my_counts.get((expert_color, Rank.LEI), 0) if expert_color else 0
    apk_lei = opp_counts.get((apk_color, Rank.LEI), 0) if apk_color else 0

    exp_total_alive = sum(my_counts.values()) if expert_color else 0
    apk_total_alive = sum(opp_counts.values()) if apk_color else 0

    return {
        "std_exp_mat": std_exp_mat,
        "std_apk_mat": std_apk_mat,
        "std_mat_delta": std_mat_delta,
        "apk_exp_mat": apk_exp_mat,
        "apk_apk_mat": apk_apk_mat,
        "apk_mat_delta": apk_mat_delta,
        "expert_board_score": expert_board_score,
        "expert_camps": expert_camps,
        "apk_camps": apk_camps,
        "unrevealed_count": unrevealed_count,
        "exp_total_alive": exp_total_alive,
        "apk_total_alive": apk_total_alive,
        "exp_si": exp_si,
        "apk_si": apk_si,
        "exp_jun": exp_jun,
        "apk_jun": apk_jun,
        "exp_shi": exp_shi,
        "apk_shi": apk_shi,
        "exp_lv": exp_lv,
        "apk_lv": apk_lv,
        "exp_zha": exp_zha,
        "apk_zha": apk_zha,
        "exp_gong": exp_gong,
        "apk_gong": apk_gong,
        "exp_lei": exp_lei,
        "apk_lei": apk_lei,
    }


def _run_eval_game(task_args: tuple) -> dict:
    game_idx, base_seed, max_plies, expert_depth = task_args
    seed = base_seed + game_idx

    # 轮流执先：偶数局 Expert 执红(seat 0)，奇数局 APK 执红(seat 0)
    expert_seat = 0 if (game_idx % 2 == 0) else 1
    apk_seat = 1 - expert_seat

    expert_agent = ExpertAgent(search=SearchConfig(depth=expert_depth), seed=seed)
    apk_agent = ApkNativeAgent(level="advanced", seed=seed)
    agents = {expert_seat: expert_agent, apk_seat: apk_agent}

    cfg = RuleConfig(max_plies=max_plies)
    st = deal(random.Random(seed), cfg)
    pos_seen = {position_key(st): 1}
    plies = 0
    exp_flips = 0
    apk_flips = 0
    exp_captures = 0
    apk_captures = 0

    t_start = time.time()
    while not st.is_terminal():
        current_turn = st.turn
        mover = agents[current_turn]
        act = mover.select_action(st)
        if act is None:
            break
        if act.kind == "flip":
            if current_turn == expert_seat:
                exp_flips += 1
            else:
                apk_flips += 1
        elif act.kind == "move" and act.to in st.board:
            if current_turn == expert_seat:
                exp_captures += 1
            else:
                apk_captures += 1

        st = st.apply(act)
        plies += 1
        pk = position_key(st)
        pos_seen[pk] = pos_seen.get(pk, 0) + 1

    game_time = time.time() - t_start

    # 计算终局指标
    eval_metrics = _evaluate_game_state(st, expert_seat)

    # 判定胜负类别
    raw_winner = st.winner
    if raw_winner == expert_seat:
        raw_result = "EXPERT_WIN"
        adj_result = "EXPERT_WIN"
    elif raw_winner == apk_seat:
        raw_result = "APK_WIN"
        adj_result = "APK_WIN"
    else:
        raw_result = "DRAW"
        # 和棋下依场面分评判
        # 优先使用专家综合态势分 (evaluate_expert)，次选标准物质分差
        if eval_metrics["expert_board_score"] > 5.0:
            adj_result = "EXPERT_WIN_BY_SCORE"
        elif eval_metrics["expert_board_score"] < -5.0:
            adj_result = "APK_WIN_BY_SCORE"
        else:
            # 微弱差距看物质净分差
            if eval_metrics["std_mat_delta"] > 0:
                adj_result = "EXPERT_WIN_BY_SCORE"
            elif eval_metrics["std_mat_delta"] < 0:
                adj_result = "APK_WIN_BY_SCORE"
            else:
                adj_result = "TRUE_DRAW"

    return {
        "game_idx": game_idx,
        "seed": seed,
        "expert_seat": expert_seat,
        "plies": plies,
        "game_time": game_time,
        "win_reason": st.win_reason or ("max_plies" if plies >= max_plies else "normal"),
        "raw_winner": raw_winner,
        "raw_result": raw_result,
        "adj_result": adj_result,
        "exp_flips": exp_flips,
        "apk_flips": apk_flips,
        "exp_captures": exp_captures,
        "apk_captures": apk_captures,
        **eval_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="ExpertAgent vs 官方 APK 难度3 对弈与场面分综合评测 (200手截断)")
    parser.add_argument("--games", type=int, default=20, help="总对局数 (默认 20 局，先后手各半)")
    parser.add_argument("--seed", type=int, default=2026, help="基础随机种子 (默认 2026)")
    parser.add_argument("--workers", type=int, default=10, help="多进程并发数 (默认 10)")
    parser.add_argument("--max_plies", type=int, default=200, help="截断步数 (默认 200 手 / 100 回合)")
    parser.add_argument("--depth", type=int, default=2, help="ExpertAgent 搜索深度 (默认 2)")
    parser.add_argument("--output", type=str, default="scripts/expert_vs_apk_200ply_results.json",
                        help="详细评估结果保存路径")
    args = parser.parse_args()

    print("=" * 70)
    print("  军棋翻棋: ExpertAgent (自研专家) vs ApkNativeAgent (官方难度3/Advanced)")
    print(f"  对局设置: 共 {args.games} 局 | 截断限制: {args.max_plies} 手 (100 回合) | 并发: {args.workers} 进程")
    print(f"  智能体配置: Expert (深度 {args.depth}) vs APK Advanced (深度 4, 限时 1000ms)")
    print("=" * 70)

    task_list = [(i, args.seed, args.max_plies, args.depth) for i in range(args.games)]

    t0 = time.time()
    if args.workers <= 1:
        results = [_run_eval_game(task) for task in task_list]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(_run_eval_game, task_list)
    total_time = time.time() - t0

    # 打印逐局明细表
    print("\n" + "-" * 110)
    print(f"{'局号':<4} | {'执先':<6} | {'步数':<5} | {'原始终局':<12} | {'裁决结果':<18} | {'存活(我/敌)':<11} | {'物质差(标/APK)':<16} | {'场面综合分':<10} | {'占营(我/敌)'}")
    print("-" * 110)

    for r in results:
        seat_str = "Expert" if r["expert_seat"] == 0 else "APK"
        surv_str = f"{r['exp_total_alive']:02d} / {r['apk_total_alive']:02d}"
        mat_str = f"{r['std_mat_delta']:+6.1f} / {r['apk_mat_delta']:+6.0f}"
        camp_str = f"{r['expert_camps']} / {r['apk_camps']}"
        print(f"{r['game_idx']+1:<4} | {seat_str:<6} | {r['plies']:<5} | {r['raw_result']:<12} | {r['adj_result']:<18} | {surv_str:<11} | {mat_str:<16} | {r['expert_board_score']:+10.1f} | {camp_str}")
    print("-" * 110)

    # 聚合汇总统计
    total_games = len(results)
    raw_wins_exp = sum(1 for r in results if r["raw_result"] == "EXPERT_WIN")
    raw_wins_apk = sum(1 for r in results if r["raw_result"] == "APK_WIN")
    raw_draws = sum(1 for r in results if r["raw_result"] == "DRAW")

    adj_wins_exp = sum(1 for r in results if "EXPERT" in r["adj_result"])
    adj_wins_apk = sum(1 for r in results if "APK" in r["adj_result"])
    adj_true_draws = sum(1 for r in results if r["adj_result"] == "TRUE_DRAW")

    avg_std_mat_delta = sum(r["std_mat_delta"] for r in results) / total_games
    avg_apk_mat_delta = sum(r["apk_mat_delta"] for r in results) / total_games
    avg_board_score = sum(r["expert_board_score"] for r in results) / total_games

    avg_exp_alive = sum(r["exp_total_alive"] for r in results) / total_games
    avg_apk_alive = sum(r["apk_total_alive"] for r in results) / total_games

    exp_si_alive_rate = sum(r["exp_si"] for r in results) / total_games
    apk_si_alive_rate = sum(r["apk_si"] for r in results) / total_games

    exp_jun_alive_rate = sum(r["exp_jun"] for r in results) / total_games
    apk_jun_alive_rate = sum(r["apk_jun"] for r in results) / total_games

    exp_zha_alive_rate = sum(r["exp_zha"] for r in results) / (total_games * 2)  # 每方2颗
    apk_zha_alive_rate = sum(r["apk_zha"] for r in results) / (total_games * 2)

    exp_gong_alive_rate = sum(r["exp_gong"] for r in results) / (total_games * 3)  # 每方3颗
    apk_gong_alive_rate = sum(r["apk_gong"] for r in results) / (total_games * 3)

    avg_exp_camps = sum(r["expert_camps"] for r in results) / total_games
    avg_apk_camps = sum(r["apk_camps"] for r in results) / total_games

    avg_exp_flips = sum(r["exp_flips"] for r in results) / total_games
    avg_apk_flips = sum(r["apk_flips"] for r in results) / total_games
    avg_exp_captures = sum(r["exp_captures"] for r in results) / total_games
    avg_apk_captures = sum(r["apk_captures"] for r in results) / total_games

    exp_shi_alive_rate = sum(r["exp_shi"] for r in results) / (total_games * 2)  # 每方2师
    apk_shi_alive_rate = sum(r["apk_shi"] for r in results) / (total_games * 2)

    print("\n" + "=" * 65)
    print("                     对战结果汇总与评判")
    print("=" * 65)
    print(f" 总对局数          : {total_games} 局 (总耗时: {total_time:.1f} 秒)")
    print(f" 原始硬判终局      : Expert 胜 {raw_wins_exp} | APK 胜 {raw_wins_apk} | 步数/规则和棋 {raw_draws}")
    print(f" 场面分裁决终局    : Expert 胜 {adj_wins_exp} ({adj_wins_exp/total_games*100:.1f}%) | APK 胜 {adj_wins_apk} ({adj_wins_apk/total_games*100:.1f}%) | 真正均势 {adj_true_draws}")
    print("-" * 65)
    print(f" 平均标准物质净胜分: {avg_std_mat_delta:+.2f} 分 (标称价值)")
    print(f" 平均官方等比净胜分: {avg_apk_mat_delta:+.1f} 分 (APK价值)")
    print(f" 平均专家综合态势分: {avg_board_score:+.2f} 分")
    print(f" 平均存活棋子数    : Expert {avg_exp_alive:.1f} 枚 vs APK {avg_apk_alive:.1f} 枚")
    print(f" 战术攻防频率对比  : 翻棋 Exp {avg_exp_flips:.1f} 次 vs APK {avg_apk_flips:.1f} 次 | 吃子 Exp {avg_exp_captures:.1f} 次 vs APK {avg_apk_captures:.1f} 次")
    print("-" * 65)
    print(" 核心子力存活与控制对比:")
    print(f"   - 司令 (SI) 存活率   : Expert {exp_si_alive_rate*100:.1f}% vs APK {apk_si_alive_rate*100:.1f}%")
    print(f"   - 军长 (JUN) 存活率  : Expert {exp_jun_alive_rate*100:.1f}% vs APK {apk_jun_alive_rate*100:.1f}%")
    print(f"   - 师长 (SHI) 存活率  : Expert {exp_shi_alive_rate*100:.1f}% vs APK {apk_shi_alive_rate*100:.1f}%")
    print(f"   - 炸弹 (ZHA) 留存率  : Expert {exp_zha_alive_rate*100:.1f}% vs APK {apk_zha_alive_rate*100:.1f}%")
    print(f"   - 工兵 (GONG) 留存率 : Expert {exp_gong_alive_rate*100:.1f}% vs APK {apk_gong_alive_rate*100:.1f}%")
    print(f"   - 行营平均占领数     : Expert {avg_exp_camps:.2f} 营 vs APK {avg_apk_camps:.2f} 营 (满营10)")
    print("=" * 65)

    # 保存 JSON
    summary = {
        "total_games": total_games,
        "total_time_seconds": total_time,
        "raw_results": {"expert_wins": raw_wins_exp, "apk_wins": raw_wins_apk, "draws": raw_draws},
        "adjudicated_results": {"expert_wins": adj_wins_exp, "apk_wins": adj_wins_apk, "draws": adj_true_draws},
        "averages": {
            "std_material_delta": avg_std_mat_delta,
            "apk_material_delta": avg_apk_mat_delta,
            "board_score": avg_board_score,
            "expert_alive": avg_exp_alive,
            "apk_alive": avg_apk_alive,
            "expert_flips": avg_exp_flips,
            "apk_flips": avg_apk_flips,
            "expert_captures": avg_exp_captures,
            "apk_captures": avg_apk_captures,
            "expert_camps": avg_exp_camps,
            "apk_camps": avg_apk_camps,
            "expert_si_alive": exp_si_alive_rate,
            "apk_si_alive": apk_si_alive_rate,
            "expert_jun_alive": exp_jun_alive_rate,
            "apk_jun_alive": apk_jun_alive_rate,
            "expert_shi_alive": exp_shi_alive_rate,
            "apk_shi_alive": apk_shi_alive_rate,
            "expert_zha_alive": exp_zha_alive_rate,
            "apk_zha_alive": apk_zha_alive_rate,
            "expert_gong_alive": exp_gong_alive_rate,
            "apk_gong_alive": apk_gong_alive_rate,
        },
        "games": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 完整评测明细已成功导出至: {out_path}")


if __name__ == "__main__":
    main()
