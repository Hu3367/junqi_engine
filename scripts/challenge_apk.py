"""官方 APK 原生假想敌 (ApkNativeAgent) 命令行挑战评测脚本。

支持多进程并发对弈加速，充分释放多核 CPU 算力。

使用示例:
  # 默认自动开启多进程并发，挑战 40 局高级假想敌
  venv\\Scripts\\python.exe scripts/challenge_apk.py --agent expert --games 40 --level advanced

  # 指定 8 个并发进程
  venv\\Scripts\\python.exe scripts/challenge_apk.py --agent expert --games 40 --level advanced --workers 8

  # 单进程排查问题
  venv\\Scripts\\python.exe scripts/challenge_apk.py --agent expert --games 2 --level beginner --workers 1
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from junqi.ai import Agent, ApkNativeAgent, ExpertAgent, HybridAgent, NNAgent
from junqi.config import RuleConfig, SearchConfig
from junqi.state import GameState, deal, position_key


def _run_single_match_game(task_args: tuple) -> dict:
    """独立 Worker 子进程：执行单局镜像轮换对战。"""
    game_idx, agent_type, level, base_seed, max_plies, depth, model_path, device = task_args
    seed = base_seed + game_idx
    # 偶数局 Candidate 执先(0)，奇数局 APK 执先(0)
    a_seat = 0 if (game_idx % 2 == 0) else 1
    b_seat = 1 - a_seat

    # 子进程独立初始化智能体实例
    if agent_type == "expert":
        candidate = ExpertAgent(search=SearchConfig(depth=depth), seed=seed)
    elif agent_type == "hybrid":
        if os.path.exists(model_path):
            candidate = HybridAgent(model_path=model_path, search_depth=depth, device=device, seed=seed)
        else:
            candidate = ExpertAgent(search=SearchConfig(depth=depth), seed=seed)
    elif agent_type == "search":
        candidate = Agent(search=SearchConfig(depth=depth), seed=seed)
    elif agent_type == "nn":
        candidate = NNAgent(model_path=model_path, simulations=0, device=device, seed=seed)
    else:
        candidate = ExpertAgent(search=SearchConfig(depth=depth), seed=seed)

    opponent = ApkNativeAgent(level=level, seed=seed)
    agents = {a_seat: candidate, b_seat: opponent}

    cfg = RuleConfig(max_plies=max_plies) if max_plies > 0 else RuleConfig()
    st = deal(random.Random(seed), cfg)
    pos_seen = {position_key(st): 1}
    plies = 0

    while not st.is_terminal():
        mover = agents[st.turn]
        act = mover.select_action(st)
        if act is None:
            break
        st = st.apply(act)
        plies += 1
        pk = position_key(st)
        pos_seen[pk] = pos_seen.get(pk, 0) + 1

    is_rep = (st.winner == -1 and st.win_reason == "repetition")
    return {
        "game_idx": game_idx,
        "winner": st.winner,
        "a_seat": a_seat,
        "plies": plies,
        "is_rep": is_rep,
    }


def main():
    import torch
    default_workers = max(1, min(16, os.cpu_count() or 4))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model = "models/bc_best.pt" if os.path.exists("models/bc_best.pt") else "models/best.pt"

    parser = argparse.ArgumentParser(description="自研 AI vs 官方 APK 原生假想敌挑战评测 (支持多核 CPU 并发与 GPU 加速)")
    parser.add_argument("--agent", choices=["expert", "hybrid", "search", "nn"], default="expert",
                        help="参评的自研智能体类型 (默认 expert; hybrid/nn 支持 GPU)")
    parser.add_argument("--level", choices=["beginner", "intermediate", "advanced"], default="intermediate",
                        help="官方 APK 假想敌难度 (默认 intermediate)")
    parser.add_argument("--games", type=int, default=10,
                        help="对抗对局数 (先后手各半，默认 10)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="基础发牌随机种子 (默认 2026)")
    parser.add_argument("--model", type=str, default=default_model,
                        help=f"神经网络模型路径 (默认 {default_model})")
    parser.add_argument("--depth", type=int, default=1,
                        help="自研搜索/混合引擎的战术搜索深度 (默认 1)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=default_device,
                        help=f"神经网络计算设备 (默认 {default_device})")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"并发对战 Worker 进程数 (默认 {default_workers})")
    parser.add_argument("--max_plies", type=int, default=0,
                        help="单局限步强制截断 (0 代表使用规则默认 1000 步)")
    args = parser.parse_args()

    dev_info = f", GPU设备: {args.device.upper()}" if args.agent in ("hybrid", "nn") else ""
    print(f"=== 正在初始化参评智能体: {args.agent.upper()} (并发进程数: {args.workers}{dev_info}) ===")
    print(f"=== 开始基准对抗: {args.agent} vs APK {args.level} (共 {args.games} 局) ===")

    t0 = time.time()
    task_list = [
        (i, args.agent, args.level, args.seed, args.max_plies, args.depth, args.model, args.device)
        for i in range(args.games)
    ]

    if args.workers <= 1:
        results = [_run_single_match_game(task) for task in task_list]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(_run_single_match_game, task_list)

    elapsed = time.time() - t0

    # 统计指标
    wins_a = sum(1 for r in results if r["winner"] == r["a_seat"])
    wins_b = sum(1 for r in results if r["winner"] == (1 - r["a_seat"]))
    draws = sum(1 for r in results if r["winner"] == -1)
    rep_draws = sum(1 for r in results if r["is_rep"])
    total_plies = sum(r["plies"] for r in results)

    score_rate_a = (wins_a + 0.5 * draws) / args.games if args.games else 0.5
    draw_rate = draws / args.games if args.games else 0.0
    rep_rate = rep_draws / args.games if args.games else 0.0
    avg_len = total_plies / args.games if args.games else 0.0

    clipped_score = max(0.01, min(0.99, score_rate_a))
    elo_diff = -400.0 * math.log10(1.0 / clipped_score - 1.0)

    print("\n" + "=" * 55)
    print(f"           对战战报 (总耗时: {elapsed:.1f} 秒)")
    print("=" * 55)
    print(f" 对战双方     : {args.agent.upper()} vs 官方 APK 原生 AI ({args.level})")
    print(f" 并发核心数   : {args.workers} 个进程")
    print(f" 总局数       : {args.games} 局 (先后手各半)")
    print(f" 战绩         : {wins_a} 胜 / {draws} 和 / {wins_b} 负")
    print(f" 得分率       : {score_rate_a:.1%} (胜=1, 和=0.5, 负=0)")
    print(f" 和棋率       : {draw_rate:.1%} (其中循环和棋: {rep_draws} 局)")
    print(f" 平均对局手数 : {avg_len:.1f} 手")
    print(f" 相对 Elo 分差: {elo_diff:+.1f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
