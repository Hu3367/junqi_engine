"""新老引擎基准对抗测试：ExpertAgent (暗棋专家引擎) vs Agent (旧 PIMC 引擎)。

依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §7 评测协议：
- 双方先后手各半（镜像对局）
- 固定随机种子
- 统计胜、和、负、吃旗胜率、无吃子和棋率、平均对局手数
"""
from __future__ import annotations

import time
from collections import Counter

from junqi.config import RuleConfig
from junqi.selfplay import play_game


def run_benchmark(n_pairs: int = 15, depth: int = 2):
    print(f"================================================================")
    print(f" 引擎对抗评测: ExpertAgent (expert{depth}) vs 旧 PIMC (search{depth})")
    print(f" 总局数: {n_pairs * 2} 局 (先后手各半镜像)")
    print(f"================================================================")

    cfg = RuleConfig()
    results = {"expert_win": 0, "draw": 0, "pimc_win": 0}
    reasons = Counter()
    total_plies = 0

    t0 = time.perf_counter()

    for i in range(n_pairs):
        seed = 2026 + i

        # Game 1: Expert 先手 (seat 0) vs PIMC 后手 (seat 1)
        rec1 = play_game(f"expert{depth}", f"search{depth}", seed=seed, cfg=cfg)
        w1 = rec1["winner"]
        reasons[rec1["reason"]] += 1
        total_plies += rec1["plies"]
        if w1 == 0:
            results["expert_win"] += 1
        elif w1 == 1:
            results["pimc_win"] += 1
        else:
            results["draw"] += 1

        # Game 2: PIMC 先手 (seat 0) vs Expert 后手 (seat 1)
        rec2 = play_game(f"search{depth}", f"expert{depth}", seed=seed, cfg=cfg)
        w2 = rec2["winner"]
        reasons[rec2["reason"]] += 1
        total_plies += rec2["plies"]
        if w2 == 1:
            results["expert_win"] += 1
        elif w2 == 0:
            results["pimc_win"] += 1
        else:
            results["draw"] += 1

        print(f"  [轮次 {i+1:2d}/{n_pairs}] 局1(先手): {rec1['reason']}(胜者{rec1['winner']}), 局2(后手): {rec2['reason']}(胜者{rec2['winner']}) | 当前总战绩: Expert {results['expert_win']} 胜 / {results['draw']} 和 / {results['pimc_win']} 负", flush=True)

    elapsed = time.perf_counter() - t0
    total_games = n_pairs * 2
    expert_score = results["expert_win"] + 0.5 * results["draw"]
    expert_score_rate = expert_score / total_games * 100.0

    print(f"\n====================== 评测统计结果 ======================", flush=True)
    print(f"总对局数: {total_games}", flush=True)
    print(f"耗时: {elapsed:.2f} 秒 (平均 {elapsed / total_games:.2f} 秒/局)", flush=True)
    print(f"ExpertAgent 战绩: {results['expert_win']} 胜 / {results['draw']} 和 / {results['pimc_win']} 负", flush=True)
    print(f"ExpertAgent 得分率 (胜=1,和=0.5): {expert_score_rate:.1f}%", flush=True)
    print(f"终局原因分布: {dict(reasons)}", flush=True)
    print(f"平均每局手数: {total_plies / total_games:.1f}", flush=True)
    print(f"==========================================================\n", flush=True)


if __name__ == "__main__":
    run_benchmark(n_pairs=6, depth=2)
