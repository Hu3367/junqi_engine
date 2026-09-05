"""P2 阶段混合引擎 (HybridAgent) 对抗基准与实战评估脚本。"""
from __future__ import annotations

import os
import random
import time

from junqi.ai import ExpertAgent, HybridAgent
from junqi.config import RuleConfig
from junqi.selfplay import make_strategy
from junqi.state import deal


def run_benchmark_match(n_pairs: int = 5):
    """运行 HybridAgent vs Greedy 与 HybridAgent vs Expert2 的对抗评估。"""
    print(f"=== P2 阶段实战对抗评测 (对局数: {n_pairs * 2} 局镜像) ===")

    strat_hybrid = make_strategy("hybrid2", seed=2026)
    strat_greedy = make_strategy("greedy", seed=2026)
    strat_expert = make_strategy("expert2", seed=2026)

    # 1. HybridAgent vs Greedy
    print("\n--- 1. HybridAgent vs Greedy 对抗 ---")
    h_v_g_wins = 0
    h_v_g_losses = 0
    h_v_g_draws = 0

    for i in range(n_pairs):
        # 顺盘: 先手 Hybrid, 后手 Greedy
        seed = 1000 + i
        st = deal(rng=random.Random(seed))
        rng_h = random.Random(seed * 2 + 1)
        rng_g = random.Random(seed * 2 + 2)

        while not st.is_terminal() and st.ply < 200:
            if st.turn == 0:
                act = strat_hybrid.choose(st, rng_h)
            else:
                act = strat_greedy.choose(st, rng_g)
            st = st.apply(act)

        if st.winner == 0:
            h_v_g_wins += 1
            res = "Hybrid 胜"
        elif st.winner == 1:
            h_v_g_losses += 1
            res = "Greedy 胜"
        else:
            h_v_g_draws += 1
            res = "和棋"
        print(f"  对局 {i*2+1:02d} (先手Hybrid): {res} (ply={st.ply}, reason={st.win_reason})")

        # 逆盘: 先手 Greedy, 后手 Hybrid
        st = deal(rng=random.Random(seed))
        while not st.is_terminal() and st.ply < 200:
            if st.turn == 0:
                act = strat_greedy.choose(st, rng_g)
            else:
                act = strat_hybrid.choose(st, rng_h)
            st = st.apply(act)

        if st.winner == 1:
            h_v_g_wins += 1
            res = "Hybrid 胜"
        elif st.winner == 0:
            h_v_g_losses += 1
            res = "Greedy 胜"
        else:
            h_v_g_draws += 1
            res = "和棋"
        print(f"  对局 {i*2+2:02d} (后手Hybrid): {res} (ply={st.ply}, reason={st.win_reason})")

    total_g = n_pairs * 2
    score_g = (h_v_g_wins + 0.5 * h_v_g_draws) / total_g
    print(f"\n>> Hybrid vs Greedy 结果: 胜 {h_v_g_wins} / 负 {h_v_g_losses} / 和 {h_v_g_draws} (得分率: {score_g*100:.1f}%)")

    # 2. HybridAgent vs Expert2
    print("\n--- 2. HybridAgent vs Expert2 对抗 ---")
    h_v_e_wins = 0
    h_v_e_losses = 0
    h_v_e_draws = 0

    for i in range(n_pairs):
        seed = 2000 + i
        st = deal(rng=random.Random(seed))
        rng_h = random.Random(seed * 2 + 1)
        rng_e = random.Random(seed * 2 + 2)

        while not st.is_terminal() and st.ply < 200:
            if st.turn == 0:
                act = strat_hybrid.choose(st, rng_h)
            else:
                act = strat_expert.choose(st, rng_e)
            st = st.apply(act)

        if st.winner == 0:
            h_v_e_wins += 1
            res = "Hybrid 胜"
        elif st.winner == 1:
            h_v_e_losses += 1
            res = "Expert 胜"
        else:
            h_v_e_draws += 1
            res = "和棋"
        print(f"  对局 {i*2+1:02d} (先手Hybrid): {res} (ply={st.ply}, reason={st.win_reason})")

        # 逆盘
        st = deal(rng=random.Random(seed))
        while not st.is_terminal() and st.ply < 200:
            if st.turn == 0:
                act = strat_expert.choose(st, rng_e)
            else:
                act = strat_hybrid.choose(st, rng_h)
            st = st.apply(act)

        if st.winner == 1:
            h_v_e_wins += 1
            res = "Hybrid 胜"
        elif st.winner == 0:
            h_v_e_losses += 1
            res = "Expert 胜"
        else:
            h_v_e_draws += 1
            res = "和棋"
        print(f"  对局 {i*2+2:02d} (后手Hybrid): {res} (ply={st.ply}, reason={st.win_reason})")

    total_e = n_pairs * 2
    score_e = (h_v_e_wins + 0.5 * h_v_e_draws) / total_e
    print(f"\n>> Hybrid vs Expert2 结果: 胜 {h_v_e_wins} / 负 {h_v_e_losses} / 和 {h_v_e_draws} (得分率: {score_e*100:.1f}%)")


if __name__ == "__main__":
    run_benchmark_match(n_pairs=3)
