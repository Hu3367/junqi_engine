"""生成分阶段基准评测集（Opening 200 局, Midgame 200 局, Endgame 200 局）。

保存于 eval_sets/ 目录下，用于客观衡量各阶段棋力。
"""
from __future__ import annotations

import json
import os
import random

from junqi.analysis import detect_phase, PHASE_OPENING, PHASE_MIDGAME
from junqi.config import RuleConfig
from junqi.endgame_gen import gen_endgame
from junqi.state import deal


def gen_all_eval_sets(out_dir: str = "eval_sets", n_per_set: int = 200, seed: int = 2026):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    cfg = RuleConfig()

    print(f"正在生成分阶段评测集 (每套 {n_per_set} 局)...")

    # 1. Opening 评测集 (首翻后 1~5 手)
    opening_path = os.path.join(out_dir, "opening.jsonl")
    with open(opening_path, "w", encoding="utf-8") as f:
        for i in range(n_per_set):
            st = deal(random.Random(seed * 100 + i), cfg)
            # 随机走 1~5 步翻子
            flips = rng.randint(1, 5)
            for _ in range(flips):
                acts = [a for a in st.legal_actions() if a.kind == "flip"]
                if acts:
                    st = st.apply(rng.choice(acts))
            f.write(st.to_json() + "\n")
    print(f"  已生成 Opening 评测集: {opening_path}")

    # 2. Midgame 评测集 (暗子 6~20 局)
    midgame_path = os.path.join(out_dir, "midgame.jsonl")
    with open(midgame_path, "w", encoding="utf-8") as f:
        for i in range(n_per_set):
            st = deal(random.Random(seed * 200 + i), cfg)
            # 随机推演若干手直至进入中盘
            for _ in range(120):
                if 6 <= len(st.hidden_positions()) <= 20:
                    break
                acts = st.legal_actions()
                if not acts or st.is_terminal():
                    break
                st = st.apply(rng.choice(acts))
            f.write(st.to_json() + "\n")
    print(f"  已生成 Midgame 评测集: {midgame_path}")

    # 3. Endgame 评测集 (含死区/非死区各半)
    endgame_path = os.path.join(out_dir, "endgame.jsonl")
    with open(endgame_path, "w", encoding="utf-8") as f:
        for i in range(n_per_set):
            fortress_flag = (i % 2 == 0)
            bal = rng.uniform(-0.5, 0.5)
            st = gen_endgame(
                rng, cfg,
                material_balance=bal,
                hidden_k=rng.randint(2, 5),
                my_engineers=rng.randint(0, 1),
                opp_engineers=0 if fortress_flag else rng.randint(0, 1),
                fortress=fortress_flag
            )
            f.write(st.to_json() + "\n")
    print(f"  已生成 Endgame 评测集: {endgame_path}")


if __name__ == "__main__":
    gen_all_eval_sets()
