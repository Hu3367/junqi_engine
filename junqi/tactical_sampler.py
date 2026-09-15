"""实战交火局面采样与多进程专家战术打标模块。

依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §5 P2 规范：
1. 从清洗后的真实人类复盘（datasets/p1_v3 来源）中直接提取中盘与残局真实交火局面；
2. 完整保留人类走法上下文与牌面状态；
3. 多进程并发调用 ExpertAgent 进行深度搜索（depth=3 + QSearch），打上高置信度胜负分值与最佳招法标签；
4. 支持增量缓存与 GUI 人工核验数据交互。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

from .analysis import detect_phase
from .ai import ExpertAgent
from .config import RuleConfig, SearchConfig
from .replay import SPECIAL_EVENT, cell_rc, parse_sav, replay_sav
from .rules import Rank, other
from .state import Action, GameState, Piece, WIN_SCORE

SCORE_SCALE = 600.0
CLASS_THRESHOLD = 0.25


def score_to_class(score: float) -> int:
    """专家搜索分数（走子方视角）→ W/D/L 类别（0=Win, 1=Draw, 2=Loss）。"""
    if score >= WIN_SCORE / 2:
        return 0
    if score <= -WIN_SCORE / 2:
        return 2
    v = math.tanh(score / SCORE_SCALE)
    if v > CLASS_THRESHOLD:
        return 0
    if v < -CLASS_THRESHOLD:
        return 2
    return 1


def state_to_dict(st: GameState) -> dict:
    """序列化 GameState 为 JSON 友好格式。"""
    return {
        "turn": st.turn,
        "seat_color": {int(k): v for k, v in st.seat_color.items()},
        "first_flip_done": getattr(st, "first_flip_done", True),
        "board": {f"{r},{c}": {"color": p.color, "rank": p.rank.name, "revealed": p.revealed}
                  for (r, c), p in st.board.items()},
        "dead": [{"color": p.color, "rank": p.rank.name, "revealed": p.revealed} for p in st.dead],
        "ply": st.ply,
        "quiet": st.quiet,
    }


def dict_to_state(d: dict, cfg: Optional[RuleConfig] = None) -> GameState:
    """从 JSON 字典完整还原 GameState。"""
    cfg = cfg or RuleConfig()
    board = {}
    for pos_str, p_dict in d["board"].items():
        r, c = map(int, pos_str.split(","))
        board[(r, c)] = Piece(p_dict["color"], Rank[p_dict["rank"]], revealed=p_dict["revealed"])
    dead = tuple(Piece(p["color"], Rank[p["rank"]], revealed=p.get("revealed", True))
                 for p in d.get("dead", []))
    first_flip_done = d.get(
        "first_flip_done",
        any(p.revealed for p in board.values()) or any(v is not None for v in d.get("seat_color", {}).values())
    )
    return GameState(
        board=board,
        turn=d["turn"],
        seat_color={int(k): v for k, v in d["seat_color"].items()},
        dead=dead,
        first_flip_done=first_flip_done,
        ply=d.get("ply", 0),
        quiet=d.get("quiet", 0),
        cfg=cfg,
    )


def action_to_dict(act: Optional[Action]) -> Optional[dict]:
    if act is None:
        return None
    return {
        "kind": act.kind,
        "frm": list(act.frm),
        "to": list(act.to) if act.to is not None else None,
    }


def dict_to_action(d: Optional[dict]) -> Optional[Action]:
    if not d:
        return None
    return Action(
        kind=d["kind"],
        frm=tuple(d["frm"]),
        to=tuple(d["to"]) if d.get("to") is not None else None,
    )


# ---------------------------------------------------------------- 局面采集与筛选

def is_active_tactical_contact(st: GameState) -> bool:
    """判断当前局面是否处于活跃交火/对峙状态（排除双方无明子或无实质接触局面）。"""
    my_c = st.my_color()
    if not my_c:
        return False
    opp_c = other(my_c)

    # 双方均需有已翻开明子
    has_my_revealed = any(p.revealed and p.color == my_c for p in st.board.values())
    has_opp_revealed = any(p.revealed and p.color == opp_c for p in st.board.values())
    if not (has_my_revealed and has_opp_revealed):
        return False

    # 盘面至少应有走子能力（不仅仅是翻棋）
    acts = st.legal_actions()
    has_moves = any(a.kind == "move" for a in acts)
    return has_moves


def sample_tactical_positions(sav_dir: str = "军旗复盘",
                              target_count: int = 6000,
                              min_plies: int = 20,
                              seed: int = 2026) -> List[dict]:
    """从复盘库中均匀抽取指定数量的中后盘真实交火局面。"""
    cfg = RuleConfig()
    rng = random.Random(seed)

    if not os.path.exists(sav_dir):
        if os.path.exists("../军旗复盘"):
            sav_dir = "../军旗复盘"

    sav_files = sorted(set(glob.glob(os.path.join(sav_dir, "*.sav")) +
                           glob.glob(os.path.join(sav_dir, "**", "*.sav"), recursive=True)))

    print(f"[采样] 找到 {len(sav_files)} 局复盘文件，开始重演筛选中后盘真实交火手...")

    candidates_mid = []
    candidates_end = []

    for fpath in sav_files:
        try:
            g = parse_sav(fpath)
            g = replay_sav(g, cfg=cfg, check=True)
        except Exception:
            continue

        if not g.replay_ok or len(g.moves) < min_plies:
            continue

        from .replay import board_from_table
        board = board_from_table(g.table)
        st = GameState(board=board, cfg=cfg)
        fname = os.path.basename(fpath)

        for ply_idx, (a, b, c) in enumerate(g.moves):
            if (a, b, c) == SPECIAL_EVENT:
                break
            if a == b and c == 1:
                act = Action("flip", cell_rc(a))
            elif a != b and c in (1, 3):
                act = Action("move", cell_rc(a), cell_rc(b))
            else:
                break

            if act not in st.legal_actions():
                break

            ph = detect_phase(st)
            if ph in (1, 2) and not st.is_terminal():
                if is_active_tactical_contact(st):
                    entry = {
                        "game_file": fname,
                        "ply": ply_idx,
                        "phase": ph,
                        "turn": st.turn,
                        "turn_color": st.my_color(),
                        "human_action": action_to_dict(act),
                        "state_dict": state_to_dict(st),
                    }
                    if ph == 1:
                        candidates_mid.append(entry)
                    else:
                        candidates_end.append(entry)

            st = st.apply(act)

    print(f"[采样] 采集完毕：中盘候选 {len(candidates_mid)}，残局候选 {len(candidates_end)}")

    # 均衡抽样：按比例抽取目标数量
    target_mid = int(target_count * 0.55)
    target_end = target_count - target_mid

    rng.shuffle(candidates_mid)
    rng.shuffle(candidates_end)

    selected_mid = candidates_mid[:target_mid]
    selected_end = candidates_end[:target_end]

    sampled = selected_mid + selected_end
    rng.shuffle(sampled)

    # 赋予唯一 ID
    for idx, item in enumerate(sampled):
        item["id"] = idx + 1
        item["human_verified"] = False
        item["verified_label"] = None
        item["comment"] = ""

    print(f"[采样] 最终精选出 {len(sampled)} 个真实交火局面 (中盘 {len(selected_mid)}, 残局 {len(selected_end)})")
    return sampled


# ---------------------------------------------------------------- 专家搜索打标

def _expert_label_task(task_args: tuple) -> dict:
    """独立 Worker 子进程：对单局面执行 Expert 搜索打标。"""
    item_dict, depth, time_limit_ms, seed = task_args
    st = dict_to_state(item_dict["state_dict"])
    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms), seed=seed)

    scored = agent.choose_actions(st, topn=1)
    if scored:
        best_act, best_score = scored[0]
        expert_cls = score_to_class(best_score)
    else:
        best_act = None
        best_score = -WIN_SCORE
        expert_cls = 2  # 无合法走法困毙负

    human_act = dict_to_action(item_dict["human_action"])
    match = (best_act == human_act) if best_act and human_act else False

    res = dict(item_dict)
    res["expert_action"] = action_to_dict(best_act)
    res["expert_score"] = float(best_score)
    res["expert_class"] = int(expert_cls)
    res["human_expert_match"] = bool(match)
    return res


def label_positions_parallel(positions: List[dict],
                            depth: int = 3,
                            time_limit_ms: int = 0,
                            workers: int = 0,
                            seed: int = 2026,
                            cache_path: Optional[str] = None,
                            overwrite: bool = False) -> List[dict]:
    """使用多进程并发为采集的局面打上专家搜索标签。"""
    if workers is None or workers <= 0:
        workers = min(16, os.cpu_count() or 4)

    total = len(positions)
    print(f"[打标] 启动 {workers} 个并发 Worker 进程执行专家深度推演 (depth={depth}, 共 {total} 局)...")

    # 检查已有缓存（支持断点续打）
    cached_map = {}
    if not overwrite and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_map = {item["id"]: item for item in cached_data if "expert_class" in item}
                print(f"[打标] 从缓存恢复已打标局面: {len(cached_map)} / {total}")
        except Exception:
            pass

    to_process = [p for p in positions if p["id"] not in cached_map]
    tasks = [(item, depth, time_limit_ms, seed + i) for i, item in enumerate(to_process)]

    start_t = time.time()
    labeled_results = list(cached_map.values())

    if tasks:
        with mp.Pool(processes=workers) as pool:
            # 使用 chunksize 优化调度
            chunksize = max(1, len(tasks) // (workers * 4))
            for i, res in enumerate(pool.imap_unordered(_expert_label_task, tasks, chunksize=chunksize)):
                labeled_results.append(res)
                if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - start_t
                    speed = (i + 1) / max(elapsed, 0.1)
                    eta = (len(tasks) - (i + 1)) / max(speed, 0.1)
                    print(f"  --> 已完成打标 {len(labeled_results)}/{total} "
                          f"({(len(labeled_results)/total)*100:.1f}%) | 速度: {speed:.1f} 局/s | ETA: {eta:.0f}s")
                    # 增量保存缓存
                    if cache_path:
                        save_distill_dataset(sorted(labeled_results, key=lambda x: x["id"]), cache_path)

    # 保证按 ID 有序
    labeled_results.sort(key=lambda x: x["id"])

    # 统计标签分布与吻合率
    win_c = sum(1 for p in labeled_results if p.get("expert_class") == 0)
    draw_c = sum(1 for p in labeled_results if p.get("expert_class") == 1)
    loss_c = sum(1 for p in labeled_results if p.get("expert_class") == 2)
    match_c = sum(1 for p in labeled_results if p.get("human_expert_match"))

    print(f"\n[打标] 打标全部完成！总耗时: {time.time() - start_t:.1f}s")
    print(f"  - 专家标签分布: 胜={win_c} ({win_c/total*100:.1f}%), "
          f"和={draw_c} ({draw_c/total*100:.1f}%), 负={loss_c} ({loss_c/total*100:.1f}%)")
    print(f"  - 专家与人类走法吻合率: {match_c}/{total} ({match_c/total*100:.1f}%)")

    if cache_path:
        save_distill_dataset(labeled_results, cache_path)

    return labeled_results


def save_distill_dataset(positions: List[dict], out_path: str):
    """保存数据集为 JSON 文件。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)


def load_distill_dataset(path: str) -> List[dict]:
    """读取已存储的数据集。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- CLI 入口

def main():
    parser = argparse.ArgumentParser(description="军棋实战交火局面采样与多进程专家打标")
    parser.add_argument("--sav-dir", default="军旗复盘", help="复盘文件夹")
    parser.add_argument("--out", default="datasets/distill_tactical_labeled.json", help="输出打标文件路径")
    parser.add_argument("--count", type=int, default=6000, help="抽样局面总数 (5000~10000)")
    parser.add_argument("--depth", type=int, default=3, help="专家打标搜索深度")
    parser.add_argument("--time-limit-ms", type=int, default=0, help="单局时间预算（ms，0 为不限时纯按深度搜透）")
    parser.add_argument("--workers", type=int, default=0, help="并行 Worker 数 (0 为自动)")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--only-sample", action="store_true", help="仅执行采样，不启动打标")
    parser.add_argument("--relabel-file", default=None, help="基于现有文件重新进行专家打标（保留已有的人工核验标签）")
    parser.add_argument("--overwrite", action="store_true", help="强制重新打标，忽略已有专家标签")

    args = parser.parse_args()

    if args.relabel_file:
        if not os.path.exists(args.relabel_file):
            print(f"错误: 找不到指定重打标文件: {args.relabel_file}")
            return
        positions = load_distill_dataset(args.relabel_file)
        out_path = args.out if args.out != "datasets/distill_tactical_labeled.json" else args.relabel_file
        print(f"[重打标] 成功加载 {len(positions)} 个局面，保留已有的人工核验标签，开始专家重算...")
        label_positions_parallel(positions, depth=args.depth, time_limit_ms=args.time_limit_ms,
                                 workers=args.workers, seed=args.seed, cache_path=out_path,
                                 overwrite=True)
        return

    # 1. 采样实战交火局面
    sampled = sample_tactical_positions(sav_dir=args.sav_dir, target_count=args.count,
                                        seed=args.seed)

    if args.only_sample:
        save_distill_dataset(sampled, args.out)
        print(f"采样数据已保存至 {args.out}")
        return

    # 2. 多进程专家打标
    label_positions_parallel(sampled, depth=args.depth, time_limit_ms=args.time_limit_ms,
                             workers=args.workers, seed=args.seed, cache_path=args.out,
                             overwrite=args.overwrite)


if __name__ == "__main__":
    main()
