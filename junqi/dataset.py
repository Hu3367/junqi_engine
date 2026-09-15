"""P1 复盘数据集生成、切分、校验与基线评估模块。

严格遵守 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §6 与 §5 P1 规范：
1. 按对局（Game）切分 Train (80%) / Val (10%) / Test (10%)，禁止按 ply 随机切分；
2. 公共 Policy 样本：公共 36 通道状态、3650 维合法动作掩码、人类动作、游戏阶段标签；
3. 终局 Value 样本：仅使用明确胜负 (+1/-1) 与规则和棋 (0)，未终局与特殊中止局标记 has_value=False；
4. 数据集可重复生成，附带版本号、SHA256 哈希与分阶段覆盖率统计。

数据集版本约定（R6 修复，2026-09-15 审查）：
  v1/v2 生成于 2026-09-06，早于 2026-09-13 的 "code 24 断线不得赋 Value" 修正，
  口径不同、**不可与新代码混用**。所有默认路径统一指向 `DEFAULT_P1_DIR`。
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

import torch
from torch.utils.data import Dataset

# R6：唯一权威数据集目录与最低可接受版本。
# v3.0.0 = 2026-09-13 修正 code 24 标签口径后重新生成（760/95/96 局切分）。
DEFAULT_P1_DIR = "datasets/p1_v3"
MIN_P1_VERSION = (3, 0, 0)

import sys
if __package__ is None or not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from junqi.analysis import detect_phase
    from junqi.config import RuleConfig, SearchConfig
    from junqi.encoder import (ACTION_SPACE_SIZE, NUM_CHANNELS, action_to_index,
                              encode_state_np, legal_action_mask)
    from junqi.replay import SPECIAL_EVENT, SavGame, cell_rc, parse_sav, replay_sav
    from junqi.rules import Rank
    from junqi.state import Action, GameState
else:
    from .analysis import detect_phase
    from .config import RuleConfig, SearchConfig
    from .encoder import (ACTION_SPACE_SIZE, NUM_CHANNELS, action_to_index,
                          encode_state_np, legal_action_mask)
    from .replay import SPECIAL_EVENT, SavGame, cell_rc, parse_sav, replay_sav
    from .rules import Rank
    from .state import Action, GameState


class NpzReplayDataset(Dataset):
    """从 npz 加载行为克隆样本的 PyTorch 数据集（支持 38 通道与 3 分类 Value）。"""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.states = data["states"]        # (N, C, 12, 5) float32
        self.masks = data["masks"]          # (N, 3650) bool
        self.actions = data["actions"]      # (N,) int32
        self.values = data["values"]        # (N,) float32
        self.has_values = data["has_values"]# (N,) bool
        self.phases = data["phases"]        # (N,) int8
        if "val_classes" in data:
            self.val_classes = data["val_classes"]
        else:
            # 兼容旧版标量 values
            v_cls = np.full_like(self.actions, -1, dtype=np.int8)
            for i in range(len(self.actions)):
                if self.has_values[i]:
                    val = float(self.values[i])
                    if val > 0.5:
                        v_cls[i] = 0  # Win
                    elif val < -0.5:
                        v_cls[i] = 2  # Loss
                    else:
                        v_cls[i] = 1  # Draw
            self.val_classes = v_cls
        self.length = len(self.actions)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.tensor(int(self.actions[idx]), dtype=torch.long),
            torch.tensor(float(self.values[idx]), dtype=torch.float32),
            torch.tensor(int(self.val_classes[idx]), dtype=torch.long),
            torch.tensor(bool(self.has_values[idx]), dtype=torch.bool),
            torch.tensor(int(self.phases[idx]), dtype=torch.long),
        )


@dataclass
class DatasetStats:
    version: str = "1.0.0"
    created_at: str = ""
    seed: int = 2026
    min_plies: int = 20
    total_sav_files: int = 0
    valid_games: int = 0
    invalid_games: int = 0
    filtered_short_games: int = 0
    total_plies: int = 0
    train_games: int = 0
    val_games: int = 0
    test_games: int = 0
    train_plies: int = 0
    val_plies: int = 0
    test_plies: int = 0
    value_samples_count: int = 0
    policy_samples_count: int = 0
    phase_distribution: dict = None
    outcome_distribution: dict = None
    hashes: dict = None


def compute_file_sha256(filepath: str) -> str:
    """计算单个文件的 SHA-256 哈希。"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


DES_KEY = bytes.fromhex("2c250ed4141278e7")


def load_list_cfg_metadata(cfg_path: str = "军旗复盘/list.cfg") -> dict[str, dict]:
    """使用硬编码 DES 密钥解密 list.cfg 获取官方终局真值数据库。"""
    if not os.path.exists(cfg_path):
        return {}
    try:
        from Crypto.Cipher import DES
    except ImportError:
        return {}

    with open(cfg_path, "rb") as fh:
        raw_data = fh.read()

    cipher = DES.new(DES_KEY, DES.MODE_ECB)
    decrypted = cipher.decrypt(raw_data)
    import struct
    length = struct.unpack_from("<I", decrypted, 0)[0]
    text = decrypted[4:4 + length].decode("gbk", errors="replace")

    records = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        fname = parts[0]
        records[fname] = {
            "filename": fname,
            "mode": int(parts[1]),
            "timestamp": int(parts[2]),
            "player1": parts[3],
            "player1_rating": int(parts[4]),
            "player2": parts[5],
            "player2_rating": int(parts[6]),
            "flag1": int(parts[7]),
            "flag2": int(parts[8]),
            "moves_count": int(parts[9]),
            "winner": int(parts[10]),  # 1=P1(seat 0)胜, 2=P2(seat 1)胜, 3=和棋
            "reason_code": int(parts[11]),
            "duration": int(parts[12]) if len(parts) > 12 else 0,
        }
    return records


def terminal_label_from_meta(reason_code: int, winner_code: int) -> tuple[str, int]:
    """P1 标签规则（AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §6 权威口径）。

    根据官方 list.cfg 终局码判定该局的 Value 标签类别。
    返回 (label_kind, final_winner_seat)：
      - ("decided", seat)  明确胜负（+1/-1）：code 1 常规终局、21 主动认输、
        22 长捉判负、23 超时判负（seat = 胜者座位 0/1）；
      - ("draw", -1)       正规和棋（0）：code 40 协议和棋、42 循环和棋、
        43 限步判和，以及无终局码但官方记 w=3 的对局；
      - ("none", -1)       不赋 Value（仅保留合法 Policy 动作）：code 20
        中途强退/逃跑与 code 24 断线——中止事件不反映棋力高低，不得作为
        终局 Value（2026-09-13 修正：此前 code 24 被无条件视为明确胜负，
        与基线计划 §6 冲突，现与 code 20 同口径废弃）。
    """
    rc, w = reason_code, winner_code
    if rc in (1, 21, 22, 23) and w in (1, 2):
        return "decided", (0 if w == 1 else 1)
    if rc in (20, 24):
        return "none", -1
    if rc in (40, 42, 43) or w == 3:
        return "draw", -1
    return "none", -1


def outcome_bucket_from_meta(meta: dict) -> str:
    """把官方终局元数据映射到 outcome_distribution 的桶名。

    C7 修复（2026-09-15）：原实现在统计循环里**另写一遍**终局码判定，并把
    code 24（断线）算进 `decided_win`，与 `terminal_label_from_meta` 判其为
    "none"（不赋 Value）直接冲突，使 metadata.json 的 decided_win 虚高。
    现在统计与标签共用同一张码表，避免再次漂移。
    """
    rc = int(meta.get("reason_code", 0) or 0)
    wc = int(meta.get("winner", 3) if meta.get("winner") is not None else 3)
    kind, _seat = terminal_label_from_meta(rc, wc)
    return {"decided": "decided_win", "draw": "rule_draw"}.get(
        kind, "special_or_unfinished")


def process_single_game(game: SavGame, cfg: RuleConfig,
                         meta_record: Optional[dict] = None) -> list[dict]:
    """重演单局并提取每个 ply 的训练样本。

    meta_record: 可选来自 list.cfg 的官方终局元数据记录，包含真实胜负、原因码与对局模式。
    返回: list[dict] 每个 ply 的公共样本字典
    """
    samples = []
    board = None
    try:
        from .replay import board_from_table
        board = board_from_table(game.table)
    except Exception:
        return []

    st = GameState(board=board, cfg=cfg)

    # 预先判断整局最终结果（用于 Value 标签赋值）
    # 优先使用 2026-09-06 官方 list.cfg 权威真值（解封 456 局认输与 265 局和棋）
    has_meta = meta_record is not None
    is_decided_winner = False
    is_rule_draw = False
    final_winner = -1

    if has_meta:
        label_kind, final_winner = terminal_label_from_meta(
            int(meta_record.get("reason_code", 0)),
            int(meta_record.get("winner", 3)))
        is_decided_winner = label_kind == "decided"
        is_rule_draw = label_kind == "draw"
    else:
        # 回退到无元数据时的纯引擎判定
        is_decided_winner = game.winner in (0, 1) and not game.stopped_on_event
        is_rule_draw = game.winner == -1 and not game.stopped_on_event
        final_winner = game.winner

    from collections import Counter
    from .state import position_key
    seen_counts = Counter([position_key(st)])

    for i, (a, b, c) in enumerate(game.moves):
        if (a, b, c) == SPECIAL_EVENT:
            break

        if a == b and c == 1:
            act = Action("flip", cell_rc(a))
        elif a != b and c in (1, 3):
            act = Action("move", cell_rc(a), cell_rc(b))
        else:
            break

        legal_acts = st.legal_actions()
        if act not in legal_acts:
            # 遇到非法动作，截断退出
            break

        # 1. 公共状态张量编码 (38, 12, 5) float32 (含重复计数)
        tensor = encode_state_np(st, seat=st.turn, world=None, history_counts=seen_counts)

        # 2. 合法动作掩码 (3650,) bool
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for la in legal_acts:
            mask[action_to_index(la)] = True

        # 3. 动作索引 (0..3649)
        act_idx = action_to_index(act)

        # 4. 阶段标签 (0=Opening, 1=Midgame, 2=Endgame)
        phase = detect_phase(st)

        # 5. Value 标签 (三分类: 0=Win, 1=Draw, 2=Loss, -1=None)
        has_val = False
        target_val = 0.0
        val_class = -1
        if is_decided_winner:
            has_val = True
            val_class = 0 if final_winner == st.turn else 2
            target_val = 1.0 if final_winner == st.turn else -1.0
        elif is_rule_draw:
            has_val = True
            val_class = 1
            target_val = 0.0

        samples.append({
            "state": tensor,
            "mask": mask,
            "action": act_idx,
            "phase": phase,
            "value": target_val,
            "val_class": val_class,
            "has_value": has_val,
        })

        st = st.apply(act)
        seen_counts[position_key(st)] += 1

    return samples


def export_replay_dataset(sav_dir: str, out_dir: str = DEFAULT_P1_DIR,
                          split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                          seed: int = 2026, version: str = "2.0.0",
                          max_games: Optional[int] = None,
                          list_cfg_path: Optional[str] = None,
                          min_plies: int = 20) -> DatasetStats:
    """从 .sav 文件目录构建标准 P1/P2 高质量行为克隆数据集（按对局切分，附带版本和哈希）。
    支持接入 list.cfg 官方权威终局判定真值数据库。
    包含数据清洗：自动过滤损坏对局与总步数小于 min_plies 的开局失衡秒退异常局。
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = RuleConfig()

    # 尝试加载 list.cfg 官方元数据真值表
    meta_dict = {}
    if list_cfg_path is not None and os.path.exists(list_cfg_path):
        meta_dict = load_list_cfg_metadata(list_cfg_path)
    elif os.path.isfile(sav_dir):
        parent_dir = os.path.dirname(sav_dir)
        cand = os.path.join(parent_dir, "list.cfg")
        if os.path.exists(cand):
            meta_dict = load_list_cfg_metadata(cand)
    else:
        cand = os.path.join(sav_dir, "list.cfg")
        if os.path.exists(cand):
            meta_dict = load_list_cfg_metadata(cand)
        elif os.path.exists("军旗复盘/list.cfg"):
            meta_dict = load_list_cfg_metadata("军旗复盘/list.cfg")

    # 1. 查找全部 .sav 文件（去重扫描）
    if not os.path.exists(sav_dir):
        if os.path.exists("军旗复盘"):
            sav_dir = "军旗复盘"
        elif os.path.exists("../军旗复盘"):
            sav_dir = "../军旗复盘"

    if os.path.isfile(sav_dir):
        sav_files = [sav_dir]
    else:
        sav_files = sorted(set(glob.glob(os.path.join(sav_dir, "*.sav")) +
                               glob.glob(os.path.join(sav_dir, "**", "*.sav"), recursive=True)))

    total_files = len(sav_files)
    if total_files == 0:
        raise FileNotFoundError(f"在 {sav_dir} 下未找到任何 .sav 文件")

    # 2. 前置数据清洗与重演有效性验证
    valid_game_entries = []  # tuple: (fpath, meta, game_samples)
    invalid_count = 0
    filtered_short_count = 0

    print(f"[Dataset] 开始重演与清洗 {total_files} 局复盘文件 (min_plies={min_plies})...")
    for fpath in sav_files:
        try:
            g = parse_sav(fpath)
            g = replay_sav(g, cfg=cfg, check=True)
        except Exception:
            invalid_count += 1
            continue

        if not g.replay_ok:
            invalid_count += 1
            continue

        base_fname = os.path.basename(fpath)
        meta = meta_dict.get(base_fname)
        if meta is None and base_fname.endswith(".sav"):
            meta = meta_dict.get(base_fname[:-4])

        game_samples = process_single_game(g, cfg, meta_record=meta)
        if len(game_samples) < min_plies:
            filtered_short_count += 1
            continue

        valid_game_entries.append((fpath, meta, game_samples))

    print(f"[Dataset] 清洗完成: 原始 {total_files} 局, 有效保留 {len(valid_game_entries)} 局, "
          f"损坏滤除 {invalid_count} 局, 短步数弃赛滤除 {filtered_short_count} 局")

    if max_games is not None and max_games > 0:
        valid_game_entries = valid_game_entries[:max_games]

    # 3. 确定性按对局洗牌并划分 Train / Val / Test
    rng = random.Random(seed)
    rng.shuffle(valid_game_entries)

    n_total_valid = len(valid_game_entries)
    n_train = int(n_total_valid * split_ratios[0])
    n_val = int(n_total_valid * split_ratios[1])

    train_entries = valid_game_entries[:n_train]
    val_entries = valid_game_entries[n_train:n_train + n_val]
    test_entries = valid_game_entries[n_train + n_val:]

    splits = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries
    }

    stats = DatasetStats(
        version=version,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed,
        min_plies=min_plies,
        total_sav_files=total_files,
        valid_games=n_total_valid,
        invalid_games=invalid_count,
        filtered_short_games=filtered_short_count,
        train_games=len(train_entries),
        val_games=len(val_entries),
        test_games=len(test_entries),
        phase_distribution={"opening": 0, "midgame": 0, "endgame": 0},
        outcome_distribution={"decided_win": 0, "rule_draw": 0, "special_or_unfinished": 0},
        hashes={}
    )

    for split_name, entries in splits.items():
        all_states = []
        all_masks = []
        all_actions = []
        all_phases = []
        all_values = []
        all_val_classes = []
        all_has_values = []

        for fpath, meta, game_samples in entries:
            if meta is not None:
                # C7 修复：与 terminal_label_from_meta 同源，不再各写一遍码表
                stats.outcome_distribution[outcome_bucket_from_meta(meta)] += 1
            else:
                has_win = any(s["has_value"] and s["val_class"] in (0, 2) for s in game_samples)
                has_draw = any(s["has_value"] and s["val_class"] == 1 for s in game_samples)
                if has_win:
                    stats.outcome_distribution["decided_win"] += 1
                elif has_draw:
                    stats.outcome_distribution["rule_draw"] += 1
                else:
                    stats.outcome_distribution["special_or_unfinished"] += 1

            for s in game_samples:
                all_states.append(s["state"])
                all_masks.append(s["mask"])
                all_actions.append(s["action"])
                all_phases.append(s["phase"])
                all_values.append(s["value"])
                all_val_classes.append(s["val_class"])
                all_has_values.append(s["has_value"])

                if s["phase"] == 0:
                    stats.phase_distribution["opening"] += 1
                elif s["phase"] == 1:
                    stats.phase_distribution["midgame"] += 1
                else:
                    stats.phase_distribution["endgame"] += 1

        n_samples = len(all_actions)
        if split_name == "train":
            stats.train_plies = n_samples
        elif split_name == "val":
            stats.val_plies = n_samples
        elif split_name == "test":
            stats.test_plies = n_samples

        stats.total_plies += n_samples
        stats.policy_samples_count += n_samples
        stats.value_samples_count += sum(all_has_values)

        # 压缩存储为 npz 文件
        out_npz = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(
            out_npz,
            states=np.array(all_states, dtype=np.float32),
            masks=np.array(all_masks, dtype=bool),
            actions=np.array(all_actions, dtype=np.int32),
            phases=np.array(all_phases, dtype=np.int8),
            values=np.array(all_values, dtype=np.float32),
            val_classes=np.array(all_val_classes, dtype=np.int8),
            has_values=np.array(all_has_values, dtype=bool)
        )
        stats.hashes[f"{split_name}.npz"] = compute_file_sha256(out_npz)
        print(f"  [Split: {split_name}] 写入 {len(entries)} 局, {n_samples} plies -> {out_npz}")

    # 写入元数据 JSON
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2, ensure_ascii=False)

    return stats


def evaluate_dataset_policy(npz_path: str, max_samples: int = 1000) -> dict:
    """在导出的数据集上评估 Policy 基准指标（Top-1, Top-3, 交叉熵与分阶段覆盖）。"""
    data = np.load(npz_path)
    masks = data["masks"]
    actions = data["actions"]
    phases = data["phases"]
    n = min(len(actions), max_samples)

    # 1. 均匀随机策略（在合法动作中随机）基准
    rand_correct_top1 = 0
    rand_correct_top3 = 0
    rand_log_loss = 0.0

    phase_counts = {0: 0, 1: 0, 2: 0}
    phase_rand_top1 = {0: 0, 1: 0, 2: 0}

    for i in range(n):
        target_act = actions[i]
        mask = masks[i]
        phase = phases[i]
        phase_counts[phase] += 1

        legal_indices = np.where(mask)[0]
        n_legal = len(legal_indices)

        if n_legal > 0:
            rand_log_loss += math.log(n_legal)
            rand_correct_top1 += (1.0 / n_legal)
            rand_correct_top3 += min(1.0, 3.0 / n_legal)
            phase_rand_top1[phase] += (1.0 / n_legal)

    return {
        "samples_evaluated": n,
        "phase_counts": {
            "opening": phase_counts[0],
            "midgame": phase_counts[1],
            "endgame": phase_counts[2]
        },
        "random_baseline": {
            "top1_acc": rand_correct_top1 / n if n else 0.0,
            "top3_acc": rand_correct_top3 / n if n else 0.0,
            "cross_entropy": rand_log_loss / n if n else 0.0,
            "phase_top1": {
                "opening": phase_rand_top1[0] / phase_counts[0] if phase_counts[0] else 0.0,
                "midgame": phase_rand_top1[1] / phase_counts[1] if phase_counts[1] else 0.0,
                "endgame": phase_rand_top1[2] / phase_counts[2] if phase_counts[2] else 0.0,
            }
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P1 复盘行为克隆数据集生成与评估")
    sub = parser.add_subparsers(dest="cmd")

    p_export = sub.add_parser("export", help="从复盘数据导出标准行为克隆数据集 (npz)")
    p_export.add_argument("--sav-dir", default="军旗复盘", help=".sav 复盘文件目录")
    p_export.add_argument("--out-dir", default=DEFAULT_P1_DIR, help="输出 npz 目录")
    p_export.add_argument("--min-plies", type=int, default=20, help="异常短局过滤最小步数阈值 (默认 20)")
    p_export.add_argument("--seed", type=int, default=2026, help="随机种子")
    p_export.add_argument("--version", default=".".join(map(str, MIN_P1_VERSION)),
                          help="数据集版本号（低于 3.0.0 的数据集会被训练侧拒绝加载）")

    p_eval = sub.add_parser("eval", help="评估数据集 Policy 基准指标")
    p_eval.add_argument("--dataset", default=f"{DEFAULT_P1_DIR}/val.npz",
                        help="npz 数据集文件路径")
    p_eval.add_argument("--samples", type=int, default=1000, help="评估样本量")

    args = parser.parse_args()
    if args.cmd == "export":
        stats = export_replay_dataset(args.sav_dir, out_dir=args.out_dir, seed=args.seed,
                                      version=args.version, min_plies=args.min_plies)
        print(f"P1/P2 高质量数据集已成功生成至 {args.out_dir}:")
        print(f"  - 总局数: {stats.total_sav_files} (有效: {stats.valid_games}, 损坏滤除: {stats.invalid_games}, 短局滤除: {stats.filtered_short_games})")
        print(f"  - 切分: Train {stats.train_games} 局 ({stats.train_plies} plies), Val {stats.val_games} 局 ({stats.val_plies} plies), Test {stats.test_games} 局 ({stats.test_plies} plies)")
        print(f"  - Policy 样本数: {stats.policy_samples_count}, Value 样本数: {stats.value_samples_count}")
        print(f"  - 阶段覆盖率: {stats.phase_distribution}")
    elif args.cmd == "eval":
        report = evaluate_dataset_policy(args.dataset, max_samples=args.samples)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
