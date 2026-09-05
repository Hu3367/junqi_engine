"""P1 复盘数据集生成、切分、校验与基线评估模块。

严格遵守 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §6 与 §5 P1 规范：
1. 按对局（Game）切分 Train (80%) / Val (10%) / Test (10%)，禁止按 ply 随机切分；
2. 公共 Policy 样本：公共 36 通道状态、3650 维合法动作掩码、人类动作、游戏阶段标签；
3. 终局 Value 样本：仅使用明确胜负 (+1/-1) 与规则和棋 (0)，未终局与特殊中止局标记 has_value=False；
4. 数据集可重复生成，附带版本号、SHA256 哈希与分阶段覆盖率统计。
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
    total_sav_files: int = 0
    valid_games: int = 0
    invalid_games: int = 0
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


def process_single_game(game: SavGame, cfg: RuleConfig) -> list[dict]:
    """重演单局并提取每个 ply 的训练样本。

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
    # 严格规则：仅当真实正常终局时才提取 Value，特殊中止或未终局绝不赋 Value
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


def export_replay_dataset(sav_dir: str, out_dir: str = "datasets/p1_v1",
                          split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                          seed: int = 2026, version: str = "1.0.0",
                          max_games: Optional[int] = None) -> DatasetStats:
    """从 .sav 文件目录构建标准 P1 数据集（按对局切分，附带版本和哈希）。"""
    os.makedirs(out_dir, exist_ok=True)
    cfg = RuleConfig()

    # 1. 查找全部 .sav 文件
    if os.path.isfile(sav_dir):
        sav_files = [sav_dir]
    else:
        sav_files = sorted(glob.glob(os.path.join(sav_dir, "*.sav")) +
                           glob.glob(os.path.join(sav_dir, "**", "*.sav"), recursive=True))

    if max_games is not None and max_games > 0:
        sav_files = sav_files[:max_games]

    total_files = len(sav_files)
    if total_files == 0:
        raise FileNotFoundError(f"在 {sav_dir} 下未找到任何 .sav 文件")

    # 2. 确定性按对局文件名哈希洗牌
    rng = random.Random(seed)
    shuffled_files = list(sav_files)
    rng.shuffle(shuffled_files)

    n_train = int(total_files * split_ratios[0])
    n_val = int(total_files * split_ratios[1])

    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    stats = DatasetStats(
        version=version,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed,
        total_sav_files=total_files,
        train_games=len(train_files),
        val_games=len(val_files),
        test_games=len(test_files),
        phase_distribution={"opening": 0, "midgame": 0, "endgame": 0},
        outcome_distribution={"decided_win": 0, "rule_draw": 0, "special_or_unfinished": 0},
        hashes={}
    )

    for split_name, files in splits.items():
        all_states = []
        all_masks = []
        all_actions = []
        all_phases = []
        all_values = []
        all_val_classes = []
        all_has_values = []

        for fpath in files:
            try:
                g = parse_sav(fpath)
                g = replay_sav(g, cfg=cfg, check=True)
            except Exception:
                stats.invalid_games += 1
                continue

            if not g.replay_ok:
                stats.invalid_games += 1
                continue

            stats.valid_games += 1

            if g.winner in (0, 1) and not g.stopped_on_event:
                stats.outcome_distribution["decided_win"] += 1
            elif g.winner == -1 and not g.stopped_on_event:
                stats.outcome_distribution["rule_draw"] += 1
            else:
                stats.outcome_distribution["special_or_unfinished"] += 1

            game_samples = process_single_game(g, cfg)
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
