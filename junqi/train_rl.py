"""深度强化学习自对弈训练流水线（V2.3 P0 收尾版：多进程并发 + 批量推理 + 阶段加权 + 残局课程）。

核心特性：
1. 多进程并发 (Multiprocessing Actors)：并行运行自对弈 Worker；
2. 批量根节点推理 (Batched Root Evaluation)：充分释放 RTX 4080 SUPER 算力；
3. 残局课程混入 (Curriculum Endgame Mix)：混入残局局面，强化死区与拖和；
4. 阶段感知分层经验池；
5. P0 修复（§3.1.3）：candidate 与 best 分离——门控失败不再回滚训练进度，
   best.pt 只在晋升时更新；每轮保存完整 checkpoint（网络+优化器+随机源+元数据）；
6. P0 修复（§3.1.4）：门控重设计——候选 vs 已发布 best 在三套固定评测集 + 随机
   完整发牌上对抗（先后手各半、固定种子），按阶段拆分胜/和/负与终局原因，
   Wilson 区间判定晋升；另附 search2 参考对抗（仅记录，不阻塞）；
7. P0 修复（§3.1.5）：Python/NumPy/PyTorch/Worker 随机源全部由实验种子派生。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .analysis import (PHASE_ENDGAME, PHASE_MIDGAME, PHASE_OPENING,
                       detect_phase)
from .config import RuleConfig
from .encoder import (ACTION_SPACE_SIZE, NUM_CHANNELS, encode_state_np,
                      legal_action_mask)
from .endgame_gen import gen_endgame
from .eval_gate import run_gate            # P0 修复 R3：统一门控实现（唯一真源）
from .mcts import MCTS
from .net import JunqiNet
from .selfplay import play_game
from .state import GameState, deal, position_key

# S2：温度表上调（AlphaZero 前中期高温探索；原尾盘 0.2 在趋和局面下等于关闭探索，
# 导致破死区/围猎暗子的决胜样本永远采不到）。
TEMP_BY_PHASE = {
    PHASE_OPENING: 1.2,
    PHASE_MIDGAME: 1.0,
    PHASE_ENDGAME: 0.5,
}

# S2：Value 批内公共模式样本目标占比（与世界模式叶子约 1:1 混合，修复模式失配）
PUBLIC_VALUE_RATIO = 0.5
# S2：辅助回归损失权重（监督 material_diff，非奖励塑形）
AUX_LOSS_WEIGHT = 0.1

# S1 修复：显式对手配比（原代码“25% best 对抗”因池扩张依赖晋升而从不生效）。
# mirror=同模型镜像、best=已发布模型/池内快照、expert=专家搜索、
# greedy=贪心期望搜索、random=随机扰动。
OPP_MIX = {"mirror": 0.50, "best": 0.25, "expert": 0.10,
           "greedy": 0.10, "random": 0.05}

# P3 批次 3（2026-09-14）：对手结构预置。动机——mirror 自对弈对抗梯度接近零，
# 而 expert 对局是**客观终局 Value 样本的最廉价来源**且提供真实对抗压力；
# 门控实测同源模型间 72-91.5% 对局以 no_capture 判和（区分度枯竭），
# 故降低 mirror 占比、提高 expert 占比。一次只改这一个变量。
OPP_MIX_PRESETS: Dict[str, Dict[str, float]] = {
    "baseline": dict(OPP_MIX),
    "diverse": {"mirror": 0.30, "best": 0.30, "expert": 0.20,
                "greedy": 0.15, "random": 0.05},
}

# 生成/评测夹具分离（2026-09-13 计划修订）：自对弈"生成侧"调稀平局触发器
# （70→120 步无吃子、循环判和 3→4 次），迫使对局必须分出胜负或真死锁，
# 大幅压缩 z=0 垃圾样本占比；"评测侧"（gate/靶场）仍严格使用官方
# RuleConfig() 默认规则（70/1000/循环 3），策略若学会在生成规则下钻空子，
# 由门控直接暴露。此为数据生成夹具，不改变任何规则定义与终局奖励语义。
GENERATION_CFG = RuleConfig(no_capture_draw_plies=120, repetition_draw_count=4)

# 和棋局的连续无吃子尾部（quiet >= 60）样本视为垃圾段丢弃：
# 该段的 z 恒为 0（生成夹具下必达 120 步判和），策略头无从学习；
# 决胜局的尾部样本保留（z=±1 有信号）。
QUIET_TAIL_CUTOFF = 60

# 每轮 Value 重锚（P3 修订 2026-09-14，见 docs/SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md）
#
# 受控实验证据（同一 6 块共享主干、同一 p1_v3 客观标签数据、联合策略+价值训练）：
#   lr=1e-3 微调 1 轮（711 步）即把 Value 平衡准确率 0.763 → 0.567，3 轮稳定 0.546，
#           且策略模仿 top-1 反而更低（0.218 < 0.261）；
#   lr=1e-4 保持 0.713，策略 top-1 更高（0.261）。
# 结论：1e-3 对 6 块共享主干做微调过高（既毁校准又不利于策略），故
#   ① DEFAULT_LR 下调至 1e-4（根因修复）；
#   ② 每轮联合训练后追加一次"冻结主干、仅训 Value 头"的重锚（保险丝），
#      锚定数据 = 回放池客观终局样本 + p1_v3 官方客观标签。
# 另需知：一旦主干已漂移，头-only 重锚无法恢复 OOD 校准（实测上限 ≈ 0.35 平衡准确率，
# 即随机水平），故重锚是预防而非修复——已漂移的候选不可救，只能从健康基座重启。
DEFAULT_LR = 1e-4
ANCHOR_P1_RATIO = 0.30            # 锚定集中 p1_v3 官方客观标签的样本占比
ANCHOR_POOL_PER_CLASS = 12000     # 每类回放池样本上限（Win/Draw/Loss 各取）
ANCHOR_EPOCHS = 3
ANCHOR_LR = 5e-4
ANCHOR_VAL_DIR = "datasets/p1_v3"  # 官方客观标签数据集目录（train/val/test 三划分）
# lr 调度（P3 批次 3 第三项）：constant = 已验证基线；cosine = 从 base_lr 余弦衰减到
# base_lr*LR_FLOOR_RATIO。方案原文写"1e-3 恒定 → 余弦衰减到 3e-4"，但批次 1 已据受控
# 实验把 base_lr 下调到 1e-4（1e-3 会毁 Value 校准），此时"衰减到 3e-4"反而是升 lr、
# 与证据矛盾，故按同一意图改为"衰减到 base 的 1/5"（1e-4 → 2e-5）。
LR_FLOOR_RATIO = 0.2


def lr_for_epoch(base_lr: float, ep: int, start_epoch: int, end_epoch: int,
                 schedule: str = "constant", floor_ratio: float = LR_FLOOR_RATIO) -> float:
    """按轮次计算学习率（纯函数，可单测）。cosine 从 base_lr 衰减到 base_lr*floor_ratio。"""
    if schedule != "cosine":
        return float(base_lr)
    total = max(1, end_epoch - start_epoch)
    t = min(1.0, max(0.0, (ep - start_epoch) / total))
    return float(base_lr) * (floor_ratio + (1.0 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


# Value 验收线（在 p1_v3/test 独立留出集上，指标 = 平衡准确率）
VALUE_ACCEPT_MAE = 0.55
VALUE_ACCEPT_BALANCED_ACC = 0.45


class ResignTracker:
    """自博弈认输判定器（2026-09-13 计划修订，P3 数据质量改造）。

    走子方根 Value <= threshold 连续 consecutive 次己方回合（且 ply >= min_ply）
    时判该方认输——语义对齐官方 code 21（主动认输属明确胜负，Value=±1）。
    终局奖励定义不变：这不是中间奖励，而是用 Value 头自身输出捷径一个
    已定的结局；回滚 = resign_enabled=False（或 threshold=-2.0 永不触发）。
    """

    def __init__(self, threshold: float = -0.95, consecutive: int = 8,
                 min_ply: int = 40):
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self.min_ply = int(min_ply)
        self._streak = {0: 0, 1: 0}

    def observe(self, seat: int, value: Optional[float], ply: int) -> bool:
        """记录 seat 方在其回合的根估值，返回是否应判该方认输。

        value=None（强制单着无估值）不计数也不清零；ply < min_ply 阶段
        开局评估不确定性大，一律不触发认输。
        """
        if ply < self.min_ply:
            return False
        if value is None:
            return self._streak[seat] >= self.consecutive
        if value <= self.threshold:
            self._streak[seat] += 1
        else:
            self._streak[seat] = 0
        return self._streak[seat] >= self.consecutive


def _drop_draw_tail(items: list, final_winner, quiet_idx: int,
                    cutoff: int = QUIET_TAIL_CUTOFF) -> list:
    """和棋局的 quiet >= cutoff 尾部样本段丢弃（决胜局全保留）。"""
    if final_winner is not None and final_winner != -1:
        return items
    return [it for it in items if it[quiet_idx] < cutoff]


def opponent_type_for(r: float, net1_available: bool = True,
                      mix: Optional[Dict[str, float]] = None) -> str:
    """按对手配比把均匀随机数映射为对手类型（纯函数，可单测）。
    net1_available=False 时 'best' 区间降级为 'expert'（防御分支，主循环已无条件注入）。
    默认使用 OPP_MIX；P3 批次 3 起可传入预置配比（见 OPP_MIX_PRESETS）。"""
    mix = OPP_MIX if mix is None else mix
    if r < mix["mirror"]:
        return "mirror"
    if r < mix["mirror"] + mix["best"]:
        return "best" if net1_available else "expert"
    if r < mix["mirror"] + mix["best"] + mix["expert"]:
        return "expert"
    if r < 1.0 - mix["random"]:
        return "greedy"
    return "random"


# ------------------------------------------------------------- 经验回放数据集
#
# 审查 C10（2026-09-15）：原 `PolicyDataset` / `ValueDataset` 两个 Dataset 子类
# 从未被任何代码引用——`train_epoch` 直接从 `StratifiedReplayBuffer` 采样
# numpy 数组再转 tensor，不经过 DataLoader。连同 `DataLoader` 导入一并删除，
# 避免后来者误以为存在一条基于 DataLoader 的训练路径。

def effective_policy_weights(counts, base_weights=(0.2, 0.5, 0.3),
                             mode: str = "fixed") -> np.ndarray:
    """计算 Policy 分桶采样权重（纯函数，可单测）。

    `fixed`（默认，已验证基线）= 固定 (0.2, 0.5, 0.3)；
    `adaptive` = 按 √桶容量 归一化。动机（2026-09-14 实测）：实战池严重失衡
    （opening 8,589 / midgame 3,208 / endgame 66,409），固定权重下 50% 的 batch
    从仅 3,208 条 midgame 样本里抽，单样本每轮被重复曝光约 12 次，而 endgame
    （占池 85%）只过 0.35 遍——既过拟合 midgame 又浪费多数数据。√容量权重把
    曝光比压缩到约 3-4 倍以内，同时不饿死小桶。
    """
    counts = np.asarray(counts, dtype=np.float64)
    nz = counts > 0
    if not nz.any():
        return np.zeros(3, dtype=np.float64)
    if mode == "adaptive":
        w = np.sqrt(counts)
        w[~nz] = 0.0
        return w / w.sum()
    w = np.asarray(base_weights, dtype=np.float64).copy()
    w[~nz] = 0.0
    if w.sum() <= 0:
        w = nz.astype(np.float64)
    return w / w.sum()


class StratifiedReplayBuffer:
    """按阶段与胜负类别分桶的经验回放池（解决 Draw 样本淹没与类别塌缩）。"""

    def __init__(self, capacity: int = 200000, rng: random.Random | None = None):
        self.capacity = capacity
        self.rng = rng or random.Random()
        # P3 批次 3：Policy 分桶权重模式（"fixed" = 已验证基线；"adaptive" = 池失衡修复）
        self.policy_weight_mode = "fixed"
        # Policy 流按阶段分桶: 0=opening, 1=midgame, 2=endgame
        self.policy_buckets: Dict[int, deque] = {
            PHASE_OPENING: deque(maxlen=capacity // 3),
            PHASE_MIDGAME: deque(maxlen=capacity // 3),
            PHASE_ENDGAME: deque(maxlen=capacity // 3),
        }
        # Value 流按胜负类别分桶: 0=Win, 1=Draw, 2=Loss
        self.value_buckets: Dict[int, deque] = {
            0: deque(maxlen=capacity // 3),
            1: deque(maxlen=capacity // 3),
            2: deque(maxlen=capacity // 3),
        }

    def add_policy(self, sample: Tuple[np.ndarray, np.ndarray, np.ndarray, int]):
        phase = sample[3]
        self.policy_buckets[phase].append(sample)

    def add_value(self, sample: Tuple[np.ndarray, int, int]):
        z_cls = sample[1]
        if z_cls in self.value_buckets:
            self.value_buckets[z_cls].append(sample)

    def total_policy_samples(self) -> int:
        return sum(len(b) for b in self.policy_buckets.values())

    def total_value_samples(self) -> int:
        return sum(len(b) for b in self.value_buckets.values())

    def sample_policy_batch(self, batch_size: int, weights: Tuple[float, float, float] = (0.2, 0.5, 0.3)) -> List:
        """按阶段配比采样 Policy 数据。"""
        counts = [len(self.policy_buckets[p]) for p in (0, 1, 2)]
        tot = sum(counts)
        if tot == 0:
            return []
        w = np.array(weights, dtype=np.float32)
        for i in range(3):
            if counts[i] == 0:
                w[i] = 0.0
        if w.sum() == 0:
            w = np.ones(3, dtype=np.float32)
        w = w / w.sum()
        # 分桶权重模式（P3 批次 3）：fixed = 已验证基线；adaptive = 按 √桶容量（池失衡修复）
        w = effective_policy_weights(counts, tuple(w), mode=self.policy_weight_mode)

        samples = []
        for p in (0, 1, 2):
            n_p = int(round(batch_size * w[p]))
            if n_p > 0 and len(self.policy_buckets[p]) > 0:
                sampled = self.rng.sample(self.policy_buckets[p], min(n_p, len(self.policy_buckets[p])))
                samples.extend(sampled)

        while len(samples) < batch_size and tot > 0:
            p = self.rng.choices([0, 1, 2], weights=counts, k=1)[0]
            if len(self.policy_buckets[p]) > 0:
                samples.append(self.rng.choice(self.policy_buckets[p]))
        return samples

    def sample_value_batch(self, batch_size: int, weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
                           prob_public: float = PUBLIC_VALUE_RATIO) -> List:
        """按类别均衡采样 Value 数据（Win : Draw : Loss 各占 weights，彻底破除 Draw 塌缩）。
        S2：双层配额抽样——先按类别分配配额（稀有类保证不被淹没），再在每个类内按
        prob_public 在公共模式根样本与世界模式叶子间分配（修复训练/评测模式失配）；
        某模式存量不足时由另一模式补足。旧版 3 元组样本按世界模式处理。"""
        counts = [len(self.value_buckets[c]) for c in (0, 1, 2)]
        tot = sum(counts)
        if tot == 0:
            return []
        w = np.array(weights, dtype=np.float32)
        for i in range(3):
            if counts[i] == 0:
                w[i] = 0.0
        if w.sum() == 0:
            w = np.ones(3, dtype=np.float32)
        w = w / w.sum()

        def _mode(s) -> int:
            return s[3] if len(s) > 3 else 1

        samples = []
        for c in (0, 1, 2):
            n_c = int(round(batch_size * w[c]))
            if n_c <= 0 or counts[c] == 0:
                continue
            pub = [s for s in self.value_buckets[c] if _mode(s) == 0]
            world = [s for s in self.value_buckets[c] if _mode(s) == 1]
            n_pub = int(round(n_c * prob_public))
            n_world = n_c - n_pub
            # 存量不足时跨模式补足，保证类配额不被接受过滤损耗（稀有类不被淹没）
            if n_pub > len(pub):
                n_world += n_pub - len(pub)
                n_pub = len(pub)
            if n_world > len(world):
                n_pub += n_world - len(world)
                n_world = len(world)
                n_pub = min(n_pub, len(pub))
            if n_pub > 0:
                samples.extend(self.rng.sample(pub, min(n_pub, len(pub))))
            if n_world > 0:
                samples.extend(self.rng.sample(world, min(n_world, len(world))))

        while len(samples) < batch_size and tot > 0:
            c = self.rng.choices([0, 1, 2], weights=counts, k=1)[0]
            if len(self.value_buckets[c]) > 0:
                samples.append(self.rng.choice(self.value_buckets[c]))
        self.rng.shuffle(samples)
        return samples[:batch_size]

    def stats(self) -> dict:
        return {
            "policy_buckets": {p: len(b) for p, b in self.policy_buckets.items()},
            "value_buckets": {c: len(b) for c, b in self.value_buckets.items()},
            "total_policy": self.total_policy_samples(),
            "total_value": self.total_value_samples(),
        }

    def save(self, path: str):
        """序列化持久化经验池内容。"""
        import pickle
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "policy": {p: list(b) for p, b in self.policy_buckets.items()},
            "value": {c: list(b) for c, b in self.value_buckets.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        """从文件恢复经验池内容。"""
        import pickle
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for p, lst in data.get("policy", {}).items():
                if p in self.policy_buckets:
                    self.policy_buckets[p] = deque(lst, maxlen=self.capacity // 3)
            for c, lst in data.get("value", {}).items():
                if c in self.value_buckets:
                    self.value_buckets[c] = deque(lst, maxlen=self.capacity // 3)
        except Exception as exc:                      # noqa: BLE001
            # C8 修复：原为静默 `pass`。经验池是断点续训的核心状态，
            # 加载失败却继续训练＝用空池起步，且外部完全看不到（只能从
            # "样本数突然归零"倒推）。现在显式告警，让故障可诊断。
            print(f"⚠️ 经验池加载失败：{path}（{type(exc).__name__}: {exc}）"
                  f"——将以空池继续，断点续训的历史样本已丢失", flush=True)


# ------------------------------------------------------------- 每轮 Value 重锚（P3 修订）

_ANCHOR_P1_CACHE: Dict[str, dict] = {}


def _load_anchor_p1(p1_dir: str, split: str) -> dict:
    """缓存加载 p1 客观标签划分（train 约 690MB，只加载一次）。"""
    key = f"{p1_dir}:{split}"
    if key not in _ANCHOR_P1_CACHE:
        from .train_value_distill import load_p1_arrays
        _ANCHOR_P1_CACHE[key] = load_p1_arrays(p1_dir, split)
    return _ANCHOR_P1_CACHE[key]


def build_anchor_dataset(buffer: StratifiedReplayBuffer, *,
                         p1_dir: str = ANCHOR_VAL_DIR,
                         per_class: int = ANCHOR_POOL_PER_CLASS,
                         p1_ratio: float = ANCHOR_P1_RATIO,
                         seed: int = 42) -> dict:
    """组装重锚数据集：回放池客观终局样本（类均衡抽取）+ p1_v3 官方客观标签。

    两类样本都是**客观终局**标签：池样本来自困毙/拔旗/真死锁/规则和棋（认输局样本
    在生成侧已丢弃），p1 样本来自官方 list.cfg 终局码。不引入任何中间奖励，
    也不读取真实暗子身份（池样本编码与训练同口径）。
    返回 {"train": {x,y,v}, "val": {x,y,v}, "sources": {...}}；val 固定为 p1 验证划分。
    """
    rng = random.Random(seed)
    xs, ys, per_cls = [], [], {}
    for c, bucket in buffer.value_buckets.items():
        items = list(bucket)
        if not items:
            continue
        k = min(per_class, len(items))
        pick = rng.sample(items, k)
        xs.extend(it[0] for it in pick)
        ys.extend(int(it[1]) for it in pick)
        per_cls[int(c)] = k
    n_pool = len(ys)
    if n_pool == 0:
        raise RuntimeError("回放池无 Value 样本，无法重锚")

    p1_tr = _load_anchor_p1(p1_dir, "train")
    n_p1 = int(round(n_pool * p1_ratio / max(1e-6, 1.0 - p1_ratio)))
    n_p1 = min(n_p1, len(p1_tr["y"]))
    idx = rng.sample(range(len(p1_tr["y"])), n_p1)
    x_pool = np.stack(xs) if xs else np.zeros((0, NUM_CHANNELS, 12, 5), dtype=np.float32)
    tr = {
        "x": np.concatenate([x_pool, p1_tr["x"][idx]], axis=0),
        "y": np.concatenate([np.asarray(ys, dtype=np.int64),
                             p1_tr["y"][idx]], axis=0),
        "v": np.concatenate([np.asarray([1.0 if c == 0 else (0.0 if c == 1 else -1.0)
                                         for c in ys], dtype=np.float32),
                             p1_tr["v"][idx]], axis=0),
    }
    order = np.random.RandomState(seed).permutation(len(tr["y"]))
    tr = {k: v[order] for k, v in tr.items()}
    return {"train": tr, "val": _load_anchor_p1(p1_dir, "val"),
            "sources": {"pool_per_class": per_cls, "pool_total": n_pool,
                        "p1_total": int(n_p1), "p1_ratio_actual":
                            round(n_p1 / max(1, n_p1 + n_pool), 4)}}


def reanchor_value_head(net: JunqiNet, buffer: StratifiedReplayBuffer, device: str, *,
                        epochs: int = ANCHOR_EPOCHS, lr: float = ANCHOR_LR,
                        batch_size: int = 128, seed: int = 42,
                        p1_dir: str = ANCHOR_VAL_DIR,
                        p1_ratio: float = ANCHOR_P1_RATIO,
                        verbose: bool = True) -> dict:
    """冻结主干、仅训 Value 头（重锚），对抗联合训练造成的主干特征漂移。

    机制与 value_distilled 离线蒸馏同源（train_value_head_only），数据为
    池内客观终局样本 + p1_v3 官方标签；返回验证指标供日志与验收使用。
    """
    from .train_value_distill import train_value_head_only
    data = build_anchor_dataset(buffer, p1_dir=p1_dir, p1_ratio=p1_ratio, seed=seed)
    res = train_value_head_only(net, data["train"], data["val"], epochs=epochs, lr=lr,
                                batch_size=batch_size, seed=seed, device=device,
                                tag="[重锚]", verbose=verbose)
    res["sources"] = data["sources"]
    return res


def probe_value_health(net: JunqiNet, device: str, p1_dir: str = ANCHOR_VAL_DIR,
                       split: str = "test") -> dict:
    """Value 健康度探针：p1_v3/test 独立留出集（9,381 条官方客观标签）。

    指标口径 = 平衡准确率（主）+ MAE + 预测分布 + 塌缩告警。理由：该集和棋占 54%，
    恒定预测单一类别的 MAE 仅约 0.50，单看 MAE 无法识别塌缩（实测全 Loss 塌缩模型
    MAE=0.502 < 0.55 阈值）。
    """
    from .train_value_distill import evaluate_value_health
    d = _load_anchor_p1(p1_dir, split)
    return evaluate_value_health(net, d["x"], d["y"], d["v"], device=device)


def value_acceptance(health: dict) -> Tuple[bool, str]:
    """Value 验收判定：平衡准确率 + MAE 双达标（设计依据见探针注释）。"""
    ok = (health["balanced_acc"] >= VALUE_ACCEPT_BALANCED_ACC
          and health["mae"] < VALUE_ACCEPT_MAE and not health["collapse_warning"])
    detail = (f"平衡acc={health['balanced_acc']:.3f} (线 {VALUE_ACCEPT_BALANCED_ACC})，"
              f"MAE={health['mae']:.3f} (线 {VALUE_ACCEPT_MAE})，"
              f"塌缩={health['collapse_warning']}")
    return ok, detail


# ------------------------------------------------- 权重热启动与 Worker 载入
#
# P0 修复（2026-09-15 审查 R1 / R2，详见 reviews/CODE_REVIEW_2026-09-15.md）
#   R1 热启动曾写成 `net = JunqiNet.load_from_file(...)` 重新绑定变量名，而
#      optimizer 在此之前已经构造完毕 → 两者脱钩，全部 AdamW step 因参数
#      grad is None 被跳过 → 训练全程权重零更新。
#   R2 `torch.load(net.save(path))` 得到的是 {"model_state": ...} 包装字典，
#      worker 却把它直接喂给 load_state_dict(strict=False) → 键名无一匹配、
#      静默通过 → OPP_MIX 中约 25% 的 "best 对手" 退化为随机初始化网络。

# 旧检查点没有 S2 辅助回归头，这些键缺失属正常向后兼容
TOLERATED_MISSING_PREFIXES = ("aux_head.",)


def load_weights_into_net(net: JunqiNet, payload, role: str = "net") -> bool:
    """把 Worker 收到的权重 payload 就地载入 net，返回是否**完整**载入。

    同时兼容 `net.save()` 的包装字典与裸 state_dict。不完整时返回 False 并打印
    原因，由调用方决定是报错（候选网络）还是降级（对手网络）。
    绝不静默通过——那正是 R2 的成因。
    """
    try:
        missing, unexpected = JunqiNet.load_state_dict_into(net, payload)
    except (RuntimeError, TypeError, KeyError, ValueError) as exc:
        print(f"⚠️ [{role}] 权重载入失败：{type(exc).__name__}: {exc}", flush=True)
        return False
    real_missing = [k for k in missing
                    if not k.startswith(TOLERATED_MISSING_PREFIXES)]
    if real_missing or unexpected:
        print(f"⚠️ [{role}] 权重不完全匹配：缺失 {len(real_missing)} 键"
              f"（例 {real_missing[:3]}）、多余 {len(unexpected)} 键", flush=True)
        return False
    return True


def infer_net_architecture(payload) -> Tuple[int, int, int, int]:
    """从权重 payload 推断 (in_channels, channels, num_blocks, value_out)。

    优先取 `net.save()` 写入的元信息；裸 state_dict 则从张量形状推导。
    原实现只看 `in_conv.0.weight.shape[1]`（输入通道）就构造 JunqiNet，
    一旦 any 检查点的主干宽度/深度非默认（128 / 6），同样会静默退化为随机网络。
    """
    meta = payload if isinstance(payload, dict) else {}
    sd = JunqiNet.unwrap_state_dict(payload)
    sd = sd if isinstance(sd, dict) else {}

    def _meta_int(key, default):
        v = meta.get(key) if isinstance(meta, dict) else None
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    w = sd.get("in_conv.0.weight")
    has_w = hasattr(w, "shape") and len(tuple(w.shape)) >= 2
    in_c = int(w.shape[1]) if has_w else _meta_int("in_channels", NUM_CHANNELS)
    channels = _meta_int("channels", int(w.shape[0]) if has_w else 128)

    num_blocks = _meta_int("num_blocks", -1)
    if num_blocks < 0:
        n = 0
        while f"blocks.{n}.conv1.weight" in sd:
            n += 1
        num_blocks = n or 6

    vw = sd.get("value_head.6.weight")
    val_out = (int(vw.shape[0]) if hasattr(vw, "shape") and len(tuple(vw.shape)) >= 1
               else 3)
    return in_c, channels, num_blocks, val_out


def build_opponent_net(payload, device: torch.device | str = "cpu"):
    """自博弈 Worker：由主进程传来的权重 payload 构造对手网络。

    载入失败返回 None（交由上层回落到 greedy / expert 分支），
    **绝不返回随机初始化的假对手**——那会让约 25% 的自博弈对局失去意义（R2）。
    """
    if payload is None:
        return None
    in_c, channels, num_blocks, val_out = infer_net_architecture(payload)
    net = JunqiNet(in_channels=in_c, num_blocks=num_blocks, channels=channels)
    if val_out == 1:
        net.value_head[6] = nn.Linear(128, 1)
    if not load_weights_into_net(net, payload, role="对手"):
        return None
    net.to(device)
    net.eval()
    return net


def warmstart_candidate(net: JunqiNet, optimizer, path: str, *, device,
                        lr: float, weight_decay: float = 1e-4):
    """训练主循环热启动：把检查点权重载入**已存在**的 net。

    返回 (net, optimizer)。主干超参一致时就地载入并返回原对象，保证 optimizer
    与 net 仍然绑定；仅当检查点的主干超参与当前 net 不一致、无法就地载入时才
    重建网络，并**同时**重建 optimizer（缺一即回到 R1 的"权重零更新"缺陷）。
    """
    tmp = JunqiNet.load_from_file(path, device=device)
    same_arch = (len(tmp.blocks) == len(net.blocks)
                 and tmp.in_conv[0].out_channels == net.in_conv[0].out_channels)
    if same_arch:
        missing, unexpected = JunqiNet.load_state_dict_into(net, tmp.state_dict())
        real_missing = [k for k in missing
                        if not k.startswith(TOLERATED_MISSING_PREFIXES)]
        if real_missing or unexpected:
            print(f"⚠️ 热启动权重部分缺失（缺失 {len(real_missing)} 键、"
                  f"多余 {len(unexpected)} 键）：{path}", flush=True)
        net.to(device)
        return net, optimizer

    print(f"热启动源 {path} 主干超参不一致（blocks={len(tmp.blocks)}, "
          f"channels={tmp.in_conv[0].out_channels}），重建候选网络与优化器", flush=True)
    return tmp, torch.optim.AdamW(tmp.parameters(), lr=lr, weight_decay=weight_decay)


# ------------------------------------------------------------- 模块级多进程 Worker

def _selfplay_worker_chunk(job_args):
    """子进程任务：执行 n_games 局自对弈（支持残局课程采样、中盘注入与多样化对手池）。
    P0 修复（§3.1.5）：子进程内所有随机源由 base_seed 统一派生。"""
    (net_dict, opp_net_dict, n_games, sims, c_puct, device_str,
     base_seed, cur_prob, mid_prob, resign_enabled, opp_mix) = job_args
    random.seed(base_seed)
    np.random.seed(base_seed % (2 ** 32))
    torch.manual_seed(base_seed % (2 ** 31))
    device = torch.device(device_str)

    # P0 修复（R2）：候选权重同样过一遍规范化载入，缺键即 fail fast，
    # 避免"看似在训练、实际随机初始化"的静默失败。
    net0 = build_opponent_net(net_dict, device)
    if net0 is None:
        raise RuntimeError("自博弈 Worker 无法载入候选网络权重，终止该 worker")

    # P0 修复（R2）：对手过去全部命中失败分支 → net1 实为随机网络。
    # 现在载入失败一律返回 None，由 opponent_type_for 回落到其它对手。
    net1 = build_opponent_net(opp_net_dict, device)

    all_p_samples = []
    all_v_samples = []
    opp_counter = Counter()
    game_records = []

    for i in range(n_games):
        seed = base_seed + i
        # S1 修复：按 OPP_MIX 显式配比选择对手；net1（已发布模型权重）由主循环无条件注入，
        # 不再依赖池内快照数（打破“晋升→扩池→多样性”死锁）
        rng_game = random.Random(seed * 10007 + 7)
        opp_type = opponent_type_for(rng_game.random(), net1_available=net1 is not None,
                                     mix=opp_mix)
        opp_strat = None
        target_net1 = None
        if opp_type == "best":
            target_net1 = net1          # 池快照仅 1 个时即已发布 best 本体
        elif opp_type == "expert":
            from .selfplay import ExpertStrategy
            opp_strat = ExpertStrategy(depth=2, seed=seed)
        elif opp_type == "greedy":
            from .selfplay import AgentStrategy
            opp_strat = AgentStrategy(depth=0, samples=4, seed=seed)
        elif opp_type == "random":
            from .selfplay import RandomStrategy
            opp_strat = RandomStrategy()
        opp_counter[opp_type] += 1

        p_samples, v_samples, pub_v_samples, game_record = play_selfplay_game(
            net0, net1=target_net1, opp_strategy=opp_strat,
            sims=sims, c_puct=c_puct, device=device_str,
            cfg=GENERATION_CFG,
            seed=seed, curriculum_prob=cur_prob, midgame_prob=mid_prob,
            resign_enabled=resign_enabled
        )
        all_p_samples.extend(p_samples)
        all_v_samples.extend(v_samples)
        all_v_samples.extend(pub_v_samples)
        game_records.append(game_record)

    return all_p_samples, all_v_samples, opp_counter, game_records


# ------------------------------------------------------------- 门控评测（P0 重设计，§3.1.4 / §7）
#
# 旧版门控（40 局同模型镜像、温度 0、无阶段拆分）在和棋密集规则下完全失去区分度。
# 新门控：候选模型 vs 已发布 best，在三套固定评测集 + 随机完整发牌上对抗，
# 先后手各半、固定种子（跨轮可比），按阶段拆分胜/和/负与终局原因，
# 用 Wilson 区间判定晋升；另附 vs search2 参考对抗（仅记录，不阻塞晋升）。
# 正式晋级建议将 --eval-games 提到 ≥ 200/阶段（§7 协议）；默认值为逐轮快速门控。

GATE_STAGES = ("opening", "midgame", "endgame")
_EVAL_SET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_sets")


def wilson_lower_bound(successes: float, n: int, z: float = 1.96) -> float:
    """Wilson 分数区间下界（广义：successes 可为小数，如 胜局数 + 0.5×和局数）。"""
    if n <= 0:
        return 0.0
    p = min(1.0, max(0.0, successes / n))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return max(0.0, centre - half)


def elo_update_from_score(opponent_elo: float, score_rate: float,
                          n_games: int) -> float:
    """按标准 Elo 公式把"对基准的得分率"换算成新的等级分（纯函数，可单测）。

    C9 修复（2026-09-15）：原实现是与对手等级分无关的线性随机游走
    （按得分率偏移固定 16 分、且与样本量无关），却被当成 Elo 写进
    elo_history.jsonl，任何人都无法据此判断"这一轮到底强了多少"。

    现在用标准换算：new = r_opp + 400 * log10(s / (1 - s))，
    其中 s 为候选对已发布基准的得分率（胜 1 / 和 0.5 / 负 0 口径）。
    s 会被夹到 [1/(n+1), n/(n+1)]，保证样本量小或全胜/全负时不产生 ±inf。
    """
    n = max(1, int(n_games))
    lo, hi = 1.0 / (n + 1), float(n) / (n + 1)
    s = min(max(float(score_rate), lo), hi)
    return float(opponent_elo) + 400.0 * math.log10(s / (1.0 - s))


def _stage_score(s: dict) -> Tuple[float, int]:
    n = s["wins"] + s["draws"] + s["losses"]
    return ((s["wins"] + 0.5 * s["draws"]) / n if n else 0.0), n


def decide_promotion(stage_stats: Dict[str, dict],
                     min_stage_score: float = 0.30,
                     ref_score: Optional[float] = None,
                     prev_ref_score: Optional[float] = None,
                     strict_stages: bool = False) -> Tuple[bool, str]:
    """晋升判定（纯函数，可单测）。stage_stats: {阶段名: {wins,draws,losses,reasons}}，
    必须含 "overall"。晋升条件（全部满足）：
      1. 整体得分 ≥0.5 且“不败率（胜+和）”Wilson 下界 ≥0.5（整体不显著退化）；
      2. 任一阶段得分不低于 min_stage_score（无严重退化）；
      3. S1 修订：**整体得分**的 Wilson 下界 >0.5（统计显著改善）。
         原“任一子阶段下界 >0.5”在快速门控样本量（每阶段 ≤32 局）下需得分 ≥0.75，
         数学上不可达，构成“永不晋升→池永不扩张”死锁；仅在正式晋升协议（每场景 ≥200 局）
         时通过 strict_stages=True 恢复子阶段显著性要求；
      4. S1 新增：若跑了参考对抗（ref_score 非 None），要求 ref_score ≥0.5 且不低于上轮；
         未跑参考对抗时跳过该检查。stage_stats 内的 "ref_vs_search2" 条目仅作记录。"""
    overall = stage_stats.get("overall")
    if not overall:
        return False, "缺少 overall 统计"
    _skip = {"overall", "ref_vs_search2"}
    o_score, o_n = _stage_score(overall)
    nonloss_lower = wilson_lower_bound(overall["wins"] + overall["draws"], o_n)
    if o_score < 0.5 or nonloss_lower < 0.5:
        return False, (f"整体退化风险（得分={o_score:.3f}，"
                       f"不败 Wilson 下界={nonloss_lower:.3f}）")
    for name, s in stage_stats.items():
        if name in _skip:
            continue
        sc, n = _stage_score(s)
        if n == 0:
            continue                     # 未评测的阶段不参与退化判定
        if sc < min_stage_score:
            return False, f"阶段 {name} 严重退化（得分={sc:.3f} < {min_stage_score}）"
    overall_lower = wilson_lower_bound(o_score * o_n, o_n)
    if overall_lower <= 0.5:
        return False, (f"整体得分 Wilson 下界={overall_lower:.3f} ≤ 0.5，"
                       f"无统计显著改善（得分={o_score:.3f}，n={o_n}）")
    if strict_stages:
        improved = [name for name, s in stage_stats.items()
                    if name not in _skip
                    for sc, n in [_stage_score(s)]
                    if n > 0 and wilson_lower_bound(sc * n, n) > 0.5]
        if not improved:
            return False, "正式协议：无阶段出现统计显著改善"
    if ref_score is not None:
        if ref_score < 0.5:
            return False, f"参考对抗不达标（ref_vs_search2={ref_score:.3f} < 0.5）"
        if prev_ref_score is not None and ref_score < prev_ref_score:
            return False, (f"参考对抗退化（ref_vs_search2={ref_score:.3f} "
                           f"< 上轮 {prev_ref_score:.3f}）")
    detail = f"整体得分={o_score:.3f}，Wilson 下界={overall_lower:.3f}"
    if ref_score is not None:
        detail += f"，ref_vs_search2={ref_score:.3f}"
    return True, detail


def inloop_gate_decision(stage_stats: Dict[str, dict],
                         inloop_gate_promote: bool = False,
                         ref_score: Optional[float] = None,
                         prev_ref_score: Optional[float] = None) -> Tuple[bool, str]:
    """轮内门控晋级判定（纯函数，可单测）。

    批次 1b（2026-09-14）默认行为：**只记录、不判定**。理由——轮内门控 n=16 时
    "整体得分 Wilson 下界 > 0.5" 需要得分率 ≥0.75（约 +191 Elo）才有显著性，
    对每轮 +10~30 Elo 的真实进步完全无功效，据此晋升等于用噪声改发布模型。
    正式晋级协议：`python -m junqi gate --model-a <候选> --model-b models/best.pt     --seeds 100 --promote-to-best`（配对同牌 + Wilson + 三元 SPRT，n=200）。
    回滚：inloop_gate_promote=True 恢复旧行为。
    """
    if not inloop_gate_promote:
        ov = stage_stats.get("overall") or {}
        n = ov.get("wins", 0) + ov.get("draws", 0) + ov.get("losses", 0)
        return False, (f"轮内门控仅记录（n={n}，无统计功效）；"
                       f"晋升须经正式 SPRT 门控 --promote-to-best")
    return decide_promotion(stage_stats, ref_score=ref_score,
                            prev_ref_score=prev_ref_score)


def _tally_from_report(rep: dict) -> dict:
    """把 eval_gate.run_gate 的报告折算成本模块的 stage_stats 条目。

    reason_breakdown 的原结构为 {原因: {wins, draws, losses}}，这里压平成
    {原因: 局数} 以对齐本模块 stage_stats 的历史契约。
    """
    t = rep["totals"]
    reasons = {r: sum(c.values()) for r, c in rep.get("reason_breakdown", {}).items()}
    return {"wins": t["wins"], "draws": t["draws"], "losses": t["losses"],
            "games": t["wins"] + t["draws"] + t["losses"], "reasons": reasons}


def evaluate_gate(candidate_pt: str, best_pt: str, sims: int,
                  games_per_side: int, seed: int, workers: int = 4,
                  ref_games: int = 0, device: str = "cpu") -> Tuple[Dict[str, dict], Optional[float]]:
    """门控评测。返回 (stage_stats, ref_vs_search2 得分)。
    stage_stats 含 opening/midgame/endgame/random/overall 五项，
    每项 {wins, draws, losses, games, reasons}（候选视角）。

    P0 修复（审查 R3，2026-09-15）：本函数过去是与 eval_gate.run_gate 平行的
    第二套实现，且两处都错：
      1. 两个方向用 seed+i 与 seed+100_000+i → 非配对同牌，先后手差异与发牌
         运气没有被消去，和棋密集规则下分辨力极低；
      2. 子进程 device 写死 "cpu"。
    现在统一委托给 eval_gate.run_gate（配对同牌 + 显式座位 + Wilson + 三元 SPRT），
     device 由训练主循环透传。
    """
    spec = f"nn_mcts_{sims}"
    stage_stats: Dict[str, dict] = {}
    overall = {"wins": 0, "draws": 0, "losses": 0, "games": 0, "reasons": {}}

    scenarios = [(st, _load_eval_jsons(st, games_per_side)) for st in GATE_STAGES]
    scenarios.append(("random", [None] * games_per_side))

    for name, jsons in scenarios:
        if not jsons:
            stage_stats[name] = {"wins": 0, "draws": 0, "losses": 0,
                                 "games": 0, "reasons": {}}
            continue
        # 每个种子一副牌：run_gate 内部对该种子跑先后手两局（真配对）
        seeds = [seed + i for i in range(len(jsons))]
        rep = run_gate(spec, spec, seeds=seeds, workers=workers,
                       model_a=candidate_pt, model_b=best_pt,
                       init_states=jsons, device=device)
        stage_stats[name] = _tally_from_report(rep)
        overall = _merge_tally(overall, stage_stats[name])

    stage_stats["overall"] = overall

    # search2 参考对抗（仅记录，不参与晋升判定）；同样走配对同牌口径
    ref_score: Optional[float] = None
    if ref_games > 0:
        ref_n = max(1, (ref_games + 1) // 2)          # 每+种子贡献先后手两局
        ref_seeds = [seed + 500_000 + i for i in range(ref_n)]
        ref_tally = {"wins": 0, "draws": 0, "losses": 0}
        n_ref = 0
        for st_name in GATE_STAGES:
            js_list = _load_eval_jsons(st_name, ref_n)
            if not js_list:
                continue
            rep = run_gate(spec, "search2", seeds=ref_seeds[:len(js_list)],
                           workers=workers, model_a=candidate_pt,
                           model_b=None, init_states=js_list, device=device)
            t = rep["totals"]
            ref_tally["wins"] += t["wins"]
            ref_tally["draws"] += t["draws"]
            ref_tally["losses"] += t["losses"]
            n_ref += t["wins"] + t["draws"] + t["losses"]
        ref_score = ((ref_tally["wins"] + 0.5 * ref_tally["draws"]) / n_ref
                     if n_ref else None)
        stage_stats["ref_vs_search2"] = {**ref_tally, "games": n_ref}

    return stage_stats, ref_score


def _load_eval_jsons(stage: str, limit: int) -> List[str]:
    path = os.path.join(_EVAL_SET_DIR, f"{stage}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit]


def _merge_tally(a: dict, b: dict) -> dict:
    return {"wins": a["wins"] + b["wins"], "draws": a["draws"] + b["draws"],
            "losses": a["losses"] + b["losses"],
            "games": a["games"] + b["games"],
            "reasons": dict(Counter(a["reasons"]) + Counter(b["reasons"]))}


# ------------------------------------------------------------- 完整 checkpoint（§3.1.3）

def should_save_buffer(epoch: int, end_epoch_exclusive: int,
                       every: int = 1) -> bool:
    """经验池落盘节流判据（纯函数，可单测）。

    every is None / <0：每轮保存（历史默认行为，会产生 GB 级文件）；
    every == 0       ：永不保存，但**最后一轮必须保存**，否则断点续训丢全部经验；
    every == 1       ：每轮保存；
    every  > 1       ：每 every 轮保存一次，同样保证最后一轮落盘。

    end_epoch_exclusive 为本轮之后即将执行的下一轮编号（即最后一轮 +1）。
    """
    if every is None or every < 0:
        return True
    if epoch >= end_epoch_exclusive - 1:        # 最后一轮恒写
        return True
    if every == 0:
        return False
    return epoch % every == 0


def save_checkpoint(path: str, net: JunqiNet, optimizer, epoch: int, elo: float,
                    main_rng: random.Random,
                    buffer: Optional[StratifiedReplayBuffer] = None,
                    save_buffer: bool = True):
    """完整检查点：网络 + 优化器 + 随机源状态 + 元数据 + （可选）经验池样本。

    P3 修订（审查 P1，2026-09-15）：经验池 pickle 实测约 3.8 GB/份，过去每轮
    无条件写入。现由 `save_buffer` 控制是否落盘（配合 `should_save_buffer` 节流），
    并改为**原子写**：先写 .tmp 再 os.replace，避免中断留下损坏的巨文件。
    """
    payload = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "elo": elo,
        "python_rng_state": main_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "buffer_stats": buffer.stats() if buffer is not None else None,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)
    if buffer is not None and save_buffer:
        buf_path = path.replace(".pt", "_buffer.pkl")
        buffer.save(buf_path)


def load_checkpoint(path: str, net: JunqiNet,
                    optimizer=None, device: str = "cpu",
                    buffer: Optional[StratifiedReplayBuffer] = None) -> dict:
    """加载完整检查点（自产可信文件，含随机源状态故用完整反序列化）。
    strict=False：兼容 S2 前的旧检查点（无 aux_head 键）。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["net"], strict=False)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if buffer is not None:
        buf_path = path.replace(".pt", "_buffer.pkl")
        buffer.load(buf_path)
    return ckpt


# ------------------------------------------------------------- 单局自对弈

_MIDGAME_STARTS_CACHE: Optional[List[str]] = None


def _load_midgame_starts() -> List[str]:
    """S2：缓存加载中盘评测集（供自对弈起始局面注入，Lc0 开局多样性类比）。"""
    global _MIDGAME_STARTS_CACHE
    if _MIDGAME_STARTS_CACHE is None:
        path = os.path.join(_EVAL_SET_DIR, "midgame.jsonl")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _MIDGAME_STARTS_CACHE = [ln.strip() for ln in f if ln.strip()]
        except OSError:
            _MIDGAME_STARTS_CACHE = []
    return _MIDGAME_STARTS_CACHE


def play_selfplay_game(net0: JunqiNet, net1: Optional[JunqiNet] = None,
                       opp_strategy=None,
                       sims: int = 60, c_puct: float = 0.6,
                       device: str = "cpu", cfg: RuleConfig | None = None,
                       seed: int | None = None,
                       curriculum_prob: float = 0.0,
                       midgame_prob: float = 0.0,
                       resign_enabled: bool = True,
                       resign_threshold: float = -0.95,
                       resign_consecutive: int = 8,
                       resign_min_ply: int = 40,
                       quiet_tail_cutoff: int = QUIET_TAIL_CUTOFF) -> Tuple[List, List, List, dict]:
    """执行一局自对弈（支持混合对手、课程采样与认输加速），返回
    (policy_samples, value_samples, public_value_samples, game_record)。
    - curriculum_prob > 0 时按概率从残局生成器开始对弈；S2 决胜课程：
      子力失衡 |mb|≥0.3 时禁用堡垒，优先生成可破局局面（制造 Win/Loss 样本）
    - midgame_prob > 0 时按概率从 eval_sets/midgame.jsonl 注入中盘起始局面（文件缺失时退回完整发牌）
    - Value 样本为 4 元组 (arr, z_cls, phase, is_world)：is_world=1 为世界模式叶子，
      is_world=0 为公共模式根局面（S2 修复训练/评测模式失配）；标签仍为纯终局结果，未引入任何中间奖励。
    - 认输加速（2026-09-13 计划修订）：走子方根 Value <= resign_threshold 连续
      resign_consecutive 次己方回合（ply >= resign_min_ply）判该方认输，
      对局按官方 code 21 语义记 ±1 终局标签——终局奖励定义不变，回滚 = resign_enabled=False。
    - 和棋局的 quiet >= QUIET_TAIL_CUTOFF 尾部样本段在返回前丢弃（生成夹具下该段必达判和，z 恒 0）。
    - 认输局的 Value 样本一律丢弃（标签来自模型自身判断，仅保留 Policy 样本）。
    - game_record: {"winner", "reason", "plies", "resigned_seat"} 供决胜率统计。
    """
    from .encoder import action_to_index
    cfg = cfg or RuleConfig()
    rng = random.Random(seed)
    resign = ResignTracker(resign_threshold, resign_consecutive, resign_min_ply) \
        if resign_enabled else None
    resigned_seat = None

    r_start = rng.random()
    if curriculum_prob > 0 and r_start < curriculum_prob:
        # S2 决胜课程：扩大子力失衡幅度；失衡大时不筑堡垒，避免课程局面自带和棋答案
        mb = rng.uniform(-0.6, 0.6)
        fortress_flag = rng.random() < 0.5 and abs(mb) < 0.3
        st = gen_endgame(
            rng, cfg,
            material_balance=mb,
            hidden_k=rng.randint(2, 5),
            my_engineers=rng.randint(0, 1),
            opp_engineers=0 if fortress_flag else rng.randint(0, 1),
            fortress=fortress_flag
        )
    elif midgame_prob > 0 and r_start < curriculum_prob + midgame_prob:
        starts = _load_midgame_starts()
        st = None
        if starts:
            try:
                cand = GameState.from_json(rng.choice(starts), cfg)
                if not cand.is_terminal() and cand.legal_actions():
                    st = cand
            except (ValueError, KeyError):
                st = None
        if st is None:
            st = deal(rng, cfg)
    else:
        st = deal(rng, cfg)

    mcts0 = MCTS(net0, simulations=sims, c_puct=c_puct, device=device)
    mcts1 = MCTS(net1 if net1 is not None else net0, simulations=sims, c_puct=c_puct, device=device)

    root_records = []
    root_value_records = []
    leaf_records = []
    seen = Counter()

    while not st.is_terminal():
        seen[position_key(st)] += 1
        if seen[position_key(st)] >= cfg.repetition_draw_count:
            st.winner, st.win_reason = -1, "repetition"
            break

        acts = st.legal_actions()
        if not acts:
            st.winner, st.win_reason = 1 - st.turn, "immobilized"
            break

        current_phase = detect_phase(st)
        temp = TEMP_BY_PHASE[current_phase]
        avoid = {k for k, n in seen.items() if n >= cfg.repetition_draw_count - 1}

        # 若黑方配置了外部多样化策略（专家/贪心/随机）
        if st.turn == 1 and opp_strategy is not None:
            # S1 修复：对手走子不再写入 one-hot 策略目标（违背 AlphaZero 语义：
            # 策略目标必须来自训练方自身 MCTS 访问分布），该手不产生任何训练样本。
            act = opp_strategy.choose(st, rng=rng, avoid=avoid, history_counts=seen)
        else:
            # 编码 38 通道（注入历史重复计数）——仅训练方走子时编码，省去对手手开销
            state_arr = encode_state_np(st, seat=st.turn, world=None, history_counts=seen)
            mask_arr = legal_action_mask(st)
            active_mcts = mcts0 if st.turn == 0 else mcts1
            act, pi_vec, _pi_dict, leaf_samples, root_value = active_mcts.search(
                st, temperature=temp, add_noise=True, rng=rng,
                history_counts=seen, avoid=avoid
            )
            root_records.append((state_arr, mask_arr, pi_vec, st.turn, current_phase, st.quiet))
            # S2：公共模式根局面同步作为 Value 样本（与靶场/GUI/混合引擎评测分布对齐）
            root_value_records.append((state_arr, st.turn, current_phase, st.quiet))
            for leaf_arr, leaf_turn in leaf_samples:
                leaf_records.append((leaf_arr, leaf_turn, current_phase, st.quiet))

            # 认输加速：走子方根估值持续深负时按官方 code 21 语义判该方认输
            # （终局奖励定义不变；对手策略回合不参与判定）
            if resign is not None and resign.observe(st.turn, root_value, st.ply):
                st.winner, st.win_reason = 1 - st.turn, "resign"
                resigned_seat = st.turn
                break

        st = st.apply(act)

    final_winner = st.winner
    game_record = {"winner": final_winner, "reason": st.win_reason,
                   "plies": st.ply, "resigned_seat": resigned_seat}

    # 和棋局的连续无吃子尾部样本丢弃（决胜局全保留；quiet 为各采样点快照）
    def _tail(items, quiet_idx):
        return _drop_draw_tail(items, final_winner, quiet_idx, quiet_tail_cutoff)

    policy_samples = [
        (state_arr, mask_arr, pi_vec, phase)
        for state_arr, mask_arr, pi_vec, _seat, phase, _q in
        _tail(root_records, 5)
    ]

    # Value 样本标签（0=Win, 1=Draw, 2=Loss，纯终局结果；第 4 位为模式标志）
    def _label(leaf_turn: int) -> int:
        if final_winner == -1 or final_winner is None:
            return 1
        return 0 if final_winner == leaf_turn else 2

    # 认输局的样本只进 Policy 回放池：其 ±1 标签来自模型自身 Value 判断而非
    # 客观终局，喂给 Value 头会形成"误判->认输->强化误判"的自证回路
    # （2026-09-13 首轮 500 局实证：认输开启时三分类准确率 60%->16%）。
    # Value 头只吃客观终局（困毙/拔旗/真死锁/规则和棋）标签。
    if resigned_seat is not None:
        value_samples = []
        public_value_samples = []
    else:
        value_samples = [(arr, _label(leaf_turn), phase, 1)
                         for arr, leaf_turn, phase, _q in _tail(leaf_records, 3)]
        public_value_samples = [(arr, _label(seat), phase, 0)
                                for arr, seat, phase, _q in _tail(root_value_records, 3)]

    return policy_samples, value_samples, public_value_samples, game_record


# ------------------------------------------------------------- 训练步骤

def train_epoch(net: JunqiNet, buffer: StratifiedReplayBuffer,
                optimizer: torch.optim.Optimizer,
                batch_size: int = 128, steps_per_epoch: int = 50,
                device: str = "cpu") -> Tuple[float, float, float, float]:
    """从分层回放池中执行一个 Epoch 的联合优化。
    返回 (loss, p_loss, v_loss, aux_loss)；S2 新增辅助回归损失（监督 material_diff，
    目标直接取自编码器通道 25，无需额外存储）与公共/世界双模式 Value 混合采样。"""
    net.train()
    total_loss, total_p_loss, total_v_loss, total_aux_loss = 0.0, 0.0, 0.0, 0.0
    actual_steps = 0

    for _ in range(steps_per_epoch):
        p_batch = buffer.sample_policy_batch(batch_size)
        v_batch = buffer.sample_value_batch(batch_size)
        if not p_batch or not v_batch:
            break

        # 1. Policy Forward
        p_states = torch.from_numpy(np.stack([item[0] for item in p_batch])).float().to(device)
        p_masks = torch.from_numpy(np.stack([item[1] for item in p_batch])).bool().to(device)
        p_targets = torch.from_numpy(np.stack([item[2] for item in p_batch])).float().to(device)

        optimizer.zero_grad()
        p_logits, _ = net(p_states, legal_mask=p_masks)
        log_probs = F.log_softmax(p_logits, dim=-1)
        p_loss = -torch.sum(p_targets * log_probs, dim=-1).mean()

        # 2. Value Forward（S2：forward_with_aux 同步输出辅助回归值）
        v_states = torch.from_numpy(np.stack([item[0] for item in v_batch])).float().to(device)
        v_targets = torch.tensor([item[1] for item in v_batch], dtype=torch.long).to(device)

        _, v_out, aux_out = net.forward_with_aux(v_states)
        if v_out.shape[-1] == 3:
            v_loss = F.cross_entropy(v_out, v_targets)
        else:
            v_scalar = torch.where(v_targets == 0, 1.0, torch.where(v_targets == 2, -1.0, 0.0))
            v_loss = F.mse_loss(v_out.squeeze(-1), v_scalar)

        # 3. 辅助回归损失：material_diff 已广播在编码器通道 25，直接读取作监督目标。
        # 属监督信号而非奖励塑形：不改变终局回报，仅为价值头提供密集锚定。
        aux_targets = v_states[:, 25, 0, 0]
        aux_loss = F.mse_loss(aux_out, aux_targets)

        loss = p_loss + v_loss + AUX_LOSS_WEIGHT * aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
        total_aux_loss += aux_loss.item()
        actual_steps += 1

    steps = max(1, actual_steps)
    return (total_loss / steps, total_p_loss / steps,
            total_v_loss / steps, total_aux_loss / steps)


# ------------------------------------------------------------- 训练主循环（V2.3）

def run_training(epochs: int = 10, games_per_epoch: int = 40,
                 sims: int = 60, eval_games: int = 32,
                 ref_games: int = 6,
                 workers: int = 4, curriculum_prob: float = 0.3,
                 midgame_prob: float = 0.1,
                 batch_size: int = 128, lr: float = DEFAULT_LR,
                 buffer_size: int = 200000, out_dir: str = "models",
                 seed: int = 42, device: str | None = None,
                 fresh: bool = False,
                 rebase_baseline: bool = False,
                 resign_enabled: bool = True,
                 reanchor_enabled: bool = True,
                 anchor_p1_ratio: float = ANCHOR_P1_RATIO,
                 pool_weights: str = "fixed",
                 opp_preset: str = "baseline",
                 inloop_gate_promote: bool = False,
                 buffer_save_every: int = 5,
                 lr_schedule: str = "constant") -> JunqiNet:
    """深度强化学习自对弈训练主闭环（V2.3 + S0/S1/S2 整改）。

    candidate/best 分离（§3.1.3）：net 是持续训练的候选模型，门控失败不回滚；
    best.pt 是发布模型，仅在晋升时更新。每轮保存完整 checkpoint，支持断点续训。
    S0：rebase_baseline=True 时用 bc_best.pt 重建发布基线；热启动优先 bc_best。
    S1：对手池无条件注入；逐轮记录实际对手占比；晋升判据整体 Wilson + ref 硬条件；Elo 解耦。
    S2：公共模式 Value 样本与世界模式叶子约 1:1 混合；决胜课程与中盘注入；
    温度表上调；辅助回归头监督 material_diff；热启动优先 value_distilled.pt（若存在）。
    P3 修订（2026-09-14，据 docs/SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md 及其验证）：
    默认学习率降至 1e-4；热启动优先 value_distilled_v2.pt；每轮联合训练后追加
    "冻结主干、仅训 Value 头"的重锚；Value 验收改用 p1_v3/test 独立留出集的平衡准确率。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 军棋深度强化学习系统 V2.3（P3.3 门控增强版）启动 ===")
    print(f"计算设备: {device.upper()} | 并发 Workers: {workers} | 随机种子: {seed}")
    print(f"自博弈认输: {'开启 (threshold=-0.95 x8 回合)' if resign_enabled else '关闭（回滚开关，Value 重校准模式）'}")
    print(f"学习率: {lr} | 每轮 Value 重锚: {'开启' if reanchor_enabled else '关闭（--no-reanchor）'}"
          f" | Value 验收探针: p1_v3/test 平衡准确率 ≥{VALUE_ACCEPT_BALANCED_ACC}")
    if opp_preset not in OPP_MIX_PRESETS:
        raise ValueError(f"未知对手配比预置: {opp_preset}（可选 {sorted(OPP_MIX_PRESETS)}）")
    opp_mix = OPP_MIX_PRESETS[opp_preset]
    print(f"对手配比预置: {opp_preset} -> {opp_mix}")
    print(f"学习率调度: {lr_schedule}" + ("" if lr_schedule == "constant"
          else f"（{lr} -> {lr * LR_FLOOR_RATIO:.2e} 余弦衰减）"))
    print(f"Policy 分桶权重模式: {pool_weights}"
          + ("（√桶容量自适应，修复池失衡）" if pool_weights == "adaptive" else "（已验证基线 0.2/0.5/0.3）"))
    if device == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    # 全局随机源由实验种子派生（§3.1.5）
    torch.manual_seed(seed)
    main_rng = random.Random((seed * 2654435761) % (2 ** 32))

    os.makedirs(out_dir, exist_ok=True)
    pool_dir = os.path.join(out_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)

    best_path = os.path.join(out_dir, "best.pt")
    ckpt_path = os.path.join(out_dir, "candidate_latest.pt")
    elo_log_path = os.path.join(out_dir, "elo_history.jsonl")

    net = JunqiNet().to(device)                       # candidate：持续训练
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    buffer = StratifiedReplayBuffer(capacity=buffer_size,
                                    rng=random.Random(seed + 777))
    buffer.policy_weight_mode = pool_weights
    start_epoch = 1
    current_elo = 1500.0
    if fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    bc_path = os.path.join("models", "bc_best.pt")
    if rebase_baseline:
        # S0 基线重建：旧 best 系旧缺陷产物（50 题靶场全 Win 塌缩，见审查报告根因 2），
        # 备份后用 BC 模型（策略头 Top-1 81.5%）重建发布基线，并清空候选进度强制重训。
        if os.path.exists(bc_path):
            legacy = best_path.replace("best.pt", "best_legacy.pt")
            if os.path.exists(best_path) and not os.path.exists(legacy):
                os.replace(best_path, legacy)
                print(f"旧发布模型已备份: {legacy}")
            JunqiNet.load_from_file(bc_path, device=device).save(best_path)
            print(f"发布基线已用 bc_best 重建: {best_path}")
            # 必须同时清空候选检查点与经验池：否则 load_checkpoint 会把旧池（Draw 主导、
            # 无模式标志）整体读回，污染新起点（实证：重训 3 轮后池内 100% 旧式样本）。
            for stale in (ckpt_path, ckpt_path.replace(".pt", "_buffer.pkl"),
                          os.path.join(out_dir, "_candidate_gate.pt")):
                if os.path.exists(stale):
                    os.remove(stale)
            print("候选检查点、经验池与门控快照已清空，将从新基线重新训练")
        else:
            print(f"--rebase-baseline 需要 {bc_path}，文件不存在，跳过基线重建")
    if os.path.exists(ckpt_path):
        # 断点续训：恢复候选权重 + 优化器 + 随机源 + 经验池（§3.1.3）
        ckpt = load_checkpoint(ckpt_path, net, optimizer, device=device, buffer=buffer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        current_elo = float(ckpt.get("elo", 1500.0))
        try:
            main_rng.setstate(ckpt["python_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
        except (KeyError, TypeError):
            pass
        print(f"已从检查点续训: epoch {start_epoch} 起，Elo={current_elo:.1f}，恢复经验池: {buffer.stats()}")
    elif (vd_path := next((p for p in ("models/value_distilled_v2.pt",
                                        "models/value_distilled.pt")
                           if os.path.exists(p)), None)) is not None:
        # S2：专家价值蒸馏预热产物优先（价值头已有锚定，避免冷启动塌缩）
        # P3 修订（2026-09-14）：优先 value_distilled_v2.pt —— 独立留出集（p1_v3/test）
        # 实测平衡准确率 0.735 / MAE 0.288，而 value_distilled.pt 为 0.318 / 0.599
        # （84% 预测画和棋的塌缩头）。旧接线只认 value_distilled.pt。
        print(f"无检查点，从价值蒸馏模型热启动候选: {vd_path}")
        # P0 修复（R1）：就地载入，禁止重新绑定 net，否则 optimizer 脱钩
        net, optimizer = warmstart_candidate(net, optimizer, vd_path,
                                             device=device, lr=lr, weight_decay=1e-4)
        if not os.path.exists(best_path):
            net.save(best_path)
    elif os.path.exists(bc_path):
        # S0：热启动优先 bc_best（策略头可用），避免继承旧缺陷 best.pt 的塌缩权重；
        # 发布基线保持既有 best.pt，仅在晋升时更新。
        print(f"无检查点，从行为克隆模型热启动候选: {bc_path}")
        # S2 提示：BC 价值头未校准（靶场开局 MAE≈1.0），直接自对弈初期会变弱；
        # 建议先跑 `python -m junqi distill_value` 生成 value_distilled.pt 再训练。
        if not os.path.exists(os.path.join("models", "value_distilled.pt")):
            print("⚠️ 未检测到 value_distilled.pt：建议先执行 distill_value 蒸馏预热，"
                  "否则候选初期棋力可能弱于旧版模型（BC 价值头未锚定）")
        # P0 修复（R1）：就地载入，禁止重新绑定 net
        net, optimizer = warmstart_candidate(net, optimizer, bc_path,
                                             device=device, lr=lr, weight_decay=1e-4)
        if not os.path.exists(best_path):
            net.save(best_path)
    elif os.path.exists(best_path):
        print(f"无检查点，从已发布模型热启动候选: {best_path}")
        # P0 修复（R1）：同上
        net, optimizer = warmstart_candidate(net, optimizer, best_path,
                                             device=device, lr=lr, weight_decay=1e-4)
    else:
        net.save(best_path)
    if not os.path.exists(best_path):
        net.save(best_path)

    pool_models: List[str] = [best_path]
    gate_seed = seed * 90000 + 7                       # 门控固定种子，跨轮可比（§7）
    end_epoch = start_epoch + epochs
    accept_streak = 0                                  # Value 验收连续达标轮数（P3 修订）

    def _prev_ref_score() -> Optional[float]:
        """从 elo_history.jsonl 读取上一轮 ref_vs_search2（用于晋升不退化硬条件）。"""
        if not os.path.exists(elo_log_path):
            return None
        try:
            with open(elo_log_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return None
            return json.loads(lines[-1]).get("ref_vs_search2")
        except (OSError, json.JSONDecodeError):
            return None

    for ep in range(start_epoch, end_epoch):
        t0 = time.time()
        print(f"\n==================== Epoch {ep} ====================", flush=True)
        print(f"执行多进程自对弈采样 ({games_per_epoch} 局, 并发 {workers} 进程, MCTS Sims={sims})...", flush=True)

        # 准备子进程任务（Worker 运行在 CPU 避免 CUDA 上下文争用）
        chunk_size = max(1, games_per_epoch // workers)
        jobs = []
        for w in range(workers):
            n_g = chunk_size if w < workers - 1 else games_per_epoch - chunk_size * (workers - 1)
            if n_g > 0:
                # S1 修复：无条件注入已发布模型权重（池快照 >1 时随机选一个），
                # 使 Worker 的 25% "best 对抗"分支真实生效。
                opp_path = main_rng.choice(pool_models)
                opp_dict = torch.load(opp_path, map_location="cpu", weights_only=True)

                w_seed = seed * 1_000_000 + ep * 10_000 + w * 500
                jobs.append((
                    net.state_dict(), opp_dict, n_g, sims, 0.6, "cpu", w_seed,
                    curriculum_prob, midgame_prob, resign_enabled, opp_mix
                ))

        if workers > 1:
            with Pool(workers) as pool:
                results = pool.map(_selfplay_worker_chunk, jobs)
        else:
            results = [_selfplay_worker_chunk(j) for j in jobs]

        opp_mix_total = Counter()
        reason_total = Counter()
        decisive_games = 0
        total_plies = 0
        n_finished = 0
        for p_samples, v_samples, opp_counts, game_records in results:
            opp_mix_total.update(opp_counts)
            for s in p_samples:
                buffer.add_policy(s)
            for s in v_samples:
                buffer.add_value(s)
            for rec in game_records:
                reason_total[rec.get("reason") or "unknown"] += 1
                if rec.get("winner") is not None and rec["winner"] != -1:
                    decisive_games += 1
                total_plies += rec.get("plies", 0)
                n_finished += 1

        opp_total_games = max(1, sum(opp_mix_total.values()))
        opponent_mix = {k: round(v / opp_total_games, 3)
                        for k, v in sorted(opp_mix_total.items())}
        decisive_rate = decisive_games / max(n_finished, 1)
        reason_mix = {k: round(v / max(n_finished, 1), 3)
                      for k, v in reason_total.most_common()}
        print(f"自对弈完成 | 总 Policy 样本: {buffer.total_policy_samples()}, 总 Value 样本: {buffer.total_value_samples()} | 实际对手占比: {opponent_mix}", flush=True)
        print(f"对局质量 | 决胜率: {decisive_rate:.1%} ({decisive_games}/{n_finished}) | "
              f"平均局长: {total_plies / max(n_finished, 1):.0f} 手 | 终局原因分布: {reason_mix}", flush=True)

        # 网络参数优化（GPU 训练）
        cur_lr = lr_for_epoch(lr, ep, start_epoch, end_epoch, schedule=lr_schedule)
        for _g in optimizer.param_groups:
            _g["lr"] = cur_lr
        print(f"优化神经网络参数 (Policy + Value + Aux on {device.upper()}, lr={cur_lr:.2e})...",
              flush=True)
        loss, p_loss, v_loss, aux_loss = train_epoch(
            net, buffer, optimizer, batch_size=batch_size,
            steps_per_epoch=max(20, buffer.total_policy_samples() // batch_size),
            device=device
        )
        print(f"优化完成 | 总 Loss: {loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Aux: {aux_loss:.4f})", flush=True)

        # 每轮 Value 重锚（P3 修订 2026-09-14）：联合训练漂移主干特征 → Value 头失准。
        # 冻结主干、仅训 Value 头（池客观终局样本 + p1_v3 官方客观标签），成本约 2-3s/轮。
        anchor_res = None
        if reanchor_enabled:
            try:
                anchor_res = reanchor_value_head(net, buffer, device,
                                                 seed=seed * 31 + ep,
                                                 p1_dir=ANCHOR_VAL_DIR,
                                                 p1_ratio=anchor_p1_ratio)
                print(f"每轮 Value 重锚 | 验证平衡acc={anchor_res['val_balanced_acc']:.3f} "
                      f"MAE={anchor_res['val_mae']:.4f} | 数据源 {anchor_res['sources']}",
                      flush=True)
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"⚠️ Value 重锚跳过：{type(exc).__name__}: {exc}", flush=True)

        # Value 健康度探针：p1_v3/test 独立留出集（9,381 条官方客观标签，未参与早停）
        health = probe_value_health(net, device)
        health_ok, health_detail = value_acceptance(health)
        accept_streak = accept_streak + 1 if health_ok else 0
        print(f"Value 健康探针(p1_v3/test) | 平衡acc={health['balanced_acc']:.3f} "
              f"MAE={health['mae']:.3f} 原始acc={health['class_acc']:.3f} "
              f"预测分布={health['pred_counts']} 塌缩={health['collapse_warning']} | "
              f"验收 {'✅达标' if health_ok else '❌未达标'} 连续 {accept_streak} 轮（{health_detail}）",
              flush=True)

        # 运行靶场基准评测（降级为固定回归探针，仅 36 题，不作为 Value 验收依据）
        from .benchmark import evaluate_net_benchmark, save_metrics_report
        bm_res = evaluate_net_benchmark(net, device=device)
        print(f"靶场回归探针 | Value MAE: {bm_res['value_mae_overall']:.4f}, 三分类准确率: {bm_res['value_class_acc']*100:.1f}%", flush=True)
        save_metrics_report({
            "value_mae": bm_res,
            "training_status": {"epoch": ep, "loss": loss, "p_loss": p_loss, "v_loss": v_loss, "elo": current_elo}
        }, output_dir="metrics")

        # 门控评测（§3.1.4 重设计）：候选落盘后与已发布 best 对抗
        cand_pt = os.path.join(out_dir, "_candidate_gate.pt")
        net.save(cand_pt)
        print(f"执行门控评测 (候选 vs 已发布 best，每阶段/方向 {eval_games} 局)..." )
        stage_stats, ref_score = evaluate_gate(
            cand_pt, best_path, sims=sims, games_per_side=eval_games,
            seed=gate_seed, workers=workers, ref_games=ref_games,
            device=device)
        ov = stage_stats["overall"]
        ov_score, _ = _stage_score(ov)
        print(f"门控总体: 胜 {ov['wins']} / 和 {ov['draws']} / 负 {ov['losses']} | 得分 {ov_score:.3f}", flush=True)
        for name in (*GATE_STAGES, "random"):
            sc, n = _stage_score(stage_stats[name])
            print(f"  阶段 {name:<8} 得分 {sc:.3f} ({n} 局) 终局原因: {stage_stats[name]['reasons']}", flush=True)
        if ref_score is not None:
            print(f"  参考 vs search2: 得分 {ref_score:.3f}（仅记录）", flush=True)

        promote, reason = inloop_gate_decision(
            stage_stats, inloop_gate_promote=inloop_gate_promote,
            ref_score=ref_score, prev_ref_score=_prev_ref_score())
        if not inloop_gate_promote:
            print(f"轮内门控：{reason}", flush=True)
        if promote:
            print(f"🎉 晋升成功：{reason} → 更新 best.pt", flush=True)
            net.save(best_path)
            snapshot_path = os.path.join(pool_dir, f"step_ep{ep}.pt")
            net.save(snapshot_path)
            pool_models.append(snapshot_path)
            if len(pool_models) > 8:
                pool_models.pop(0)
        else:
            # §3.1.3：门控失败不回滚候选训练进度，仅不更新发布模型
            print(f"⚠️ 未达晋升条件：{reason}（候选训练进度保留，不回滚）", flush=True)
        # C9 修复：Elo 与晋升解耦（每轮都更新，恢复曲线信号），但必须用标准公式
        # 由"对已发布基准的得分率"换算，而不是与对手无关的线性随机游走。
        # 基准侧沿用当前 current_elo（同一血脉的等级分），得分率来自本轮门控 overall。
        prev_elo = current_elo
        current_elo = elo_update_from_score(current_elo, ov_score, stage_stats["overall"]["games"])
        print(f"Elo 更新（vs 已发布基准）：{prev_elo:.1f} → {current_elo:.1f} "
              f"（得分率 {ov_score:.3f}，n={stage_stats['overall']['games']}）", flush=True)

        # 每轮保存完整检查点（无论是否晋升）
        # P1 瘦身（审查 P1）：经验池 pickle 约 3.8 GB/份，按节流判据落盘
        save_buf = should_save_buffer(ep, end_epoch, buffer_save_every)
        if not save_buf:
            print(f"跳过经验池落盘（每 {buffer_save_every} 轮一次），"
                  f"仅保存网络/优化器状态", flush=True)
        save_checkpoint(ckpt_path, net, optimizer, ep, current_elo, main_rng, buffer,
                        save_buffer=save_buf)

        elapsed = time.time() - t0
        log_entry = {
            "epoch": ep,
            "loss": round(loss, 4),
            "p_loss": round(p_loss, 4),
            "v_loss": round(v_loss, 4),
            "aux_loss": round(aux_loss, 4),
            "lr": round(cur_lr, 8),
            "lr_schedule": lr_schedule,
            "opp_preset": opp_preset,
            "inloop_gate_promote": inloop_gate_promote,
            "value_health": {
                "balanced_acc": round(health["balanced_acc"], 4),
                "mae": round(health["mae"], 4),
                "class_acc": round(health["class_acc"], 4),
                "pred_counts": health["pred_counts"],
                "per_class_recall": health["per_class_recall"],
                "collapse_warning": health["collapse_warning"],
            },
            "value_accepted": health_ok,
            "value_accept_streak": accept_streak,
            "reanchor": None if anchor_res is None else {
                "val_balanced_acc": anchor_res["val_balanced_acc"],
                "val_mae": anchor_res["val_mae"],
                "best_epoch": anchor_res["best_epoch"],
                "sources": anchor_res["sources"],
            },
            "benchmark_regression": {
                "value_mae": round(bm_res["value_mae_overall"], 4),
                "value_class_acc": round(bm_res["value_class_acc"], 4),
            },
            "gate_score": round(ov_score, 3),
            "gate_wins": ov["wins"], "gate_draws": ov["draws"], "gate_losses": ov["losses"],
            "stage_scores": {k: round(_stage_score(v)[0], 3)
                             for k, v in stage_stats.items() if k != "ref_vs_search2"},
            "gate_reasons": ov["reasons"],
            "opponent_mix": opponent_mix,
            "ref_vs_search2": None if ref_score is None else round(ref_score, 3),
            "promoted": promote,
            "gate_decision": reason,
            "elo": round(current_elo, 1),
            "time_sec": round(elapsed, 1),
        }
        with open(elo_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"Epoch {ep} 耗时: {elapsed:.1f}s | 当前估算 Elo: {current_elo:.1f}", flush=True)

    return net


def main():
    default_workers = min(8, os.cpu_count() or 4)
    parser = argparse.ArgumentParser(description="军棋翻棋深度强化学习自训练 V2.3 (P0 收尾版)")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--games", type=int, default=40, help="每轮自对弈局数")
    parser.add_argument("--sims", type=int, default=60, help="MCTS 每手模拟次数")
    parser.add_argument("--eval-games", type=int, default=32,
                        help="门控评测：每阶段每方向局数（S1 起默认 32；正式晋级建议 ≥200）")
    parser.add_argument("--ref-games", type=int, default=6,
                        help="vs search2 参考对抗：每阶段局数（仅记录，0=关闭）")
    parser.add_argument("--workers", type=int, default=default_workers, help="并发 Worker 进程数")
    parser.add_argument("--curriculum-prob", type=float, default=0.3, help="残局课程采样概率")
    parser.add_argument("--midgame-prob", type=float, default=0.1,
                        help="S2：中盘评测集起始局面注入概率（开局多样性）")
    parser.add_argument("--batch-size", type=int, default=128, help="批处理大小")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="学习率（P3 修订：默认 1e-4。受控实验证据——1e-3 微调一轮即把 "
                             "Value 平衡准确率 0.763→0.567 且策略学得更差；1e-4 保持 0.713）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out-dir", type=str, default="models", help="模型输出目录")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cuda/cpu)")
    parser.add_argument("--fresh", action="store_true", help="忽略已有检查点，从头训练")
    parser.add_argument("--no-reanchor", action="store_true",
                        help="P3 修订回滚开关：关闭每轮 Value 重锚（此时仅低学习率起作用）")
    parser.add_argument("--anchor-p1-ratio", type=float, default=ANCHOR_P1_RATIO,
                        help="重锚数据集中 p1_v3 官方客观标签占比（默认 0.30）")
    parser.add_argument("--pool-weights", type=str, default="fixed",
                        choices=("fixed", "adaptive"),
                        help="Policy 分桶权重模式：fixed=已验证基线；adaptive=√桶容量（修复池失衡）")
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=("constant", "cosine"),
                        help="学习率调度：constant=已验证基线；cosine=余弦衰减到 base/5")
    parser.add_argument("--opp-preset", type=str, default="baseline",
                        choices=tuple(OPP_MIX_PRESETS),
                        help="对手配比预置：baseline=已验证基线（mirror .5/expert .1）；"
                             "diverse=mirror .3/expert .2（批次 3 对手结构）")
    parser.add_argument("--inloop-gate-promote", action="store_true",
                        help="回滚开关：恢复轮内 n=16 门控的晋升判定（默认已降级为只记录，"
                             "晋升改由正式 SPRT 门控裁定）")
    parser.add_argument("--rebase-baseline", action="store_true",
                        help="S0：用 bc_best.pt 重建发布基线（旧 best 备份），并清空候选进度")

    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        games_per_epoch=args.games,
        sims=args.sims,
        eval_games=args.eval_games,
        ref_games=args.ref_games,
        workers=args.workers,
        curriculum_prob=args.curriculum_prob,
        midgame_prob=args.midgame_prob,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
        fresh=args.fresh,
        rebase_baseline=args.rebase_baseline,
        reanchor_enabled=not args.no_reanchor,
        anchor_p1_ratio=args.anchor_p1_ratio,
        pool_weights=args.pool_weights,
        opp_preset=args.opp_preset,
        inloop_gate_promote=args.inloop_gate_promote,
        lr_schedule=args.lr_schedule,
    )


if __name__ == "__main__":
    main()
