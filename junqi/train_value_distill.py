"""S2 专家价值蒸馏预热（Value Distillation Warmup）。

定位：价值头冷启动解药（审查报告根因 5 / 路线图 S2-2）。
自博弈在弱模型阶段产出的终局标签 86% 为 Draw，价值头缺乏锚定信号即塌缩；
本模块用已有的专家搜索引擎（Star1 期望极大极小 + QSearch + 死区/行营评估）
为采样局面打 W/D/L 伪标签，**仅监督训练价值头**（冻结策略头与主干），
为后续自对弈提供有锚定的价值初始化。

合规性说明：这是监督蒸馏信号，不是奖励塑形——自对弈的终局回报
z ∈ {+1, 0, -1} 保持不变，本模块不参与 RL 目标函数。

用法：
    python -m junqi distill_value --base models/bc_best.pt --out models/value_distilled.pt
蒸馏产物会被 train_rl 的热启动链自动优先采用。
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .analysis import detect_phase
from .ai import ExpertAgent
from .config import RuleConfig, SearchConfig
from .encoder import encode_state_np
from .endgame_gen import gen_endgame
from .net import JunqiNet
from .state import WIN_SCORE, GameState, deal

# 专家分数 → 类别映射：|tanh(score/SCALE)| 超过阈值判胜/负，否则判和
SCORE_SCALE = 600.0
CLASS_THRESHOLD = 0.25


# ---------------------------------------------------------------- 局面采样

def gen_positions(rng: random.Random, cfg: RuleConfig, *,
                  n_opening: int = 300, n_midgame: int = 500,
                  n_endgame: int = 400) -> List[Tuple[GameState, int]]:
    """采样蒸馏局面：开局/中盘来自随机走子推进（产生真实战斗结构），
    尾盘来自残局生成器（含子力失衡局面）。返回 [(state, phase)]。"""
    positions: List[Tuple[GameState, int]] = []

    def _rollout(target: int) -> Optional[GameState]:
        st = deal(rng, cfg)
        for _ in range(target):
            if st.is_terminal():
                return None
            acts = st.legal_actions()
            if not acts:
                return None
            st = st.apply(rng.choice(acts))
        if st.is_terminal() or st.my_color() is None or not st.legal_actions():
            return None
        return st

    for _ in range(n_opening + n_midgame):
        target = rng.randint(6, 14) if len(positions) < n_opening else rng.randint(15, 40)
        st = _rollout(target)
        if st is not None:
            positions.append((st, detect_phase(st)))

    for i in range(n_endgame):
        mb = rng.uniform(0.2, 0.6) * (1 if rng.random() < 0.5 else -1)
        fortress = rng.random() < 0.3 and abs(mb) < 0.3
        st = gen_endgame(rng, cfg, material_balance=mb,
                         hidden_k=rng.randint(2, 5),
                         my_engineers=rng.randint(0, 1),
                         opp_engineers=0 if fortress else rng.randint(0, 1),
                         fortress=fortress)
        if not st.is_terminal() and st.legal_actions():
            positions.append((st, detect_phase(st)))

    return positions


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


def label_with_expert(states: List[GameState], depth: int = 3,
                      time_limit_ms: int = 300,
                      seed: int = 2026) -> List[int]:
    """用专家搜索为局面打伪标签（走子方视角）。无合法走法判负（困毙）。"""
    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms),
                        seed=seed)
    labels = []
    for st in states:
        scored = agent.choose_actions(st, topn=1)
        if not scored:
            labels.append(2)
            continue
        labels.append(score_to_class(scored[0][1]))
    return labels


# ---------------------------------------------------------------- 蒸馏训练

def train_value_distill(base_model: str = "models/bc_best.pt",
                        out_path: str = "models/value_distilled.pt",
                        n_samples: int = 1200, epochs: int = 3,
                        batch_size: int = 128, lr: float = 5e-4,
                        val_ratio: float = 0.15, seed: int = 2026,
                        depth: int = 3, time_limit_ms: int = 300,
                        device: str | None = None) -> dict:
    """执行价值蒸馏：采样局面 → 专家打标 → 仅训练价值头 → 保存。
    策略头与主干冻结，保证 BC 策略能力不被破坏。返回指标字典。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = RuleConfig()

    # 1. 局面采样与专家打标
    n_open = int(n_samples * 0.25)
    n_mid = int(n_samples * 0.40)
    n_end = n_samples - n_open - n_mid
    print(f"[蒸馏] 采样局面：开局 {n_open} / 中盘 {n_mid} / 尾盘 {n_end} ...")
    positions = gen_positions(rng, cfg, n_opening=n_open, n_midgame=n_mid,
                              n_endgame=n_end)
    print(f"[蒸馏] 有效局面 {len(positions)}，开始专家打标 "
          f"(depth={depth}, time_limit={time_limit_ms}ms) ...")
    labels = label_with_expert([st for st, _ in positions], depth=depth,
                               time_limit_ms=time_limit_ms, seed=seed)

    from collections import Counter
    label_dist = dict(Counter(labels))
    print(f"[蒸馏] 伪标签分布: Win={label_dist.get(0, 0)} "
          f"Draw={label_dist.get(1, 0)} Loss={label_dist.get(2, 0)}")

    samples = [(encode_state_np(st, seat=st.turn, world=None), z, phase)
               for (st, phase), z in zip(positions, labels)]

    # 2. 确定性切分训练/验证
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _batch(indices):
        xs = torch.from_numpy(np.stack([samples[i][0] for i in indices])).float().to(device)
        ys = torch.tensor([samples[i][1] for i in indices], dtype=torch.long).to(device)
        return xs, ys

    # 3. 冻结主干与策略头，仅训练价值头
    net = JunqiNet.load_from_file(base_model, device=device)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("value_head")
    optimizer = torch.optim.AdamW(net.value_head.parameters(), lr=lr,
                                  weight_decay=1e-4)

    def _eval(indices):
        net.eval()
        correct, total = 0, 0
        pred_cnt = Counter()
        with torch.no_grad():
            for b in range(0, len(indices), batch_size):
                xs, ys = _batch(indices[b:b + batch_size])
                _, v_logits = net(xs)
                if v_logits.shape[-1] == 3:
                    pred = v_logits.argmax(dim=-1)
                else:
                    v = torch.tanh(v_logits).squeeze(-1)
                    pred = torch.where(v > CLASS_THRESHOLD, 0,
                                       torch.where(v < -CLASS_THRESHOLD, 2, 1))
                correct += int((pred == ys).sum())
                total += int(ys.numel())
                for c in pred.tolist():
                    pred_cnt[c] += 1
        return correct / max(total, 1), dict(pred_cnt)

    best_acc, best_sd = -1.0, None
    for ep in range(1, epochs + 1):
        net.train()
        rng.shuffle(train_idx)
        total_loss, n_b = 0.0, 0
        for b in range(0, len(train_idx), batch_size):
            xs, ys = _batch(train_idx[b:b + batch_size])
            optimizer.zero_grad()
            _, v_logits = net(xs)
            if v_logits.shape[-1] == 3:
                loss = F.cross_entropy(v_logits, ys)
            else:
                tgt = torch.where(ys == 0, 1.0, torch.where(ys == 2, -1.0, 0.0))
                loss = F.mse_loss(v_logits.squeeze(-1), tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        val_acc, val_pred = _eval(val_idx)
        print(f"[蒸馏] Epoch {ep}: loss={total_loss / max(n_b, 1):.4f} "
              f"val_acc={val_acc * 100:.1f}% val_pred={val_pred}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_sd = {k: v.detach().clone()
                       for k, v in net.value_head.state_dict().items()}

    if best_sd is not None:
        net.value_head.load_state_dict(best_sd)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    net.save(out_path)
    print(f"[蒸馏] 完成：最佳验证准确率 {best_acc * 100:.1f}%，已保存 {out_path}")

    return {
        "samples": len(samples),
        "label_dist": label_dist,
        "best_val_acc": best_acc,
        "out_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="S2 专家价值蒸馏预热（仅训练价值头）")
    parser.add_argument("--base", default="models/bc_best.pt", help="基座模型路径")
    parser.add_argument("--out", default="models/value_distilled.pt", help="输出权重路径")
    parser.add_argument("--samples", type=int, default=1200, help="蒸馏局面数")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集占比")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--depth", type=int, default=3, help="专家搜索深度")
    parser.add_argument("--time-limit-ms", type=int, default=300,
                        help="专家搜索单步时间预算（毫秒）")
    parser.add_argument("--device", default=None, help="计算设备")
    args = parser.parse_args()
    train_value_distill(base_model=args.base, out_path=args.out,
                        n_samples=args.samples, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        val_ratio=args.val_ratio, seed=args.seed,
                        depth=args.depth, time_limit_ms=args.time_limit_ms,
                        device=args.device)


if __name__ == "__main__":
    main()
