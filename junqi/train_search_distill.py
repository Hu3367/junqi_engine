"""P2 搜索蒸馏管线 (Search-as-Teacher Distillation)。

定位（REPLAY_ANALYSIS_AND_SYSTEMIC_IMPROVEMENT_PLAN_20260913.md §4.3）：
纯行为克隆会全盘接收人类复盘噪声；本模块以带 QSearch 的专家搜索引擎
（Star1 期望极大极小 + 吃子截断）为教师，在采样局面与固定评测局面上产出
根节点动作软分布 pi_search，用交叉熵反向蒸馏给神经网络 Policy 头，
使网络单步前向直觉自带深层战术记忆。

合规性（AGENTS.md）：
- 教师/学生均只使用公共状态（不读取真实暗子身份）；
- 蒸馏是监督信号，不改变自博弈终局奖励 z ∈ {+1, 0, -1}；
- 产物为独立候选权重，绝不覆盖 best.pt。

用法：
    python -m junqi distill_search --base models/bc_best.pt \\
        --out models/search_distilled.pt --states 1200 --depth 3
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import RuleConfig, SearchConfig
from .encoder import action_to_index, encode_state_np, legal_action_mask
from .net import JunqiNet
from .state import WIN_SCORE, Action, GameState

# 教师根节点分数 -> 软分布的 softmax 温度（搜索分值量纲：quiet ±600，战术 ±WIN_SCORE）
DEFAULT_TEACHER_TEMPERATURE = 120.0


# ---------------------------------------------------------------- 教师打标

def _teacher_label_worker(task: tuple) -> List[Tuple[int, float]]:
    """子进程教师打标：返回 [(action_index, score)]（走子方视角根节点搜索分）。"""
    from .ai import ExpertAgent
    state_json, depth, time_limit_ms, seed = task
    st = GameState.from_json(state_json)
    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms), seed=seed)
    scored = agent.choose_actions(st, topn=len(st.legal_actions()))
    out = []
    for a, s in scored:
        idx = action_to_index(a)
        if idx >= 0:
            out.append((idx, float(s)))
    return out


def teacher_soft_targets(state: GameState, depth: int = 3,
                         time_limit_ms: int = 300,
                         temperature: float = DEFAULT_TEACHER_TEMPERATURE,
                         seed: int = 2026,
                         root_scores: Optional[List[Tuple[Action, float]]] = None
                         ) -> Tuple[np.ndarray, int]:
    """单局面教师软分布：softmax(root_scores / T) 投影到全动作空间。

    返回 (targets [ACTION_SPACE_SIZE] float32, teacher_top1_index)。
    根节点分数含终局 ±WIN_SCORE 时 softmax 自然饱和到制胜/防败动作。
    """
    from .ai import ExpertAgent
    if root_scores is None:
        agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms),
                            seed=seed)
        scored = agent.choose_actions(state, topn=len(state.legal_actions()))
    else:
        scored = root_scores

    targets = np.zeros(3650, dtype=np.float32)
    if not scored:
        # 无根节点评分（罕见）：均匀回退到合法动作
        mask = legal_action_mask(state)
        targets[mask] = 1.0 / max(int(mask.sum()), 1)
        return targets, -1

    scores = np.array([min(max(s, -WIN_SCORE), WIN_SCORE) for _, s in scored],
                      dtype=np.float64)
    scores -= scores.max()
    exps = np.exp(scores / max(temperature, 1e-6))
    probs = exps / exps.sum()
    for (a, _), p in zip(scored, probs):
        idx = action_to_index(a)
        if idx >= 0:
            targets[idx] = p
    top1 = int(np.argmax([action_to_index(a) for a, _ in scored]))
    return targets, int(np.argmax(targets)) if targets.sum() > 0 else top1


def gen_distill_positions(rng: random.Random, cfg: RuleConfig,
                          n_states: int,
                          eval_sets: Optional[List[str]] = None
                          ) -> List[GameState]:
    """采样蒸馏局面：随机走子推进（真实战斗结构）+ 可选固定评测集起始局面。"""
    from .train_value_distill import gen_positions

    states: List[GameState] = []
    if eval_sets:
        for path in eval_sets:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            states.append(GameState.from_json(json.loads(line)))
                        except Exception:
                            continue
    need = max(n_states - len(states), 0)
    attempts = 0
    while need > 0 and attempts < 3:
        # 复用价值蒸馏的局面采样器（开局/中盘随机推进 + 尾盘残局生成器）；
        # rollout 可能命中终局被跳过，故按缺口补采，最多重试 3 轮
        rolled = gen_positions(rng, cfg,
                               n_opening=int(need * 0.3),
                               n_midgame=int(need * 0.3),
                               n_endgame=need - int(need * 0.6))
        states.extend(st for st, _ in rolled)
        need = max(n_states - len(states), 0)
        attempts += 1
    return states[:n_states]


# ---------------------------------------------------------------- 蒸馏训练

def train_search_distill(base_model: str = "models/bc_best.pt",
                         out_path: str = "models/search_distilled.pt",
                         n_states: int = 1200,
                         eval_sets: Optional[List[str]] = None,
                         epochs: int = 6, batch_size: int = 128,
                         lr: float = 3e-4, val_ratio: float = 0.15,
                         depth: int = 3, time_limit_ms: int = 300,
                         temperature: float = DEFAULT_TEACHER_TEMPERATURE,
                         seed: int = 2026, workers: int = 0,
                         device: str | None = None) -> dict:
    """执行搜索蒸馏：教师软分布 -> Policy 头交叉熵。返回指标 dict。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = RuleConfig()

    # 1. 采样局面
    states = gen_distill_positions(rng, cfg, n_states, eval_sets)
    print(f"[蒸馏] 有效局面 {len(states)}，教师打标 (depth={depth}, "
          f"time_limit={time_limit_ms}ms, T={temperature}) ...")

    # 2. 教师打标（支持多进程；公共状态序列化传递，不泄漏暗子身份）
    labels: List[Tuple[np.ndarray, int]] = []
    if workers and workers > 1 and len(states) > 10:
        import multiprocessing as mp
        tasks = [(st.to_json(), depth, time_limit_ms, seed + i)
                 for i, st in enumerate(states)]
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_teacher_label_worker, tasks)
        for st, scored in zip(states, results):
            targets = np.zeros(3650, dtype=np.float32)
            if scored:
                scores = np.array([min(max(s, -WIN_SCORE), WIN_SCORE)
                                   for _, s in scored], dtype=np.float64)
                scores -= scores.max()
                exps = np.exp(scores / max(temperature, 1e-6))
                probs = exps / exps.sum()
                for (idx, _), p in zip(scored, probs):
                    if idx >= 0:
                        targets[idx] = p
                top1 = int(np.argmax(targets))
            else:
                mask = legal_action_mask(st)
                targets[mask] = 1.0 / max(int(mask.sum()), 1)
                top1 = -1
            labels.append((targets, top1))
    else:
        from .ai import ExpertAgent
        agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms),
                            seed=seed)
        for i, st in enumerate(states):
            scored = agent.choose_actions(st, topn=len(st.legal_actions()))
            targets, top1 = teacher_soft_targets(
                st, root_scores=scored, temperature=temperature)
            labels.append((targets, top1))
            if (i + 1) % 100 == 0:
                print(f"[蒸馏] 教师打标进度 {i + 1}/{len(states)}")

    # 3. 特征与确定性切分
    feats = [encode_state_np(st, seat=st.turn, world=None) for st in states]
    idx_all = list(range(len(states)))
    rng.shuffle(idx_all)
    n_val = max(1, int(len(states) * val_ratio))
    val_idx, train_idx = idx_all[:n_val], idx_all[n_val:]

    def _batch(indices):
            xs = torch.from_numpy(np.stack([feats[i] for i in indices])).float().to(device)
            ms = torch.from_numpy(np.stack([legal_action_mask(states[i])
                                            for i in indices])).bool().to(device)
            ts = torch.from_numpy(np.stack([labels[i][0] for i in indices])).float().to(device)
            return xs, ms, ts

    # 4. 仅训练 Policy 头（主干与价值头冻结）
    net = JunqiNet.load_from_file(base_model, device=device)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("policy_head")
    optimizer = torch.optim.AdamW([p for p in net.policy_head.parameters()
                                   if p.requires_grad], lr=lr, weight_decay=1e-4)

    def _soft_ce(logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        return -(targets * log_p).sum(dim=-1).mean()

    def _eval(indices):
        net.eval()
        tot_kl, top1_hit, n = 0.0, 0, 0
        with torch.no_grad():
            for b in range(0, len(indices), batch_size):
                xs, ms, ts = _batch(indices[b:b + batch_size])
                logits, _ = net(xs, legal_mask=ms)
                log_p = F.log_softmax(logits, dim=-1)
                tot_kl += float((ts * (torch.log(ts.clamp_min(1e-9)) - log_p)).sum(-1).sum())
                top1_hit += int(((log_p.argmax(-1) == ts.argmax(-1)).float()).sum())
                n += int(xs.size(0))
        return tot_kl / max(n, 1), top1_hit / max(n, 1)

    best_kl, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        net.policy_head.train()
        rng.shuffle(train_idx)
        tot_loss, nb = 0.0, 0
        for b in range(0, len(train_idx), batch_size):
            xs, ms, ts = _batch(train_idx[b:b + batch_size])
            optimizer.zero_grad()
            logits, _ = net(xs, legal_mask=ms)
            loss = _soft_ce(logits, ts)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item())
            nb += 1
        val_kl, val_top1 = _eval(val_idx)
        print(f"[蒸馏] Epoch {ep}: loss={tot_loss / max(nb, 1):.4f} "
              f"val_KL={val_kl:.4f} val_teacher_top1={val_top1:.3f}")
        if val_kl < best_kl:
            best_kl = val_kl
            best_sd = {k: t.detach().clone() for k, t in net.state_dict().items()}

    if best_sd is not None:
        net.load_state_dict(best_sd)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "in_channels": net.in_channels,
        "num_blocks": len(net.blocks),
        "channels": net.in_conv[0].out_channels,
        "base_model": base_model,
        "distill_states": len(states),
        "teacher_depth": depth,
        "teacher_temperature": temperature,
        "val_kl": best_kl,
        "label_rule": "search-as-teacher soft targets (public state only)",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_path)
    print(f"[蒸馏] 完成：最佳验证 KL={best_kl:.4f}，候选已保存 {out_path}（未触碰 best.pt）")
    return {"states": len(states), "val_kl": best_kl, "out_path": out_path}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="P2: 搜索蒸馏（QSearch 教师 -> Policy 头）")
    parser.add_argument("--base", default="models/bc_best.pt", help="基座模型路径")
    parser.add_argument("--out", default="models/search_distilled.pt", help="输出权重路径")
    parser.add_argument("--states", type=int, default=1200, help="蒸馏局面数")
    parser.add_argument("--eval-sets", nargs="*", default=None,
                        help="附加固定评测集 jsonl（如 eval_sets/opening.jsonl）")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--depth", type=int, default=3, help="教师搜索深度")
    parser.add_argument("--time-limit-ms", type=int, default=300, help="教师单步时间预算")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEACHER_TEMPERATURE,
                        help="教师软分布温度（搜索分值量纲）")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--workers", type=int, default=0, help="教师打标并行进程数 (0=单进程)")
    parser.add_argument("--device", default=None, help="计算设备")
    args = parser.parse_args()
    train_search_distill(base_model=args.base, out_path=args.out,
                         n_states=args.states, eval_sets=args.eval_sets,
                         epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr, depth=args.depth,
                         time_limit_ms=args.time_limit_ms,
                         temperature=args.temperature, seed=args.seed,
                         workers=args.workers, device=args.device)


if __name__ == "__main__":
    main()
