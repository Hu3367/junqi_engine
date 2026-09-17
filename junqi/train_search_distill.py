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
import os
import random
import time
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

def _teacher_label_worker(task: tuple) -> Tuple[List[Tuple[int, float]], bool]:
    """子进程教师打标：返回 ([(action_index, score)], degraded)（走子方视角根节点搜索分）。"""
    from .ai import ExpertAgent
    state_json, depth, time_limit_ms, seed = task
    st = GameState.from_json(state_json)
    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms), seed=seed)
    agent.engine.tt.clear()
    scored, stats = agent.choose_actions(st, topn=len(st.legal_actions()), return_stats=True)
    out = []
    for a, s in scored:
        idx = action_to_index(a)
        if idx >= 0:
            out.append((idx, float(s)))
    return out, bool(stats.degraded)


def teacher_soft_targets(state: GameState, depth: int = 3,
                         time_limit_ms: int = 300,
                         temperature: float = DEFAULT_TEACHER_TEMPERATURE,
                         seed: int = 2026,
                         root_scores: Optional[List[Tuple[Action, float]]] = None,
                         return_degraded: bool = False,
                         degraded: bool = False
                         ) -> Tuple[np.ndarray, int] | Tuple[np.ndarray, int, bool]:
    """单局面教师软分布：softmax(root_scores / T) 投影到全动作空间。

    返回 (targets [ACTION_SPACE_SIZE] float32, teacher_top1_index)。
    若 return_degraded=True，额外返回 degraded: bool。
    根节点分数含终局 ±WIN_SCORE 时 softmax 自然饱和到制胜/防败动作。
    """
    from .ai import ExpertAgent
    if root_scores is None:
        agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms),
                            seed=seed)
        agent.engine.tt.clear()
        scored, stats = agent.choose_actions(state, topn=len(state.legal_actions()), return_stats=True)
        degraded = bool(stats.degraded)
    else:
        scored = root_scores

    targets = np.zeros(3650, dtype=np.float32)
    if not scored:
        # 无根节点评分（罕见）：均匀回退到合法动作
        mask = legal_action_mask(state)
        targets[mask] = 1.0 / max(int(mask.sum()), 1)
        if return_degraded:
            return targets, -1, degraded
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

    # 新增 3 修复：正确计算 top1 动作索引（若 targets 无质量则回退到 scored 首个最高分动作）
    if targets.sum() > 0:
        top1 = int(np.argmax(targets))
    elif scored:
        top1 = action_to_index(scored[0][0])
    else:
        top1 = -1

    if return_degraded:
        return targets, top1, degraded
    return targets, top1


def teacher_confidence(scored, min_spread: float = 0.0) -> float:
    """教师对该局面**是否有明确意见**：根分值展布达标才给训练权重 1，否则 0。

    为什么需要它（2026-09-16 实测）：
    随机推进采样的局面里，教师（depth 3）根分值极差中位仅 **54.5**，
    配 T=120 时首选/最差走法权重比只有 **1.58×** —— 软分布接近均匀
    （实测 94% 最大熵、最大概率 0.095）。在教师"没意见"的局面上强行拟合，
    等于把策略推向均匀分布：实测人类测试集 top-1 从 0.529 掉到 0.204，
    门控裁决判分 0.267（CI 上界 < 0.5）。
    ⇒ 只在教师有明确偏好的局面上学习，其余局面让锚点/基座策略自己说了算。

    判据取 `max - median`（而非 max - min）：对单个离群差着法更稳健。
    `min_spread <= 0` 表示不过滤（保持旧行为）。
    """
    if min_spread <= 0 or not scored or len(scored) < 2:
        return 1.0 if (scored and min_spread <= 0) else 0.0
    v = np.array([min(max(s, -WIN_SCORE), WIN_SCORE) for _, s in scored],
                 dtype=np.float64)
    spread = float(v.max() - np.median(v))
    return 1.0 if spread >= min_spread else 0.0


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
                         device: str | None = None,
                         anchor_weight: float = 0.0,
                         tac_min_spread: float = 0.0) -> dict:
    """执行搜索蒸馏：教师软分布 -> Policy 头交叉熵。返回指标 dict。

    `anchor_weight`（**2026-09-16 新增，默认 0 = 保持旧行为**）：
    人类策略锚点权重 β。损失为
        loss = CE(student, teacher_soft) + β · CE(student, base_soft)
    其中 base_soft 是**基座模型自身**的 policy 软分布（冻结副本）。
    第二项在常数意义下等价于 KL(base ‖ student)，即 trust-region 正则。

    为什么需要它（2026-09-16 实测）：
    锚点缺失时，蒸馏会把 BC 学到的策略**整体覆盖**成搜索偏好 ——
    人类测试集 top-1 从 0.529 掉到 0.204/0.353，且门控判分（裁决式）
    仅 0.267（CI 上界 0.390 < 0.5）、符号检验 p=0.0446，**明显更弱**。
    加锚点后网络只能在"保留人类模仿"的约束内叠加战术偏好。
    """
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
    # 每样本训练权重（教师置信度过滤；min_spread<=0 时恒为 1）
    weights: List[float] = []
    samples_degraded = 0
    if workers and workers > 1 and len(states) > 10:
        import multiprocessing as mp
        tasks = [(st.to_json(), depth, time_limit_ms, seed + i)
                 for i, st in enumerate(states)]
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_teacher_label_worker, tasks)
        samples_degraded = sum(1 for _, deg in results if deg)
        for st, (scored, deg) in zip(states, results):
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
                top1 = int(np.argmax(targets)) if targets.sum() > 0 else (scored[0][0] if scored else -1)
            else:
                mask = legal_action_mask(st)
                targets[mask] = 1.0 / max(int(mask.sum()), 1)
                top1 = -1
            labels.append((targets, top1))
            weights.append(teacher_confidence(scored, tac_min_spread))
    else:
        from .ai import ExpertAgent
        agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms),
                            seed=seed)
        for i, st in enumerate(states):
            # A3-1: 保证单进程与多进程冷 TT 行为一致
            agent.engine.tt.clear()
            scored, stats = agent.choose_actions(st, topn=len(st.legal_actions()), return_stats=True)
            deg = bool(stats.degraded)
            if deg:
                samples_degraded += 1
            targets, top1 = teacher_soft_targets(
                st, root_scores=scored, temperature=temperature, degraded=deg)
            labels.append((targets, top1))
            weights.append(teacher_confidence(scored, tac_min_spread))
            if (i + 1) % 100 == 0:
                print(f"[蒸馏] 教师打标进度 {i + 1}/{len(states)}")

    print(f"[蒸馏] 教师打标完成: 有效局面 {len(states)}, 时限截断 (degraded) {samples_degraded} "
          f"({samples_degraded / max(len(states), 1):.1%})")

    kept = sum(1 for w in weights if w > 0.0)
    if tac_min_spread > 0:
        print(f"[蒸馏] 教师置信度过滤 min_spread={tac_min_spread}: "
              f"保留 {kept}/{len(weights)} 个样本"
              f"（{kept / max(len(weights), 1):.1%}），其余权重置 0")

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
        ws = torch.tensor([weights[i] for i in indices],
                          dtype=torch.float32, device=device)
        return xs, ms, ts, ws

    # 4. 仅训练 Policy 头（主干与价值头冻结）
    net = JunqiNet.load_from_file(base_model, device=device)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("policy_head")
    optimizer = torch.optim.AdamW([p for p in net.policy_head.parameters()
                                   if p.requires_grad], lr=lr, weight_decay=1e-4)

    # 4.1 人类策略锚点：基座 policy 的冻结副本（只在 β>0 时加载）
    ref_net = None
    if anchor_weight > 0:
        ref_net = JunqiNet.load_from_file(base_model, device=device)
        ref_net.eval()
        for p in ref_net.parameters():
            p.requires_grad = False

    def _soft_ce(logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        return -(targets * log_p).sum(dim=-1).mean()

    def _weighted_soft_ce(logits, targets, ws):
        """按样本权重加权的软交叉熵（权重 0 的样本不参与训练）。

        权重来自 teacher_confidence：教师对该局面没有明确偏好时不学，
        避免把策略推向均匀分布。
        """
        log_p = F.log_softmax(logits, dim=-1)
        per = -(targets * log_p).sum(dim=-1)
        denom = ws.sum().clamp_min(1e-6)
        return (per * ws).sum() / denom

    def _eval(indices):
        net.eval()
        tot_kl, top1_hit, n = 0.0, 0, 0
        with torch.no_grad():
            for b in range(0, len(indices), batch_size):
                xs, ms, ts, _ws = _batch(indices[b:b + batch_size])
                logits, _ = net(xs, legal_mask=ms)
                log_p = F.log_softmax(logits, dim=-1)
                tot_kl += float((ts * (torch.log(ts.clamp_min(1e-9)) - log_p)).sum(-1).sum())
                top1_hit += int(((log_p.argmax(-1) == ts.argmax(-1)).float()).sum())
                n += int(xs.size(0))
        return tot_kl / max(n, 1), top1_hit / max(n, 1)

    def _eval_vs_ref(ref, indices, bs):
        """学生 argmax 与**基座**策略 argmax 的一致率 = BC 知识保留度。

        这是本轮改造的核心监测量：无锚点时它会大幅下滑（策略被覆盖）。
        """
        net.eval()
        ref.eval()
        hit, n = 0, 0
        with torch.no_grad():
            for b in range(0, len(indices), bs):
                xs, ms, _ts, _ws = _batch(indices[b:b + bs])
                lo, _ = net(xs, legal_mask=ms)
                ro, _ = ref(xs, legal_mask=ms)
                hit += int((lo.argmax(-1) == ro.argmax(-1)).float().sum())
                n += int(xs.size(0))
        return hit / max(n, 1)

    best_kl, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        net.eval()
        net.policy_head.train()
        rng.shuffle(train_idx)
        tot_loss, nb = 0.0, 0
        for b in range(0, len(train_idx), batch_size):
            xs, ms, ts, ws = _batch(train_idx[b:b + batch_size])
            optimizer.zero_grad()
            logits, _ = net(xs, legal_mask=ms)
            loss = _weighted_soft_ce(logits, ts, ws)
            if ref_net is not None:
                # 锚点：把学生拉回基座策略（常数意义下 = KL(base ‖ student)）
                with torch.no_grad():
                    ref_logits, _ = ref_net(xs, legal_mask=ms)
                loss = loss + anchor_weight * _soft_ce(
                    logits, F.softmax(ref_logits, dim=-1))
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item())
            nb += 1
        val_kl, val_top1 = _eval(val_idx)
        extra = ""
        if ref_net is not None:
            extra = f" val_base_top1={_eval_vs_ref(ref_net, val_idx, batch_size):.3f}"
        print(f"[蒸馏] Epoch {ep}: loss={tot_loss / max(nb, 1):.4f} "
              f"val_KL={val_kl:.4f} val_teacher_top1={val_top1:.3f}{extra}")
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
        "samples_degraded": samples_degraded,
        "teacher_depth": depth,
        "teacher_temperature": temperature,
        "anchor_weight": anchor_weight,
        "tac_min_spread": tac_min_spread,
        "samples_kept": kept,
        "val_kl": best_kl,
        "label_rule": "search-as-teacher soft targets (public state only)",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_path)
    print(f"[蒸馏] 完成：最佳验证 KL={best_kl:.4f}，候选已保存 {out_path}（未触碰 best.pt）")
    return {"states": len(states), "val_kl": best_kl, "out_path": out_path,
            "samples_degraded": samples_degraded}


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
    parser.add_argument("--anchor-weight", type=float, default=0.0,
                        help="人类策略锚点权重 β（0=旧行为）。损失 = CE(教师软分布) "
                             "+ β·CE(基座软分布)，即把学生拉回基座策略以防 BC 被覆盖")
    parser.add_argument("--tac-min-spread", type=float, default=0.0,
                    help="教师置信度过滤阈值：根分值 max−median 低于此值的局面不参与训练（0=不过滤）。教师没意见时学不到东西，只会把策略推向均匀分布")
    parser.add_argument("--device", default=None, help="计算设备")
    args = parser.parse_args()
    train_search_distill(base_model=args.base, out_path=args.out,
                         n_states=args.states, eval_sets=args.eval_sets,
                         epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr, depth=args.depth,
                         time_limit_ms=args.time_limit_ms,
                         temperature=args.temperature, seed=args.seed,
                         workers=args.workers, device=args.device,
                         anchor_weight=args.anchor_weight,
                         tac_min_spread=args.tac_min_spread)


if __name__ == "__main__":
    main()
