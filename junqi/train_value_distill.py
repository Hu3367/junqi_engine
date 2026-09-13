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


def _label_worker(task: tuple) -> int:
    """子进程独立打标：使用独立种子和 SearchConfig 对单局面执行搜索。"""
    st, depth, time_limit_ms, seed = task
    agent = ExpertAgent(SearchConfig(depth=depth, time_limit_ms=time_limit_ms), seed=seed)
    scored = agent.choose_actions(st, topn=1)
    if not scored:
        return 2
    return score_to_class(scored[0][1])


def label_with_expert(states: List[GameState], depth: int = 3,
                      time_limit_ms: int = 300,
                      seed: int = 2026, workers: int = 0) -> List[int]:
    """用专家搜索为局面打伪标签（走子方视角）。无合法走法判负（困毙）。支持多进程加速。"""
    if workers is None or workers <= 0:
        workers = min(8, os.cpu_count() or 4)
    if workers > 1 and len(states) > 10:
        import multiprocessing as mp
        tasks = [(st, depth, time_limit_ms, seed + i) for i, st in enumerate(states)]
        print(f"[蒸馏] 启动 {workers} 个并行 Worker 进程加速打标...")
        with mp.Pool(processes=workers) as pool:
            return pool.map(_label_worker, tasks)

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
                        data_path: Optional[str] = None,
                        n_samples: Optional[int] = None,
                        epochs: int = 8,
                        batch_size: int = 128, lr: float = 5e-4,
                        val_ratio: float = 0.15, seed: int = 2026,
                        depth: int = 3, time_limit_ms: int = 300,
                        workers: int = 0,
                        device: str | None = None) -> dict:
    """执行价值蒸馏：支持加载实战预打标数据（或随机推进采样） → 仅训练价值头 → 保存。
    策略头与主干冻结，保证 BC 策略能力不被破坏。返回指标字典。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = RuleConfig()

    # 1. 局面采样与专家打标（优先读取高质量预打标实战数据）
    from collections import Counter
    if data_path and os.path.exists(data_path):
        import json
        from .tactical_sampler import dict_to_state
        print(f"[蒸馏] 从实战预打标数据源加载：{data_path} ...")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if n_samples is not None and n_samples < len(raw_data):
            rng.shuffle(raw_data)
            raw_data = raw_data[:n_samples]

        samples = []
        labels = []
        for item in raw_data:
            st = dict_to_state(item["state_dict"], cfg=cfg)
            feat = encode_state_np(st, seat=st.turn, world=None)
            cls = int(item["expert_class"])
            score = float(item.get("expert_score", 0.0))
            phase = int(item.get("phase", detect_phase(st)))
            samples.append((feat, cls, score, phase))
            labels.append(cls)

        label_dist = dict(Counter(labels))
        print(f"[蒸馏] 成功加载 {len(samples)} 个实战局面，标签分布: "
              f"Win={label_dist.get(0, 0)} Draw={label_dist.get(1, 0)} Loss={label_dist.get(2, 0)}")
    else:
        n_sm = n_samples or 1200
        n_open = int(n_sm * 0.25)
        n_mid = int(n_sm * 0.40)
        n_end = n_sm - n_open - n_mid
        print(f"[蒸馏] 采样局面：开局 {n_open} / 中盘 {n_mid} / 尾盘 {n_end} ...")
        positions = gen_positions(rng, cfg, n_opening=n_open, n_midgame=n_mid,
                                  n_endgame=n_end)
        print(f"[蒸馏] 有效局面 {len(positions)}，开始专家打标 "
              f"(depth={depth}, time_limit={time_limit_ms}ms) ...")
        labels = label_with_expert([st for st, _ in positions], depth=depth,
                                   time_limit_ms=time_limit_ms, seed=seed, workers=workers)
        label_dist = dict(Counter(labels))
        print(f"[蒸馏] 伪标签分布: Win={label_dist.get(0, 0)} "
              f"Draw={label_dist.get(1, 0)} Loss={label_dist.get(2, 0)}")

        samples = [(encode_state_np(st, seat=st.turn, world=None), z,
                    (600.0 if z == 0 else (-600.0 if z == 2 else 0.0)), phase)
                   for (st, phase), z in zip(positions, labels)]

    # 2. 确定性切分训练/验证
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _batch(indices):
        xs = torch.from_numpy(np.stack([samples[i][0] for i in indices])).float().to(device)
        ys = torch.tensor([samples[i][1] for i in indices], dtype=torch.long).to(device)
        vs = torch.tensor([math.tanh(samples[i][2] / SCORE_SCALE) for i in indices], dtype=torch.float32).to(device)
        return xs, ys, vs

    # 3. 冻结主干与策略头，仅训练价值头
    net = JunqiNet.load_from_file(base_model, device=device)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("value_head")
    optimizer = torch.optim.AdamW([p for p in net.value_head.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=1e-4)

    def _eval(indices):
        net.eval()
        correct, total = 0, 0
        total_mse = 0.0
        pred_cnt = Counter()
        with torch.no_grad():
            for b in range(0, len(indices), batch_size):
                xs, ys, vs = _batch(indices[b:b + batch_size])
                _, v_logits = net(xs)
                if v_logits.shape[-1] == 3:
                    pred = v_logits.argmax(dim=-1)
                    probs = F.softmax(v_logits, dim=-1)
                    pred_v = probs[:, 0] - probs[:, 2]
                else:
                    v = torch.tanh(v_logits).squeeze(-1)
                    pred = torch.where(v > CLASS_THRESHOLD, 0,
                                       torch.where(v < -CLASS_THRESHOLD, 2, 1))
                    pred_v = v
                correct += int((pred == ys).sum())
                total += int(ys.numel())
                total_mse += float(F.mse_loss(pred_v, vs).item()) * ys.size(0)
                for c in pred.tolist():
                    pred_cnt[c] += 1
        return correct / max(total, 1), total_mse / max(total, 1), dict(pred_cnt)

    best_acc, best_mse, best_sd = -1.0, 999.0, None
    for ep in range(1, epochs + 1):
        net.eval()
        net.value_head.train()
        rng.shuffle(train_idx)
        total_loss, n_b = 0.0, 0
        for b in range(0, len(train_idx), batch_size):
            xs, ys, vs = _batch(train_idx[b:b + batch_size])
            optimizer.zero_grad()
            _, v_logits = net(xs)
            if v_logits.shape[-1] == 3:
                loss_ce = F.cross_entropy(v_logits, ys)
                probs = F.softmax(v_logits, dim=-1)
                pred_v = probs[:, 0] - probs[:, 2]
                loss_mse = F.mse_loss(pred_v, vs)
                loss = loss_ce + 0.5 * loss_mse
            else:
                loss = F.mse_loss(v_logits.squeeze(-1), vs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        val_acc, val_mse, val_pred = _eval(val_idx)
        print(f"[蒸馏] Epoch {ep}: loss={total_loss / max(n_b, 1):.4f} "
              f"val_acc={val_acc * 100:.1f}% val_mse={val_mse:.4f} val_pred={val_pred}")
        if val_acc > best_acc or (abs(val_acc - best_acc) < 1e-4 and val_mse < best_mse):
            best_acc = val_acc
            best_mse = val_mse
            best_sd = {k: v.detach().clone()
                       for k, v in net.value_head.state_dict().items()}

    if best_sd is not None:
        net.value_head.load_state_dict(best_sd)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    import time
    out_dict = {
        "model_state": net.state_dict(),
        "in_channels": net.in_channels,
        "num_blocks": len(net.blocks),
        "channels": net.in_conv[0].out_channels,
        "base_model": base_model,
        "distill_samples": len(samples),
        "val_acc": best_acc,
        "val_mse": best_mse,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        base_ckpt = torch.load(base_model, map_location="cpu", weights_only=False)
        if isinstance(base_ckpt, dict):
            for k in ("val_top1", "val_top3", "val_metrics"):
                if k in base_ckpt:
                    out_dict[k] = base_ckpt[k]
    except Exception:
        pass

    torch.save(out_dict, out_path)
    print(f"[蒸馏] 完成：最佳验证准确率 {best_acc * 100:.1f}% (MSE={best_mse:.4f})，已保存 {out_path}")

    # 同步更新 models/pool/value_distilled.pt（若 pool 目录存在）
    pool_path = os.path.join("models", "pool", os.path.basename(out_path))
    if os.path.exists(os.path.dirname(pool_path)):
        import shutil
        shutil.copyfile(out_path, pool_path)
        print(f"[蒸馏] 已同步更新对手池权重: {pool_path}")

    return {
        "samples": len(samples),
        "label_dist": label_dist,
        "best_val_acc": best_acc,
        "best_val_mse": best_mse,
        "out_path": out_path,
    }


# ---------------------------------------------------------------- 真实终局标签训练 (P1)

def train_value_from_p1_dataset(p1_dir: str = "datasets/p1_v2",
                                base_model: str = "models/bc_best.pt",
                                out_path: str = "models/value_distilled_v2.pt",
                                epochs: int = 8, batch_size: int = 512,
                                lr: float = 5e-4, weight_decay: float = 1e-4,
                                seed: int = 2026,
                                device: str | None = None) -> dict:
    """P1：用官方 list.cfg 解密真实终局标签重训 Value 头。

    数据源为 export_replay_dataset 导出的 p1_v2 npz（按对局切分、带哈希）：
      - val_classes: 0=Win / 1=Draw / 2=Loss（按每手行动方视角，官方终局码真值）；
      - has_values:  False 为无价值标签样本（早期强退 code 20 等），仅保留 Policy；
    标签规则严格符合 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §6：
      code 1/21/22/23 -> ±1；code 40/42/43 -> 0；code 20 -> 不赋值。

    仅训练价值头（主干与策略头冻结，BC 策略能力不受影响）。
    输出为独立候选权重，绝不覆盖 best.pt（AGENTS.md 硬约束）。
    返回 Value 健康度指标 dict（三分类准确率 / RMSE / 预测分布）。
    """
    import json
    import time
    from collections import Counter

    import numpy as np
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    def _load_split(name: str):
        d = np.load(os.path.join(p1_dir, name))
        return (d["states"], d["val_classes"], d["values"], d["has_values"])

    print(f"[P1-Value] 加载 {p1_dir} (train/val npz，官方终局码标签) ...")
    x_tr, y_tr, v_tr, h_tr = _load_split("train.npz")
    x_va, y_va, v_va, h_va = _load_split("val.npz")

    # has_values=True 的样本才进入 Value 训练（AGENTS.md：未终局/早期逃跑不赋 Value）
    m_tr, m_va = h_tr.astype(bool), h_va.astype(bool)
    x_tr, y_tr, v_tr = x_tr[m_tr], y_tr[m_tr], v_tr[m_tr]
    x_va, y_va, v_va = x_va[m_va], y_va[m_va], v_va[m_va]
    dist = Counter(int(c) for c in y_tr)
    print(f"[P1-Value] 训练样本 {len(y_tr)} (Win={dist.get(0, 0)} Draw={dist.get(1, 0)} "
          f"Loss={dist.get(2, 0)})，验证样本 {len(y_va)}")

    net = JunqiNet.load_from_file(base_model, device=device)
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("value_head")
    optimizer = torch.optim.AdamW([p for p in net.value_head.parameters()
                                   if p.requires_grad], lr=lr, weight_decay=weight_decay)

    def _run(split_x, split_y, split_v, train_mode: bool):
        net.train(mode=train_mode)
        net.value_head.train(mode=train_mode)
        agg = {"loss": 0.0, "correct": 0, "n": 0, "mse": 0.0}
        pred_cnt = Counter()
        order = np.random.RandomState(seed if not train_mode else seed + epoch).permutation(len(split_y)) \
            if train_mode else range(len(split_y))
        with torch.set_grad_enabled(train_mode):
            for b in range(0, len(split_y), batch_size):
                idx = order[b:b + batch_size]
                xs = torch.from_numpy(split_x[idx]).float().to(device)
                ys = torch.from_numpy(np.asarray(split_y[idx])).long().to(device)
                vs = torch.from_numpy(np.asarray(split_v[idx])).float().to(device)
                _, v_logits = net(xs)
                probs = F.softmax(v_logits, dim=-1)
                pred_v = probs[:, 0] - probs[:, 2]
                loss = (F.cross_entropy(v_logits, ys)
                        + 0.5 * F.mse_loss(pred_v, vs))
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                agg["loss"] += float(loss.item()) * ys.size(0)
                agg["correct"] += int((probs.argmax(dim=-1) == ys).sum())
                agg["mse"] += float(F.mse_loss(pred_v, vs).item()) * ys.size(0)
                agg["n"] += int(ys.size(0))
                if not train_mode:
                    for c in probs.argmax(dim=-1).tolist():
                        pred_cnt[int(c)] += 1
        n = max(agg["n"], 1)
        return (agg["loss"] / n, agg["correct"] / n, agg["mse"] / n, dict(pred_cnt))

    best_acc, best_mae, best_sd = -1.0, 9.9, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _, _ = _run(x_tr, y_tr, v_tr, train_mode=True)
        va_loss, va_acc, va_mse, va_pred = _run(x_va, y_va, v_va, train_mode=False)
        rmse = va_mse ** 0.5
        print(f"[P1-Value] Epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} val_MAE={mae:.4f} "
              f"val_pred(0/1/2)={va_pred}")
        # 晋级依据 = 验证集三分类准确率（并列时看更低 MAE），不是训练 loss
        if va_acc > best_acc or (abs(va_acc - best_acc) < 1e-6 and rmse < best_mae):
            best_acc, best_mae = va_acc, rmse
            best_sd = {k: t.detach().clone() for k, t in net.value_head.state_dict().items()}

    if best_sd is not None:
        net.value_head.load_state_dict(best_sd)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "model_state": net.state_dict(),
        "in_channels": net.in_channels,
        "num_blocks": len(net.blocks),
        "channels": net.in_conv[0].out_channels,
        "base_model": base_model,
        "p1_dir": p1_dir,
        "value_train_samples": int(len(y_tr)),
        "val_acc": best_acc,
        "val_rmse": best_mae,
        "label_rule": "list.cfg terminal codes: 1/21/22/23=+-1, 40/42/43=0, 20=no-value",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_path)
    print(f"[P1-Value] 完成：最佳验证三分类准确率 {best_acc:.3f} (MAE={best_mae:.4f})，"
          f"候选已保存 {out_path}（未触碰 best.pt）")
    return {"val_acc": best_acc, "val_rmse": best_mae,
            "train_samples": int(len(y_tr)), "out_path": out_path}


def main():
    default_data = "datasets/distill_tactical_labeled.json" if os.path.exists("datasets/distill_tactical_labeled.json") else None
    parser = argparse.ArgumentParser(description="S2 专家价值蒸馏预热（仅训练价值头）")
    parser.add_argument("--base", default="models/bc_best.pt", help="基座模型路径")
    parser.add_argument("--out", default="models/value_distilled.pt", help="输出权重路径")
    parser.add_argument("--data", default=default_data, help="预打标数据集 JSON 路径")
    parser.add_argument("--samples", type=int, default=None, help="蒸馏局面数（默认使用全部数据或1200）")
    parser.add_argument("--epochs", type=int, default=8, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集占比")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--depth", type=int, default=3, help="专家搜索深度")
    parser.add_argument("--time-limit-ms", type=int, default=300,
                        help="专家搜索单步时间预算（毫秒）")
    parser.add_argument("--workers", type=int, default=0, help="并行打标进程数 (0 为自动)")
    parser.add_argument("--device", default=None, help="计算设备")
    args = parser.parse_args()
    train_value_distill(base_model=args.base, out_path=args.out,
                        data_path=args.data,
                        n_samples=args.samples, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr,
                        val_ratio=args.val_ratio, seed=args.seed,
                        depth=args.depth, time_limit_ms=args.time_limit_ms,
                        workers=args.workers,
                        device=args.device)


if __name__ == "__main__":
    main()
