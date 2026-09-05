"""根因审查实证脚本：对比三个关键模型的 50 题靶场表现 + 经验池类别分布统计。

属于 P3 收尾 / P4 准入复审的只读诊断，不修改任何模型文件。
"""
import os
import pickle
from collections import Counter

import torch

from junqi.benchmark import evaluate_net_benchmark
from junqi.net import JunqiNet

MODELS = {
    "best.pt": "models/best.pt",
    "bc_best.pt": "models/bc_best.pt",
    "_candidate_gate.pt": "models/_candidate_gate.pt",
}

print("=" * 70)
print("1) 50 题靶场：三个模型 Value 表现对比")
print("=" * 70)
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"{name}: 不存在")
        continue
    try:
        net = JunqiNet.load_from_file(path)
    except Exception as e:
        print(f"{name}: 加载失败 {e}")
        continue
    r = evaluate_net_benchmark(net)
    print(f"{name}:")
    print(f"  MAE={r['value_mae_overall']:.4f} acc={r['value_class_acc']*100:.1f}% "
          f"pred={r['pred_class_counts']} true={r['true_class_counts']}")
    print(f"  opening={r['value_mae_opening']:.4f} midgame={r['value_mae_midgame']:.4f} "
          f"endgame={r['value_mae_endgame']:.4f}")

print()
print("=" * 70)
print("2) 经验池类别分布（candidate_latest_buffer.pkl）")
print("=" * 70)
buf_path = "models/candidate_latest_buffer.pkl"
if os.path.exists(buf_path):
    # 该文件为本地训练流水线自产的经验池序列化（与 train_rl.load_checkpoint 同口径），可信。
    with open(buf_path, "rb") as f:
        data = pickle.load(f)
    v = data.get("value", {})
    cls_names = {0: "Win", 1: "Draw", 2: "Loss"}
    total = 0
    dist = Counter()
    z_hist = Counter()
    for c, lst in v.items():
        total += len(lst)
        dist[cls_names.get(int(c), str(c))] = len(lst)
    print(f"Value 样本总数: {total}")
    for k in ("Win", "Draw", "Loss"):
        n = dist.get(k, 0)
        print(f"  {k}: {n} ({100.0*n/max(total,1):.1f}%)")
    p = data.get("policy", {})
    print(f"Policy 样本: opening={len(p.get(0,[]))} midgame={len(p.get(1,[]))} "
          f"endgame={len(p.get(2,[]))}")
else:
    print("buffer 文件不存在")

print()
print("=" * 70)
print("3) 门控统计功效测算（Wilson 下界，每阶段 n = 2*eval_games）")
print("=" * 70)
from junqi.train_rl import wilson_lower_bound

for n in (20, 32, 200):
    print(f"n={n} 局/阶段:")
    for score in (0.55, 0.60, 0.65, 0.70, 0.75):
        succ = score * n
        lb = wilson_lower_bound(succ, n)
        print(f"  得分 {score:.2f} -> Wilson 下界 {lb:.3f} {'(>0.5 可晋升)' if lb > 0.5 else '(<0.5 必被否决)'}")
