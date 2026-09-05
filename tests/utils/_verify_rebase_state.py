"""S3 前置诊断：基线重建是否生效 + 经验池是否被旧数据污染（只读，不改任何文件）。"""
import json
import os
import pickle

import torch

from junqi.benchmark import evaluate_net_benchmark
from junqi.net import JunqiNet

# 1. best.pt 是否已被 BC 重建（与 bc_best.pt 权重逐键比对），以及与审查基线的哈希对照
base = json.load(open("metrics/baseline.json", encoding="utf-8"))

def sd_equal(a: str, b: str) -> bool:
    sa = JunqiNet.load_from_file(a).state_dict()
    sb = JunqiNet.load_from_file(b).state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa if k in sb)

print("best.pt == bc_best.pt 权重:", sd_equal("models/best.pt", "models/bc_best.pt"))
print("best.pt == best_legacy.pt 权重:", sd_equal("models/best.pt", "models/best_legacy.pt"))

# 2. 当前 best.pt 靶场表现（对照 metrics/baseline.json 中的旧基线哈希）
import hashlib
h = hashlib.sha256(open("models/best.pt", "rb").read()).hexdigest()[:16]
old = base["models"]["best.pt"]["sha256"]
print(f"best.pt 哈希: {h} | 审查时旧基线哈希: {old} | 同一文件: {h == old}")
r = evaluate_net_benchmark(JunqiNet.load_from_file("models/best.pt"))
print(f"best.pt 靶场: MAE={r['value_mae_overall']:.4f} acc={r['value_class_acc']*100:.1f}% "
      f"pred={r['pred_class_counts']} collapse={r['collapse_warning']}")

# 3. 经验池构成：旧数据（3 元组/世界模式）占比
buf_path = "models/candidate_latest_buffer.pkl"
size_gb = os.path.getsize(buf_path) / 2 ** 30
# 本地训练流水线自产的经验池序列化（与 train_rl.load_checkpoint 同口径），可信。
with open(buf_path, "rb") as f:
    data = pickle.load(f)
v = data.get("value", {})
total = legacy = 0
cls = {0: 0, 1: 0, 2: 0}
for c, lst in v.items():
    for s in lst:
        total += 1
        cls[int(c)] += 1
        if len(s) < 4:
            legacy += 1
p = data.get("policy", {})
print(f"buffer 文件: {size_gb:.2f} GB")
print(f"Value 样本: 总 {total} | 旧式3元组(残留旧数据) {legacy} ({100*legacy/max(total,1):.1f}%)")
print(f"Value 类别: Win={cls[0]} Draw={cls[1]} Loss={cls[2]} "
      f"(Draw 占 {100*cls[1]/max(total,1):.1f}%)")
print(f"Policy 样本: {sum(len(x) for x in p.values())}")
