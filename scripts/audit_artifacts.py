#!/usr/bin/env python3
"""训练产物有效性核查（只读）。

用途：在重训/重测之前，先回答"手上这些 .pt / 日志 / 数据集到底还有多少可信"。

    python scripts/audit_artifacts.py                 # 查 models/
    python scripts/audit_artifacts.py --models-dir models --p1 datasets/p1_v3
    python scripts/audit_artifacts.py --probe         # 额外做 Value 头行为探针（较慢）

产出三部分：
  A. 权重文件两两比对（是否只是互相的副本；checkpoint 会取 ckpt["net"]）
  B. Value 头在 p1_v3/test 上的平衡准确率 / MAE / 预测分布 / 塌缩告警
  C. elo_history.jsonl 逐轮门控与晋升记录

设计背景（2026-09-15 审查）：
  · `models/best.pt` 曾与 `models/pool/bc_best.pt` **逐位相同** —— 说明它从未被
    训练或晋升更新过，只是 BC 基线副本，而 BC 的价值头未校准（平衡 acc≈随机、
    Draw 恒为 0）。只看文件名无法发现这一点，必须逐位比对。
  · `candidate_latest.pt` 是完整 checkpoint（{"net","optimizer",...}），
    直接与 `net.save()` 的包装字典比对会得到毫无意义的结论。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from junqi.net import JunqiNet                                    # noqa: E402

DEFAULT_MODELS = "models"
DEFAULT_P1 = "datasets/p1_v3"

KEY_FILES = [
    "best.pt", "bc_best.pt", "value_distilled_v2.pt", "value_distilled.pt",
    "candidate_latest.pt", "_candidate_gate.pt",
]


# ------------------------------------------------------------------ 纯函数（可单测）

def load_state_dict_any(path: str):
    """统一取出裸 state_dict。

    支持三种形态：`net.save()` 的包装字典 {"model_state":...}、
    `save_checkpoint()` 的完整检查点 {"net":..., "optimizer":...}、
    以及裸 state_dict / 直接 pickle 的 nn.Module。
    返回 (state_dict, 形态说明)。
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if isinstance(obj.get("net"), dict):
            return obj["net"], f"checkpoint(epoch={obj.get('epoch')})"
        if isinstance(obj.get("model_state"), dict):
            return obj["model_state"], "net.save() 包装"
        return obj, "裸 state_dict"
    return obj.state_dict(), "pickle(nn.Module)"


def compare_state_dicts(a: dict, b: dict) -> dict:
    """逐位比对两个 state_dict。

    返回 {"same","diff","max_delta","missing","shape_mismatch","identical"}。
    missing / shape_mismatch 分别记录只在一边出现的键与形状不符的键 ——
    这两个字段在判断"是否为同一模型"时与 diff 同等重要。
    """
    same = diff = 0
    max_delta = 0.0
    missing, shape_mismatch = [], []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            missing.append(k)
            continue
        ta, tb = a[k], b[k]
        if tuple(ta.shape) != tuple(tb.shape):
            shape_mismatch.append(k)
            continue
        if torch.equal(ta, tb):
            same += 1
        else:
            diff += 1
            max_delta = max(max_delta, (ta.float() - tb.float()).abs().max().item())
    return {"same": same, "diff": diff, "max_delta": max_delta,
            "missing": missing, "shape_mismatch": shape_mismatch,
            "identical": diff == 0 and not missing and not shape_mismatch}


# ------------------------------------------------------------------ 报告

def report_weights(models_dir: str) -> dict:
    print("=" * 78)
    print(f"A. 权重文件（{models_dir}）")
    print("=" * 78)
    sds, kinds = {}, {}
    candidates = list(KEY_FILES)
    pool = os.path.join(models_dir, "pool")
    if os.path.isdir(pool):
        candidates += [os.path.join("pool", f) for f in sorted(os.listdir(pool))
                       if f.endswith(".pt")]
    for name in candidates:
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            continue
        try:
            sd, kind = load_state_dict_any(path)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name:<30} 读取失败: {type(exc).__name__}: {exc}")
            continue
        sds[name], kinds[name] = sd, kind
        first = tuple(sd["in_conv.0.weight"].shape) if "in_conv.0.weight" in sd else "?"
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {name:<30} {size:7.2f} MB  {kind:<22} 键数={len(sd):<4} 首层={first}")

    print()
    ref = "value_distilled_v2.pt"
    if ref in sds:
        print(f"  与 {ref} 对比：")
        for name in sds:
            if name == ref:
                continue
            r = compare_state_dicts(sds[name], sds[ref])
            tag = ("逐位相同（只是副本）" if r["identical"]
                   else f"不同 {r['diff']} 键（最大差 {r['max_delta']:.3e}）")
            extra = ""
            if r["missing"]:
                extra += f" 仅一侧有 {len(r['missing'])} 键"
            if r["shape_mismatch"]:
                extra += f" 形状不符 {len(r['shape_mismatch'])} 键"
            print(f"    {name:<26} -> {tag}{extra}")

    # 关键组合：发布模型是否只是某个基线的副本
    print()
    for a, b in (("best.pt", os.path.join("pool", "bc_best.pt")),
                 ("best.pt", "bc_best.pt"),
                 ("best.pt", "candidate_latest.pt")):
        if a in sds and b in sds:
            r = compare_state_dicts(sds[a], sds[b])
            verdict = ("**逐位相同 → 该文件不是训练产物**" if r["identical"]
                       else f"不同 {r['diff']} 键（最大差 {r['max_delta']:.3e}）")
            print(f"  {a} vs {b}: {verdict}")
    return sds


def report_value_probe(models_dir: str, p1_dir: str) -> None:
    print()
    print("=" * 78)
    print(f"B. Value 头行为探针（{p1_dir}/test，指标=平衡准确率）")
    print("=" * 78)
    try:
        from junqi.train_value_distill import (evaluate_value_health,
                                               load_p1_arrays)
        d = load_p1_arrays(p1_dir, "test")
    except Exception as exc:                            # noqa: BLE001
        print(f"  跳过：{type(exc).__name__}: {exc}")
        return
    print(f"  测试集 {d['n_labeled']} 条有标签样本")
    print(f"  {'模型':<28}{'平衡acc':>9}{'MAE':>8}{'原始acc':>9}  预测分布(W/D/L)      塌缩")
    for name in KEY_FILES:
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            continue
        try:
            net = JunqiNet.load_from_file(path, device="cpu")
            h = evaluate_value_health(net, d["x"], d["y"], d["v"], device="cpu")
        except Exception as exc:                        # noqa: BLE001
            print(f"  {name:<28} 探针失败: {type(exc).__name__}: {exc}")
            continue
        pc = h["pred_counts"]
        print(f"  {name:<28}{h['balanced_acc']:>9.3f}{h['mae']:>8.3f}"
              f"{h['class_acc']:>9.3f}  "
              f"{pc['Win']:>5}/{pc['Draw']:<5}/{pc['Loss']:<6}      "
              f"{h['collapse_warning']}")
    print("  参考：随机水平 0.333；健康参考 ≈ 0.735")


def report_history(models_dir: str) -> None:
    print()
    print("=" * 78)
    print("C. elo_history.jsonl（门控与晋升）")
    print("=" * 78)
    path = os.path.join(models_dir, "elo_history.jsonl")
    if not os.path.exists(path):
        print("  (不存在)")
        return
    with open(path, encoding="utf-8") as f:
        for ln in (x.strip() for x in f if x.strip()):
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                print("  (无法解析的行)")
                continue
            vh = e.get("value_health") or {}
            print(f"  ep{e.get('epoch')}: loss={e.get('loss')} "
                  f"门控={e.get('gate_wins')}/{e.get('gate_draws')}/{e.get('gate_losses')} "
                  f"得分={e.get('gate_score')} 晋升={e.get('promoted')} "
                  f"值平衡acc={vh.get('balanced_acc')}")
    print("  注意：2026-09-15 之前的 elo 字段是线性随机游走，不可作为实力依据；")
    print("        2026-09-15 之前的轮内门控使用非配对同牌种子，统计效力不足。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练产物有效性核查（只读）")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--p1", default=DEFAULT_P1, help="p1 数据集目录（用于 Value 探针）")
    ap.add_argument("--probe", action="store_true",
                    help="执行 Value 头行为探针（需加载模型与数据集，较慢）")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.models_dir):
        print(f"[ERROR] 目录不存在: {args.models_dir}")
        return 1

    report_weights(args.models_dir)
    report_history(args.models_dir)
    if args.probe:
        report_value_probe(args.models_dir, args.p1)
    else:
        print()
        print("（加 --probe 可执行 Value 头行为探针）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
