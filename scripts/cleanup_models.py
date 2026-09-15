#!/usr/bin/env python3
"""models/ 目录清理工具（默认 dry-run）。

⚠️ 安全性改造（2026-09-15 审查）：
  原实现的删除模式包含 `*_distilled.pt` / `*_latest.pt`，会连带删除
  `models/value_distilled_v2.pt` —— 那正是 P3 修订后 `train_rl` 热启动链
  **首选**的健康 Value 头（p1_v3/test 平衡准确率 0.735）；一旦删掉，训练会
  静默退回 `bc_best.pt`（价值头未校准，靶场开局 MAE≈1.0），候选初期反而变弱。
  本版改动：
    1. PROTECTED_EXACT 显式保护热启动链与发布链上的必需文件；
    2. **默认只打印计划**，必须显式加 `--yes` 才真正删除；
    3. 同时报告被忽略的大目录（pool/、evidence_*/），避免"清完还是 8GB"的困惑。
"""
import argparse
import fnmatch
import sys
from datetime import datetime
from pathlib import Path

# 热启动链 / 发布链上的必需文件，任何情况下都不删
PROTECTED_EXACT = {
    "best.pt",                  # 发布模型（唯一晋级出口写入）
    "bc_best.pt",               # --rebase-baseline 与 S0 热启动基线
    "bc_best_history.json",
    "value_distilled_v2.pt",    # P3 热启动链首选（健康 Value 头）
    "value_distilled.pt",       # 热启动链备选
    "MANIFEST.jsonl",           # 模型注册表
    "elo_history.jsonl",        # 训练曲线（含晋级判据依赖的 ref_vs_search2）
}

# 可清理的中间产物
DELETE_PATTERNS = [
    "*.pkl",                    # 经验池快照（3.8GB/份，可按需重建）
    "_*.pt",                    # _candidate_gate.pt 等临时快照
    "*_gate.pt",
    "*_legacy_*.pt",            # 带时间戳的旧模型备份
    "best_legacy.pt",
]

# 明确不删（保留实验证据与模型池）
KEEP_PATTERNS = [
    "candidate_latest.pt",      # 断点续训入口
    "search_distilled_*.pt",    # P2 蒸馏产物
]


def get_file_info(filepath: Path):
    try:
        stat = filepath.stat()
        return {
            "name": filepath.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
    except OSError:
        return None


def should_delete(name: str) -> bool:
    """是否可删。保护名单与保留模式优先级最高。"""
    if name in PROTECTED_EXACT:
        return False
    for pat in KEEP_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return False
    for pat in DELETE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def scan(models_dir: Path):
    to_delete, to_keep = [], []
    for f in sorted(models_dir.iterdir(), key=lambda x: x.stat().st_size, reverse=True):
        if not f.is_file():
            continue
        info = get_file_info(f)
        if not info:
            continue
        (to_delete if should_delete(f.name) else to_keep).append(info)
    return to_delete, to_keep


def dir_sizes(models_dir: Path):
    out = []
    for d in sorted(models_dir.iterdir()):
        if d.is_dir():
            total = sum(x.stat().st_size for x in d.rglob("*") if x.is_file())
            out.append((d.name, total / 1024 / 1024))
    return sorted(out, key=lambda t: -t[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="清理 models/ 中间产物（默认 dry-run）")
    ap.add_argument("--models-dir", default="models", help="模型目录")
    ap.add_argument("--yes", action="store_true", help="真正执行删除（默认只打印计划）")
    args = ap.parse_args(argv)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"[ERROR] 目录不存在: {models_dir}")
        return 1

    to_delete, to_keep = scan(models_dir)
    total = sum(f["size_mb"] for f in to_delete + to_keep)
    delete_size = sum(f["size_mb"] for f in to_delete)

    print("=" * 64)
    print(f"MODELS 清理计划（{'执行' if args.yes else 'DRY-RUN'}）")
    print("=" * 64)
    print(f"当前顶层文件总大小: {total:.2f} MB")
    print(f"\n可删除 ({len(to_delete)}):")
    for info in to_delete:
        print(f"  [DEL] {info['name']:<44s} {info['size_mb']:9.2f} MB")
    print(f"\n保留 ({len(to_keep)}):")
    for info in to_keep:
        mark = " <-- 保护" if info["name"] in PROTECTED_EXACT else ""
        print(f"  [OK ] {info['name']:<44s} {info['size_mb']:9.2f} MB{mark}")

    subdirs = dir_sizes(models_dir)
    if subdirs:
        print("\n子目录占用（本脚本不动子目录）:")
        for name, mb in subdirs[:8]:
            print(f"  {name:<44s} {mb:9.2f} MB")

    print(f"\n预计释放: {delete_size:.2f} MB")
    if not args.yes:
        print("\n[dry-run] 未删除任何文件。确认无误后加 --yes 执行。")
        return 0

    deleted, freed = 0, 0.0
    for info in to_delete:
        try:
            (models_dir / info["name"]).unlink()
            deleted += 1
            freed += info["size_mb"]
            print(f"  [DELETED] {info['name']}")
        except OSError as exc:
            print(f"  [ERROR] {info['name']}: {exc}")
    print(f"\n完成：删除 {deleted} 个文件，释放 {freed:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
