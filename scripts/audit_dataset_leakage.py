"""跨版本数据集切分泄漏审计（2026-09-17 复核暴露的系统性缺陷的守卫）。

要解决的问题
------------
同一批 1072 个 `.sav` 被反复重导出成 p1_v1 / p1_v2 / p1_v3 等多个版本。
各版本都用 `seed=2026` 洗牌，但**洗的是不同长度的列表**（有效局数不同），
所以划分完全不同；一个版本的 train 会覆盖另一个版本 test 的大约 80%。
而 `metadata.json` 只保存 npz 的 SHA-256、**不保存文件清单**，
因此这种重叠在工程上无法被发现 —— 实测结果就是"同一个模型在自己训过的划分上
Top-1 45~52%，在真正未见的划分上只有 24~25%"（详见
`reviews/BC_ACCEPTANCE_VERDICT_2026-09-17.md`）。

本脚本做三件事
--------------
1. **单数据集内部**：train / val / test 必须两两不相交（按对局切分的基本要求）。
2. **跨版本**：给出所有数据集两两之间的 3x3 切分交集矩阵，并按危害程度分级：
   - **ERROR**：一方的 `train/val` 与另一方的 **`test`** 有交集
     （模型在自己的评测局上训练过 ⇒ 测试分数彻底无效）；
   - **WARN**：`train/val` 彼此有交集（只影响"模型选择"层面的可比性，
     test 分数本身仍合法），或两方的 `test` 有交集且至少一方 `leak_free=False`
     （两者的测试集不独立）；
   - `legacy-exempt`：命中 ERROR 但该数据集在 `--exempt` 名单里
     （历史污染是既成事实，交集数字照打印，只是不再判 ERROR）；
   - OK：仅 `train∩train` 有交集（正常）。
   任一方缺少清单 ⇒ 该对标记 `un-auditable`（历史数据集可用
   `export_dataset --legacy-reconstruct <dir>` 反推清单）。
3. **canonical test 守卫**：若存在 `datasets/canonical_test.json`，
   则断言**任何**数据集的 train/val 都不包含冻结局；启用冻结的版本其 test 必须覆盖冻结清单。

用法
----
    # 审计 datasets/ 下全部数据集
    python scripts/audit_dataset_leakage.py

    # 指定数据集
    python scripts/audit_dataset_leakage.py --datasets datasets/p1_v3 datasets/p1_v4

    # 把某个数据集的 test 划分冻结为 canonical test（写入 datasets/canonical_test.json）
    python scripts/audit_dataset_leakage.py --freeze-from datasets/p1_v4

退出码：存在 ERROR 时为 1，否则 0（可直接接进 CI）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------- 读取

def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(ds_dir: str) -> Dict[str, Any]:
    """读取一个数据集的 metadata 与切分清单。

    清单来源优先级：`split_files.json`（反推/新导出） > `metadata.json.split_files`。
    两者都没有时 `files=None`（不可审计）。
    """
    meta = _read_json(os.path.join(ds_dir, "metadata.json"))
    sidecar = _read_json(os.path.join(ds_dir, "split_files.json"))
    files, source = None, "无"
    if sidecar and isinstance(sidecar.get("files"), dict):
        files = {s: list(sidecar["files"].get(s) or []) for s in SPLITS}
        source = "split_files.json" + ("" if sidecar.get("verified", True)
                                       else "（未通过复现校验！）")
    elif meta and isinstance(meta.get("split_files"), dict):
        files = {s: list(meta["split_files"].get(s) or []) for s in SPLITS}
        source = "metadata.json.split_files"
    return {
        "dir": ds_dir,
        "name": os.path.basename(ds_dir.rstrip("\\/")),
        "version": (meta or {}).get("version"),
        "seed": (meta or {}).get("seed"),
        "min_plies": (meta or {}).get("min_plies"),
        "leak_free": bool((meta or {}).get("leak_free", False)),
        "counts": {s: (meta or {}).get(f"{s}_games") for s in SPLITS},
        "manifest_source": source,
        "files": files,
        "sets": ({s: set(files[s]) for s in SPLITS} if files else None),
    }


def discover(datasets_root: str) -> List[str]:
    out = []
    if not os.path.isdir(datasets_root):
        return out
    for name in sorted(os.listdir(datasets_root)):
        d = os.path.join(datasets_root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metadata.json")):
            out.append(d)
    return out


# ----------------------------------------------------------------- 审计

def audit(datasets: List[Dict[str, Any]], canonical: List[str],
          canonical_path: str, exempt: Optional[set] = None) -> Dict[str, Any]:
    """`exempt`：历史污染数据集的名单（名称或目录）。

    为什么需要豁免机制：p1_v1/p1_v2/p1_v3 与后续版本的重叠是**既成事实**，
    无法追溯消除；如果让它永远把审计判红，守卫很快就会被无视。
    被豁免的数据集在跨版本检查中降级为 `legacy-exempt`（WARN），
    **交集数字照样打印**，只是不再判 ERROR；其**内部**切分仍照常检查。
    豁免只说明"该版本的跨版本指标不可比"，不说明它没有泄漏。
    """
    exempt = exempt or set()
    errors: List[str] = []
    warnings: List[str] = []
    infos: List[str] = []

    # --- 1. 单数据集内部 ---
    internal = []
    n_auditable = 0
    for ds in datasets:
        if ds["sets"] is None:
            internal.append({"dataset": ds["name"], "auditable": False})
            ver = str(ds.get("version") or "?")
            hint = (f"可反推：`export_dataset --legacy-reconstruct {ds['dir']}`"
                    if ver.startswith("3.") else
                    f"该版本有效局判定口径已变（v{ver}），**无法反推**，"
                    f"只能标注为历史不可审计")
            warnings.append(
                f"⚠️ {ds['name']}: 无切分清单，内部切分与跨版本重叠**均不可审计**。{hint}")
            continue
        n_auditable += 1
        row = {"dataset": ds["name"], "auditable": True, "pairs": {}}
        ok = True
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                inter = ds["sets"][a] & ds["sets"][b]
                row["pairs"][f"{a}&{b}"] = len(inter)
                if inter:
                    ok = False
                    errors.append(f"❌ {ds['name']}: {a} ∩ {b} = {len(inter)} 局"
                                  f"（同一数据集内部泄漏；示例 {sorted(inter)[:3]}）")
        row["disjoint"] = ok
        internal.append(row)

    # --- 2. 跨版本两两 ---
    cross = []
    for i, A in enumerate(datasets):
        for B in datasets[i + 1:]:
            if A["sets"] is None or B["sets"] is None:
                cross.append({"a": A["name"], "b": B["name"], "status": "un-auditable",
                              "detail": (f"{A['name'] if A['sets'] is None else B['name']}"
                                         f" 缺切分清单")})
                continue
            mat = {f"{sa}&{sb}": len(A["sets"][sa] & B["sets"][sb])
                   for sa in SPLITS for sb in SPLITS}
            # 致命：一方的 train/val 与另一方的 **test** 有交集
            # ⇒ 该模型在自己的 test 局上训练过，测试分数彻底无效。
            fatal = {k: v for k, v in mat.items()
                     if v and k.split("&")[0] in ("train", "val")
                     and k.split("&")[1] == "test"}
            # 次致命：train/val 之间的交集 ⇒ 只影响"模型选择"层面的可比性
            # （A 训过的局被 B 用来选模型），B 的 test 分数本身仍然合法。
            selection = {k: v for k, v in mat.items()
                         if v and (k in ("train&val", "val&train", "val&val"))}
            status = "ok"
            is_exempt = (A["name"] in exempt or B["name"] in exempt)
            if fatal:
                if is_exempt:
                    status = "legacy-exempt"
                    warnings.append(
                        f"⚠️ {A['name']} 与 {B['name']}: 训练/选择划分与 **test** 重叠 "
                        f"{fatal} —— 已按历史豁免放行，但 "
                        f"**这两个版本之间的任何指标对比都无效**")
                else:
                    status = "error"
                    errors.append(
                        f"❌ {A['name']} 与 {B['name']}: 训练/选择划分与 **test** 重叠 "
                        f"{fatal}；{A['name']} 作训、{B['name']} 作评（或反向）时测试分数无效")
            elif selection:
                status = "warn"
                warnings.append(
                    f"⚠️ {A['name']} 与 {B['name']}: train/val 之间重叠 {selection}"
                    f"——test 分数合法，但两者的**模型选择不可比**"
                    f"（A 训过的局被 B 用于选模型）")
            elif mat["test&test"] and not (A["leak_free"] and B["leak_free"]):
                status = "warn"
                warnings.append(
                    f"⚠️ {A['name']} 与 {B['name']}: test 交集 {mat['test&test']} 局，"
                    f"但至少一方 leak_free=False ⇒ 两者的 test 指标不独立")
            cross.append({"a": A["name"], "b": B["name"], "status": status,
                          "matrix": mat})

    # --- 3. canonical test 守卫 ---
    canon_checks = []
    if canonical:
        canon_set = set(canonical)
        infos.append(f"canonical test 清单 {canonical_path}: {len(canon_set)} 局")
        for ds in datasets:
            row = {"dataset": ds["name"], "leak_free": ds["leak_free"],
                   "train_val_hits": None, "test_coverage": None}
            if ds["sets"] is None:
                row["status"] = "un-auditable"
                canon_checks.append(row)
                continue
            hits = (ds["sets"]["train"] | ds["sets"]["val"]) & canon_set
            row["train_val_hits"] = len(hits)
            ds_exempt = ds["name"] in exempt
            if hits and not ds_exempt:
                errors.append(f"❌ {ds['name']}: canonical test 有 {len(hits)} 局"
                              f"出现在 train/val 中（示例 {sorted(hits)[:3]}）")
            elif hits:
                row["exempt"] = True
                warnings.append(f"⚠️ {ds['name']}: canonical test 有 {len(hits)} 局落在 "
                                f"train/val（历史豁免）——该版本不能用于 canonical 评测")
            if ds["leak_free"]:
                missing = canon_set - ds["sets"]["test"]
                row["test_coverage"] = len(canon_set) - len(missing)
                if missing and not ds_exempt:
                    errors.append(f"❌ {ds['name']}: 标记 leak_free 但 test 未覆盖"
                                  f" canonical 清单的 {len(missing)} 局")
                row["status"] = "ok" if not hits and not missing else "warn"
            else:
                row["status"] = "ok" if not hits else "warn"
            canon_checks.append(row)
    else:
        infos.append(f"未找到 canonical test 清单（{canonical_path}）："
                     f"跨版本 test 可比性无守卫。可用 --freeze-from 创建。")

    # --- 4. 审计有效性守卫 ---
    # "没有可审计对象"不等于"没有泄漏"。若不设这道守卫，一个全是旧数据集的工程
    # 会得到一个绿色的"未发现泄漏"，把系统性缺陷包装成合规。
    if n_auditable == 0:
        errors.append(f"❌ 没有任何数据集携带切分清单：本次审计**无法判定**是否存在泄漏"
                      f"（这不等于无泄漏）。请先导出带清单的新版本，"
                      f"或用 --legacy-reconstruct 反推历史版本。")
    elif n_auditable < len(datasets):
        infos.append(f"本次审计覆盖 {n_auditable}/{len(datasets)} 个数据集；"
                     f"其余为历史不可审计版本，它们与其它版本的重叠仍然未知。")

    # --- 5. 训练默认数据集是否受保护 ---
    # 审计绿了但训练用的还是未冻结的数据集 ⇒ 守卫形同虚设，必须显式说明。
    try:
        from junqi.dataset import DEFAULT_P1_DIR
        cur = [ds for ds in datasets if ds["dir"].replace("\\", "/") ==
               str(DEFAULT_P1_DIR).replace("\\", "/")]
        if not cur:
            warnings.append(f"⚠️ 训练默认数据集 {DEFAULT_P1_DIR} 不在本次审计范围内")
        elif not cur[0]["leak_free"]:
            warnings.append(
                f"⚠️ 训练默认数据集 {DEFAULT_P1_DIR} 的 leak_free=False："
                f"它的 train/val 未与 canonical test 隔离，"
                f"用它训练得到的模型不能拿 canonical 分数做晋级对比")
        else:
            infos.append(f"训练默认数据集 {DEFAULT_P1_DIR}: leak_free=True ✔")
    except Exception:      # noqa: BLE001  审计不应因导入失败而中断
        pass

    # 报告里只放可 JSON 序列化的视图（`sets` 是 Python set，不能直接 dump）
    ds_public = [{k: v for k, v in ds.items() if k != "sets"} for ds in datasets]
    return {"datasets": ds_public, "internal": internal, "cross": cross,
            "canonical": canon_checks, "errors": errors,
            "warnings": warnings, "infos": infos,
            "n_auditable": n_auditable,
            "passed": not errors}


# ----------------------------------------------------------------- 报告

def render_md(rep: Dict[str, Any], canonical_path: str) -> str:
    lines = [
        "# 数据集切分泄漏审计",
        "",
        f"- 生成时间：{rep['generated_at']}",
        f"- 审计数据集：{', '.join(d['name'] for d in rep['datasets'])}",
        f"- canonical test 清单：`{canonical_path}`",
        f"- 可审计数据集：{rep.get('n_auditable', 0)}/{len(rep['datasets'])}"
        f"（0 个 ⇒ 判定为**无法判定**，不是「无泄漏」）",
        f"- 结论：**{'✅ 未发现泄漏' if rep['passed'] else '❌ 存在泄漏/无法判定'}**"
        f"（ERROR {len(rep['errors'])} / WARN {len(rep['warnings'])}）",
        "",
        "## 一、数据集清单与清单来源",
        "",
        "| 数据集 | 版本 | seed | min_plies | 清单来源 | leak_free | train | val | test |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in rep["datasets"]:
        c = ds["counts"]
        lines.append(f"| {ds['name']} | {ds['version']} | {ds['seed']} | {ds['min_plies']} | "
                     f"{ds['manifest_source']} | {ds['leak_free']} | "
                     f"{c['train']} | {c['val']} | {c['test']} |")
    lines.append("")

    lines += ["## 二、单数据集内部切分", "",
              "| 数据集 | 可审计 | train∩val | train∩test | val∩test | 合格 |",
              "|---|---|---|---|---|---|"]
    for row in rep["internal"]:
        if not row.get("auditable"):
            lines.append(f"| {row['dataset']} | ❌ 无清单 | - | - | - | ❌ |")
            continue
        p = row["pairs"]
        lines.append(f"| {row['dataset']} | ✅ | {p['train&val']} | {p['train&test']} | "
                     f"{p['val&test']} | {'✅' if row['disjoint'] else '❌'} |")
    lines.append("")

    lines += ["## 三、跨版本切分交集矩阵", "",
              "矩阵单元 = `A 的行 × B 的列` 的交集局数。",
              "",
              "- `ERROR`：`A.train/val × B.test` 非零（或反向）⇒ 测试分数无效；",
              "- `WARN`：仅 `train/val` 之间重叠（模型选择层面不可比）或 test 交集未受冻结保护；",
              "- `legacy-exempt`：命中 ERROR 但已列入历史豁免名单。",
              ""]
    for pair in rep["cross"]:
        lines.append(f"### {pair['a']}  ×  {pair['b']} —— `{pair['status']}`")
        lines.append("")
        if pair["status"] == "un-auditable":
            lines.append(f"{pair['detail']}")
            lines.append("")
            continue
        lines += ["| A \\ B | B.train | B.val | B.test |", "|---|---|---|---|"]
        for sa in SPLITS:
            cells = [str(pair["matrix"][f"{sa}&{sb}"]) for sb in SPLITS]
            lines.append(f"| **A.{sa}** | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    if rep["canonical"]:
        lines += ["## 四、canonical test 守卫", "",
                  "| 数据集 | leak_free | canonical 落入 train/val | test 覆盖 canonical | 结论 |",
                  "|---|---|---|---|---|"]
        for row in rep["canonical"]:
            lines.append(f"| {row['dataset']} | {row.get('leak_free')} | "
                         f"{row.get('train_val_hits', '-')} | "
                         f"{row.get('test_coverage', '-')} | {row.get('status')} |")
        lines.append("")

    if rep["errors"]:
        lines += ["## 五、ERROR", ""] + [f"- {e}" for e in rep["errors"]] + [""]
    if rep["warnings"]:
        lines += ["## 六、WARN", ""] + [f"- {w}" for w in rep["warnings"]] + [""]
    if rep["infos"]:
        lines += ["## 七、说明", ""] + [f"- {i}" for i in rep["infos"]] + [""]
    return "\n".join(lines)


# ----------------------------------------------------------------- 冻结

def freeze_from(ds_dir: str, canonical_path: str, note: str = "",
                stamp: bool = True) -> dict:
    """把某个数据集的 test 划分冻结为 canonical test 清单。

    `stamp=True`（默认）会回写源数据集的 `leak_free=True` 与 `frozen_test_files`。
    这样做的依据是**构造性事实**：canonical 清单就是该数据集的 test 划分，
    而该数据集 train/val ∩ test = ∅ 已由审计脚本单独校验（内部切分检查），
    所以 `train/val ∩ canonical = ∅` 必然成立。
    回写内容会带上 `leak_free_source` 说明，便于事后追溯这个标记从哪来。
    """
    ds = load_dataset(ds_dir)
    if ds["sets"] is None:
        raise SystemExit(f"❌ {ds_dir} 没有切分清单，无法冻结（先导出或反推清单）")
    files = sorted(ds["sets"]["test"])
    payload = {
        "note": note or ("canonical test 冻结集：来自该数据集导出时的 test 划分。"
                         "后续所有版本导出都必须通过 --frozen-test 传本清单，"
                         "使其整批排除出 train/val（leak_free=True），"
                         "以保证跨版本 test 指标可比、且不存在 train→test 泄漏。"),
        "source_dataset": ds_dir,
        "source_version": ds["version"],
        "source_seed": ds["seed"],
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_files": len(files),
        "files": files,
    }
    from junqi.dataset import write_json_file
    write_json_file(canonical_path, payload)

    if stamp:
        meta_path = os.path.join(ds_dir, "metadata.json")
        meta = _read_json(meta_path) or {}
        meta["leak_free"] = True
        meta["frozen_test_files"] = files
        meta["frozen_test_missing"] = sorted(set(files) - ds["sets"]["test"])
        meta["leak_free_source"] = (
            f"canonical 清单即本数据集 test 划分（{len(files)} 局），"
            f"冻结于 {payload['created_at']}；train/val ∩ canonical = ∅ 由构造与"
            f"内部切分检查共同保证（见 scripts/audit_dataset_leakage.py）")
        write_json_file(meta_path, meta)
        print(f"[freeze] 已回写 {meta_path} 的 leak_free=True"
              f"（frozen_test_files={len(files)}）")
    return payload


# ----------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="跨版本数据集切分泄漏审计")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="要审计的数据集目录（默认 datasets/ 下全部含 metadata.json 的目录）")
    ap.add_argument("--datasets-root", default="datasets", help="数据集根目录")
    ap.add_argument("--canonical", default="datasets/canonical_test.json",
                    help="canonical test 清单路径")
    ap.add_argument("--freeze-from", default=None,
                    help="把该数据集的 test 划分冻结为 canonical 清单后写入 --canonical")
    ap.add_argument("--note", default="", help="冻结时的备注")
    ap.add_argument("--exempt", nargs="*", default=None,
                    help="历史污染数据集名单（名称或目录），其跨版本重叠降级为 WARN")
    ap.add_argument("--no-stamp", action="store_true",
                    help="--freeze-from 时不回写源数据集的 leak_free 标记")
    ap.add_argument("--out-dir", default="reports", help="报告输出目录")
    ap.add_argument("--out-name", default=None, help="报告文件名前缀")
    args = ap.parse_args()

    if args.freeze_from:
        payload = freeze_from(args.freeze_from, args.canonical, args.note,
                              stamp=not args.no_stamp)
        print(f"✅ 已冻结 canonical test: {args.canonical}"
              f"（{payload['n_files']} 局，来自 {args.freeze_from}）")

    exempt = {os.path.basename(str(x).rstrip("\\/")) for x in (args.exempt or [])}
    dirs = args.datasets or discover(args.datasets_root)
    if not dirs:
        raise SystemExit("未找到任何数据集（含 metadata.json 的目录）")

    datasets = [load_dataset(d) for d in dirs]
    canonical = []
    if os.path.exists(args.canonical):
        with open(args.canonical, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("files") if isinstance(data, dict) else data
        canonical = sorted(os.path.basename(str(x)) for x in (raw or []))

    rep = audit(datasets, canonical, args.canonical, exempt)
    rep["generated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rep["canonical_path"] = args.canonical
    rep["canonical_size"] = len(canonical)
    rep["exempt"] = sorted(exempt)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.out_name or ("dataset_leakage_audit_"
                               + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    json_path = os.path.join(args.out_dir, f"{prefix}.json")
    md_path = os.path.join(args.out_dir, f"{prefix}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_md(rep, args.canonical))

    print("=" * 74)
    for ds in datasets:
        print(f"  {ds['name']:<10} version={str(ds['version']):<7} "
              f"leak_free={str(ds['leak_free']):<5} 清单={ds['manifest_source']}")
    for i in rep["infos"]:
        print(f"  · {i}")
    for e in rep["errors"]:
        print(f"  {e}")
    for w in rep["warnings"]:
        print(f"  {w}")
    if rep["passed"]:
        verdict = "✅ 未发现泄漏"
    elif rep.get("n_auditable", 0) == 0:
        verdict = "⚠️ 无法判定（没有任何数据集携带切分清单）"
    else:
        verdict = "❌ 存在泄漏"
    print(f"  结论: {verdict}")
    print(f"  JSON: {json_path}\n  MD:   {md_path}")
    print("=" * 74)
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
