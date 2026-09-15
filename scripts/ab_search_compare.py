#!/usr/bin/env python3
"""传统搜索引擎 A/B 对照工具：验证某次改动是否改变了搜索结果。

用途：改动 `junqi/search.py` / `ai.py` / `apk_engine.py` 后，最该回答的问题是
"结果变了吗、变成更好还是更差"。只靠读代码容易得出相反结论——本项目就发生过一次：
C1 想修"限时搜索浪费预算"，第一版用比值外推，实测反而在 endgame 局面**少搜一层**
（1000ms 预算只用了 171ms）。这个工具把当时的对照流程固化下来。

做法：把 git 里的旧版本导出到临时目录，用**两个独立进程**对同一批固定局面各跑一遍
（独立进程可避免模块串味），再逐项比对。

    python scripts/ab_search_compare.py --ref-commit HEAD~1
    python scripts/ab_search_compare.py --ref-commit d6f0820 --deals 6

判读要点：
  · `默认 depth=2, time_limit=0` 应**完全一致**——默认深度搜索不该受性能优化影响；
  · `限时 1000ms` 分四种情况：`更深(改善)` / `一致` /
    `更浅(仅记录口径)`（节点数与决策都相同，只是 `max_depth` 记法不同）
    / `更浅(回归!)`（真的少搜了）。**只看 max_depth 会误判**：本项目就发生过一次
    —— 修复"根循环超时中断"时把降级结果退回浅一层，工具报 5 处"更浅(回归)"，
    逐层追踪后发现是决策真的退化了（处女局面 d=1 全同分 0.0，退回等于弃权），
    于是实现改为"保留部分层最优 + 置 degraded"。所以本工具现在**同时比对
    决策（动作+分值）与节点数**，而不只看深度数字；
  · `ApkSearchEngine` 一致说明 TT 键校验在这批局面未触发（碰撞本就罕见）；
  · `degraded` 变化单独统计：它只应"由 False 变 True"（新增的诚实标记），
    反向变化才是回归。

⚠️ 本工具不做任何写操作（不 checkout、不建 worktree），只 `git archive` 到临时目录。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r'''
import argparse, json, os, random, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--eval-sets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--deals", type=int, default=6)
a = ap.parse_args()
PKG = os.path.abspath(a.pkg)
sys.path.insert(0, PKG)
import junqi
assert os.path.abspath(os.path.dirname(os.path.dirname(junqi.__file__))) == PKG, junqi.__file__
from junqi.apk_engine import ApkSearchEngine
from junqi.config import RuleConfig
from junqi.search import ExpertSearchEngine
from junqi.state import GameState, deal

def build():
    out = []
    for name in ("opening", "midgame", "endgame"):
        p = os.path.join(a.eval_sets, name + ".jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i >= 3: break
                ln = ln.strip()
                if ln:
                    try: out.append((f"{name}#{i}", GameState.from_json(ln)))
                    except Exception: pass
    for s in range(a.deals):
        out.append((f"deal{s}", deal(random.Random(1000 + s), RuleConfig())))
        cur = deal(random.Random(1000 + s), RuleConfig()); rng = random.Random(5000 + s)
        for _ in range(12):
            acts = cur.legal_actions()
            if not acts or cur.is_terminal(): break
            cur = cur.apply(rng.choice(acts))
        out.append((f"mid{s}", cur))
    return out

def key(x):
    return None if x is None else f"{x.kind}:{x.frm}->{x.to}"

recs = []
for name, st in build():
    r = {"pos": name, "plies": st.ply, "n_legal": len(st.legal_actions())}
    try:
        e = ExpertSearchEngine(seed=7); t0 = time.time()
        act, sc, s = e.search(st, max_depth=2, time_limit_ms=0)
        r["depth2"] = {"act": key(act), "score": round(float(sc), 6),
                       "max_depth": s.max_depth, "nodes": s.nodes}
    except Exception as exc: r["depth2"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        e = ExpertSearchEngine(seed=7); t0 = time.time()
        act, sc, s = e.search(st, max_depth=4, time_limit_ms=1000)
        r["timed"] = {"act": key(act), "score": round(float(sc), 6),
                      "max_depth": s.max_depth, "nodes": s.nodes,
                      "ms": round((time.time() - t0) * 1000),
                      "degraded": bool(getattr(s, "degraded", False))}
    except Exception as exc: r["timed"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        apk = ApkSearchEngine(seed=7)
        act, sc, _ = apk.search(st, depth=2, time_limit_ms=0)
        r["apk"] = {"act": key(act), "score": round(float(sc), 6)}
    except Exception as exc: r["apk"] = {"error": f"{type(exc).__name__}: {exc}"}
    recs.append(r)
json.dump(recs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", a.out, len(recs))
'''

COMPARE = r'''
import json, sys
o = {r["pos"]: r for r in json.load(open(sys.argv[1], encoding="utf-8"))}
n = {r["pos"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))}

print("=== 1) 默认深度搜索 depth=2, time_limit=0（应完全一致）===")
same = 0; bad = []
for k in o:
    a, b = o[k]["depth2"], n[k]["depth2"]
    ok = (a.get("act") == b.get("act") and abs(a.get("score", 0) - b.get("score", 0)) < 1e-9
          and a.get("nodes") == b.get("nodes"))
    same += ok
    if not ok: bad.append((k, a, b))
print(f"   动作+分值+节点数一致: {same}/{len(o)}")
for k, a, b in bad:
    print(f"   差异 {k}: {a.get('act')}/{a.get('score')} -> {b.get('act')}/{b.get('score')}")

print()
print("=== 2) 限时搜索 max_depth=4, time_limit=1000ms ===")
print("   %-11s %-18s %-18s %-18s %s" % ("局面", "旧(深度/节点/ms)", "新(深度/节点/ms)", "深度变化", "决策"))
better = samec = worse = bookkeeping = 0
act_diff = []
deg_up = deg_down = 0
for k in o:
    a, b = o[k]["timed"], n[k]["timed"]
    if "error" in a or "error" in b:
        print(f"   {k}: ERROR {a.get('error')} / {b.get('error')}"); continue
    da, db = a["max_depth"], b["max_depth"]
    same_work = a.get("nodes") == b.get("nodes")
    same_act = a.get("act") == b.get("act")
    same_sc = abs(a.get("score", 0) - b.get("score", 0)) < 1e-9
    same_dec = same_act and same_sc
    if db > da:
        dtag, better = "更深(改善)", better + 1
    elif db < da:
        if same_work and same_dec:
            # 搜索工作量与决策都没变，只是 max_depth 的记法不同（旧实现会把
            # "进入该层但一个动作都没搜完"也记成已达深度）——不是回归。
            dtag, bookkeeping = "更浅(仅记录口径)", bookkeeping + 1
        else:
            dtag, worse = "更浅(回归!)", worse + 1
    else:
        dtag, samec = "一致", samec + 1
    atag = "同" if same_dec else "不同"
    if not same_dec:
        act_diff.append((k, a, b))
    if bool(b.get("degraded")) and not bool(a.get("degraded")):
        deg_up += 1
    elif bool(a.get("degraded")) and not bool(b.get("degraded")):
        deg_down += 1
    print("   %-11s %-18s %-18s %-18s %s" % (k, "%d/%d/%d" % (da, a["nodes"], a["ms"]),
                                             "%d/%d/%d" % (db, b["nodes"], b["ms"]),
                                             dtag, atag))
print(f"   => 更深 {better} / 一致 {samec} / 更浅仅记录口径 {bookkeeping} / 更浅回归 {worse}")
print(f"   决策(动作+分值)变化: {len(act_diff)} 处"
      f"；degraded 由 False→True: {deg_up} 处，True→False（异常）: {deg_down} 处")
for k, a, b in act_diff[:8]:
    print(f"     决策差异 {k}: {a.get('act')}/{a.get('score')} -> {b.get('act')}/{b.get('score')}")

print()
print("=== 3) ApkSearchEngine depth=2 ===")
samea = 0; diffsa = []
for k in o:
    a, b = o[k]["apk"], n[k]["apk"]
    ok = a.get("act") == b.get("act") and abs(a.get("score", 0) - b.get("score", 0)) < 1e-6
    samea += ok
    if not ok: diffsa.append((k, a, b))
print(f"   动作+分值一致: {samea}/{len(o)}")
for k, a, b in diffsa[:10]:
    print(f"   差异 {k}: {a.get('act')}/{a.get('score')} -> {b.get('act')}/{b.get('score')}")
'''


def export_commit_pkg(commit: str, dest: str) -> str:
    tar = os.path.join(dest, "pkg.tar")
    p = subprocess.run(["git", "archive", "--format=tar", commit, "junqi", "-o", tar],
                       cwd=BASE, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"git archive 失败: {p.stderr.decode('utf-8', 'replace')[:300]}")
    with tarfile.open(tar) as tf:
        tf.extractall(dest)
    os.remove(tar)
    return dest


def run_probe(python: str, pkg: str, eval_sets: str, out: str, deals: int) -> None:
    p = subprocess.run([python, "-c", PROBE, "--pkg", pkg,
                        "--eval-sets", eval_sets, "--out", out,
                        "--deals", str(deals)], capture_output=True)
    if p.returncode != 0:
        raise SystemExit("探针失败:\n"
                         + (p.stdout + p.stderr).decode("utf-8", "replace")[:1500])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="传统搜索引擎 A/B 对照（只读）")
    ap.add_argument("--ref-commit", default="HEAD~1",
                    help="对照基准的 git 提交（默认 HEAD~1）")
    ap.add_argument("--eval-sets", default=os.path.join(BASE, "eval_sets"))
    ap.add_argument("--deals", type=int, default=6, help="额外随机发牌/中盘局面数")
    ap.add_argument("--python", default=sys.executable, help="解释器路径")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="junqi_ab_") as tmp:
        old_dir = os.path.join(tmp, "old")
        os.makedirs(old_dir)
        export_commit_pkg(args.ref_commit, old_dir)
        old_json = os.path.join(tmp, "old.json")
        new_json = os.path.join(tmp, "new.json")
        print(f"[1/3] 旧版（{args.ref_commit}）探针 …", flush=True)
        run_probe(args.python, old_dir, args.eval_sets, old_json, args.deals)
        print("[2/3] 新版（工作区）探针 …", flush=True)
        run_probe(args.python, BASE, args.eval_sets, new_json, args.deals)
        print("[3/3] 比对\n", flush=True)
        p = subprocess.run([args.python, "-c", COMPARE, old_json, new_json])
        return p.returncode


if __name__ == "__main__":
    sys.exit(main())
