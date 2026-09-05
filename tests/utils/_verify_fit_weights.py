"""拟合权重晋级验证（P1，一次性脚本）：
按 §7 协议做 fit_weights.json(新) vs 默认权重(旧) 的镜像对抗——
先后手各半、固定种子、纯得分(胜1/和0.5/负0)与胜/和/负拆分、Wilson 下界。

  python _verify_fit_weights.py [spec] [games]   # 默认 search2 200 局
"""
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")

from junqi.config import EvalWeights, RuleConfig
from junqi.selfplay import play_game
from junqi.train_rl import wilson_lower_bound

SPEC = sys.argv[1] if len(sys.argv) > 1 else "search2"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WORKERS = 8
SEED = 20260831


def chunk_job(args):
    cand_d, base_d, spec, seed0, n, start = args
    cand, base = EvalWeights.from_dict(cand_d), EvalWeights.from_dict(base_d)
    w = d = l = 0
    for i in range(n):
        seat = (start + i) % 2           # 先后手各半
        w0, w1 = (cand, base) if seat == 0 else (base, cand)
        rec = play_game(spec, spec, SEED + start + i, RuleConfig(),
                        weights0=w0, weights1=w1)
        if rec["winner"] == seat:
            w += 1
        elif rec["winner"] in (-1, None):
            d += 1
        else:
            l += 1
    return w, d, l


if __name__ == "__main__":
    fit = json.load(open("reports/fit_weights.json", encoding="utf-8"))
    cand_d, base_d = fit["eval_weights"], EvalWeights().to_dict()

    per = GAMES // WORKERS
    jobs = [(cand_d, base_d, SPEC, SEED, per, w * per) for w in range(WORKERS)]
    extra = GAMES - per * WORKERS
    if extra:
        jobs.append((cand_d, base_d, SPEC, SEED, extra, WORKERS * per))

    with Pool(len(jobs)) as pool:
        parts = pool.map(chunk_job, jobs)
    w = sum(p[0] for p in parts)
    d = sum(p[1] for p in parts)
    l = sum(p[2] for p in parts)
    n = w + d + l
    score = (w + 0.5 * d) / n
    nonloss = wilson_lower_bound(w + d, n)

    print(f"=== 拟合权重晋级验证 | {SPEC} | {n} 局 | seed={SEED} ===")
    print(f"新权重: 胜 {w} / 和 {d} / 负 {l}")
    print(f"纯得分 = {score:.3f}（>0.5 为优于旧权重）")
    print(f"不败率 Wilson 下界 = {nonloss:.3f}")
    verdict = score > 0.5 and nonloss >= 0.5
    print(f"晋级判定: {'通过 → 可启用新权重' if verdict else '不通过 → 保留默认权重'}")

    out = {"spec": SPEC, "games": n, "seed": SEED, "wins": w, "draws": d,
           "losses": l, "score": round(score, 4),
           "nonloss_wilson_lower": round(nonloss, 4), "promote": verdict}
    rep = dict(fit)
    rep.setdefault("metrics", {})["mirror_verify"] = out
    with open("reports/fit_weights.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("验证结果已写入 reports/fit_weights.json 的 metrics.mirror_verify")
