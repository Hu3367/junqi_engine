"""A3 效果验收（一次性脚本）：传统搜索显式判断增强的镜像对抗。
search2(增强：三新键默认值) vs search2(旧行为：三新键置 0)，
其余口径完全相同（同为默认子力价值、同为修复后的公共暗子池口径），
唯一变量即 A2 三项 —— 保证可归因。

§7 协议：≥200 局、先后手各半、固定种子、纯得分 + 胜/和/负拆分 + Wilson 下界。

  python _verify_a2.py [games]   # 默认 200 局
"""
import json
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")

from junqi.config import EvalWeights, RuleConfig
from junqi.selfplay import play_game
from junqi.train_rl import wilson_lower_bound

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
WORKERS = 8
SEED = 20260901
SPEC = "search2"


def chunk_job(args):
    cand_d, base_d, seed0, n, start = args
    cand, base = EvalWeights.from_dict(cand_d), EvalWeights.from_dict(base_d)
    w = d = l = 0
    for i in range(n):
        seat = (start + i) % 2           # 先后手各半
        w0, w1 = (cand, base) if seat == 0 else (base, cand)
        rec = play_game(SPEC, SPEC, SEED + start + i, RuleConfig(),
                        weights0=w0, weights1=w1)
        if rec["winner"] == seat:
            w += 1
        elif rec["winner"] in (-1, None):
            d += 1
        else:
            l += 1
    return w, d, l


if __name__ == "__main__":
    cand = EvalWeights()                 # 三新键默认值（启用增强）
    base = EvalWeights()
    base.camp_zone = 0.0                 # 旧行为：关闭三项
    base.fortress = 0.0
    base.hidden_tempo = 0.0
    cand_d, base_d = cand.to_dict(), base.to_dict()

    per = GAMES // WORKERS
    jobs = [(cand_d, base_d, SEED, per, w * per) for w in range(WORKERS)]
    extra = GAMES - per * WORKERS
    if extra:
        jobs.append((cand_d, base_d, SEED, extra, WORKERS * per))

    with Pool(len(jobs)) as pool:
        parts = pool.map(chunk_job, jobs)
    w = sum(p[0] for p in parts)
    d = sum(p[1] for p in parts)
    l = sum(p[2] for p in parts)
    n = w + d + l
    score = (w + 0.5 * d) / n
    nonloss = wilson_lower_bound(w + d, n)

    print(f"=== A3 验收 | {SPEC} | {n} 局 | seed={SEED} ===")
    print(f"增强权重: 胜 {w} / 和 {d} / 负 {l}")
    print(f"纯得分 = {score:.3f}（>0.5 为三项增强净贡献为正）")
    print(f"不败率 Wilson 下界 = {nonloss:.3f}")
    verdict = score > 0.5 and nonloss >= 0.5
    print(f"A3 判定: {'通过 → 保留三项默认值' if verdict else '不通过 → 需回调权重或关闭单项复查'}")

    out = {"spec": SPEC, "games": n, "seed": SEED, "wins": w, "draws": d,
           "losses": l, "score": round(score, 4),
           "nonloss_wilson_lower": round(nonloss, 4), "pass": verdict,
           "note": "候选=默认权重(含A2三项)，基线=三新键置0；其余口径一致"}
    with open("reports/a2_mirror_verify.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("结果已写入 reports/a2_mirror_verify.json")
