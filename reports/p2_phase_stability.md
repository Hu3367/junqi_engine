# P2 验收第 2 条：固定开局 / 中盘 / 尾盘集合「无明显崩溃」压测

- 生成时间：2026-09-17 23:22:29
- 口径：**决策级全量覆盖**（每阶段全部快照，不抽样），每快照对每个引擎做一次单步决策
- 引擎：hybrid2, expert2（候选 hybrid2，对照 expert2）
- 复现：`python scripts/audit_p2_phase_stability.py --out-dir reports --out-name p2_phase_stability`

> 崩溃定义：① 抛异常；② 存在合法动作却返回空；③ 首选动作不在 `legal_actions()` 中。
> 终局 / 无合法动作的快照单独计数（其正确行为就是返回空），不计入崩溃。

## 一、逐项判定

| 阶段 | 引擎 | 快照 | 终局 | 无合法动作 | 异常 | 返回空 | 非法 | 判定 |
|---|---|---|---|---|---|---|---|---|
| opening | hybrid2 | 200 | 0 | 0 | 0 | 0 | 0 | ✅ PASS |
| opening | expert2 | 200 | 0 | 0 | 0 | 0 | 0 | ✅ PASS |
| midgame | hybrid2 | 200 | 0 | 0 | 0 | 0 | 0 | ✅ PASS |
| midgame | expert2 | 200 | 0 | 0 | 0 | 0 | 0 | ✅ PASS |
| endgame | hybrid2 | 200 | 0 | 0 | 0 | 0 | 0 | ✅ PASS |
| endgame | expert2 | 200 | 0 | 0 | 0 | 0 | 0 | ✅ PASS |

## 二、单步耗时（有合法动作的快照，毫秒）

| 阶段 | 引擎 | p50 | p95 | max | 平均合法动作数 |
|---|---|---|---|---|---|
| opening | hybrid2 | 18.7 | 39.9 | 71.3 | 48.37 |
| opening | expert2 | 10.8 | 31.2 | 54.3 | 48.37 |
| midgame | hybrid2 | 21.8 | 74.6 | 317.7 | 38.89 |
| midgame | expert2 | 45.9 | 430.4 | 2180.3 | 38.89 |
| endgame | hybrid2 | 8.1 | 10.9 | 13.1 | 22.84 |
| endgame | expert2 | 2.5 | 5.1 | 8.1 | 22.84 |

## 三、首选动作 kind 分布（旁证：集合非退化）

| 阶段 | 引擎 | flip | move | 合计 |
|---|---|---|---|---|
| opening | hybrid2 | 70 | 130 | 200 |
| opening | expert2 | 70 | 130 | 200 |
| midgame | hybrid2 | 0 | 200 | 200 |
| midgame | expert2 | 0 | 200 | 200 |
| endgame | hybrid2 | 2 | 198 | 200 |
| endgame | expert2 | 5 | 195 | 200 |

## 六、局限

- 本压测只验证**单步决策**不崩溃，不验证连续对局中的行为质量；后者由 `scripts/audit_p2_game_quality.py`（对局级）覆盖，两者互补。
- 每个快照只测一次、`seed` 固定；两个默认引擎的搜索是确定性的（`ExpertSearchEngine.rng` 全仓只赋值不读取），因此结果可复现。
- 快照来自真实复盘，只覆盖「实战出现过的局面分布」。
