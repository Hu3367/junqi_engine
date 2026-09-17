# 数据集切分泄漏审计

- 生成时间：2026-09-17 22:29:53
- 审计数据集：p1_v1, p1_v2, p1_v3, p1_v4
- canonical test 清单：`datasets/canonical_test.json`
- 可审计数据集：2/4（0 个 ⇒ 判定为**无法判定**，不是「无泄漏」）
- 结论：**✅ 未发现泄漏**（ERROR 0 / WARN 3）

## 一、数据集清单与清单来源

| 数据集 | 版本 | seed | min_plies | 清单来源 | leak_free | train | val | test |
|---|---|---|---|---|---|---|---|---|
| p1_v1 | 1.0.0 | 2026 | None | 无 | False | 1600 | 200 | 200 |
| p1_v2 | 2.0.0 | 2026 | 20 | 无 | False | 668 | 83 | 84 |
| p1_v3 | 3.0.0 | 2026 | 20 | split_files.json | True | 760 | 95 | 96 |
| p1_v4 | 4.0.0 | 2026 | 20 | metadata.json.split_files | True | 760 | 95 | 96 |

## 二、单数据集内部切分

| 数据集 | 可审计 | train∩val | train∩test | val∩test | 合格 |
|---|---|---|---|---|---|
| p1_v1 | ❌ 无清单 | - | - | - | ❌ |
| p1_v2 | ❌ 无清单 | - | - | - | ❌ |
| p1_v3 | ✅ | 0 | 0 | 0 | ✅ |
| p1_v4 | ✅ | 0 | 0 | 0 | ✅ |

## 三、跨版本切分交集矩阵

矩阵单元 = `A 的行 × B 的列` 的交集局数。

- `ERROR`：`A.train/val × B.test` 非零（或反向）⇒ 测试分数无效；
- `WARN`：仅 `train/val` 之间重叠（模型选择层面不可比）或 test 交集未受冻结保护；
- `legacy-exempt`：命中 ERROR 但已列入历史豁免名单。

### p1_v1  ×  p1_v2 —— `un-auditable`

p1_v1 缺切分清单

### p1_v1  ×  p1_v3 —— `un-auditable`

p1_v1 缺切分清单

### p1_v1  ×  p1_v4 —— `un-auditable`

p1_v1 缺切分清单

### p1_v2  ×  p1_v3 —— `un-auditable`

p1_v2 缺切分清单

### p1_v2  ×  p1_v4 —— `un-auditable`

p1_v2 缺切分清单

### p1_v3  ×  p1_v4 —— `warn`

| A \ B | B.train | B.val | B.test |
|---|---|---|---|
| **A.train** | 678 | 82 | 0 |
| **A.val** | 82 | 13 | 0 |
| **A.test** | 0 | 0 | 96 |

## 四、canonical test 守卫

| 数据集 | leak_free | canonical 落入 train/val | test 覆盖 canonical | 结论 |
|---|---|---|---|---|
| p1_v1 | False | None | None | un-auditable |
| p1_v2 | False | None | None | un-auditable |
| p1_v3 | True | 0 | 96 | ok |
| p1_v4 | True | 0 | 96 | ok |

## 六、WARN

- ⚠️ p1_v1: 无切分清单，内部切分与跨版本重叠**均不可审计**。该版本有效局判定口径已变（v1.0.0），**无法反推**，只能标注为历史不可审计
- ⚠️ p1_v2: 无切分清单，内部切分与跨版本重叠**均不可审计**。该版本有效局判定口径已变（v2.0.0），**无法反推**，只能标注为历史不可审计
- ⚠️ p1_v3 与 p1_v4: train/val 之间重叠 {'train&val': 82, 'val&train': 82, 'val&val': 13}——test 分数合法，但两者的**模型选择不可比**（A 训过的局被 B 用于选模型）

## 七、说明

- canonical test 清单 datasets/canonical_test.json: 96 局
- 本次审计覆盖 2/4 个数据集；其余为历史不可审计版本，它们与其它版本的重叠仍然未知。
- 训练默认数据集 datasets/p1_v3: leak_free=True ✔
