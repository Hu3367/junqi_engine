# 补测 P2 验收第 3、4 条 + 数据集切分可审计化

> **阶段归属**：`P2`（行为克隆与搜索蒸馏）
> **承接**：`reviews/BC_ACCEPTANCE_VERDICT_2026-09-17.md` §6「先补 P2 缺口、再修数据切分」
> **执行时间**：2026-09-17 22:00–23:20（§7 修复与复测为 23:00–23:20 追加）
> **产物**：
> - `reports/gate_hybrid2_vs_search2_p2accept.json`（③）
> - `reports/p2_game_quality.json` / `.md`（④，修复后）
> - `reports/p2_game_quality_prefix_20260917.json` / `.md`（④ 修复前留档）
> - `datasets/p1_v3/split_files.json`、`datasets/canonical_test.json`、`datasets/p1_v4/`（切分）
> - `reports/dataset_leakage_audit.json` / `.md`（切分审计）
> - 新代码：`scripts/audit_p2_game_quality.py`、`scripts/audit_dataset_leakage.py`、
>   `tests/test_dataset_split_manifest.py`
> - 修复：`junqi/ai.py`（`HybridAgent` 的 avoid 判定）+ `tests/test_hybrid_agent.py` 新回归测试

---

## 0. 结论速览

| 验收条款 | 结论 | 关键数字 |
|---|---|---|
| ③ 混合代理在相同计算预算下**不弱于** `search2` | ✅ **通过（非劣）** | 得分率 0.5225（Wilson [0.4535, 0.5906]，δ=5% 非劣界成立）；配对 Δ 检验 p<0.0001 支持更强；但**不能**宣称"显著更强"（SPRT `accept_h0`） |
| ④ 非法 / 送旗 / 无意义循环 | ⚠️ **首次未通过 → 修复后通过** | 首次：非法 0/4165 ✅、送旗 1/4165 ✅、**重复判和 24/60 = 40%**（对照 `expert2` 4/60 = 6.7%）❌。定位到 `junqi/ai.py:538` 的 `avoid` 判定缺陷并修复后复测：**2/60 = 3.3%** ✅（详见 §7） |
| 数据切分可审计化 | ✅ 完成 | p1_v3 清单反推**逐项吻合**（760/95/96 局、90645/11360/11811 plies）；冻结 96 局 canonical；`p1_v4` 导出 `leak_free=True` |

**P2 仍未整体通过**：① 满足、② 仍缺（未在 `eval_sets/{opening,midgame}.jsonl` 上测）、
③ 通过（非劣）、④ 修复后通过（§7）。**因此仍不建议现在启动 P3 长训练** ——
先把 ② 补齐并处理 §6 的残余风险。

---

## 1. ③ 混合代理 vs `search2`（同预算门控）

```bat
python -m junqi gate --a hybrid2 --b search2 --seeds 100 --workers 4 ^
    --init-set eval_sets/endgame.jsonl --model-a models/bc_best.pt ^
    --out reports --out-name gate_hybrid2_vs_search2_p2accept
```

- 100 组种子（配对同牌 + 先后手互换）= **200 局**，全部从 `eval_sets/endgame.jsonl`
  取起始局面（纯随机发牌下自对局大面积循环判和，门控无分辨力 —— 这一点已在上一轮验证）。
- 总战绩：**24 胜 / 161 和 / 15 负**，即 176 和预期中的 161 和 + 39 胜负局。

| 口径 | 值 | 95% 区间 | 读法 |
|---|---|---|---|
| 官方得分率（胜1/和0.5/负0） | **0.5225** | [0.4535, 0.5906] | 点估计优于 0.5，区间含 0.5 |
| 仅计胜负局胜率 | 0.6154 | [0.4590, 0.7511]，n=39 | 样本太少，不做判据 |
| 配对检验（每种子先后手合计 vs 1.0） | mean 1.0450 | z=+1.993, p=0.0463 | 边缘显著 |
| SPRT（elo0=0, elo1=65） | LLR −28.45 | 界 [−2.94, +2.94] | **`accept_h0`** |
| 配对 Δ 检验（保留估值差幅度） | **+243.2** | t=+4.747, p<0.0001；符号 正72/负28 | **显著为正** |
| 裁决式判分 | 0.5100 | [0.4412, 0.5784] | 受座位标签偏差拉回 0.5（见下） |

终局原因拆分（候选视角）：`no_capture` 86 和、`repetition` 75 和、
`immobilized` 20（16 胜 4 负）、`flag` 19（8 胜 11 负）。

### 1.1 为什么"裁决式判分"和"配对 Δ 检验"结论相反

工程早已记录 `evaluate_expert` 存在**按座位标签的加性偏差**（先手 +90 量级）。
- 裁决式判分在同**一对**局内比较 `final_eval_sym0 vs final_eval_sym1`，两局的下法不同 ⇒
  终局局面不同 ⇒ 座位偏差**没有**被抵消，反而把真实强度差（≈121 分）压过半；
  所以它给出 0.5100 而不是更高。
- 配对 Δ 检验用 `d = Δ(r0) − Δ(r1)`，同一副牌先后手互换 ⇒ 座位偏差**严格抵消**，
  于是 243.2 的净优势显现出来（p<0.0001）。

**应以配对 Δ 检验为准**（它的设计目的就是消掉该偏差）；裁决式判分在本配置下是保守下界。

### 1.2 判定

条款原文是"**不弱于** `search2`"。据此：
- 点估计 0.5225 > 0.5；
- 非劣性检验：要拒绝"得分率 ≤ 0.5 − δ"，需 Wilson 下界 > 0.5 − δ。
  0.4535 > 0.45 ⇒ **δ = 5% 的非劣界成立**；
- 配对 Δ 检验独立地给出 p<0.0001 的正向证据。

⇒ ③ **通过（非劣）**。但必须同时记录：**"显著优于 search2"不成立**
（SPRT 接受 H0，Wilson 区间含 0.5）。长期记忆已定证：分辨 0.05 的得分率差需约 1500 局，
本次 200 局只能支撑"不弱于"，不能支撑"更强"。

---

## 2. ④ 非法 / 送旗 / 无意义循环（新增审计）

新增 `scripts/audit_p2_game_quality.py`。此前的唯一证据是 `eval_bc.py` 里
硬编码的一行"非法动作预测率 0.00%"，**没有分母**、也没有覆盖送旗与循环。

### 2.1 口径（分子分母都写清，可复算）

- 分母 `decisions` = 双方**实际做出选择**的次数（= 各局 plies 之和）；另有局级分母。
- **非法**：① 策略返回的动作不在 `legal_actions()` 中（命中即兜底但计数）；
  ② `apply()` 后状态不变量被破坏（`ply` 未恰好 +1 / 子力总数不守恒 / 越界坐标）。
- **送旗**：军旗 `Rank.QI` **不可移动**（`legal_actions` 明确排除 LEI/QI），
  所以"送旗"只能表现为"本方这一手**新造出**对方一步可吃己方军旗的窗口"：
  `(走子前对方不能吃) 且 (走子后对方能吃)`。与"最后是否真被吃"解耦，避免幸存者偏差。
  判定复用 `legal_actions()`（规则唯一真源），本脚本**不重写** `flag_gong_only` /
  `flag_needs_mines_cleared` / 暗子不可攻击等门控。
- **无意义循环**：`end_repetition`（相同局面重复判和）/ `end_max_plies`（触顶）/
  往复踱步探针（复用 `scripts/mine_blunders.py`，按**事件段**计数而非按 ply，避免长循环被放大）
  / 龟缩拒翻探针。
- **判定**：非法类硬性为 0；其余与对照配置（项目既有传统强引擎 `expert2`，同一起始局面、
  同一对手 `search2`、同 60 局）做 **Wilson 非劣性检验**（与 `eval_gate` 同一统计真源）——
  判 FAIL 需同时"候选 Wilson 下界 > 对照 Wilson 上界"且"点估计 > 对照 × 1.5"。

⚠️ 与条款原文的差异：条款说"**GUI 中**出现…"。本审计在**无头自对弈**下跑，
走的是与 GUI 相同的策略层与规则层（`junqi.selfplay.make_strategy` +
`GameState.legal_actions/apply`），但不经过 GUI 的事件循环与坐标映射。
GUI 侧的点击合法性由 P0 的 `legal_actions` 门控保证，二者共享同一真源。

### 2.2 结果（60 局，`eval_sets/endgame.jsonl` 起始）

| 指标 | `hybrid2` vs `search2` | 对照 `expert2` vs `search2` | 判定 |
|---|---|---|---|
| 非法动作 | **0 / 4165** | 0 / 4975 | ✅ |
| 非法状态转移 | **0 / 4165** | 0 / 4975 | ✅ |
| 送旗（每 100 决策） | 1 次 = 0.024 | 0 次 | ✅（不显著更差，远低于 2/100 上限） |
| 送旗致失旗（局占比） | 1/60 = 1.7% | 0/60 | ✅ |
| 可吃旗窗口出现次数 | 5 | 6–7 | （说明该指标有信号、未被规则饱和） |
| 往复踱步事件段（每 100 决策） | **0.192**（8 段） | 0.422（21 段） | ✅ 优于对照 |
| 龟缩拒翻事件段 | 0 | 0 | ✅ |
| **重复判和+触顶（局占比）** | **24/60 = 40.0%** | **4/60 = 6.7%** | ❌ **显著且幅度更大** |

### 2.3 关键发现：和棋总数相同，但**判和的方式**少了 6 倍

| 终局原因 | `hybrid2` | `expert2` |
|---|---|---|
| `flag`（被吃旗） | 5 | 5 |
| `immobilized`（困毙） | 8 | 9 |
| `no_capture`（70 手无吃子） | 23 | 42 |
| **`repetition`（局面重复）** | **24** | **4** |
| 合计和棋 | 55 | 55 |
| 平均手数 | 125.1 | 138.6 |

- **总判和局数完全相同（55/60）**，所以这不是"和棋更多"，而是"**判和的方式变了**"：
  混合引擎明显更倾向把对局带进**局面重复**，而不是被 70 手无吃子规则判和。
- 往复踱步探针反而更低（8 vs 21）⇒ 这些重复**不是**两格来回的经典踱步，
  而是更长的循环（多子绕圈），经典探针抓不到。
- ⇒ 条款里的"无意义循环"一项对混合引擎**不合格**。对人类的可感知后果是：
  AI 会在已经无进展的局面上反复回摆，而不是尝试打开局面（或干脆认输），
  这正是条款想禁止的行为。

### 2.4 ④ 判定（**修复前**）

**未通过**：非法 ✅、送旗 ✅、无意义循环 ❌。

> ⚠️ 本节及 2.2/2.3 记录的是**修复前**的数据，用于留证。
> 该缺陷的根因已定位到 `junqi/ai.py:538` 并修复，修复后的复测见 **§7**（结论：通过）。

---

## 3. 数据集切分可审计化

### 3.1 问题回顾

上一轮定证：同一批 1072 个 `.sav` 被反复重导出为 p1_v1/v2/v3，
各版用同一 `seed=2026` 洗**不同长度**的列表 ⇒ 划分完全不同，
旧版 train 覆盖新版 test 约 80%；而 `metadata.json` **只存 SHA-256、没有文件名清单**
⇒ 重叠无法被审计。实测同一模型在"自己训过的划分"上 Top-1 45–52%、
在真正未见的划分上只有 24–25%。

### 3.2 修复（三处）

1. **清单落盘**：`DatasetStats` 新增 `split_files` / `sav_dir` / `frozen_test_files` /
   `frozen_test_missing` / `leak_free`，导出时把每个 split 的 `.sav` 文件名写进
   `metadata.json`（**向后兼容**：老 metadata 没有这些字段，审计脚本会标为"不可审计"）。
2. **冻结 test 机制**（`--frozen-test`，默认 `datasets/canonical_test.json`）：
   冻结局**整批排除出 train/val**，且 **`test` 恒等于 canonical 清单**。
   第二点很关键：若 test 里还掺入"随版本变化的额外局"，那些局又会与其它版本的 train 重叠，
   等于把刚修好的洞重新挖开。非冻结局只在 train/val 间按 ratios **相对**分配
   （0.8/0.1 ⇒ 88.89% 进 train）。
3. **审计脚本** `scripts/audit_dataset_leakage.py`：
   - 单数据集内部 train/val/test 两两互斥；
   - 跨版本 3×3 交集矩阵：`(train|val) × test` 命中 = **ERROR**（test 分数无效）；
     `train×val`、`val×val` 命中 = **WARN**（只影响模型选择的可比性）；
   - canonical 守卫：任何数据集 train/val 不得含冻结局；`leak_free` 版本的 test 必须覆盖清单；
   - **兜底**：没有任何数据集携带清单时判 **ERROR"无法判定"**，
     而不是绿灯"未发现泄漏" —— 本次缺陷的根源就是"审计盲区被当成合规"；
   - `--freeze-from <dir>`：冻结某数据集的 test 划分并回写其 `leak_free`（附 `leak_free_source` 说明）；
   - `--exempt`：历史豁免名单，交集数字照打印、只降级 WARN，避免守卫因长期报红而被无视。

### 3.3 反推历史切分（把 p1_v3 变成可审计）

`_reconstruct_legacy_manifests` 用**同一份有效局列表**按 metadata 的 seed/ratios 复算，
且**只有局数与 plies 逐项吻合才落盘**（错清单比没有清单更危险）。
对 `datasets/p1_v3` 的结果：

| split | 局数（反推 / metadata） | plies（反推 / metadata） | 吻合 |
|---|---|---|---|
| train | 760 / 760 | 90645 / 90645 | ✅ |
| val | 95 / 95 | 11360 / 11360 | ✅ |
| test | 96 / 96 | 11811 / 11811 | ✅ |

产出 `datasets/p1_v3/split_files.json`（`verified: true`）。
**副产品**：p1_v3 的诚实基线 —— `bc_best.pt` 在 p1_v3/test 上 Top-1 **24.90%** ——
成为一份受保护的 canonical 分数，不需要重测就能与后续版本对比。

### 3.4 冻结 canonical test 与 p1_v4

- `datasets/canonical_test.json`：**96 局**，与 p1_v3 的 test 划分**逐文件一致**。
- `datasets/p1_v4`（version 4.0.0，`--frozen-test datasets/canonical_test.json`）：

  | split | 局数 | plies |
  |---|---|---|
  | train | 760 | 90645 |
  | val | 95 | 11360 |
  | test | 96 | 11811 |

  `leak_free=True`，冻结命中 96/96、缺失 0；总体本量（951 有效局、113816 plies、
  phase 分布）与 p1_v3 完全一致 —— 只是**重新划分**。已用 `NpzReplayDataset` 验证可加载，
  且 p1_v4/test 与 p1_v3/test 的 Value 类别分布**逐位相同**（2119 / 5144 / 2118）。

### 3.5 审计结论（`reports/dataset_leakage_audit.md`）

```
p1_v1  1.0.0  leak_free=False  清单=无      → 历史不可审计（口径已变，无法反推）
p1_v2  2.0.0  leak_free=False  清单=无      → 历史不可审计
p1_v3  3.0.0  leak_free=True   清单=split_files.json（反推并逐项校验）
p1_v4  4.0.0  leak_free=True   清单=metadata.json.split_files
p1_v3 × p1_v4: test∩test=96（同一 canonical）；train∩train=678；
               train∩val=82（双向）、val∩val=13
               → WARN：test 分数合法，但**模型选择不可比**
训练默认数据集 datasets/p1_v3: leak_free=True ✔
结论：✅ 未发现泄漏（无 (train|val) × test 交集）
```

⚠️ **残余风险（未消除）**：冻结模式只钉住 test，train/val 仍由 seed 洗牌决定
⇒ 不同版本的 `train ∩ val` 仍可能重叠（p1_v3 × p1_v4 实测 82 局）。
彻底消除需要把 train/val 也一起冻结，并把新增语料放进"只允许进 train 的扩展池"。
这是**后续工作**，已写进 CHANGELOG 的未解决问题。

### 3.6 仍然不可比的数字

- `p1_v1` / `p1_v2` 无法反推（p1_v2 只认 835 局有效、把 116 局判为损坏，口径已变），
  已列入 `--exempt`。派生自它们的模型（`models/pool/bc_best.pt` 88.18%、
  `models/pool/distilled.pt`、`bc_best_legacy_36ch_20260906.pt` 52.41%）**任何跨版本对比仍然无效**。
- `reports/eval_bc_best.md`（52.90%/74.61%）、`eval_bc_distilled20k_*.md`（40.82% 等）继续作废。
- `junqi/train_rl.py:1280-1282` 基于泄漏值（0.735/0.288）优先取
  `models/value_distilled_v2.pt` 的热启动决策**本轮未改** —— 仍建议改为按诚实 val 指标选基座。

---

## 4. 复现

```bat
:: ③ 门控（约 7 分钟 / 4 workers）
python -m junqi gate --a hybrid2 --b search2 --seeds 100 --workers 4 ^
    --init-set eval_sets/endgame.jsonl --model-a models/bc_best.pt ^
    --out reports --out-name gate_hybrid2_vs_search2_p2accept

:: ④ 对局质量审计（约 13 分钟，2 配置 × 60 局）
python scripts/audit_p2_game_quality.py --games 60 ^
    --a hybrid2 --b search2 --model-a models/bc_best.pt ^
    --compare-a expert2 --init-set eval_sets/endgame.jsonl --out-dir reports

:: 切分反推（一次性，约 2 分钟；--no-write-npz 不落盘）
python -m junqi export_dataset --sav-dir 军旗复盘 --out-dir scratch/_unused ^
    --frozen-test none --legacy-reconstruct datasets/p1_v3 --no-write-npz

:: 冻结 canonical + 导出 p1_v4
python scripts/audit_dataset_leakage.py --freeze-from datasets/p1_v3
python -m junqi export_dataset --sav-dir 军旗复盘 --out-dir datasets/p1_v4 ^
    --version 4.0.0 --frozen-test datasets/canonical_test.json

:: 审计
python scripts/audit_dataset_leakage.py --exempt p1_v1 p1_v2 --out-dir reports ^
    --out-name dataset_leakage_audit

:: 测试
python -m pytest tests/test_dataset_split_manifest.py -q     :: 17 passed
```

---

## 5. 下一步建议（按优先级）

1. ✅ **已做（见 §7）**：④ 暴露的行为缺陷已修 —— 根因是 `junqi/ai.py:538`
   `HybridAgent.choose_actions` 把 avoid 判定写成 `a in avoid`（`a` 是 `Action` dataclass，
   而 `avoid` 装的是 `position_key()` 产出的局面键）⇒ **恒不命中，混合引擎从不规避重复局面**。
   修复后重复判和 24/60 → **2/60**，④ 转为通过。
2. **补 ②**：在 `eval_sets/{opening,midgame}.jsonl` 上跑同一套审计与门控
   （`--init-set` 换成对应文件即可），确认三个阶段都没有崩溃。
3. **迁移训练默认数据集到 p1_v4**（可选但建议）：需同步改 `junqi/dataset.py::DEFAULT_P1_DIR`、
   `junqi/train_rl.py::ANCHOR_VAL_DIR`、`scripts/audit_artifacts.py::DEFAULT_P1`
   以及断言 p1_v3 的两个测试。p1_v3 当前也是 `leak_free=True`，所以**不迁移也不会泄漏**；
   迁移的收益是"训练划分与 canonical 的关系写死在 metadata 里"。
4. **把 train/val 也冻结**（3.5 的残余风险），并把新语料设计成 train-only 扩展池。
5. **改 `train_rl` 的热启动选择依据**：不要再用泄漏值 0.735/0.288，
   改为读 `value_distilled_v2.pt` 自己记录的诚实 val 指标（现在诚实值 0.3247 < 1/3 随机，
   等于从"策略 24.9% + 价值低于随机"起步）。

## 6. 未解决风险

- ⚠️ **③ 的结论测于修复前**：`gate_hybrid2_vs_search2_p2accept`（200 局、0.5225）
  跑在 `ai.py:538` 修复**之前**，也就是"从不规避重复"的那一版 hybrid2。
  修复改变了混合引擎的走法分布，**严格说 ③ 需要重跑一次门控**（本报告未重跑）。
  修复只让引擎多了一条"别走进历史局面"的约束，方向上不应削弱棋力，但这是推断、不是实测 ——
  **按本工程的纪律，推断不算证据**。
- ④ 只在**残局起始局面**上做了 60 局；开局/中盘尚未测（条款 ② 也还欠着）。
  无头自对弈 vs GUI 的差异已在 2.1 说明。
- 修复后的重复判和率（3.3%）已优于对照 `expert2`（6.7%），但仍未做
  "同一模型开/关重复惩罚"的门控 A/B 来量化它对**胜率**的影响。
- `p1_v1` / `p1_v2` 的切分清单**原理上不可恢复**，其派生模型的跨版本数字永久不可信。
- 冻结 test 的 96 局是**同一批语料**的一部分，因此 canonical 分数只能衡量"在已有语料分布上的
  拟合/泛化"，不能衡量对新棋风的适应能力。

---

## 7. ④ 根因修复与复测（2026-09-17 23:15 追加）

### 7.1 根因

`junqi/ai.py:538`（`HybridAgent.choose_actions`）原为：

```python
if avoid and a in avoid:          # ← 恒为 False
    tactical_score = -WIN_SCORE + 100.0
```

`avoid` 是**局面键集合** —— 由 `junqi/state.py::position_key()` 产出的
`(tuple(obs), turn, seat_color[0], seat_color[1])` 四元嵌套 tuple；
而 `a` 是 `Action` **dataclass**（`kind/frm/to`）。两者类型不同 ⇒ 判定**永远不命中**
⇒ 混合引擎**从不规避重复局面**，这正是 40% 重复判和的直接原因。

同仓其余 5 处消费者写法**都是正确的**（用 `position_key(state.apply(a)) in avoid`）：

| 位置 | 写法 |
|---|---|
| `junqi/ai.py:208-212`（`Agent`） | ✅ `if a.kind == "move" and position_key(...) in avoid` |
| `junqi/ai.py:451`（`ExpertAgent`） | ✅ 同上 |
| `junqi/search.py:1096-1099`（专家搜索根节点） | ✅ `if avoid and a.kind == "move"` |
| `junqi/mcts.py:155-158` | ✅（对全部动作，含翻子） |
| `junqi/hybrid_engine.py:195` | ✅ `if a.kind == "move" and position_key(...) in avoid` |
| **`junqi/ai.py:538`（`HybridAgent`）** | ❌ **`a in avoid`** ← 唯一一处 |

⇒ 这是一个**孤例错误**，不是设计意图。

### 7.2 修复

```python
nxt = state.apply(a) if a.kind == "move" else None
if avoid and nxt is not None and position_key(nxt) in avoid:
    tactical_score = -WIN_SCORE + 100.0
elif a.kind == "move":
    if nxt.is_terminal() and nxt.winner == state.turn:
        ...
```

两点说明：

1. **只对走子判定**（`a.kind == "move"`），与多数同侪一致。这不只是"随大流"——
   翻子会**永久增加公开信息**，而 `position_key` 记录每格的 `(color, rank)` 或 `None`，
   因此翻子后的局面键**不可能与任何历史键重合**，重复判和在数学上不可能由翻子造成。
   加上该限制既正确又省一次 `apply`。
2. **`nxt` 提到循环顶部只算一次**，顺手消掉了原先"avoid 分支不算、常规分支才算"的重复
   `apply`（原实现若修好后仍写在分支内，会在命中路径上多付一次全盘复制）。

### 7.3 回归测试（先写测试、再看它红、再改实现）

`tests/test_hybrid_agent.py::TestHybridAgent::test_hybrid_agent_penalizes_repetition_position`

- 构造"1 个明子（可走）+ 1 个暗子（可翻）"的局面，先用**不传 `avoid`** 的完整打分挑出
  "本来最优的那步走子"，再把它的后继局面键放进 `avoid` 重问一次。
- 断言：该走法仍在打分表里（只是垫底）、**必须排末位**、且分数 `< -1e5`（WIN_SCORE 量级重罚）。
- **自校准设计**：目标由"无 avoid 时的最优走子"动态选出，因此修复前必然失败
  （最优走子不可能自己垫底），不会出现"碰巧不被选中而假通过"。

修复前实测（留证）：

```
tests/test_hybrid_agent.py::...::test_hybrid_agent_penalizes_repetition_position FAILED
E  AssertionError: Action(kind='move', frm=(3, 1), to=(3, 0)) != Action(kind='move', frm=(3, 1), to=(3, 2))
   : 命中 avoid 的走法必须被罚到末位
1 failed, 3 passed
```

修复后：

```
tests/test_hybrid_agent.py ....                                            [100%]
4 passed in 1.64s
```

回归面：`tests/test_hybrid_agent.py` + `tests/test_hybrid_tactical_pricing.py` +
`tests/test_p1_advanced_enhancements.py`（内含 `TestP1DHybridAgentIntegration`）
共 **21 passed**。

### 7.4 复测（同一命令、同一种子基）

```
python scripts/audit_p2_game_quality.py --games 60 --a hybrid2 --b search2 \
    --model-a models/bc_best.pt --compare-a expert2 \
    --init-set eval_sets/endgame.jsonl --seed-base 900000
```

修复前留档：`reports/p2_game_quality_prefix_20260917.{json,md}`
修复后（= 现 `reports/p2_game_quality.{json,md}`，另存带时间戳副本）。

| 指标 | 修复前 | **修复后** | 对照 `expert2` | 判定 |
|---|---|---|---|---|
| 非法动作 / 状态转移 | 0 / 0 | **0 / 0**（分母 5798） | 0 / 0 | ✅ |
| 送旗（每 100 决策） | 1/4165 = 0.024 | **1/5798 = 0.017** | 0 | ✅ |
| 送旗致失旗（局） | 1/60 | **1/60** | 0/60 | ✅ |
| 往复踱步事件段（每 100 决策） | 0.192 | **0.138** | 0.422 | ✅ 优于对照 |
| 龟缩拒翻 | 0 | **0** | 0 | ✅ |
| **重复判和 + 触顶（局占比）** | **24/60 = 40.0%** | **2/60 = 3.3%** | 4/60 = 6.7% | ✅ **从 6 倍劣势变为优于对照** |
| 平均手数 | 125.1 | **152** | 138.6 | — |

终局原因分布（局）：

| 配置 | flag | immobilized | no_capture | repetition | 和棋合计 | 分胜负 |
|---|---|---|---|---|---|---|
| hybrid2（修复前） | 5 | 8 | 23 | **24** | 47 | 13 |
| **hybrid2（修复后）** | 5 | 12 | 41 | **2** | 43 | **17** |
| expert2（对照） | 5 | 9 | 42 | 4 | 46 | 14 |

**总判定：✅ 通过**（7 项判据全部 PASS）。

读法：`repetition` 的 22 局大多转成了 `no_capture`（23→41），
即"多子绕圈"被消除后，对局改由既有的 70 手无吃子规则收尾；
同时 `immobilized` 8→12、分胜负局 13→17，**决定性略有提升**。
"平均手数 125→152"与"分胜负局增多"方向一致，说明引擎不再提前把局面走死。

### 7.5 改动文件

| 文件 | 改动 |
|---|---|
| `junqi/ai.py` | `HybridAgent.choose_actions` 第 535-545 行：avoid 判定改用 `position_key(nxt)`，限定走子，`nxt` 提取到循环顶部 |
| `tests/test_hybrid_agent.py` | 新增 `test_hybrid_agent_penalizes_repetition_position`；导入补 `position_key` |
| `reports/p2_game_quality.{json,md}` | 更新为修复后复测结果 |
| `reports/p2_game_quality_prefix_20260917.{json,md}` | 修复前留档（新增） |
| `reports/p2_game_quality_hybrid2_vs_search2_20260917_225838.{json,md}` | 本轮原始输出（新增） |

