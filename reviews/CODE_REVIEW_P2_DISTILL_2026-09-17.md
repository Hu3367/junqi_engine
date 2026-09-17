# 蒸馏 / 行为克隆代码审查报告

> **审查日期**：2026-09-17
> **所属阶段**：`P2`（行为克隆 + 搜索蒸馏），依据 `AGENTS.md` 与 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`
> **审查范围**（只读，未修改任何生产代码）：
> `junqi/train_bc.py`、`junqi/eval_bc.py`、`junqi/train_search_distill.py`、
> `junqi/train_value_distill.py`，及其依赖 `junqi/dataset.py`、`junqi/net.py`、
> `junqi/ai.py`、`junqi/search.py`、`junqi/__main__.py` 的相关路径。
> **审查方法**：逐行阅读 + 3 个可复现探针实测（拒绝"读代码推断"下结论）。

---

## 0. 结论摘要

代码整体结构清晰、注释里保留了充分的历史事故背景，合规边界（公共状态、不覆盖 `best.pt`）
基本守住。但发现 **3 项 A 级问题（影响训练正确性 / 与自身承诺不符）**、3 项 B 级（合规与一致性）、
5 项 C 级（死代码 / 可读性），其中 **A1、A3、B3 会直接影响已产出的模型与蒸馏的可复现性**。

| 编号 | 级别 | 问题 | 影响面 |
|---|---|---|---|
| A1 | 高 | `train_value_head_only` 用 `net.train()`，冻结模块的 BatchNorm 统计被改写 | 已污染 `models/best.pt`（`value_distilled_v2.pt`）的产出过程 |
| A2 | 高 | 教师 `degraded`（限时降级）分数被静默当成精确分 | 蒸馏/价值伪标签被系统性偏置 |
| A3 | 高 | 教师打标不可复现：`--workers` 语义不同，且 `--seed` 是空转的 | 蒸馏结果无法复现、无法对比 |
| B1 | 中 | `train_bc` / `eval_bc` 完全绕过数据集版本守卫 | 可静默在口径不同的旧数据集上训练 |
| B2 | 中 | W/D/L 码表散落 4 份，其中 2 份是不可达死代码 | 违反项目"唯一真源"硬约束 |
| B3 | 中 | `train_value_distill` 两条数据分支的 MSE 目标量纲不一致（差 0.238） | 系统性压低价值头置信度 |
| C1–C5 | 低 | 死代码 / 空转统计 / 未用导入 / 静默覆写对手池 / 缩进 | 可维护性 |

**一个重要前提**：`docs` 与长期记忆中已定论 —— P2 搜索蒸馏**对局强度无稳健提升，路线已证伪**
（门控 0.46~0.57，CI ±0.12）。因此本报告**不主张**修完这些问题就能让蒸馏变强；
修它们的价值在于（a）A1 影响已发布权重，(b) 重跑蒸馏时结果必须可信、可复现。

---

## A 级：影响训练正确性

### A1. `train_value_head_only` 的"冻结"承诺不成立：BatchNorm 统计被改写

**位置**：`junqi/train_value_distill.py:467`

```python
for name, p in net.named_parameters():
    saved_flags[name] = p.requires_grad
    p.requires_grad = name.startswith("value_head")   # 只冻 requires_grad
...
for epoch in range(1, epochs + 1):
    net.train()          # ← 问题：整网进入 training 模式
```

**根因**：`requires_grad=False` 只阻止**权重**更新，**不阻止 BatchNorm 更新
`running_mean` / `running_var`** —— 只要 `module.training is True`，BN 在前向时就会更新
运行统计（与梯度无关）。而 `forward()` 会同时跑 `in_conv`、全部 `blocks`、`policy_head`
和 `value_head`，因此主干与策略头的 BN 统计全部被改写。

**实测证据**（`scratch/probe_freeze_bn.py`，单 epoch，故意喂分布不同的批次以放大效应）：

```
== BatchNorm running_mean 是否被改动 ==
  in_conv.1.running_mean            max|Δ|=9.660e-01  改动
  blocks.0.bn1.running_mean         max|Δ|=1.472e-01  改动
  blocks.0.bn2.running_mean         max|Δ|=1.531e-01  改动
  policy_head.1.running_mean        max|Δ|=4.552e-01  改动
  value_head.1.running_mean         max|Δ|=2.266e-01  改动
== 冻结的策略头权重 ==
  policy_head.4.weight max|Δ| = 0.000000e+00      ← 权重确实没变
```

**危害**：docstring 承诺"冻结主干与策略头，BC 策略能力不受影响"。权重确实没动，
但**策略头的输出分布会随 BN 统计漂移** —— 这恰恰是承诺要避免的事。
`train_value_from_p1_dataset` 复用本函数，因此当前发布模型 `models/best.pt`
（= `value_distilled_v2.pt`）的产出过程带有这个偏差。

**对照**：同一文件的 `train_value_distill`（第 246 行）写法是**正确**的：
```python
net.eval()
net.value_head.train()      # 只让 value_head 处于 train 模式
```
两条路径不一致 —— 说明这是遗漏而非有意设计。

**说明**：探针刻意用了 OOD 批次放大效应；真实数据下幅度更小，但方向一致且非零。

**建议修法**（改动 2 行）：
```python
net.eval()
net.value_head.train()
```
并把 `tests/test_s2_fixes.py:191` 的断言从"只比 `in_conv.0.weight` / `policy_head.4.weight`"
扩展为**同时断言所有 BN 的 `running_mean` / `running_var` 不变**（现在没有任何测试守卫这一点，
全仓 grep `running_mean` 只命中 `net.py` 的定义处）。

---

### A2. 教师 `degraded` 分数被静默采信，软分布被系统性偏置

**位置**：`train_search_distill.py:58-94`（`teacher_soft_targets`）、
`train_value_distill.py:109-131`（`label_with_expert`）

**问题链**：

1. `junqi/search.py:985` 的 docstring 明确要求：
   > `time_limit_ms > 0` 时应先查 `stats.degraded` …… `stats.root_scores` 可能不覆盖全部合法动作

2. `junqi/search.py:1168-1170` 的代码注释也明确写了：
   > 消费方（GUI top-N、**train_search_distill 的 softmax**）应结合 `stats.degraded` 判断

3. **但 `ExpertAgent.choose_actions` 只返回 `[(action, score)]`，把 `stats` 丢掉了**
   （`junqi/ai.py:304-337`），调用方**拿不到** `degraded`。

4. 于是 `teacher_soft_targets` 直接对 `root_scores` 做全量 softmax 并**重归一化到 1**。
   若根循环层内被时限打断，`root_scores` 是**截断集合** —— 重归一化会把概率质量
   集中到"恰好被搜到的那批动作"，未搜到的动作概率被抹成 0。这不是"教师偏好"，
   是**预算的产物**。

同样的失效发生在价值链路：`label_with_expert` 取 `scored[0][1]`，
`degraded` 时该分是真实值的**下界**，却被 `score_to_class` 当作精确分判胜/和/负。

**讽刺的是**，仓库里已经有探测工具 `scratch/probe_distill_teacher.py:41-48`
**专门统计 degraded 比例**（注释还写着"C++ 移植后教师能在 300ms 内搜完 depth 3"），
但这个比例从未回流到训练管线，也没进日志或 checkpoint。

**建议修法**：
- `ExpertAgent.choose_actions` 增加 `return_stats: bool = False` 或暴露 `agent.engine.stats`；
- 教师打标处累计 degraded 计数并 **打印到日志 + 写入 checkpoint 元数据**
  （`samples_degraded`），至少让"这次蒸馏的教师有多少局面是不可信的"可被审计；
- 可选：degraded 局面直接丢弃或降权（与既有的 `tac_min_spread` 权重机制天然兼容）。

---

### A3. 教师打标不可复现：`--workers` 改变语义，`--seed` 完全空转

这是两个独立缺陷的叠加，合起来使**蒸馏结果无法复现、也无法比较**。

#### A3-1. serial 与 mp 路径的置换表生命周期不同

`train_search_distill.py:197-232`：

```python
if workers and workers > 1 and len(states) > 10:
    # mp 路径：每个 task 新建 ExpertAgent（→ 全新 TT）
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_teacher_label_worker, tasks)
else:
    # serial 路径：**复用一个 agent 跑完所有局面**
    agent = ExpertAgent(SearchConfig(...), seed=seed)
    for i, st in enumerate(states):
        scored = agent.choose_actions(st, topn=...)
```

而 `junqi/search.py:1000` 明确写着：

> 切片 3：C++ 子树每轮搜索重置统计与停止位（**置换表跨轮保留**，与 Python 一致）

`tt_size_power=18` ⇒ 262,144 条目。1200 个局面 × depth-3 搜索远超出这个容量，
必然发生淘汰。因此：

- **serial**：第 k 个局面的教师分取决于**前 k−1 个局面**填充/淘汰 TT 的方式；
- **mp**：每个局面都是冷 TT。

同一 `--seed`、同一 `--states`，`--workers 0` 与 `--workers 8` 会得到**不同的教师软分布**。
`train_value_distill.label_with_expert` 结构完全同构（`train_value_distill.py:113-131`）。

#### A3-2. `seed` 对搜索结果没有任何影响

`seed + i` 看似是为并行补偿随机性，但：

- `junqi/search.py:161` 的 `self.rng = random.Random(seed)` —— **全仓 grep 确认只此一处赋值，从未被读取**；
- `junqi/ai.py:298` 的 `ExpertAgent.rng` —— 同样未被读取。

即专家搜索是**完全确定性**的，`seed` 参数在 `ExpertSearchEngine` / `ExpertAgent` 上是死参数。
它给了调用方"可以用 seed 复现"的错觉，而这层保障并不存在。

**建议修法**（二选一，需明确写进 docstring）：
- **逐局面冷启动**：打标前调用 `engine.tt.clear()`，让 serial 与 mp 语义一致
  （这也与长期记忆中"单节点语义等价性验证＝每个子节点用全新引擎"的口径一致）；或
- 让 serial 路径也每局面新建 agent。
- 另：要么让 `ExpertSearchEngine` 真正使用 seed，要么删掉该参数并在 `--seed` 的 help 里
  说明它只影响**局面采样**（`random.Random(seed)` 用于 `gen_distill_positions`）不影响教师打分。

---

## B 级：合规与一致性

### B1. `train_bc` / `eval_bc` 完全绕过数据集版本守卫

- `train_value_distill.load_p1_arrays`（第 341-359 行）会读同目录 `metadata.json` 并
  调 `check_p1_version`，低于 `3.0.0` **直接拒载**，理由写在 docstring 里：
  p1_v1/v2 生成于 2026-09-06，早于 code 24 标签口径修正，"同一份指标、两套真值"。
- 但 `train_bc.py:147-148` 与 `eval_bc.py:33` 走的是 `NpzReplayDataset`，
  它在 `dataset.py:59` 直接 `np.load(npz_path)` —— **没有任何版本校验**。

⇒ 可以静默在 p1_v1/v2 上训练/评测"BC 基线"，产出与 P1 口径不一致的指标。

**grep 确认**：`check_p1_version` 全仓只有 `load_p1_arrays` 一个调用点。

**现有测试覆盖不到**：`tests/test_p1_dataset_label_consistency.py:203-206` 只断言源码里
**没有硬编码** `p1_v1`/`p1_v2` 字符串，不检查守卫是否生效。

**建议修法**：把版本校验下移到 `NpzReplayDataset.__init__`（读同目录 `metadata.json`，
可加 `check_version: bool = True` 参数），让所有 npz 消费者共享同一道门。

---

### B2. W/D/L 码表散落 4 份，其中 2 份是不可达死代码

项目硬约束（`AGENTS.md` / 计划 §6 / 长期记忆）：

> `dataset.terminal_label_from_meta` 是唯一真源，统计走 `outcome_bucket_from_meta`，
> **禁止各写一份码表**

现状：

| # | 位置 | 内容 |
|---|---|---|
| 1 | `dataset.py:69-80` | `NpzReplayDataset` 无 `val_classes` 时由 `values` ±0.5 推 0/1/2 |
| 2 | `train_bc.py:68-76` | `evaluate_model` 里 `pred_val.shape[-1]==3 and val_classes is None` → 现推码表 |
| 3 | `train_bc.py:203-212` | `train_bc` 主循环里同一段逻辑再写一遍 |
| 4 | `eval_bc.py:78-85` | `evaluate_test_set` 里第三遍 |

**而且 #2/#3/#4 是不可达的**：`NpzReplayDataset.__getitem__` **恒定返回 7 元组**，
`default_collate` 后 `len(batch_data) == 7` 恒为真 ⇒ `train_bc.py:181-186`、
`eval_bc.py:54-59` 的 `else` 分支（含 `val_classes is None` 的 fallback）永不执行。

**另一个祸首是分派条件本身**：用 `len(batch_data) == 7` 来判断数据形态是脆弱的隐式契约
（数据集有 8 个字段时就会静默走错分支），而 `NpzReplayDataset` 明明有 `val_classes` 属性。

**建议修法**：删掉 #2/#3/#4 三处回退 + `if len(batch_data) == 7` 分派，直接解包 7 元组；
若真要保留旧数据兼容，应集中在 `NpzReplayDataset` 一处。

---

### B3. `train_value_distill` 两条数据分支的 MSE 目标量纲不一致

`train_value_distill.py` 有两条数据来源，回归目标算法不同：

**预打标数据分支**（第 171-174 行）用真实 `expert_score`：
```
vs = tanh(expert_score / SCORE_SCALE)
```
判胜时 `expert_score >= WIN_SCORE/2`，而 `WIN_SCORE = 1_000_000`（`state.py:16`）⇒
`tanh(500000/600) = 1.0000`

**采样分支**（第 196-198 行）写死常数：
```python
samples = [..., (600.0 if z == 0 else (-600.0 if z == 2 else 0.0)), ...]
```
⇒ `tanh(600/600) = 0.7616`

**实测**（`scratch/probe_freeze_bn.py` 末段）：
```
WIN_SCORE=1000000, SCORE_SCALE=600.0
采样分支 tanh(600/600)   = 0.7616
预打标分支 tanh(500000/600) = 1.0000
两条分支的 MSE 目标相差 0.2384
```

**危害**：预测值 `pred_v = P(win) − P(loss)` 的值域是 `[-1, 1]`。
若模型对必胜局面正确给出 1.0，采样分支会以每样本 `(1−0.7616)² = 0.0568` 的 MSE
把它**往 0.7616 拉** ⇒ 系统性训练出**过于保守**的价值头。
而分类项 `F.cross_entropy(v_logits, ys)` 的监督又是"这是必胜" ⇒ 两项目标互相拉扯。

`600.0` 看起来是误把 `SCORE_SCALE`（tanh 压缩尺度）当成了"分数本身"。

**建议修法**：把写死值改为语义正确的目标（`±1.0` 直接给回归目标，或
`math.tanh(WIN_SCORE / SCORE_SCALE)`），与数据分支对齐；
并在 `tests/test_value_reanchor.py` 加一条"回归目标值域"断言。

---

## C 级：死代码 / 可读性 / 性能

### C1. 死代码与空转统计

- **`train_bc.py`**：`train_p_loss`(L172)、`train_v_loss`(L173)、`v_samples`(L176)
  只有累加、**从不读取**。`t_loss` 仅由 `train_loss` 得出。
  后果：**训练期的 Value loss 完全不可观测**（日志与返回值里都没有），
  与 `train_value_distill` 每轮打印 `val_acc/val_mse` 的透明度不一致。
- **`train_search_distill.py`**：`teacher_soft_targets` 返回元组的第 2 项 `top1`
  被写进 `labels[i][1]` 后**从未被消费** —— `_batch` 只用 `labels[i][0]`；
  日志里的 `val_teacher_top1` 是 `student.argmax` vs `ts.argmax(-1)` 另算的。
  因此第 93-94 行那段 `top1` 推导（含 `targets.sum() > 0` 三元）整段可删。
- **同类小问题**：`teacher_soft_targets` 里"无根节点评分 → 均匀回退到合法动作"
  （L78-82）实际上**恒无效**：该情形下 `teacher_confidence` 返回权重 0，
  均匀目标乘 0 权重没有任何梯度贡献。

### C2. 未使用的导入（AST 扫描，`scratch/probe_distill_code_audit.py`）

```
junqi/train_bc.py:19              import torch.nn as nn
junqi/eval_bc.py:8                import json
junqi/eval_bc.py:12               import numpy as np
junqi/train_search_distill.py:21  import math
junqi/train_search_distill.py:25  from collections import Counter
```

### C3. `train_value_distill` 静默覆写对手池权重

`train_value_distill.py:303-308`：只要 `models/pool/` 目录存在就
`shutil.copyfile(out_path, models/pool/value_distilled.pt)`。

这会**不经任何门控**地改动自博弈对手池里的权重，而函数 docstring 与 CLI help
**完全没提**这个副作用。若对手池构成评测/训练基线，等于悄悄换了基线。

**建议**：改为显式开关（如 `--sync-pool`，默认关），或至少打印醒目警告。

### C4. `_batch` 每批重算 `legal_action_mask`（**已实测：不值得优化**）

`train_search_distill.py:247-254` 的 `_batch` 每个 epoch 每个 batch 都对
`states[i]` 重算一次 `legal_action_mask`。直觉上"掩码与局面绑定，应该预计算"。

**但实测结果否定了这个直觉**：
```
重算 360 次掩码耗时 0.01s（单次 0.040ms）
折算到 1200 局面 × 6 epoch：~0.3s
```
相对 300ms × 1200 的教师打标（≈6 分钟起）可**忽略不计**。

**记录在此的目的恰恰是阻止**后来者把它当优化点 —— 与本项目"读代码推断会得出相反结论"
的既有教训一致。

### C5. 其他不一致（低优先级）

- **缩进**：`train_search_distill.py:247-254` `_batch` 的函数体多出 8 个空格
  （历史 `Edit` 工具的模糊匹配产物），建议修正。
- **尾批**：`train_value_head_only` 用 `range(0, len−bs+1, bs)` **丢尾批**，
  而 `train_value_distill` 用 `range(0, len, bs)` **保留尾批** —— 两者不一致（影响很小）。
- **checkpoint 字段**：`train_bc` 保存 `optimizer_state`，`train_search_distill` 不保存；
  模型选择指标分别是 `val_top1` 与 `val_kl`。各自合理，但建议在文档里统一说明。

---

## 建议行动顺序

1. **A1（2 行改动 + 1 个测试断言）** —— 成本最低、直接影响已发布权重链路，优先。
2. **A3-2（删死参数或让其生效）** —— 成本低，消除"seed 可复现"的假象。
3. **A3-1 / A2（教师打标冷启动 + degraded 审计）** —— 若**不打算**再跑蒸馏，优先级可降；
   若打算重跑，必须先修，否则产出不可信。
4. **B3（目标量纲对齐）** —— 影响价值头置信度；与 A1 同在 `train_value_distill`，可一并修。
5. **B1 / B2（守卫下移 + 删死码表）** —— 合规性，改动面稍大，建议单独一个切片。
6. **C1–C5** —— 随手清理。

---

## 附：本次审查使用的探针（可复跑）

| 脚本 | 验证内容 |
|---|---|
| `scratch/probe_freeze_bn.py` | A1（BN 统计是否被改写）、B3（目标标尺差异） |
| `scratch/probe_distill_code_audit.py` | C1（死数据）、C2（未用导入）、C4（掩码重算成本） |
| `scratch/probe_distill_teacher.py`（已有） | A2 的 degraded 比例前置探测 |

复跑方式（**不要用 PATH 上的 `python`**）：
```
E:\Local code\军棋\venv_junqi_engine\Scripts\python.exe scratch\probe_freeze_bn.py
```

---

## 未做 / 未验证

- **未修改任何生产代码**（本次为只读审查）。
- 未评估 A1 对 `models/best.pt` 实际棋力的影响幅度。可做的量化实验：
  取 `models/bc_best.pt`（A1 之前的干净产品），复制两份跑 `train_value_from_p1_dataset`
  （一份修好、一份保持现状），在 `datasets/p1_v3/test.npz` 上比 `policy top1/CE`
  与 `evaluate_value_health` 的平衡准确率。**但这需要 `gate` 级对局验证，
  单看指标不可下结论**（长期记忆：指标可比性 + 多重比较纪律）。
- 未审查 P3/P4 自博弈与门控链路（不在本次范围）。
- A2 的 degraded 实际比例**未在本机复测**（已有工具可跑，但那是一次真实的
  300ms × N 次搜索，耗时约数分钟，且会写 CPU 负载）—— 建议在决定是否修 A2 前先跑
  `probe_distill_teacher.py 20 3 300` 拿到当次环境的真实比例。
