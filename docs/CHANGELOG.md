# CHANGELOG

## [2026-09-15] 第七批 — 清理作废产物 + 修复后首次端到端跑通（含两处新修复）

阶段归属：**P0（正确性验证）+ P3（运营清理）**。用户决策：清理三个 `*_buffer.pkl`、
用健康基座直接覆盖 `best.pt`、按既定顺序执行重训/蒸馏/门控。

### 一、清理与基线重置

- 删除三个作废的自对弈经验池（R2 使 25% 对手为随机网络 + C6 改了 MCTS 重复阈值语义），
  释放 **11.18 GB**：`models/`(3835 MB)、`models_b3/`(3822 MB)、`models_b3opp/`(3792 MB)。
  `models/evidence_collapsed_20260914/`(3.58 GB) 属另一份证据归档，未在授权范围，保留。
- **`models/best.pt` 已用 `models/value_distilled_v2.pt` 覆盖**
  （md5 由 `73164845…` → `26675f331…`，与 v2 一致）。原 best.pt 的**权重**与
  `models/pool/bc_best.pt` 逐位相同（文件 md5 不同仅因 pickle 元信息），因此旧内容天然保留。

### 二、修复后首次端到端验证（2 轮 × 60 局，sims=10，8 workers，RTX 4080 SUPER）

全链路跑通，且**五项判据全部通过**：

| 判据 | 结果 |
|---|---|
| R1 候选权重是否真被更新 | ✅ 与热启动源 **106/106 键全部不同**（最大差 3.15e3）——修复前这里恒为 0 |
| best.pt 是否被轮内门控改动 | ✅ 未改动（md5 不变） |
| 每轮 Value 重锚是否执行 | ✅ 两轮均 `reanchor=OK`（此前的 `NameError` 会让它直接崩） |
| 新候选 Value 头是否塌缩 | ✅ 未塌缩，p1_v3/test 平衡 acc **0.619**（线 0.45） |
| Elo 口径 | ✅ 由得分率按标准公式换算（1500→1471→1456.5） |

其他观测（均为修复生效的正面证据）：
- 对手配比真实生效：`best 0.33~0.35 / mirror 0.533 / expert 0.05~0.067 / greedy 0.033~0.05 /
  random 0.017~0.033` —— 修复前 `best` 分支拿到的是随机网络；
- 门控为**配对同牌**（每阶段 12 局 = 6 seeds × 先后手），Elo 每轮更新；
- 正式门控单独复跑（`gate --seeds 12 --workers 8`，24 局）：
  得分率 0.5833、Wilson [0.388, 0.755]、SPRT `continue`、
  **座位拆分 as_first 0.5833 = as_second 0.5833**（配对同牌消掉先后手差异的直接证据）、
  `promote=False` → 未触碰 `best.pt`（晋级出口的"未通过不动"语义正确）；
- 搜索蒸馏冒烟（80 局面 / depth 2 / 2 epochs）跑通并保存候选，未触碰 best.pt。

**需要关注但不属本轮修复的问题**：自对弈终局分布异常偏向 `immobilized`
（71.7% / 53.3%），`flag` 仅 1.7~5%，与人类复盘的终局分布（认输 45.6% / 协议和棋 26.5%）
差异很大；候选对 `search2` 参考得分仅 0.083~0.167，远弱于传统搜索——这与基线计划
§4.1 的判断一致（达到 search2 以上水平要靠搜索蒸馏 + 混合引擎，而非纯 RL）。

### 三、验证过程中发现并修复的两个新问题

1. **`JunqiNet.load_from_file` 不认检查点格式 → 静默随机网络**（与 R2 同族）
   `save_checkpoint()` 写出的是 `{"net":..., "optimizer":...}`，而 `load_from_file`
   只判 `"model_state"` 与"裸 dict"：传入 `models/candidate_latest.pt` 会把整个检查点
   当 state_dict，`strict=False` **键名无一匹配、静默载入零个权重**。
   评测/策略构造路径一旦踩到就会得到"看似正常、实为随机"的模型。
   现已识别检查点分支；并在"超过半数键未载入"时打印明确告警。
   另：`save_checkpoint` 补写 `in_channels / num_blocks / channels`，让检查点自描述
   （否则非默认主干会因形状不符而 RuntimeError）。架构不符时保持**大声报错**。
2. **`--buffer-save-every` 缺"从不落盘"档位**（P1 修复的可用性缺口）
   冒烟跑（2 轮）在"最后一轮恒写"规则下重新生成了 **2.37 GB** 的经验池，对验证毫无价值。
   现将语义定为：`<0` = 从不写（冒烟/验证用）、`0` = 仅最后一轮、`1` = 每轮、
   `>1` = 每 N 轮且末轮恒写、`None` = 每轮。本次冒烟产生的池已删除。

### 四、测试

新增 `tests/test_p0_load_checkpoint_format.py`（6 项）；`test_p3_buffer_persistence.py`
的节流判据按新语义更新。全量：**441 passed / 3 skipped → 449 passed / 3 skipped**。

---

## [2026-09-15] 第六批 — 修复自引入的 NameError + 新增产物核查工具

阶段归属：**P0（正确性）**。起因：回答"修复后是否需要重训"时做的产物核查，
过程中发现并修掉了一个**上一批自己引入的真 bug**。

### 一、修复：`train_value_distill` 缺模块级 `import json`（回归）

第四批给 `load_p1_arrays` 加了数据集版本守卫 `check_p1_version`，其内部使用
`json.load`，但该模块**只在某个函数内部** `import json`，模块级没有。
后果：`load_p1_arrays("datasets/p1_v3", ...)` 直接 `NameError` —— 而这正是
`train_rl` 每轮 Value 重锚（`reanchor_value_head`）与健康探针（`probe_value_health`）
的必经路径；`NameError` 又不在 `except (RuntimeError, FileNotFoundError)` 内，
会**让训练直接崩**。

- 已补模块级 `import json`；
- 旧测试漏检原因：只用了"临时目录 + 无 metadata"的 npz，恰好绕过版本守卫分支。
  现已新增三项守卫：真实 `datasets/p1_v3` 可加载、真实 metadata 版本校验通过、
  以及**静态扫描（精确到函数作用域）**"某处 `X.attr` 用法所在函数未 import X"，
  防止同类问题再生（该扫描确认 `junqi/` 现已无真实残留）。

### 二、新增 `scripts/audit_artifacts.py`：训练产物有效性核查（只读）

回答"重训前手上这些产物还有多少可信"。三部分：
A. 权重两两比对（自动区分 `save()` 包装 / 完整 checkpoint / 裸 state_dict）；
B. Value 头行为探针（p1_v3/test 平衡准确率、MAE、预测分布、塌缩告警）；
C. `elo_history.jsonl` 逐轮门控与晋升记录。

**核查发现（关键）**：

| 产物 | Value 平衡acc | MAE | 预测分布(W/D/L) | 结论 |
|---|---|---|---|---|
| `best.pt` | **0.330**（≈随机 0.333） | 0.871 | 4799 / **0** / 4582 | 退化：Draw 恒为 0 |
| `value_distilled_v2.pt` | **0.735** | **0.288** | 2339 / 4731 / 2311 | 健康，可用基座 |
| `candidate_latest.pt`(ep5) | 0.333 | 0.475 | **0 / 0 / 9381** | **完全塌缩**（100% 预测 Loss） |
| `_candidate_gate.pt` | 0.639 | 0.408 | 2249 / 4753 / 2379 | 相对健康 |

- **`models/best.pt` 与 `models/pool/bc_best.pt` 逐位相同** → 发布模型**从未被训练或
  晋升更新过**，只是 BC 基线副本（无 aux_head、96 键），而 BC 价值头本就未校准。
  与 `elo_history.jsonl` 中**每一轮都是 `promoted=False`** 完全吻合。
- 该工具的比对逻辑有单元测试覆盖（`tests/test_p3_audit_artifacts.py`，12 项），
  含"只用一侧存在的键必须判为不同"与"形状不符应单列"这两个曾导致误判的场景。

### 三、测试

新增 `tests/test_p3_audit_artifacts.py`（12 项）、`test_p1_dataset_label_consistency.py`
增 3 项。全量：**426 passed / 3 skipped → 441 passed / 3 skipped**。

> 说明：本批**未改动任何训练语义**，也未触碰 `models/` 下的产物文件（纯只读核查）。

---

## [2026-09-15] 第五批 — 激活路径易错修复 + PowerShell 编码陷阱

阶段归属：**P0（可运行性）**。承接第四批的启动链路修复。

### 一、用户实测报错与诊断

```console
(venv) PS E:\Local code\军棋\junqi_engine> .\venv_junqi_engine\Scripts\Activate.ps1
无法将".\venv_junqi_engine\Scripts\Activate.ps1"项识别为 cmdlet...
```

两处问题：

1. **路径少一个点**：虚拟环境在工程**上一级**（`..\venv_junqi_engine`），不是工程内；
   `.\` 指当前目录，故找不到。正确写法 `..\venv_junqi_engine\Scripts\Activate.ps1`。
2. **会话里残留着旧激活状态**（提示符显示 `(venv)`）：该会话此前激活的是**已被移走的**
   `junqi_engine\venv`，其 `Scripts` 目录已不存在。于是 PATH 首项指向死路径，
   `python` 静默落到别的解释器上——这正是第四批那个 `ModuleNotFoundError: torch` 的来源。
   仅"重新激活正确路径"还不够，必须先 `deactivate` 或重开终端。

### 二、修复

1. **新增 `activate_env.ps1` / `activate_env.cmd`**（工程根目录）：自动定位虚拟环境
   （工程内 `venv` → 同级 `..\venv_junqi_engine`），供用户直接
   `.\activate_env.ps1` 调用，彻底不必记相对层级；激活后追加 `import torch` 依赖自检。
2. **残留激活自动清理**：两个脚本都会检测 `VIRTUAL_ENV` 指向的解释器是否还存在，
   不存在则从 PATH 中剔除该死路径并清空 `VIRTUAL_ENV`，并提示重开窗口。
3. **修复 `activate_env.ps1` 的解释器路径拼接错误**：原写成 `<venv>\python.exe`，
   导致依赖自检被静默跳过；正确为 `<venv>\Scripts\python.exe`。
4. **README** 环境准备章节改写：给出方式 A（`activate_env.ps1`）与方式 B（`..\` 手动），
   并把"路径少一个点"和"残留 `(venv)` 激活"两个高频报错连同处置方法写进提示框。

### 三、PowerShell 编码陷阱（顺带修掉一个真 bug）

`activate_env.ps1` 首次实测直接报
`ParseException: 语句块或类型定义中缺少右"}"` —— Power**Shell 5.1 在没有 BOM 时
按本地编码（GBK）读取脚本**，中文注释被解成乱码，随即产生"括号不匹配"这类
看起来毫不相干的语法错误。

- 已给 `activate_env.ps1`、`scripts/run_test.ps1` 补上 **UTF-8 BOM**；
- 新增守卫：仓库内任何含非 ASCII 的 `.ps1` 必须带 UTF-8 BOM；
- 新增守卫：用 **PowerShell 自身的 Parser** 校验这两个脚本语法，错误数必须为 0
  （该测试在本轮实际捕获到了上述 ParseException）。

### 四、测试

`tests/test_p0_env_and_launchers.py` 由 8 项扩充至 **15 项**：新增
activate_env 双布局定位、解释器路径必须拼到 `Scripts\python.exe`（回归上述拼接 bug）、
非 ASCII `.ps1` 必须有 BOM、PowerShell Parser 语法校验、README 需提及 activate_env。
全量：**419 passed / 3 skipped → 426 passed / 3 skipped**。

---

## [2026-09-15] 第四批 — 启动链路修复：`ModuleNotFoundError: No module named 'torch'`

阶段归属：**P0（可运行性）**。这是第三批"虚拟环境移出工程目录"的直接后遗症修复。

### 一、故障诊断（用户实测）

```console
$ python -m junqi gui
  File "junqi\__init__.py", line 19, in <module>
    from .hybrid_engine import HybridDecisionEngine
  File "junqi\hybrid_engine.py", line 28, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**不是误删启动文件**：`junqi/__init__.py`、`junqi/__main__.py`、`cli.py`、`pytest.ini`、
`run_tests.bat` 均完好（包级导入已成功走到 `hybrid_engine`）。

真实原因：虚拟环境在第三批被移到同级 `../venv_junqi_engine/`，而这次调用用的是
PATH 上的系统 python（本机为 WorkBuddy 自带的 Python 3.13，**不含 torch**）。
`junqi/__init__.py` 会经 `ai` / `hybrid_engine` 连带 `import torch`，于是在包导入阶段就失败。
用迁移后的解释器直接跑 `pytest tests/` 全程通过（411 passed），可反证代码无损。

### 二、修复

1. **包级诊断增强**（`junqi/__init__.py`）：把"第三方依赖缺失"转成可操作提示
   （回显当前解释器 + 正确的虚拟环境路径 + `run.bat` 入口 + README 指引），
   并用 `raise ... from exc` 保留原始异常链。白名单 `_THIRD_PARTY_DEPS` 之外的
   `ModuleNotFoundError`（即项目自身的导入错误）**原样抛出**，不掩盖真问题。
2. **新增统一启动器 `run.bat`**：自动定位解释器（工程内 `venv\` → 同级
   `..\venv_junqi_engine\` → 报错并给出创建命令），支持
   `run.bat gui` / `run.bat train_rl ...` / `run.bat test`。这是避免该故障最省事的入口。
3. **`scripts/run_test.ps1` 不再裸调 `python`**：改为先定位虚拟环境并校验 torch 可用。
4. **文档路径收口**：`docs/01-GettingStarted/FINAL_REFACTORING_SUMMARY.md` 3 处旧 venv
   绝对路径改为新位置；README 环境准备/快速开始/CLI/目录结构同步 `run.bat`，
   并修正 `junqi/expert/` 的旧说明（改为"已彻底删除 + 新文档路径"）；
   "下一步行动"里失效的 `python cli.py gui` 改为 `.\run.bat gui`。

### 三、测试

新增 `tests/test_p0_env_and_launchers.py`（8 项），其中一项**直接复现用户故障**：
用一个"缺 torch 的解释器"跑 `python -m junqi --help`，断言输出必须是可操作提示
（含"缺少第三方依赖"、`venv_junqi_engine`、`run.bat`、当前解释器），而非裸报错；
另含 run.bat/run_tests.bat 双布局定位、run_test.ps1 不裸调 python、
以及"仓库内不得残留指向工程内 venv 的可执行路径"（历史记录类文档白名单豁免）。

全量：**411 passed / 3 skipped → 419 passed / 3 skipped**。

---

## [2026-09-15] 第三批 — 僵尸包彻底移除 + 虚拟环境移出工程 + 归档脚本标注

阶段归属：**P0/P4 收尾（结构性清理）**。依据用户对审查报告"未处置项"的三项决策。

### 一、`junqi/expert/` 彻底删除（原 R4 的升级处置）

上一批选择"标记废弃 + 守卫测试"，本批按决策**彻底删除**：

- 删除 `junqi/expert/` 全部 11 个模块（`__init__ / expert_engine / tactical_analyzer /
  threat_detection / search_optimizer / mobility_calculator / conditional_value /
  hidden_piece_belief / tempo_tracker / rule_validator / move_adapter`）；
- 连带删除 `scripts/test_expert_core.py`（367 行，唯一 `import junqi.expert` 的脚本）；
- 原包内 `DEPRECATED.md` 的内容迁出并改写为
  [`docs/06-References/DEPRECATED_EXPERT_PACKAGE.md`](06-References/DEPRECATED_EXPERT_PACKAGE.md)：
  记录删除原因、被引用的不存在成员清单、**git 恢复命令**、以及将来复活的前置 gates；
- 守卫测试由 `tests/test_p0_expert_deprecated.py` 替换为
  `tests/test_p0_expert_removed.py`：断言目录已消失、全仓无任何 `import junqi.expert`、
  删除记录文档仍在、在线引擎仍归属 `junqi/search.py::ExpertSearchEngine`。

### 二、虚拟环境移出工程目录

`junqi_engine/venv/`（4.7 GB / 25,054 个文件）→ 同级 `../venv_junqi_engine/`。
动机：依赖树混在源码目录里会被误当项目内容扫描/归档/统计，也拖慢全库检索。

- `venv/` 本就在 `.gitignore` 中，移动**不影响版本控制**；
- 已修正迁移后 `Scripts/activate`、`Scripts/activate.bat`、`pyvenv.cfg` 中的绝对路径
  （`Activate.ps1` 用 `$PSScriptRoot` 相对解析，无需改动）；
- `run_tests.bat` 改为**自动定位**：优先工程内 `venv\`（旧布局/自建环境），
  其次同级 `..\venv_junqi_engine\`，都找不到时回退 PATH 上的 python 并明确告警；
- `README.md` 的环境准备与测试章节同步为新路径，并注明移动原因。

### 三、归档脚本加废弃标注

`scripts/archive/`（8 个脚本）与 `tests/utils/`（4 个脚本，与前者同源）此前既无测试覆盖、
也不在 pytest 收集范围内，尤其 `tests/utils/` 位于 `tests/` 之下极易被误读为"被执行过的测试"。
现各加 `README.md` 标注：说明它们是一次性验证脚本、逐一列出当初用途与现役替代入口
（`pytest tests/` / `python -m junqi benchmark` / `python -m junqi gate` / `export_dataset`），
并明确"需要长期守门就写成 `tests/test_*.py` 正式用例"。

新增 `tests/test_p4_archived_scripts_labelled.py` 守卫：两处 README 存在且内容达标、
归档脚本文件名不得匹配 `test_*.py`、`pytest.ini` 收集范围未被放宽、
以及它们依赖的 `train_rl` 内部原语若被删除须先在此失败（而非静默 ImportError）。

### 四、未处置（按决策"暂时不进行清理"）

- `models/` 中间产物（约 8.0 GB，其中三个 `*_buffer.pkl` 合计约 11.4 GB 的历史残留）
  保持不变；需要时执行 `python scripts/cleanup_models.py`（**默认 dry-run**，加 `--yes` 才删除；
  `best.pt` / `bc_best.pt` / `value_distilled*.pt` 在保护名单内）。

### 五、测试

`tests/test_p0_expert_deprecated.py` → `tests/test_p0_expert_removed.py`（5 项，含"目录已消失"断言）；
新增 `tests/test_p4_archived_scripts_labelled.py`（6 项）。
全量：**403 passed / 3 skipped**（用迁移后的 `../venv_junqi_engine` 解释器执行）。

> ⚠️ **过程记录（供后来者避坑）**：本批使用 `git rm -r` 删除 `junqi/expert/` 时，命令被中途
> 终止（SIGTERM）并遗留 `.git/index.lock`，随后 `junqi/` 与 `scripts/` 共 96 个文件从工作区
> 消失（`git status` 显示为 ` D`）。已通过清除 stale lock + `git checkout HEAD -- junqi scripts`
> 完整恢复，全部源码改动经关键词核验无损。**教训：本仓库的删除操作请用普通文件系统操作
> （`os.remove` / `shutil.rmtree`）完成，不要用 `git rm`；确需 git 操作时避免长链命令。**

---

## [2026-09-15] 第二批 — 搜索/数据/健壮性/性能全量收口（审查 C1–C12、P2–P6 修复完毕）

阶段归属：**P0（正确性）+ P1（数据口径）+ P3（运营/性能）**。承接同日第一批
（R1/R2/R3/P1/R4，见下条）。至此 `reviews/CODE_REVIEW_2026-09-15.md` 中的
全部条目均已处置。**未改动** `RuleConfig` 规则定义、终局奖励语义（胜 +1 / 和 0 / 负 −1）、
公共 Policy 信息边界；未触碰 `models/best.pt`。

### 一、搜索正确性（C1–C5）

| 编号 | 问题 | 修复 |
|---|---|---|
| C1 | IDS 早停判据写成"当轮总耗时 > 预算 25%"，与"下一深度预计超时才停"的语义完全不符——1000ms 预算下 depth-1 用掉 260ms 就退出，剩余 740ms 全浪费 | 新增纯函数 `should_stop_ids` / `ids_growth_estimate`：按"已用 + 本层耗时×实测增长因子(夹在 2~12) > 预算"判定 |
| C2 | 首层未完成即超时时返回 `(acts[0], -inf)`；该 `-inf` 经 `hybrid_engine` 取负成 `+inf` 参与排序，并被 `train_search_distill` 全量 softmax 蒸馏 | 超时降级返回 `(兜底动作, 0.0)` 且置 `stats.degraded=True`，**永不返回 ±inf** |
| C3 | 根节点后续动作在收窄的 `[alpha, beta)` 窗口内搜索，fail-low 返回的是**上界**，却被当精确分写进 `root_scores` 供 top-N 展示与蒸馏 | 新增 `search(exact_root_scores=True)`：根节点全窗口搜索；同时新增 `stats.root_scores_bounded` 标记非精确分（PVS 模式）。`ai.ExpertAgent.choose_actions` 在 `topn>1` 时自动启用 |
| C4 | `ai.py` 的 topn 回退分支把 1e5 量级的 move-ordering 分当估值分返回（还 `+= 100_000`） | 回退只诚实地返回 `[(best_act, 0.0)]`，不再伪造候选与分值 |
| C5 | `hybrid_engine.select_action` 兜底 `Action("pass")` 缺必填 `frm`，触发即 TypeError | 返回 `None`；签名改为 `Optional[Action]` |

### 二、MCTS 与推理热路径（C6、P2、P3）

- **C6**：树内重复判和阈值原硬编码 `>= 3`，与生成夹具 `GENERATION_CFG`（4）不一致。新增
  `tree_repetition_limit(state)` 跟随 `cfg.repetition_draw_count`（下限夹到 2）。
- **P2**：`net.predict_state` 新增 `acts` / `mask` 可选参数，消除每次推理内**两遍**
  `legal_actions()`；MCTS 叶子评估复用已算好的动作列表（用 `inspect.signature` 做能力探测，
  不破坏旧签名/测试替身）。MCTS 根节点 `avoid` 命中判定从算两遍改为算一遍复用。
- **P3**：`hybrid_engine._price_actions` 复用已构造的 `nxt` 计算重复局面键，去掉多余的
  `state.apply(a)`。
- **P5**：`_negamax` 中 `depth <= 0` 的 QSearch 提前到 Zobrist 计算之前（叶子不再白算全盘哈希）；
  `_is_tactical` 闭包从"每动作每深度重建"提升到方法级定义一次。

### 三、数据口径与方案文档（C7、R5、R6、C12）

- **C7**：`dataset.py` 的 outcome 统计另写了一遍终局码判定并把 code 24（断线）计入
  `decided_win`，与标签函数判其为 `none` 冲突，导致 `metadata.json` 的 `decided_win` 虚高
  （p1_v3 记 488）。新增 `outcome_bucket_from_meta()`，统计与标签**共用同一张码表**。
- **R6**：新增 `DEFAULT_P1_DIR = "datasets/p1_v3"` 与 `MIN_P1_VERSION = (3,0,0)`；
  `dataset` / `train_bc` / `eval_bc` / `train_value_distill` / `__main__` 的默认数据集路径
  全部统一到 p1_v3（此前散落 p1_v1/p1_v2，而 v2 生成于 code 24 修正之前、口径不同）；
  `load_p1_arrays` 新增版本守卫 `check_p1_version`，低于 3.0.0 直接拒绝加载；
  `--version` 默认值改为 3.0.0。
- **R5**：基线文档 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §6「Value」原先写"认输局双方按 ±1
  计入 Value"，与 §5 P3 及 `train_rl.py`（认输局只进 Policy）自相矛盾。**代码是对的**
  （防 Value 自证回路：实证开启认输时三分类准确率 60%→16%），已回写方案并补齐
  「修订原因 / 影响范围 / 验证方法 / 回滚方案」四项（AGENTS.md 第 10 条要求）。
- **C12**：`train_bc` / `eval_bc` / `fit_weights` / `load_p1_arrays` 首次获得测试覆盖。

### 四、健壮性与指标口径（C8、C9）

- **C8**：`selfplay.play_game` 的专家估值失败从 `except Exception: pass` 改为计数 +
  显式告警，并把 `eval_failures` 写进对局记录（否则门控"裁决式判分"静默退化为 0.5 而无从察觉）；
  顺带修掉"为算一个布尔标志又重复调用两次 `evaluate_expert`"的浪费（4 次 → 2 次）。
  `StratifiedReplayBuffer.load` 失败不再静默，改为打印可诊断告警。
- **C9**：`train_rl` 的 Elo 更新原为 `current_elo += 16*(score-0.5)*2`（与对手无关的线性
  随机游走，却被写入 `elo_history.jsonl` 当实力曲线）。新增 `elo_update_from_score`，
  按标准公式 `r_opp + 400*log10(s/(1-s))` 换算，得分率按样本量夹紧保证数值有限；每轮打印
  前后等级分。

### 五、性能收尾（P4、P6）

- **P4**：新增 `ai.load_net_cached`（按 绝对路径+mtime+device 缓存，容量 8，仅供推理的
  eval 副本）。门控/评测默认 256 局/轮、每局都会重建策略，原先每次都要从磁盘反序列化
  ~34MB 权重（约 512 次/轮）。`NNAgent` 与 `HybridDecisionEngine` 均改走缓存。
- **P6**：`apk_engine` 置换表原把**截断后的 18 位哈希直接当键**且读取时不校验完整
  Zobrist（跨局面碰撞会静默返回错误分数与错误 PV，对比 `junqi/tt.py` 有 key 校验），
  且 `self.tt` 无界增长。现改为存完整键 + 命中校验（`tt_collisions` 可观测）、
  显式容量上界与插入序淘汰、且浅条目不得覆盖深条目。
- **C10（部分）**：删除 `train_rl` 中从未被引用的 `PolicyDataset` / `ValueDataset` 两个
  DataLoader 数据集类（训练实际直采 `StratifiedReplayBuffer`）。

### 六、附带发现并修复：`scripts/cleanup_models.py` 会删掉热启动基线

原删除模式含 `*_distilled.pt`，会连带删除 `models/value_distilled_v2.pt`——P3 修订后
`train_rl` 热启动链**首选**的健康 Value 头（平衡准确率 0.735）。删掉后训练会静默退回
`bc_best.pt`（价值头未校准），候选初期反而更弱。现改为：显式 `PROTECTED_EXACT` 保护
热启动/发布链、**默认 dry-run**（必须 `--yes` 才执行）、并报告 pool/ 与 evidence_*/ 等
子目录占用。

### 七、测试

新增 6 个测试文件（本批 66 项）：
`test_p1_search_time_and_scores.py`(16)、`test_p1_mcts_repetition_and_batching.py`(9)、
`test_p1_dataset_label_consistency.py`(21)、`test_p1_robustness_and_elo.py`(10)、
`test_p3_apk_tt.py`(8)、`test_p4_net_cache.py`(6)、`test_p4_model_cleanup_script.py`(7)。

全量：**298 passed / 3 skipped（审查前基线）→ 396 passed / 3 skipped**，无回归。

### 八、仍未处置 / 需人工决策

- `junqi/expert/` 11 个模块：选"标记废弃 + 守卫测试"（见 `junqi/expert/DEPRECATED.md`），
  **未删除**；若要彻底删除请显式确认（`git rm -r junqi/expert/`，文件已在版本控制内可回滚）。
- `models/` 现约 8.0 GB、`venv/` 约 4.7 GB：属运营清理，**未自动执行**。
  清理中间产物可用 `python scripts/cleanup_models.py`（默认 dry-run）；
  `venv/` 建议移出工程目录或加入忽略。
- C10 的"超长函数拆分"（21 个 >120 行函数，如 `run_training` 339 行）属结构性重构，
  收益低于回归风险，未在本批执行。

---

## [2026-09-15] 第一批 — P0 修复：训练闭环两处致命缺陷 + 门控统一 + 经验池瘦身（代码审查落地）

阶段归属：**P0（正确性与可复现性）为主，含 P3 运营修正**。依据 [AGENTS.md](../AGENTS.md) 与
[AI_TRAINING_AND_HUMAN_PLAY_PLAN.md](../AI_TRAINING_AND_HUMAN_PLAY_PLAN.md) §5 P0 / §7。
完整审查结论见 [reviews/CODE_REVIEW_2026-09-15.md](../reviews/CODE_REVIEW_2026-09-15.md)。
**未改动** `RuleConfig` 规则定义、终局奖励语义、信息边界，也未触碰 `models/best.pt`。

### 一、致命缺陷修复（这两条使此前自博弈训练在数学上无效）

1. **热启动 optimizer 脱钩 → 权重零更新**（`junqi/train_rl.py:1162/1174/1179`）
   `net = JunqiNet.load_from_file(...)` 重新绑定变量名，而 `optimizer` 在此之前已构造完毕，
   其 `param_groups` 仍指向旧网络参数；新网络反传后旧参数 `grad is None`，AdamW 全部跳过。
   触发条件：首次训练 / `--fresh` / `--rebase-baseline` 之后（断点续训路径不受影响）。
   唯一可观测征兆是 loss 恒定。
   **修复**：新增 `warmstart_candidate(net, optimizer, path, ...)`，主干超参一致时**就地**载入并返回原对象；
   仅当检查点主干超参不一致、无法就地载入时才重建网络并**同时**重建 optimizer。
2. **自博弈 "best 对手" 是随机初始化网络**（`junqi/train_rl.py:1217` + Worker `:537`）
   `torch.load(net.save(path))` 得到的是 `{"model_state": ...}` 包装字典，却被直接喂给
   `load_state_dict(..., strict=False)`：键名无一匹配、静默通过。OPP_MIX 中 best 约占 25%，
   即四分之一对局的"强对手"从未训练过，且日志完全看不出异常。
   **修复**：新增 `JunqiNet.unwrap_state_dict` / `adapt_state_dict` / `load_state_dict_into` 与
   `train_rl.load_weights_into_net` / `infer_net_architecture` / `build_opponent_net`；
   缺键即**降级为 None**（回落 greedy/expert）并打印原因，**绝不返回随机初始化的假对手**。
   顺带修掉原实现只看 `in_conv.0.weight.shape[1]` 推断架构的问题（非默认 128/6 的检查点同样会静默退化）。

### 二、门控统一（R3）

`train_rl.evaluate_gate` 原是与 `eval_gate.run_gate` 平行的第二套实现，且两处都错：
两个方向用 `seed+i` 与 `seed+100_000+i`（**非配对同牌**）、子进程 `device` 写死 `"cpu"`。

- `eval_gate.run_gate` 新增 `init_states`（与 seeds 等长，两局共用同一初始局面）与 `device` 透传；
- `train_rl.evaluate_gate` 改为逐场景委托 `run_gate`，保留原 `stage_stats` 返回契约；
- 删除 `train_rl` 内已失效的 `_gate_game_job` / `_run_jobs` / `_tally`；参考对抗（vs search2）同样走配对口径；
- `device` 由训练主循环传入（有 GPU 时门控不再强制 CPU）。

### 三、P3 运营：经验池落盘瘦身（P1）

`models/candidate_latest_buffer.pkl` 实测 **3.8 GB/份**（三个实验目录合计约 11.4 GB），
原实现每轮无条件写入且非原子。

- 新增 `should_save_buffer(epoch, end_epoch, every)` 纯函数节流判据；
- `save_checkpoint(..., save_buffer=...)` 控制是否落池；改为 **`.tmp` + `os.replace` 原子写**；
- CLI `--buffer-save-every`（默认 **5**，末轮恒写；`0`=除末轮外不写；`1`=恢复旧行为）。

### 四、`junqi/expert/` 标记为废弃（R4）

AST 依赖图 + 成员存在性核验：11 个模块引用 `state.Move` / `get_piece_at` / `current_turn` /
`turn_count` / `get_pieces` / `config.PIECE_RANKS` / `board.is_my_base` 等**全部不存在**的成员，
`import junqi.expert` 直接 `NameError`；fan-in 为 0，主流程零调用。在线传统搜索引擎是
`junqi/search.py::ExpertSearchEngine`。新增 `junqi/expert/DEPRECATED.md` 说明现状与复活路径，
并新增守卫测试防止它被重新接回主流程。

### 五、测试

- 新增 `tests/test_p0_hotstart_and_opponent.py`（10 项）：包装字典事实固化、旧写法"零载入"证据、
  对手权重逐位一致、垃圾 payload 返回 None、热启动保持对象同一性、optimizer 仍绑定当前 net、
  **热启动后优化步必须改变权重**、架构不一致时重建 optimizer；
- 新增 `tests/test_p0_gate_pairing.py`（6 项）：配对同牌（同 seed 同初始局面、只换模型）、
  `init_states` 长度不一致报错、`device` 透传、`evaluate_gate` 逐场景委托且每 seed 只发射一次；
- 新增 `tests/test_p3_buffer_persistence.py`（9 项）：节流判据、`save_buffer=False` 不写池、
  原子写无 `.tmp` 残留、无 buffer 检查点仍可加载、空池 pickle 可读；
- 新增 `tests/test_p0_expert_deprecated.py`（3 项）：废弃说明存在、主流程零依赖、在线引擎归属；
- 全量：**298 passed / 3 skipped → 全量通过**（修复前基线 298/3，新增 28 项）。

### 六、未处理（留待下一轮，按优先级）

`search.py` IDS 25% 早停语义错误与超时返回 `-inf`（C1/C2，会污染搜索蒸馏教师标签）、
根节点 alpha 窗口上界当精确分（C3）、`dataset.py:421` 把 code 24 计入 `decided_win`（C7）、
伪 Elo 随机游走（C9）、MCTS batch=1 前向（P2）。详见审查报告。

---

## [2026-09-14] — P3 修订：Value 对齐根因定位（学习率）+ 每轮 Value 重锚 + p1_v3/test 独立健康验收

阶段归属：**P3（数据质量改造，基线计划 2026-09-13 修订版）**，依据 [AGENTS.md](../AGENTS.md) 与 [AI_TRAINING_AND_HUMAN_PLAY_PLAN.md](../AI_TRAINING_AND_HUMAN_PLAY_PLAN.md) §6。逐条落实 [SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md](SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md) §5 方案的验证修正版（该文档 §4 的"认输投毒"假设已被 §3 实验 B 推翻；本次进一步给出了根因的受控实验证据）。未修改 `RuleConfig` 定义与终局奖励语义；未触碰 `models/best.pt`。

### 一、根因定位（受控实验，非推理）

同一 6 块共享主干、同一 p1_v3 官方客观标签、联合策略+价值训练，仅改学习率：

| lr | 起始 | 1 轮（711 步） | 2 轮 | 3 轮 | 策略模仿 top-1（3 轮后） |
|---|---|---|---|---|---|
| 1e-3（原默认） | 平衡acc 0.763 / MAE 0.266 | 0.567 / 0.454 | 0.544 / 0.491 | 0.546 / 0.486 | 0.218 |
| **1e-4（新默认）** | 同上 | 0.713 / 0.320 | 0.716 / 0.313 | **0.713 / 0.319** | **0.261** |

结论：**1e-3 对 6 块共享主干做微调过高**——一个 epoch 即砍掉约 20 个点 Value 平衡准确率，且策略学得更差。这解释了"实验 B 只训 1 轮 Value 三分类即崩回 16%"。

同时证伪了"头-only 重锚可修复"的假设：对已漂移主干（`models/candidate_latest.pt`，实测 100% 单类塌缩）做头-only 重锚 2,372 步，平衡准确率上限仅 0.344（≈随机 0.333），用健康 v2 头初始化亦被拖回 0.350，解冻末块 2 轮无恢复。**故重锚是预防（每轮保险丝），不是修复；已漂移候选不可救，只能从健康基座重启。**

### 二、代码变更

1. **默认学习率 1e-3 → 1e-4**（`DEFAULT_LR`，`junqi/train_rl.py`），CLI `--lr` 帮助文本记录证据；
2. **热启动优先 `models/value_distilled_v2.pt`**：原接线只认 `value_distilled.pt`（p1_v3/test 实测平衡acc 0.318 / MAE 0.599，84% 预测画和棋的塌缩头），而 v2 为 **0.735 / 0.288**（三类召回 0.69-0.77）；两者策略头权重逐位相同（最大差异 0.00000），换用是纯收益；
3. **每轮 Value 重锚**（`reanchor_value_head` / `build_anchor_dataset`）：每轮联合训练后冻结主干、仅训 Value 头，锚定数据 = 回放池客观终局样本（每类 ≤12000，类均衡）+ p1_v3 官方客观标签（占比 0.30），验证集为 p1_v3/val；成本实测约 2-3 秒/轮（head-only 2.7ms/步 @4080S）。回滚开关 `--no-reanchor`；
4. **Value 验收探针换口径**（`probe_value_health` / `value_acceptance`）：主指标改为 **p1_v3/test 独立留出集（9,381 条官方客观标签）的平衡准确率**（线 0.45）+ MAE（线 0.55）+ 塌缩告警。理由：该集和棋占 54%，"恒定预测单一类别"的塌缩模型 MAE 仅约 0.49-0.50，**单看 MAE 会误放行**（实测 `candidate_latest.pt` 即全 Loss 塌缩且 MAE 0.502）；原靶场仅 36 题（17/11/8），accuracy 粒度 2.8%、噪声带宽 ±13-16%，降级为固定回归探针；
5. **修复 `train_value_from_p1_dataset` 必然崩溃**（`junqi/train_value_distill.py`）：首轮打印引用未定义变量 `mae` → `NameError`，CLI `distill_value --p1-dir` 路径此前不可用且零测试覆盖；同时抽出可复用助手 `load_p1_arrays`（has_values 过滤）/ `value_health_metrics`（纯函数）/ `evaluate_value_health` / `train_value_head_only`（离线蒸馏与轮内重锚共用，含验证集平衡准确率早停与 requires_grad 恢复保护）；
6. **CLI 接线补齐**（`junqi/__main__.py`）：`train_rl` 子命令的 `--lr` 默认同步为 1e-4，新增 `--no-reanchor` 与 `--anchor-p1-ratio` 并传入 `run_training`（`train_rl.py` 内的 `main()` 非实际入口，此前新开关不会生效）；
7. **README 同步**：修正失效入口（`cli.py` 为历史壳，其 `train/`/`eval/`/`ui/` 目录已不存在）为 `python -m junqi <子命令>`，更新测试计数（258 项 + 3 跳过）。

### 三、测试与验收

- 新增 `tests/test_value_reanchor.py`（11 项）：塌缩检测（含"低 MAE 高塌缩"反例）、仅 Value 头更新（冻结参数逐位不变）、requires_grad 恢复、打乱头部可恢复且恢复最优 epoch 权重、重锚数据集混合比例/类别覆盖、空池报错、验收判据、p1 训练器端到端回归（崩溃修复）、has_values 过滤；
- 新增 `tests/test_pool_weights.py`（9 项）：分桶权重纯函数（fixed=旧基线 / adaptive=√容量）、曝光失衡比从 34.5 压到 4.5、小桶不饿死、空桶排除、端到端 batch 组成偏移；
- 全量：**258 passed / 3 skipped**（含新增 20 项），无回归；
- 回滚点：`--no-reanchor`（关闭重锚）、`--lr 1e-3`（恢复旧学习率）、`--pool-weights fixed`（默认即基线）；均为 CLI 开关，无需改代码。

### 四、批次 1 验证运行结果（5 轮 × 100 局，与历史主运行严格对齐）

配置：每轮 100 局、sims=20、10 workers、认输开启；受控变量仅三项（lr 1e-4 / 热启动 v2 / 每轮重锚）。

| Epoch | v_loss | 重锚 val 平衡acc | 探针 p1_v3/test 平衡acc | 探针 MAE | 验收 | 耗时 | Elo |
|---|---|---|---|---|---|---|---|
| 1 | 7.895 | 0.662 | 0.595 | 0.456 | ✅ 连续1 | 1466s | 1499 |
| 2 | 1.480 | 0.662 | 0.603 | 0.442 | ✅ 连续2 | 1513s | 1497 |
| 3 | 1.768 | 0.626 | 0.605 | 0.452 | ✅ 连续3 | 1250s | 1496 |
| 4 | 1.532 | 0.653 | 0.616 | 0.443 | ✅ 连续4 | 1189s | 1497 |
| 5 | 1.616 | 0.637 | **0.639** | **0.408** | ✅ 连续5 | 1249s | 1495 |

**验收线（连续 2 轮 平衡acc ≥0.45 / MAE <0.55 / 无塌缩）达成，实际连续 5 轮**；探针指标缓升、预测分布贴近真值分布。对比历史：同口径指标（靶场三分类）曾剧烈震荡 60%→12%→58%→40%→16%→28%，实验 B 单轮即崩回 16%。认输率 1-7%（历史 2-27%）、局长 296-318 手、决胜率 63-70%——对局形态不变，只有 Value 校准被修复。日志：`reports/trainrl_p3_20260914.log`。

### 五、批次 3 首项（Policy 池失衡修复，opt-in）

`--pool-weights adaptive`：按 √桶容量 归一化分桶采样权重。实测池（opening 8,589 / midgame 3,208 / endgame 66,409）下，单样本曝光率最大/最小比由约 **34.5** 压到约 **4.5**，消除"50% batch 抽自 3,208 条 midgame"的过度曝光。**默认仍为 `fixed`**（本轮已验证基线），待下一轮单独验证其效果。

### 六、批次 2：形式化 SPRT 门控（n=200）判定 **不重定 best.pt**

候选（批次 1 第 5 轮）vs `models/best.pt`（09-01），200 局配对同牌：**胜 25 / 和 144 / 负 31，得分率 0.4850（Wilson [0.4167, 0.5539]），仅计胜负局胜率 0.4464（n=56），配对 z=-0.973（p=0.331），SPRT LLR=-20.809 → accept_h0，promote=False**。依 AGENTS.md 硬约束不覆盖 `best.pt`。报告：`reports/gate_nn_mcts_20_vs_nn_mcts_20.json`。

**关键判读**：批次 1 修复的是 Value 校准（必要），但**未带来可测的棋力提升**——强度瓶颈在信号质量（sims=20 自我蒸馏、50% mirror 对手、缺外部梯度），属批次 3/4 范畴。门控同时暴露 72% 和棋率（144/200，其中 no_capture 141），使仅有 28% 对局携带胜负信息。

### 七、批次 3 首项验证（adaptive 池权重）：**无显著差异**

5 轮训练（唯一变量 `--pool-weights adaptive`）：探针平衡acc 0.595/0.605/0.625/0.624/0.623 全程达标；p_loss 一致更低（3.37-3.52 vs 3.62-3.80，按纪律不作为变强证据）。对比门控（adaptive vs fixed，200 局）：**胜 10 / 和 183 / 负 7，得分率 0.5075（Wilson [0.4387, 0.5760]），SPRT LLR=-73.322 → accept_h0，promote=False**；和棋率 91.5%，仅 17 局分出胜负。

**结论与下一步**：池失衡修复不改变棋力。批次 1-3 的改动共同确认——Value 校准已修好，但**下一个瓶颈是"区分度"本身**：同源模型间 72-91.5% 对局以 no_capture 判和，门控只剩 17-56 局胜负样本，而训练端 z=0 样本又被裁掉。故下一步优先级为：**① 批次 4 提高搜索质量/有效 sims（C++ nn_mcts 或 GPU 批量推理）；② 对手结构（mirror 0.5→0.3、expert 0.1→0.2）**，而非继续调整数据配比。

**证据归档**：`run_gate` 的 JSON 名仅由 spec 派生，两次 `nn_mcts_20` 门控互相覆盖；批次 2 文本报告已归档 `reports/archive_gate_b2_candidate_vs_best_n200.log`，批次 3 为 `reports/gate_b3_adaptive_vs_b1_fixed.{log,json}`（后续宜加 out_name 参数）。

### 八、批次 1b 补齐：轮内门控降级为「只记录、不判定」+ 正式晋级协议

- `junqi/train_rl.py` 新增纯函数 `inloop_gate_decision(stage_stats, inloop_gate_promote=False, ...)`：**默认返回 `promote=False`**（即使记录是 16/16 全胜），理由——n=16 时 Wilson 判据需得分率 ≥0.75（约 +191 Elo）才显著，对每轮 +10~30 Elo 的真实进步无功效，据此晋升等于用噪声改发布模型；回滚开关 `--inloop-gate-promote`（CLI）恢复旧行为。
- 正式晋级协议（唯一的发布模型改动入口）：`junqi/eval_gate.py` 新增 `promote_candidate(report, model_a, best_path, backup_dir)`——仅当 `report["promote"] is True` 才把候选写入 `models/best.pt`，旧模型带时间戳备份，否则完全不触碰。CLI：`python -m junqi gate --model-a <候选> --model-b models/best.pt --seeds 100 --promote-to-best`；另加 `--out-name` 修复同 spec 门控报告互相覆盖的问题。
- 测试：`tests/test_batch3_opp_and_gate.py` 中 5 项覆盖（完美战绩仍不晋升、回滚开关恢复旧行为、平庸战绩仍拒绝、通过与拒绝两条晋级路径、候选缺失拒绝）。

### 九、批次 3 第二项：对手配比预置（mirror 0.5→0.3、expert 0.1→0.2）

- `OPP_MIX_PRESETS` 新增 `diverse = {mirror .30, best .30, expert .20, greedy .15, random .05}`；`opponent_type_for` 与 worker 任务参数支持传入配比；CLI `--opp-preset {baseline,diverse}`（默认 baseline=已验证基线）。动机：mirror 自对弈对抗梯度近零，而 expert 对局同时提供真实对抗压力与客观终局 Value 样本的最廉价来源；两次 n=200 门控显示同源模型间 72-91.5% 对局以 no_capture 判和、区分度枯竭。
- 测试 7 项：预置分布合法性（和为 1、键一致）、baseline 等于既有常量、diverse 数值符合方案、抽样分布与预置一致（±2%）、无对手网络时 best 降级 expert、expert 占比从 10%→20% 的对照。
- 首轮实测（`reports/trainrl_b3opp_diverse.log`）：实际对手占比 mirror 0.26 / best 0.35 / expert 0.20 / greedy 0.13 / random 0.06（与预置一致）；**拔旗终局占比 4%→9%**、决胜率 63%、探针 0.592/0.472 达标；单轮耗时 1183.1s（未因 expert 占比翻倍而变慢）。
- **5 轮训练结果**：探针 0.592 / 0.632 / 0.613 / 0.611 / 0.608（MAE 0.472→0.431）全部达标；拔旗占比最高 11%、第 2 轮零认输；每轮耗时 1158-1318s 与基线持平。
- **对比门控（n=200，`reports/gate_b3opp_vs_b1.{log,json}`）**：胜 8 / 和 184 / 负 8，**得分率 0.5000**，Wilson [0.4314, 0.5686]，仅计胜负局 0.5000（n=16），配对 z=0.000，**SPRT LLR=-85.998 → accept_h0**，promote=False。即**对手结构显著改善决胜局质量（拔旗 4%→9-11%），但棋力无可测变化**。

### 九bis、三次形式化门控的合并判读：瓶颈是「区分度」而非配比

| # | 对比 | 战绩 | 得分率 | 决胜局 | SPRT |
|---|---|---|---|---|---|
| 1 | 批次1候选 vs 旧 best(09-01) | 25/144/31 | 0.4850 | 56（28%） | accept_h0 |
| 2 | adaptive 池权重 vs 批次1 | 10/183/7 | 0.5075 | 17（8.5%） | accept_h0 |
| 3 | diverse 对手结构 vs 批次1 | 8/184/8 | 0.5000 | 16（8%） | accept_h0 |

三次一致 `accept_h0`，和棋率依次 72% → 91.5% → 92%。**自博弈数据里几乎没有胜负信号**（92% 的 no_capture 和棋在生成端已稀薄，训练端又裁掉和棋尾部样本），故批次 1-3 的改动（Value 校准 / 池权重 / 对手结构）都无法转化为可测棋力。**下一优先项**：① 诊断 no_capture 和棋成因（策略是否系统性回避进攻）；② 用批次 4 的推理提速把 sims 提上去；③ 其余超参（lr 余弦、池权重）暂缓。

### 九ter、和棋成因诊断（`scratch/diagnose_draws.py` + 专家对照，报告 §8.10）

- **和棋不是不战斗**：182 局 no_capture 和棋局长中位 320 手、被吃子力中位 924 分（≈一整军）；净子力差中位仅 75（决胜局 168），65/182 局净差恰为 0。
- **NN 系统性回避交火**：有攻击机会时仅 5.6-6.6% 选择进攻；每局攻击 13.5 次、被吃 868 分、净差≈0、拔旗 1.5%。**专家对照**（expert2 镜像）：有机会时 18.1% 进攻、每局攻击 25.8 次、被吃 1416 分、净差 184、拔旗 12.5%——专家用不对等交换制造优势，NN 只做对等交换。
- **判定**：和棋倾向部分为博弈固有（专家对专家也 75% 和棋，分胜负空间上限约 25%），但 NN 把它压到 8% 且方式退化为困毙——**瓶颈在网络与搜索，非规则目标**。规则层攻击稀缺（暗子不可攻/禁自杀攻击/行营保护）是共同背景，每手合法攻击选项仅 0.9-2.0 个。
- 下一步指向：提高 sims（看穿进攻-反杀线）+ 批次 4 提速；数据侧可研究对"进攻后取胜"片段的采样加权（不引入中间奖励）。

### 九quater、搜索深度扫描与裁决式判分（测量口径修复，2026-09-14）

**(a) 搜索深度扫描**（`reports/diagnose_sims_sweep.log`，12 局/档，官方规则，max_plies=1000）：

| 指标 | sims=20 | sims=40 | sims=80 | 专家对照 |
|---|---|---|---|---|
| 决胜率 | 8% | **17%** | 17% | 12.5% |
| 每局攻击次数 | 15.5 | **25.0** | 25.9 | 25.8 |
| 被吃子力中位 | 911 | **1540** | 1501 | 1416 |
| 有机会时选择攻击 | 5.6% | 8.5% | 7.9% | 18.1% |

sims 20→40 显著提升交战质量（攻击 +61%、交换 +69%），**40→80 已饱和**；且 sims=20 档的 92% 和棋率与 200 局门控完全一致（插桩方法学自证）。同时专家自己在官方规则下也 88% 和棋 → **高和棋率主要是博弈+规则集固有属性**。

**(b) 测量口径修复：裁决式判分**。既然 8-12% 的决胜负率把真实强度差在"得分率"上稀释约 12 倍（+65 Elo 仅表现为 0.50→0.525，被 Wilson 半宽 ±0.069 淹没），则所有后续验证都会因测量而死：

- `junqi/selfplay.py::play_game` 在记录中写入 `final_eval0/final_eval1`（`evaluate_expert` 的公共信息估值，0.1ms/次，不参与训练奖励）；
- `junqi/eval_gate.py::adjudicate_record`：有胜负按胜负，和棋按终局估值判胜负（`margin` 可配默认为 0），把全部对局变为有效样本；
- `run_gate` 报告新增 `adjudicated` 段（战绩/得分率/Wilson/配对检验/估值差中位），**不改动官方 `promote` 判据**（AGENTS.md：评测夹具保持官方规则）；
- 测试 `tests/test_gate_adjudication.py` 6 项（胜负映射含座位互换、margin 语义、估值缺失回退、play_game 记录零和一致性、门控报告段），全量 **294 passed / 3 skipped**。
- 标定实验（镜像噪声底线 + sims20-vs-sims5 敏感性）结果见下一节。

#### 标定第一轮暴露的实现缺陷（已修）

- **镜像标定（40 组种子 = 80 局，`reports/gate_calib_mirror.{log,json}`）**：官方得分率 0.5000、裁决得分率也 0.5000，且 74 局和棋里**每一局的 `final_eval0` 都恰好是 0.0**。
- **根因**（读码定位）：`analysis._is_dead_draw_impl` 的**第一分支**是规则限步——`quiet >= no_capture_draw_plies` 即判死；而 `eval_expert.evaluate_expert` 对判死局面直接 `return 0.0`。于是"终局估值裁决"在恰好需要它的场合（限步判和的死锁终局）必然退化为 0，与结构无关。
- **修复**：(1) `_is_dead_draw_impl` / `is_dead_draw` 新增 `ignore_quiet_limit`（默认 False，专家引擎热路径语义与缓存路径不变）；(2) `evaluate_expert` 新增 `ignore_rule_draw` 透传；(3) `play_game` 每 10 手采样一次"有效估值"（`ignore_rule_draw=True`），终局若仍为结构性死锁（估值恒 0）则回退到对局中最后一次有效采样，并在记录中标注 `last_live_eval_used`。
- **敏感性对照（`reports/gate_calib_sims.{log,json}`，A=sims20 vs B=sims5，80 局）**：官方得分率 0.5062（Wilson [0.3989, 0.6130]），**80 局仅 3 局分出胜负（3.75%）**，SPRT accept_h0——搜索深度差 4 倍都无法在官方口径上表达。这是裁决判分的目标场景，修复后已重跑（结果见下节）。
- 注意：`evaluate_expert` **不是严格零和**（同一局面 seat0/seat1 视角值之和可为非零，实测例 6.98 / −21.72），故裁决比较"各自视角谁更高"仅作为一致的排序信号，报告同时给出终局估值差中位供诊断。

#### 标定第二轮：裁决指标**未通过**偏差校准；同时发现门控打分器的座位归属 bug（已修）

**(1) 裁决指标现状：不可用（诚实结论）**

| 对照（80 局，修复后裁决） | 官方得分率 | 裁决得分率 | 裁决 Wilson 95%CI |
|---|---|---|---|
| A=sims20 vs B=sims5（**已知 A 更强**） | 0.5125 | 0.5750 | [0.4657, 0.6774] |
| 镜像（**同一模型**，应≈0.5） | 0.5125 | **0.6000** | [0.4905, 0.7004] |

镜像的裁决得分（0.600）反而**高于**已知更强方的 0.575，两者 CI 几乎完全重叠 → 在 n=80 下裁决被噪声/座位偏差主导，**无法分辨真实强度差**。可能的偏差来源：`evaluate_expert` 非严格零和 + 座位/颜色的系统性偏好。结论：该指标需重新设计（例如按 pair-sum 配对统计、或构造对称化 arbiter）后再评估，当前**不用于任何判定**。

**(2) 门控打分器的座位归属 bug（严重，已修复）**

- `run_gate` 原用 spec 名推断"候选坐哪一桌"：`if rec["a"] == spec_a: seat = 0`。而**所有正式门控都使用同名 spec**（`--a nn_mcts_20 --b nn_mcts_20`，仅 `--model-a/--model-b` 不同），此时该判断恒真 → 每一对局中"候选执后手"的那一局被**反向计分**（胜记负、负记胜）→ 真实差异被系统性压回 0.5。
- **历史证据**：所有正式门控报告里 `seat_split.as_second` 恒为 `0 局`（应为一半），`as_first` 独揽全部局数。
- **影响**：批次 2（候选 vs 旧 best，报 0.4850）、批次 3-adaptive（0.5075）、批次 3-diverse（0.5000）三次"一致 accept_h0、棋力无可测差异"的结论**建立在被反向计分的半数样本上，不可信**，需用修复后的打分器重测。
- **修复**：新增 `_score_at_seat(rec, seat)`，座位由 `_run_pair` 的构造显式决定（pair[0]=候选执先、pair[1]=候选执后）；`seat_split` 改用座位索引；`adjudicate_record` 新增 `seat_a` 显式参数（未给出时才回退旧推断）。
- **测试**：`tests/test_gate_adjudication.py` 新增"同名 spec 必须正确归属两个座位方向"（修复前 `as_second` 恒 0）与 `_score_at_seat` 语义断言；门控相关测试 16 项通过。
- **下一步（需重测）**：用修复后的打分器重跑批次 2/批次 3 的模型对比与 sims 敏感性（n≥200），再据此判断 Value 校准、池权重、对手结构各自是否真有棋力影响；之后再谈批次 3 剩余的 sims 20→40 训练周期。

### 十、批次 3 第三项：学习率余弦衰减（opt-in）

- `lr_for_epoch(base_lr, ep, start_epoch, end_epoch, schedule)` 纯函数 + CLI `--lr-schedule {constant,cosine}`（默认 constant=已验证基线）；cosine 从 base_lr 余弦衰减到 base_lr/5。
- **与方案原文的偏离及理由**：方案写"1e-3 恒定 → 余弦衰减到 3e-4"，但批次 1 已据受控实验把 base_lr 从 1e-3 下调到 1e-4（1e-3 会毁 Value 校准）；在 base=1e-4 时"衰减到 3e-4"是**升** lr，与证据矛盾。故按同一意图改为"衰减到 base 的 1/5"（1e-4 → 2e-5）。
- 测试 4 项：constant 恒定、cosine 端点与单调不增、超范围轮次被 clamp、floor 比例可配。

### 十一、批次 4 第一步：推理热路径 `eval()` 短路（已验证逐位一致）

**先测后改**。cProfile 单局（301 手、sims=20、CPU）：墙钟 123.7s / 每手 411ms，其中 **NN 前向 63.3s（51%）**、`legal_actions` 15.7s（13%）、`encode_state_np` 8.1s（7%）；前向次数 6,268 = 每手 20.8 次（与 sims=20 吻合，**无冗余前向**）。但每次单状态推理都调用 `self.eval()`，而 `nn.Module.train/eval` 递归遍历全部子模块：实测 **369,812 次子模块遍历、约 8.7s 累计**（另叠加 `module.__setattr__` 4.8s），约占总墙钟 7-10%。

- 修复：`JunqiNet.train(mode)` 在 `self.training == mode` 时直接返回（`eval()` 转发到 `train(False)`），语义与 `nn.Module.train` 一致，仅跳过冗余遍历。
- **等价性证据**：新增 `scratch/nn_predict_baseline.py`（capture/verify，覆盖 24 个固定局面 + 6 组批量推理，含采样世界与重复计数通道）→ **24 单状态 + 6 批量组逐位一致 ✓**（比旧工具 `perf_baseline.py` 更贴近自博弈推理路径）。
- **提速证据**（同机同负载下前后对比）：单状态 16.236 → **12.494 ms/次（−23%）**；批量 4 条 28.305 → **22.848 ms/次（−19%）**。
- 测试：`tests/test_net_inference_mode.py` 5 项（重复 eval 只遍历一次并断言增量、train/eval 语义、直接切换子模块后仍可恢复、predict_* 仍强制 eval、eval/train 返回 self）。
- **下一步结构性提速（已量化头寸）**：批量 4 条推理折算 7.1 ms/条 vs 单条 16.2 ms/条 → **每状态快 2.3 倍**，而 MCTS 模拟主循环目前是「每模拟一次单状态前向」（`mcts.py` 选择/扩展阶段）。故下一步为叶节点批量推理（虚拟损失并行），预期自博弈吞吐 2 倍以上，从而把 sims 20→40~100。

## [2026-09-13] — P0/P1/P2 阶段：评测门控落地、hybrid_engine 定价重构（废除加性硬打分）、真实终局标签 Value 重训与搜索蒸馏管线

阶段归属：**P0（正确性与可复现性）**、**P1（复盘数据集和传统基线）** 与 **P2（行为克隆和搜索蒸馏）**，依据 [AI_TRAINING_AND_HUMAN_PLAY_PLAN.md](../AI_TRAINING_AND_HUMAN_PLAY_PLAN.md) 与 [AGENTS.md](../AGENTS.md)。未修改 `RuleConfig` 类定义与 APK 对齐常量；未触碰 `models/best.pt`（所有训练产物均为独立候选权重）。

### 一、P0：统计严谨的评测门控 `junqi/eval_gate.py`（新增）
- 针对"和棋密集环境门控失效（历史 0胜/0负/40和）"的根本性修复，方法学对齐 `reports/camp_race_report.txt`：
  1. **配对同牌**：每 seed 先后手各一局（同一副牌），消除先手结构优势（实测 +0.7 营）与发牌运气；
  2. **固定种子**：全部随机源由实验种子派生，可复现；
  3. **统计检验**：得分率 Wilson 95%CI（支持 0.5 和棋加权）、每 seed 合计得分配对 z 检验、**三元 SPRT 序贯检验**（W/D/L，fishtest 同款 LLR）；
  4. **终局原因拆分**：flag/no_capture/repetition/immobilized 分列，防止"循环和棋减少"被误读为棋力提升；
  5. **晋级判据**：得分率 Wilson 下界 > 0.5 且 SPRT 不接受 H0；
- CLI：`python -m junqi gate --a p4hybrid --b expert2 --seeds 100`（正式晋级建议 ≥100 组 = 200 局）；
- **端到端验证**：`gate p4hybrid vs expert2`（6 组种子 × 先后手 = 12 局，max_plies=250）全链路跑通——配对/座位互换/终局原因拆分（max_plies 9、repetition 1、no_capture 1、immobilized 1）/配对 z/SPRT/晋级判定均正常输出，小样本下正确判 `promote=False`（0胜11和1负，得分率 0.458，无证据不晋级即门控应有行为）；报告见 `reports/gate_p4hybrid_vs_expert2.json`；
- 测试：`tests/test_eval_gate.py`（Wilson 教科书值、配对 z、SPRT 单调性与三态判定、端到端冒烟）。

### 二、P2：`junqi/hybrid_engine.py` 定价重构（复盘报告根因 2.1/2.2/2.3 的落地修复）
- **废除全部加性硬打分常数**（原 `score ±= 15/25/50/80/100/150/300/350/400/500/5000/10000`，跨 4 个数量级淹没 Policy 概率）；
- 新定价公式：`total = (1-prior_weight)·QSearch战术估值 + prior_weight·clip(log π(a)·5, -15, 0)`：
  1. **战术估值**由 `ExpertSearchEngine`（Star1 期望极大极小 + QSearch 吃子截断，教师同源）对候选动作做浅层前向搜索定价，彻底消灭 1-ply 反杀检测的地平线盲区（军长撞炸弹、炸弹炸雷等 2~3 步战术可见）；
  2. **先验有界化**：`prior_band=15` 分 < 最小子力单位"排长 18 分"，先验只做近平手动作的平局打破器，**永不翻转搜索已定价的真实子力摆动**（量纲崩溃的核心防线）；
  3. **规则层只保留硬判定**（对齐报告 §4.1）：一步吃旗终局捷径直接返回、死锁和棋前置拦截、重复局面根节点规避（REP_PENALTY=150，与 ai.py 同口径，价值量纲）；
  4. 候选集 = 策略 Top-K（默认 12）+ 全部吃明子走法；多世界 Value 头仅负责胜/和/负概率报告；
  5. **`search.py` 修复**：`ExpertSearchEngine.search` 新增 `as_evaluator` 参数——作为子节点估值 oracle 时跳过"唯一合法走法直接返回 0 分"的决策捷径，返回强制应着的真实搜索分（修复前确定化子局的必胜/必败线会被误估为 0，如炸弹自爆后一方无子可动的终局线）；
- `selfplay.py` 新增 `p4hybrid` 策略接入 `make_strategy`，可用 gate 实测；
- 测试：`tests/test_hybrid_tactical_pricing.py`（军长撞弹火力圈被压制、行营小子出营拆弹涌现为最优——含"炸雷后工兵拔旗"的强制拆弹局面，深度 4 ply 验收）。

### 三、P2：`junqi/train_search_distill.py` 搜索蒸馏管线（新增）
- 教师（QSearch 专家搜索）在采样局面 + 固定评测集上产出根节点软分布 `softmax(root_scores/T)`，交叉熵蒸馏给 Policy 头（主干/价值头冻结）；仅公共状态，无暗子身份泄漏；支持多进程打标；
- CLI：`python -m junqi distill_search --states 1200 --depth 3`；
- 小规模实证（600 局面 × 4 epoch，T=120）：loss 5.28→4.21，val_KL 2.63→2.03（均匀分布基线 ≈4.8 nats），管线有效；正式蒸馏需扩规模（≥1200 局面、更多轮数，可评估解冻末端残差块）；
- 测试：`tests/test_search_distill.py`（软分布归一化/合法掩码/制胜分饱和、端到端冒烟）。

### 四、P1：真实终局标签 Value 头重训 `train_value_from_p1_dataset`（新增）
- 数据源：`datasets/p1_v2` npz（官方 `list.cfg` 解密终局码真值，按对局切分带哈希；`has_values` 过滤早期强退 code 20）；仅训练价值头；CLI：`python -m junqi distill_value --p1-dir datasets/p1_v2`；
- **实证结论（重要）**：64585 训练样本（来自 668 局，标签按 ply 复制）训练 8 epoch：train_acc 97.9% 但 **val 三分类准确率仅 44.7%、RMSE 0.937（比全预测和棋的 0.738 基线更差）**——价值头在记忆 668 个唯一局级标签而非学习泛化规律。**定量证实基线计划 §1 的判断：仅 800 局真实终局标签不足以支撑可泛化 Value 头**；该候选（`models/value_distilled_v2.pt`）仅可作热启动初始化，**不得用于生产定价或覆盖 best.pt**（P4.4 Value 健康度门控不会通过）。主 Value 信号应走搜索蒸馏伪标签（数据量随算力扩展）与自博弈终局回报。

### 五、既有测试契约更新（先改测试后改实现的合规记录）
- `tests/test_camp_topology_fuzz.py` 按新契约重写：废除"弃营吃子绝对分值 < -150"断言（量纲崩溃产物）；修正旧用例战术前提错误（"连长反杀工兵"在 `battle()` 下为 defender_wins，不构成反杀；伏兵不得放 row0/row11 冻结位）；新增**普遍成立的地平线修复不变式**：同一出营吃子动作在有真实反杀伏兵时的定价必须显著低于无伏兵对照（10 行营 × 3 组合全遍历）；
- 背景说明：原"绝不吃诱饵"不变式不普遍成立——诱饵紧邻行营时任何非吃子走法同样丢营（诱饵下手进营），吃诱饵被反杀常为合理最小损失；原测试靠 -150 硬扣制造了通过假象。

### 六、搜索热路径性能优化（P0/P1/P2 共用基础设施）
- **现状查证**：C++ 内核 `junqi_core`（实测 2.7ms/步）目前仅接入 `ApkNativeAgent`（APK 复刻假想敌）；自对弈教师 `expert2` 走纯 Python `ExpertSearchEngine`（~130ms/步），两者未共用；
- **cProfile 剖析**（seed=102 全局，140 手）：91% 耗时在 `evaluate_expert`（49.6 万次调用），调用量放大器是翻棋几率节点的解析期望——内层每个翻棋候选触发 ~24 种翻子结果的全量估值；
- **已落地（语义逐位等价，快照比对 EQUIVALENT：280 状态估值/死和/fortress + 30 局深度 2 搜索逐位一致）**：
  1. `rules.battle` 查表矩阵（矩阵由原分支实现逐一构建；battle 单局 ~870 万次调用）；
  2. `fortress_score` / `is_dead_draw` / `_get_alive_counts` 实例缓存（座位色元组 + quiet 守卫；GameState `__slots__` 扩展）；
  3. `evaluate_expert` 明子划分单趟化（聚合梯队计数，消 6 个重复遍历生成器）+ 热循环 `board.get` 局部绑定；
  4. 实测局时长 75.0s → 73.0s（约 3%）——**诚实地低**：估值大多发生在互不相同的状态上（翻棋几率节点每次展开 ~24 个新子局面），实例缓存命中不了，等价优化空间已到顶；
- **试验并回退（重要教训）**：内层节点翻棋候选收紧 2/3 可得 58s/局，但深度 3 下用户复盘验证的战术场景（test_camp_tactics_fix 第 19 手师长推进）决策翻转，按验收测试回退；3/5 折中（70s）因仅多 4% 收益且引入行为差异面同样回退。**教师决策质量优先于延迟**；
- **大幅降耗的真实杠杆（按性价比排序，均需立项）**：
  1. 把 `ExpertSearchEngine` 语义（Star1/QSearch/TT）移植 C++（src_cpp 构建体系已就绪，预估 20-50x，教师打标/评测全线受益）；
  2. 吞吐而非延迟：自对弈数据生成用 `workers` 并行（已线性扩展，8 worker ≈ 390 局/小时）；
  3. 自博弈陪练侧换 C++ ApkNativeAgent（50x，裁定 4:14 弱于 expert——作对手池多样性/陪练，不作教师）；
  4. 翻棋期望的跨搜索记忆化（以 (position_key, quiet, ply 阈值) 为键的跨手缓存，需严防键不完整导致估值污染，属精确优化但正确性风险需专项测试）；
- 基线快照工具留存于 `scratch/perf_baseline.py`（capture/verify 两模式），供后续任何搜索优化做等价性比对。

### 七、P3 数据质量改造（执行计划 0→1→2 落地，基线计划已同步修订）
- **背景**：专家对局实测约 75% 为无吃子判和，自对弈 z=0 样本无学习信号；"快速产出高价值数据"取代"继续加打分限制"成为核心任务（加性打分路线已在第七节关闭）；
- **0. 生成/评测夹具分离**：`train_rl.GENERATION_CFG`（no_capture_draw_plies=120、repetition_draw_count=4）仅用于自对弈生成，**评测门控/靶场保持官方 70/1000/循环 3 规则**；和棋局 quiet ≥ 60 的尾部样本段不入回放池（`_drop_draw_tail`，决胜局全保留）；课程采样（残局生成 + 中盘注入）沿用既有挂钩；
- **1. 自博弈认输机制**：`MCTS.search` 新增第 5 返回值 root_value（根走子方视角期望，子节点访问量加权）；`ResignTracker` 判定走子方根 Value ≤ −0.95 连续 8 次己方回合（ply ≥ 40）认输，按官方 **code 21 语义**记 ±1 终局标签——终局奖励定义不变，非中间奖励；回滚 = resign_enabled=False。基线计划 P3 节与 §6 Value 规则已同步修订（原因/影响/验证/回滚齐备）；
- **训练主循环新增对局质量观测**：每轮打印决胜率、平均局长、终局原因分布（`对局质量 | 决胜率: ...`），作为 P4.4 健康度证据链的一部分；
- **2. 蒸馏规模化**：`distill_search --states 20000 --workers 8`（教师打标与策略头蒸馏，见第三节管线）；实测 val_KL 0.60（600 局时 2.03），候选 `models/search_distilled_20k.pt`；
- **首轮 500 局复盘（Epoch 2-6）与回滚**：决胜率稳定 60-69%、认输自增强 2%→27%；但 Value 三分类准确率 12%→58%→16%→28% 剧烈失稳，且 Epoch 2（认输仅 2%）已崩——主因判定为 buffer 新旧标签分布冲突（旧和棋标签 vs 新决胜标签），认输为加剧因素；门控 5 轮 0.406-0.531 无趋势，未晋级。**已执行回滚开关**：`--no-resign`（CLI 旋钮，穿线 run_training→worker→play_selfplay_game）+ `--fresh` 清空受污染 buffer，从 value_distilled 热启动点重校准；验收线 MAE<0.5 / 三分类>50%，恢复后再以更严口径（对方视角互证）重开认输；
- 验收测试：`tests/test_resign_fixture.py`（ResignTracker 计满/清零/min_ply/分座位/None 处理、GENERATION_CFG 与官方规则隔离、尾部过滤、stub 网络认输集成、mcts 根 Value 返回）；`mcts.search` 返回值升级为 5 元组，全部调用方（ai.py/train_rl/各测试）已同步；
- **实验链完整记录与修正后诊断（2026-09-14）**：首轮 500 局（决胜率稳定 60-69%、Value 校准崩坏、门控无趋势）→ 对照实验 A（关认输 + fresh 重校准达标）→ 验证实验 B（认输开启 + 认输局样本只进 Policy 池，Value 仍崩、认输率仅 2%）→ **"认输投毒是主因"假设被推翻**：高认输率本身是 Value 头过自信且错误的病症（该验收标准已撤回），真因判定为联合 RL 训练下主干特征漂移摧毁 Value 校准（buffer 内低 loss vs 靶场 16% 的过拟合特征）。认输局样本只进 Policy 池的隔离措施正确且保留。**待执行修复**：每轮 Value 重锚（冻结主干短微调，验收线 MAE<0.55/三分类≥45% 连续 2 轮）。全部数据表、被推翻的假设与经验教训见 [SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md](SELFPLAY_DATA_QUALITY_EXPERIMENTS_20260914.md)。

### 八、冲突与风险记录（AGENTS.md §9 合规）

1. **口径冲突已修复（2026-09-13 第二轮）**：`dataset.py` 此前将 code 24（断线）无条件归入"明确胜负"，与基线计划 §6 冲突。已抽取纯函数 `terminal_label_from_meta` 作为唯一标签判定口径：明确胜负仅含 code 1/21/22/23；**code 24 断线与 code 20 强退同为中止事件，一律不赋 Value（仅保留 Policy）**；40/42/43 及官方记 w=3 为正规和棋。新增 `tests/test_dataset.py::test_terminal_label_from_meta_codes` 与 `test_meta_code24_disconnect_gets_no_value` 验收；修正后数据集重导出为 `datasets/p1_v3`（影响约 2.3% 超时/断线局中的断线部分，此类局保留约 9.5 万条 Policy 样本但退出 Value 训练）；
2. **测试契约冲突已修复**：`test_camp_topology_fuzz.py`（详见第五节）；`test_replay_review_fixes.py` 四条棋理断言同步更新为搜索定价语义（胜势吃子恒为正、司令反杀线折价、弃营挖雷深度 3 反驳线、炸弹自爆净负、自杀攻击由硬掩码拦截）；
3. **训练状态**：P4.4 长期挂机准入仍未通过（Value 健康度、对手占比日志、自动熔断未达标），本次全部产物均不改变该结论；
4. **GUI 延迟**：hybrid_engine 定价重构后单步含候选搜索（默认 tactical_depth=2），开局多翻棋候选时约 0.1~0.5s，在报告 §5.1 延迟预算内；如需进一步压缩可调低 `candidate_k` 或 `tactical_depth`。

---

## [2026-09-13] — P1/P2/P4 阶段：实战对弈全景复盘、底层病灶根因剖析与根本性改良方案多方审查发布

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）**、**P2 阶段（真实 Value 监督与搜索蒸馏）** 与 **P4 阶段（混合智能与在线多世界决策，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。本次为**实战复盘、根因剖析与架构演进方案多方审查**，未修改 `RuleConfig` 类定义与 APK 对齐常量。

### 一、实战对弈全景复盘与关键缺陷定位
1. **人机 40 手基准对弈（Seed: 884811）**：验证纯自然语言大模型因缺乏前向博弈树模拟与空间几何拓扑计算的根本局限；
2. **三引擎对抗对局分析 (`expert` / `p4_hybrid` / `apk`)**：
   - `p4_hybrid`：第 141 手军长直撞明炸弹白兑；第 150 手后行营驻军受限于硬编码出营扣分，拒不出营拦截铁路逼近的炸弹，导致防空虚化、地雷被炸、军旗失守；
   - `apk`：第 23 手红排长失误送吃却因根节点货币脱节（翻棋静态启发分远高于吃子）而放过吃子；
   - `expert`：单子无意义往复折返跑步数达 27.8%。

### 二、底层病因剖析与现代棋类 AI 范式对比
- 确认当前引擎瓶颈在于**“代码强制硬打分（量纲淹没 Policy）”**与**“1-ply 浅层反杀检测导致的地平线效应”**；
- 深入横向对比 Stockfish NNUE（QSearch 截断消灭地平线效应）、AlphaZero/KataGo（零中间奖励、MCTS 回溯战术自然涌现）及微软 Suphx（PIMC 信念展开与搜索反哺蒸馏）。

### 三、系统重构落地路线（详见交付文档）
- **交付文档**：[docs/REPLAY_ANALYSIS_AND_SYSTEMIC_IMPROVEMENT_PLAN_20260913.md](REPLAY_ANALYSIS_AND_SYSTEMIC_IMPROVEMENT_PLAN_20260913.md)
  1. 激活破译的官方 `list.cfg` 800+ 真实终局标签训练三分类胜率 Value 头，剥离代码手工算术加减分；
  2. 实现轻量战术静态截断搜索（QSearch）与 PIMC-MCTS 多步树搜索；
  3. 构建“搜索作为教师”的反哺蒸馏管线，推动 Policy 自进化。

---

## [2026-09-12] — P1/P4 阶段：Δ（占营数量差）战术研究 + 配对设计缺陷修复与两项结论撤回

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P4 阶段（官方 APK 逆向假想敌基准对弈，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。本次仍为**纯分析与验证**，未修改 `RuleConfig` 类定义/默认值、`eval_apk_pure`、`eval_apk_flip_root` 或任何 APK 对齐常量。

### 一、方法论缺陷修复（本轮最重要的自我更正）
上一轮的座位互换用**不同牌局**（`seed0+g*2+half`），导致配对差里残留 `0.5·[Adv(S_even) − Adv(S_odd)]` 的牌局级共同模噪声：n=400 时 SE≈0.10 营，**与要测量的战术效应（0.05~0.30 营）同量级**，会把噪声读成效应。现改为"**同一副牌局交换座位各跑一场**"的完美配对设计（`race_batch` / `delta_batch`），共同模精确归零。

**有效性验证**：新增 §15「关键效应跨 7 个种子段重复」，15 个效应的**段间 SD / 单段 SE 全部 ≤ 1.01**（多数 < 0.6），说明段间波动完全由抽样误差解释，配对设计成立。池化把 SE 从 0.11~0.14 压到 **≈0.052 营**，可分辨下限 |Δ| ≈ 0.11 营。

**据此撤回两项已发布结论**：
1. ❌ 撤回「后手另辟营簇(split) 把先手优势从 +0.777 压缩到 +0.520（降低 33%）」——更正后四种反制下的 seat 视角先手优势为 **+0.820 / +0.810 / +0.863 / +0.800**（极差仅 0.063），**反制无法压缩先手优势**；
2. ❌ 撤回「卡位进营(denial) 净收益 +0.211 营，z=+2.14 显著」——更正后为 **+0.0071 ± 0.0545 营，z=+0.13，不显著**。
3. 🔄 修正「首翻打底线死位少 0.613 营」为池化 **+0.5050 ± 0.0526 营（z=+9.60，95%CI [+0.402,+0.608]）**，方向不变、精度提升。
4. 🔄 修正「补齐 (2,2)/(9,2) 差 −0.033 营」为池化 **−0.0693 ± 0.0520（z=−1.33）**，仍不显著 ⇒ 维持不改 `CENTER_CAMP_FLIP_POSITIONS`。

另修复 `sign_flips`（领先权易手）指标定义：Δ 每步只变 ±1，易手必然经过 0，原实现只比相邻两手会**完全漏计**（+1→0→−1）。修复前恒为 0.000，修复后实测 **0.81 次/场**（分布 0 次 44% / 1 次 33% / 2 次 18% / ≥3 次 5%）——竞赛过程实为反复拉锯，而非一边倒。

### 二、Δ 战术研究（新增 `section_delta_tactics` + `section_effect_replication`）
新增 `DeltaPolicy`（5 个正交维度：`capture_scope` / `capture_judgment` / `outstrike` / `camp_pref` / `sacrifice` / `exposure` / `flip_pref`）与 30 个策略配置，**29 组对抗 × 800 场 = 23200 场**，另加 **15 个关键效应 × 7 种子段 × 400 场 = 42000 场**池化重复。记录 Δ（均值/SD/SE/配对 z/95%CI）、T_fill、吃子数、行营扑杀占比、被吃子力等级分布、损失价值、|Δ|≥2 达成前的吃子数、领先权易手次数。

**规则可行性探针（§14.0，用项目默认 `RuleConfig()` 构造验证）**：
1. **用户假设「主动送死低价值子换敌高价值子」在规则层不可行**：`allow_suicide_attack=False` 使排长撞司令、工兵撞军长、连长撞师长（均 `defender_wins`）**根本不进入 `legal_actions`**。合法牺牲只有 3 种：炸弹同尽、同衔相撞、走入敌杀区。
2. **行营单向扑杀特权确认**：营内连长可打营外排长；营外排长打营内连长**不合法**（`state.py::_attackable` 对 `is_camp(tpos)` 直接返回 False）。

**核心发现（5/15 效应池化后显著）**：
1. **「吃子挤出效应」**：合法吃子机会并不罕见（低接触 2.19 次/场、占回合 18.9%；强制接触 16.9 次/场），但 `AG_anywhere`（有吃子机会就吃）**实际吃子 0.00 次、转化率 0%** —— 进空营是优先级 1，在 10 营未填满前几乎每回合都有空营可进，吃子被完全挤出。即使放开到"接受全部合法交换"，转化率也只有 15.7%。**唯一能落地的吃子是行营扑杀**（0.41 次/场，**99.1% 源自行营内**），因为它用的是另一枚子，不与进营竞争。
2. **行营扑杀的 Δ 收益随接触密度变号**：低接触 **+0.2871 ± 0.0554（z=+5.18）**；高接触 **−0.0993 ± 0.0533（z=−1.86）**。机理：出营吃子=弃营（该营变空，需再花 1/ρ≈3.6 ply 占回），低接触时每场仅 0.41 次故净赚，高接触时 4.3 次故净亏。⇒ 可执行判据：**每场出营 ≲1 次时才启用行营扑杀**。
3. **「诱骗/暴露」对 Δ 完全无效**：暴露高价值子 +0.0193、低价值子 +0.0107、随机 +0.03（全部 |z|<0.7），实测吃子 0.00、双方损失价值 0（暴露未诱发出任何交换）。双重根因：**明子身份是公共信息**（`position_key` 只对暗子隐藏身份），且 `allow_suicide_attack=False` 使对手无法"上当撞大子"。对"贪交换"缺陷对手同样无效。**真正有收益的是自己接受全部合法交换：+0.2354 ± 0.0522（z=+4.51）**。
4. **炸弹是 Δ 目标下最高效的交换工具**：炸弹同尽 **+0.1929 ± 0.0525（z=+3.67）**，己方损失 146 价值换敌方 671 价值（净 **+525**）；`CT_all_legal` 己方损失 654（主要 ZHA 422 枚）换敌方 1021（净 +367）。**没有任何战术需要牺牲高价值子来换 Δ**。
5. **关键转折点：Δ 的拉开与吃子无关**。|Δ|≥2 达成率 96~98%，而**达成前平均吃子数 = 0.00**（跨 3 种子段一致）。故不存在"第 N 次吃子导致 Δ 跳变"——单次占营只让 Δ 变 ±1，吃子不直接改变 Δ。
6. **后手四种反制对 Δ 的净收益全部为 0**（池化 +0.0071 / −0.0171 / +0.0050 / 单段 +0.047），但**代价高度显著**：限制翻棋营簇慢 2.86~3.69 ply（+12.6%~16.3%），卡位(denial)零代价。⇒ **后手最优就是与先手同策略**。

### 三、交付物
- **文档**：[CAMP_RUSH_OPENING_PLAYBOOK.md](03-RulesAndStrategy/CAMP_RUSH_OPENING_PLAYBOOK.md) 新增 **§7 占营数量差 Δ 的最优战术路径**（7.1 规则可行性探针 / 7.2 吃子挤出效应 / 7.3 Δ 战术效应总表 / 7.4 Δ–T_fill 权衡矩阵散点图 / 7.5 诱骗为何无效 / 7.6 关键转折点 / 7.7 池化重复与配对有效性 / 7.8 最佳 Δ 策略决策树 / 7.9 结论摘要），原 §7~§10 顺延为 §8~§11；战术纪律由 D1~D9 扩展为 **D1~D14**；§0 新增 C13~C18 与「方法论更正」框；§4.3/§4.5/§6.2 按更正后数据重写并附撤回声明。
- **脚本**：`scripts/analyze_camp_opening.py` 新增 §14（规则探针 + Δ0~Δ6 共 29 组对抗 + `print_delta_scatter` 权衡矩阵 + Δ 排名）与 §15（`section_effect_replication` 池化重复）；`race_batch`/`delta_batch` 改为完美配对；`run_delta_race` 新增 `camp_origin_caps`（严格"源自行营内"）与 `camp_zone_caps`（行营邻域）分离口径、吃子挤出转化率、领先权易手（修正定义）；新增 `--delta / --delta-races / --replicate-races / --delta-log / --delta-json` 参数。
- **测试**：`tests/test_camp_opening_strategy.py` 由 28 项扩展到 **38 项**（新增规则可行性、行营单向扑杀、吃子挤出效应、领先权易手、Δ 与吃子无关、**吃子不必然增加 Δ**、诱骗无效、牺牲被规则阻断、阻挠纯亏速度、Δ 竞赛记录完整性等）；全量测试 **225 通过 / 3 跳过**，与 `test_p4_pure_apk_alignment.py`、`test_p1_camp_shuttle_fix.py` 零冲突。
- **报告**：`reports/camp_delta_report.txt`（人读）与 `reports/camp_delta_report.json`（机读：`strategies` / `key_effects_replication` / `results`）。

## [2026-09-12] — P1/P4 阶段：开局占营策略赛制重构（10 营占满即停）与旧实验资产清理

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P4 阶段（官方 APK 逆向假想敌基准对弈，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。本次仍为**纯分析与验证**，未修改 `RuleConfig` 类定义、`eval_apk_pure`、`eval_apk_flip_root` 或任何 APK 对齐常量。

### 资产清理
- **删除**：`scripts/exp_camp_first_opening.py`、`scripts/diag_apk_blind_flip.py`，以及 `reports/camp_opening_experiment.txt|.json`、`reports/diag_blind_flip_beginner.txt`、`reports/diag_blind_flip_advanced.txt`、`reports/camp_opening_analysis.txt`。
- **保留并扩展**：`scripts/analyze_camp_opening.py`（现为唯一的分析 + 竞赛脚本，13 个章节）、`tests/test_camp_opening_strategy.py`。
- 上一轮完整对局赛制得出的结论已**归档**至 `CAMP_RUSH_OPENING_PLAYBOOK.md` §9（含官方引擎诊断数据、双轨对齐校验、`has_camp_entrance_opportunity` 地雷/军旗边界缺陷），其中与几何/常量相关的部分改由构造性单元测试固化，不再依赖已删除的脚本。

### 赛制变更
不再模拟完整对局（旧赛制平均 210 ply、95.5% 判和、信号被稀释），改为 **第 10 个行营被占据的瞬间立即终止**，产出三个正交指标：`占满手数 T_fill`、`先手/后手占营数`、`以占营数判胜负的胜率`。竞赛使用 `RuleConfig(no_capture_draw_plies=0, max_plies=10**9)` 测量夹具——`config.py` 已注明 `0=关闭`，属构造夹具而非修改项目规则。

### 新增核心结论
1. **中心营必须最先占（本轮最强发现）**：中心营 `(3,2)/(8,2)` 只有 4 个非营入口（全是黄金位），角营有 7 个。先占角营会把中心营**锁死**，稳态占营速率 ρ 从 **0.2796 塌缩到 0.0615 营/ply（4.09×）**，占满 10 营从 **22.64 ply 拖到 85.83 ply（3.79×，两样本 z=+28.2）**，完成率从 100% 降到 98.5%，并额外产生 0.342 次/场弃营。
2. **"邻接暗子最多的营优先"是陷阱**：该规则隐式等价于角营优先（角营 7 个非营邻居 vs 中心营 4 个），实测 85.39 ply，与角营优先同级。辐射潜力的正确度量是"是否为簇内枢纽"，不是"邻接暗子数"。
3. **"占营/拓荒交替"是负担而非优化**：38.55 ply（比"有营就进"慢 68%，z=+72.9）且占营数不变（5.317 vs 5.343）。全因子 24 配置中 `alternate=True` 在每个组合下都降低先手占优率。
4. **10 营分配不是 5:5**：跨 7 个独立种子段（各 400 场）先手 **5.368** : 后手 **4.631**（优势 **+0.737 营**，配对 z=+6.0~+8.4）。先手 **≥6 营率 46.2%** vs 后手 **26.0%**（**1.77:1**）；5:5 仅占 27~33%。
5. **「第 10 营由谁拿下」不能作为先手优势判据**：均值 **50.4%**（7 段极差 4.25pp），接近抛硬币。此前列为结论的该指标已撤回。
6. **后手反制的重要负结果**：`follow`/`split`/`denial`/`split+denial` 四种反制的净占营收益**全部落在噪声内**（\|配对差\| ≤ 0.09 营，\|z\| < 1）。但**代价差异高度显著**：限制翻棋营簇（follow/split）使整场慢 2.7~3.5 ply（+12%~16%），而 `denial` 只改"进哪个营"不改"翻哪个位"，**速度代价为零**（22.65 vs 22.64）。⇒ 后手应首选 `denial`。
7. **修正闭合模型**：上一轮粗模型把"首营成本"外推到 10 营，低估先手优势 37%。修正为 `T_fill = C/(2ρ) + (T₁+T₂)/2`、`先手营 = ρ(T_fill−T₁)`、`后手营 = ρ(T_fill−T₂)`，其中 ρ 由基准行**直接测量**（非拟合）。预测误差：**占满手数 +1.6%、先手营 +0.9%、后手营 −1.1%**。
8. **首翻位价值在竞赛赛制下被放大**：首翻打底线性度 0 死位少 **0.613 营**（z=+5.50，旧完整对局赛制下仅 0.265 营）；补齐 `(2,2)/(9,2)` 仍无实质收益（−0.033 营，z=−0.29）⇒ 维持不改 `CENTER_CAMP_FLIP_POSITIONS`。
9. **种子稳定性分级**：`占满手数` 跨种子波动仅 **0.4%**（极稳，本方案全部速度结论建立于此）；`先手/后手占营` 4.0%/4.7%（可用）；`Δ 首占营性能差` **20.5%**（高波动，对照模型须给 ±0.6 ply 容差，单种子段结论不可用）。

### 交付物
- **文档**：重写 [docs/03-RulesAndStrategy/CAMP_RUSH_OPENING_PLAYBOOK.md](03-RulesAndStrategy/CAMP_RUSH_OPENING_PLAYBOOK.md)，新增棋盘标注图（G/g/@/O/./H）、黄金位↔行营辐射覆盖表、占营曲线 ASCII 条形图、先手/后手策略决策树、24 配置全因子扫描表、种子稳定性表，战术纪律由 D1~D7 扩展为 **D1~D9**。
- **脚本**：`scripts/analyze_camp_opening.py` 新增 §9 全因子扫描、§10 定向对比（T1 先手顺序 / T2 后手反制 / T3 占营优先级 / T4 单变量）、§11 占营曲线与典型占营顺序、§12 种子稳定性核查、§13 可视化输出；新增 `--races / --stability-races / --quick / --log / --json` 参数。
- **测试**：`tests/test_camp_opening_strategy.py` 由 17 项扩展到 **28 项**（移除 3 项对已删除脚本的依赖，新增 14 项竞赛制测试）；全量测试 **215 通过 / 3 跳过**，与 `test_p4_pure_apk_alignment.py`、`test_p1_camp_shuttle_fix.py` 无冲突。
- **报告**：`reports/camp_race_report.txt`（人读，13 章节）与 `reports/camp_race_report.json`（机读：model / targeted / curve / stability / scan）。总样本约 **25000 场**竞赛（全因子扫描 14400 + 定向对比 7200 + 曲线 600 + 稳定性核查 2800）。

## [2026-09-12] — P1/P4 阶段：开局"快速占据行营"最优策略系统性探索与验证（纯分析，未改动引擎）

> **注**：本条目所列 `scripts/exp_camp_first_opening.py`、`scripts/diag_apk_blind_flip.py` 与 4 份报告文件已在同日的赛制重构中按需求清理，结论归档于 `CAMP_RUSH_OPENING_PLAYBOOK.md` §8。

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P4 阶段（官方 APK 逆向假想敌基准对弈，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 交付物
- **策略手册**：新增 [docs/03-RulesAndStrategy/CAMP_RUSH_OPENING_PLAYBOOK.md](03-RulesAndStrategy/CAMP_RUSH_OPENING_PLAYBOOK.md)，给出先手/后手独立开局方案与 7 条战术纪律（D1~D7），并逐项对照 `+150 / +120 / +60 / -40 / -200 / +20`、`CAMP_POSITION_BONUS(+50/+40)`、`camp_abandon_penalty(250000)` 等代码常量。
- **分析脚本**：`scripts/analyze_camp_opening.py`（几何拓扑 + 打分枚举 + 闭式期望模型 + 蒙地卡罗校验）。
- **实验脚本**：`scripts/exp_camp_first_opening.py`（13 组策略 × 800 局 + 4 组官方引擎对照 × 200 局，双视角 seat/policy 统计与配对 z 检验）。
- **诊断脚本**：`scripts/diag_apk_blind_flip.py`（根节点仲裁常量分解、分数货币核对、行营扑杀占比、Python/C++ 双轨分数级对齐）。
- **回归测试**：新增 `tests/test_camp_opening_strategy.py`（17 项），全量测试 **204 通过 / 3 跳过**，零回归。
- **原始报告**：`reports/camp_opening_analysis.txt`、`reports/camp_opening_experiment.txt|.json`、`reports/diag_blind_flip_beginner.txt`、`reports/diag_blind_flip_advanced.txt`。

### 核心验证结论
1. **开局机动性定理（0 反例）**：满盘暗子时，一枚刚翻开的明子的合法非吃子走法数**恒等于**其相邻空行营数（邻营度），且每条都指向行营。邻营度分布为 度3=8位 / 度2=8位 / 度1=24位 / 度0=10位（邻接边共 64 条）；`row0`/`row11` 的 10 个度 0 底线位翻出的子**完全冻结**。
2. **营簇星形拓扑**：10 营分上下两簇，每簇为"中心营—4 角营"星形（角营互不相邻、只连中心营），由 4 个度 3 黄金位辐射并**完整覆盖该簇 5 营**；8 个黄金位**两两不相邻**，各自唯一的非营邻居恰为 8 个度 2 位。此结构解释了 `CAMP_POSITION_BONUS` 中心营 +50 > 角营 +40 的几何依据。
3. **占营竞赛闭式模型与实测吻合**：可动子 21/方（25−3雷−1旗），无放回首次命中期望 `E=(n+1)/(k+1)=50/22=2.2727`（递推精确验证），推出 `E[T_first]=3.727 ply`、`E[T_second]=6.545 ply`、结构性能差 `Δ=2.818 ply`、10 营耗尽时先手 `(C+Δr)/2=5.215` : 后手 `4.785`。5 组 800 局对称实验实测 Δ 均值 **2.824（误差 0.2%）**、占营差均值 **0.444（误差 3.0%）**。
4. **首翻位权重远高于后续翻棋位**：单变量实验显示"首翻改打底线性度 0 位"使首占营慢 1.6 ply、终局少 0.265 营（配对 z=+2.06，显著）；而"全程改打度≤1 远端位"仅慢 0.8 ply、终局差 +0.010（z=+0.08，不显著）——因为度 1 位仍可一步进营，度 0 位的子永久冻结。
5. **战术纪律量化**：弃营盲翻（有进营机会却翻棋）代价为终局占营 **7.63 : 2.28**（z=+35.4）；放开驻营子出营吃子后**多吃 1.56 子却少占 0.30 营**，先手优势从 z=+2.56 退化为 z=+1.52（失去显著性）。
6. **后手反制排序**：`另辟营簇(split)` +0.239 营（z=−1.87）> `卡位进营(denial)` +0.119 营（z=−0.94）> `同簇卡位(contest)` +0.124 营（z=−0.98）；三者互不叠加（contest vs split z=+0.11）。推荐组合：**split 选簇 + denial 选营**。

### 对既有实现的重要澄清（未修改任何代码）
1. **`-200` 弃营惩罚工作正常**：初版纪律计数把"有合法进营却翻棋"一律记为违规，测得官方引擎开局期每局约 7 次，疑似惩罚失效。原子化复算后推翻该假设——550 个实战采样局面中 **99~100% 属 `+120` 依托行营辐射拓荒（`flip−base` 实测恰为 +140.0，与 `+120+20` 精确吻合，且为 `test_07` 明确断言的设计行为）**，真弃营盲翻仅 0~1%（实测 `flip−base = −200.0`）。
2. **官方对齐引擎开局占营速率仅为占营优先策略的 ~50%**（ply30: 2.19~2.47 vs 5.25~5.36），根因是**根节点分数"货币"不同源**：翻棋分是 0-ply 静态启发（`base+140`），明子走法分是 depth-ply 极小化值（进营中位仅 `base+10.4`）。据此引擎系统性偏向继续拓荒而非去占第 2、3 个营；同时开局期每局仍有营间闲走 0.14~0.28 次、弃营 0.43~0.53 次。
3. **行营扑杀占比实测仅 19.2%（初级）/ 24.1%（高级）**，远低于方案文档引述的复盘统计 50.1%，直接由"据点密度不足"导致。
4. **`CENTER_CAMP_FLIP_POSITIONS` 几何不完备**：度 3 位实为 8 个，代码收录 6 个，缺 `(2,2)`、`(9,2)`（C++ 侧 `CENTER_CAMP_FLIPS={16,18,22,37,41,43}` 与 Python 一致，同缺）。但 8 位 vs 6 位的实战配对差仅 +0.033 营（z=+0.25，不显著）⇒ **按 AGENTS.md 硬约束不建议改动**，仅由测试固化现状。
5. **已定位低频边界缺陷**：`has_camp_entrance_opportunity` 遍历己方明子时未排除地雷/军旗（`junqi/apk_engine.py`、`src_cpp/src/eval_apk.cpp` 同），当己方唯一明子为雷/旗且紧邻空营时会误判"有进营机会"并施加 `-200` 误罚（构造性反例：`base=+40` 时远端翻棋得 `−160`）。实战采样 550 局面中出现 **0 次**，故本次仅以测试固化，不修改。
6. **双轨对齐须比分数而非比动作**：动作完全一致率仅 49.0%（初级）/ 53.3%（高级），但在容许 2×jitter 的分数等价判据下达 **99.8% / 100.0%**，证明差异全部来自同分候选 tie-break 与 jitter 扰动，估值与搜索逻辑双轨一致。

### 目标设定的重要修正
本次按"占营数为唯一胜负标准"执行验证，但同时给出反证：纯占营最大化把 **95.5%** 的对称对局拖入循环/限步判和；对官方引擎占营 **7.63 : 1.81** 碾压（占营判据胜率 88.5%），真实胜负却是 **4.0% : 5.5%**（反而略低）。故最终建议：**将"占据行营数量"作为开局期（ply ≤ 30）最高权重的过程指标，终局优化目标仍为 `state.winner`**——与 AGENTS.md"不得添加吃子、挖雷等中间奖励改变终局目标"一致。本次未据此修改任何评估权重或规则常量。

## [2026-09-08] — P1/P4 阶段：首翻子力占营据点优先与盲目远端翻棋惩罚对齐

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P4 阶段（官方 APK 逆向假想敌基准对弈，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心动机与对局实战根因
在实战对局 `games/game_20260907_234949.json` 中，AI 翻出首个己方明子（红排长）后，其相邻两处空行营均未被占领，但 AI 却视空营不见，持续在全盘远端盲翻暗棋，导致明子裸露 10 手后被敌方扑杀。深入排查发现：
1. `CENTER_CAMP_FLIP_POSITIONS` 的 +150 黄金位加分无条件贯穿全盘，压制了进营加分（+40）；
2. 博弈树内部节点（纯明子树）强制驻营子力出营送死，造成“进营估值塌陷”；
3. 缺乏“己方有未保护明子且近邻有空营时，重度抑制远端盲翻”的战术纪律。

### 核心改进与技术实现
1. **严格开局翻棋适用期**：`CENTER_CAMP_FLIP_POSITIONS` 的 +150.0 奖励严格限制在全盘无任何己方明子的开局首翻期生效，一旦翻出己方明子立即转入据点建立模式；
2. **战术纪律与远端盲翻抑制**：当场上有裸露在外的己方明子且近邻存在空行营时，所有未依托已占行营的远端盲翻一律重度扣减 200 分（`score -= 200.0`），确保 AI 优先进营建立防线；
3. **行营驻守机制 (Stand-Pat)**：在 Alpha-Beta 搜索深层，为驻营子力提供静态估值下界，消除由于不能展开翻棋而被迫走出行营导致的估值下溢；
4. **C++ 与 Python 双轨 100% 对齐**：同步更新 `junqi/apk_engine.py`、`src_cpp/src/eval_apk.cpp` 与 `src_cpp/src/apk_engine.cpp`，重新编译 MSVC 动态库，实测结果完全一致；
5. **回归测试**：在 `tests/test_p4_pure_apk_alignment.py` 中新增 `test_06`（对局 234949 复盘对齐）与 `test_07`（占营后依托行营辐射拓荒）。

## [2026-09-07] — P1/P4 阶段：C++ 核心引擎 (src_cpp) 基础设施建立与双轨热拔插网桥交付

阶段归属：**P1 阶段（传统搜索增强）** 与 **P4 阶段（官方 APK 逆向假想敌基准对弈，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心改进与技术实现
1. **建立高性能 C++ 原生引擎目录 (`src_cpp/`)**：
   - **紧凑内存布局 (`types.h`, `constants.h`, `board.h`)**：采用 60 字节连续内存数组 `cells[60]`，彻底消灭 Python 对象与堆分配开销，数据结构天然贴合 CPU L1 Data Cache；
   - **高性能走法生成器 (`rules.h`, `rules.cpp`)**：预计算公路与铁路正交邻接表，工兵铁路转弯采用栈队列广度优先算法，消除动态内存分配；
   - **1:1 原生 APK 极速搜索核 (`eval_apk.h`, `apk_engine.h`, `eval_apk.cpp`, `apk_engine.cpp`)**：将 2560 等比估值核、动态炸弹、行营偏置、PVS 零窗口探测、纯吃子 QSearch 与 Delta 剪枝全量 C++ 原生化；
   - **64 位 Zobrist 哈希与置换表 (`zobrist.h`, `zobrist.cpp`)**：实现确定性快速哈希更新与置换表高效探测；
   - **独立基准测试程序 (`tests/bench_main.cpp`)**：包含棋盘验证、吃子结算、搜索决策与百万次走法生成吞吐量测试。
2. **构建与绑定基础设施 (`CMakeLists.txt`, `setup.py`, `bindings/python_bindings.cpp`)**：
   - 编写标准 C++17/20 CMake 构建脚本与 `setup.py`，支持独立 CMake 编译和 `pip install -e .` 安装；
   - 编写 `pybind11` 接口导出 `junqi_core` 模块（包含 `JunqiBoard`, `ApkSearchEngine`, `RuleConfig` 等）。
3. **双轨热拔插网桥 (`junqi/core_bridge.py`)**：
   - 实现无缝透明降级机制：优先检测并载入 `junqi_core`，若环境尚未配置 C++ 编译器则自动透明降级到纯 Python 引擎，绝不阻断现有对弈、训练与测试。
4. **测试与保障**：
   - 新增 `tests/test_p4_core_bridge.py`，验证 C++ 源码树完整性、网桥状态探测与平滑降级决策。

## [2026-09-07] — P1/P4 阶段：极简纯净 APK 引擎 (ApkSearchEngine) 独立重构与 5 大差异 100% 对齐

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P4 阶段（官方 APK 逆向假想敌基准对弈，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心动机与逆向对齐目标
用户实测反馈 PC 端 APK 假想敌手感与手机原版仍有微妙差异。经深入逆向分析反汇编二进制（`libjunqi.so` 专有二人翻棋模块 `0x57000 - 0x5b000`），定位到 5 项关键差异并彻底还原：
1. **博弈树拓扑结构**：原实现将明子与暗子混合在全概率几率树中展开，而原版在树深层（`ply >= 1`）绝不递归展开暗子翻棋，仅在根节点独立分流启发评估，内部全为纯明子搜索；
2. **静态估值核复合特征冗余**：原估值混入了非 APK 的死区势能、火炮网、死锁和棋衰减等复合专家项，脱离了原版纯净物质等比表（2560 幂次）+ 60 格静态位 + 地雷护旗（+80）的简洁架构；
3. **判和限步规则偏差**：原规则无吃子限步判和设为 40 步，而官方标准为 70 步（`no_capture_draw_plies = 70`）；
4. **开局库偏好与随机扰动 (Jitter) 缺失**：手机原版开局首翻对中心 4 行营邻近的 6 个黄金格有强偏好，且各难度带有受控随机扰动；
5. **执行性能与 IDS**：极简静态估值大幅提速，且迭代加深在耗时达 25% 预算时安全早停，保障落地 target depth。

### 核心改进与技术实现
1. **构建极简纯净原生引擎模块 (`junqi/apk_engine.py`)**：
   - 建立 1:1 对齐 `0x59f90` / `0x124094` 的静态估值核 `eval_apk_pure`：纯净物质价值表（司令 2560 至排长 30、工兵 80、地雷 70、军旗 50）+ 动态炸弹（1/3 存活最高军衔）+ 60 格静态行营偏置（中营 +50，角营 +40）+ 地雷护旗防御加分（+80），彻底剥离复合专家特征；
   - 建立根节点翻棋启发评估 `eval_apk_flip_root` (0x5a3c0)：基础继承全盘明子估值，叠加中心 4 行营周围 6 个关键黄金暗子位偏好（`CENTER_CAMP_FLIP_POSITIONS` +150）与行营单向扑杀辐射加成（+120）；
   - 实现纯明子 Alpha-Beta 搜索 `ApkSearchEngine`：内部节点（`ply >= 1`）绝不生成翻暗棋，采用 PVS 零窗口探测与 Zobrist 哈希置换表（TT），叶子节点转入纯吃子 QSearch 与 Delta 剪枝；
   - 对齐官方三档难度与 Jitter 机制：初级 `beginner` (depth 2, 100ms, jitter ±30.0)、中级 `intermediate` (depth 3, 300ms, jitter ±10.0)、高级 `advanced` (depth 4, 1000ms, jitter ±0.5)；
   - 对齐 IDS 25% 安全早停机制（`r2 > r3 asr #2`）。
2. **假想敌代理底层全面切换 (`junqi/apk_agent.py`)**：
   - 将 `ApkNativeAgent` 内部引擎由 `ExpertSearchEngine` 切换为独立的 `ApkSearchEngine`，接口与行为 1:1 精准复刻手机手感。
3. **官方 70 步和棋规则对齐 (`junqi/config.py` 与 `configs/rules.yaml`)**：
   - 将 `RuleConfig.no_capture_draw_plies` 恢复并固定为官方标准值 70。
4. **自动化测试与对齐验证**：
   - 新增 `tests/test_p4_pure_apk_alignment.py`，全量覆盖 5 大对齐维度的独立单测；
   - 保持 `tests/test_p4_apk_agent.py`、`tests/test_p1_camp_shuttle_fix.py`、`tests/test_p1_quiescence_depth.py` 等全部测试通过。

## [2026-09-07] — P1 阶段：行营往复互窜根治与“首翻即据点、依托行营辐射拓荒”实战棋理校准

阶段归属：**P1 阶段（传统搜索增强与专家评估校准，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 问题根因与实证定位
在 `games/game_20260907_211452.json` 与 `games/game_20260907_211613.json` 复盘中，专家引擎与 APK 原生引擎在进驻中营（`(3, 2)` 或 `(8, 2)`）后，在相邻角营与中营之间无休止往复闲走（如 `(3,2) <-> (2,3)`），甚至无故出营闲走，拒绝翻暗棋推进：
1. **启发排序误判**：非吃子跨营移动被原版 `is_camp(act.to)` 误判为高优先级的“进驻空营”，赋予高达 `+250,000` 排序分，且逃脱了弃营惩罚；
2. **几率节点奇偶深度地平线截断偏差**：翻棋几率节点在单数展开时轮到对手行动，对手可立即利用刚翻出的敌子钻入相邻空角营掠夺 `camp_occ`（APK 高达 +100，默认 +10），而己方翻出友军子在该视界内因未轮到行动无法进营，导致翻棋期望被严重悲观化（翻棋期望暴跌至 -40 ~ -46，而营间闲走因免死保持在 0 ~ -2），引发“翻棋恐惧症”；
3. **单子跨营控制权多报虚胀**：原估值函数在计算 `empty_camps_score` 时，单个位于十字路口或营间的棋子同时为周围 3 个空营全额计分，导致脱离行营反而产生虚假势能；
4. **根节点战术确定性优先误伤**：根节点 `is_better` 的 `+0.5` 确定性偏置未严格限定于实质性吃子或进营，静步闲走在分值接近时强制压制翻棋；
5. **翻棋领地偏好对称性失真**：原领地偏好误将翻棋按红蓝 0..5 / 6..11 半场割裂，对另一半场翻棋惩罚 `-100,000` 分，抑制了全盘拓荒。

### 核心改进与技术实现
1. **走法排序精准定级与营间闲走严惩**：
   - 跨营闲走（`is_camp(act.frm) and is_camp(act.to)` 且无吃子）：明确判定为重度负优先级（`-200_000.0`）；
   - 从场外进驻空营（`not is_camp(act.frm) and is_camp(act.to)`）：保留 `+250_000.0` 战略进营特权；
   - 依托已控据点邻域辐射拓荒（`has_friendly_camp`）：翻开营周暗子赋予 `100_000.0 + rank_boost` 高优先级；
   - 领地偏好全盘对称化：中前场咽喉带与行营辐射带对称赋予正偏好，取消红蓝半场扣分。
2. **中营核心枢纽地位与据点辐射期权显式注入 (`junqi/eval_expert.py`)**：
   - 拥有 8 向通达辐射特权的中营 `(3, 2)` 与 `(8, 2)` 赋予 `0.85 * w.camp_occ` 额外核心枢纽加成，彻底消灭主动放弃中营窜往 4 向角营的伪收益；
   - 营内战斗子力对其周围暗子赋予每枚 `0.06 * w.camp_occ` 的就近单向扑杀期权；
   - 空营控制权引入单子独占匹配（`used_my_pieces / used_opp_pieces`），单一明子最多贡献单营控制，彻底杜绝脱营多报虚胀；空营势能严格折现为实占行营的 15%~25%。
3. **根节点决策确定性偏置严格边界化 (`junqi/search.py`)**：
   - 仅当走法属于【实质性吃子/战术制胜】或【从场外进驻空行营】时享有 0.5 分确定性优先特权；
   - 普通静步闲走无确定性特权，翻棋期望分持平或更优时翻棋必胜出。
4. **测试回归与保障**：
   - 新增 `tests/test_p1_camp_shuttle_fix.py`，覆盖营间闲走负排序、中营单子防虚胀、`game_211613` 第 5 手坚守中营翻棋辐射、`game_211452` 第 7 手拒不出营优先翻开邻营暗子；
   - 全量 180 项自动化测试 100% 通过（177 passed, 3 skipped, 0 failed）。

## [2026-09-07] — P1/P4 阶段：传统搜索专家模型与原生 APK 假想敌模型深度对齐与地平线盲区根治

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P4 阶段（假想敌对弈评测与基准对齐，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心改进与技术实现
1. **彻底根治静态搜索 (QSearch) 组合爆炸与地平线盲区丢大子**：
   - **逆向取证**：对 `libjunqi.so` 二人翻棋专有模块（`0x57000 - 0x5b000`）进行全量反汇编，确认原生 `0x5a678` 静态搜索仅对吃子动作（`0x591d8`）进行深度延伸（上限 31 层）；
   - **语义纠偏**：从 `_qsearch` 中彻底剔除非吃子的空行营进驻走法，使战术分支因子从 15~20 大幅收敛至 1~3；
   - **深度延伸与剪枝**：将默认 QSearch 深度上限从 4 扩展至 16，并引入大 Delta 与局部 Delta Pruning（当 `stand_pat + victim_val < alpha` 时剪枝），使微秒级交火线推演深度可达 8~16 层，彻底杜绝贪吃诱饵后第 5~6 步大子被反杀的地平线盲区。
2. **博弈树内部迭代加深 (IDS) 动态时间预算早停对齐**：
   - 对齐原版 `0x5ac3a`（`cmp.w r2, r3, asr #2`）：在迭代加深循环中，若当前深度搜索总耗时已超过总时限的 25%，下一层因分支倍增大概率超时截断，故在此安全早停，保持当前最高深度完整稳定的 PV 决策。
3. **原生 APK 假想敌代理 (`ApkNativeAgent`) 逆向勘误与全量重构**：
   - 彻底剔除历史逆向张冠李戴的四国军棋（17x17，`0x600ca/0x2f9f6/0x5e0a8`）误导性注释与架构，全面对接二人翻棋原生 60 格架构；
   - 对齐官方难度配置：`beginner` (depth 2, qdepth 8), `intermediate` (depth 3, qdepth 12), `advanced` (depth 4, qdepth 16)；
   - 严格遵守非透视公共信息屏障。
4. **GUI 交互防御与模型全量接入 (`junqi/gui.py`)**：
   - **根治 KeyError 回调崩溃**：修复 `update_status()` 在棋步落地后再次调用 `describe(self.last_action)` 导致从原位置取子报 `KeyError` 的严重缺陷；在 `do_action()` 中预存不可变的 `last_action_desc` 直接供界面呈现；同时将 `describe()` 重构为完全防御式实现；
   - **模型引擎面板对齐**：GUI 引擎选择默认置为“原生APK”并置顶首行，直观支持“原生APK / 专家搜索 / P4混合智能 / 混合智能”一键切换与算力映射；AI 思考及提示时明确显示当前运行引擎名称；
5. **测试回归与保障**：
   - 新增 `tests/test_p1_quiescence_depth.py`（长程陷阱、QSearch 纯吃子分支收敛、IDS 25% 动态早停、ApkNativeAgent 各难度测试）；
   - 新增 `tests/test_gui_fixes.py`（KeyError 防御、last_action_desc 传递、APK 与专家引擎决策验证）；
   - 全量自动化测试 173 项全部通过（173 passed, 3 skipped, 0 failed）。

## [2026-09-07] — P1 阶段：人机对局界面最后一步着法视觉轨迹与状态高亮

阶段归属：**P1 阶段（人机交互体验与对弈可解释性，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心改进与技术实现
1. **棋盘最后一步着法轨迹与落点高亮（`junqi/gui.py`、`junqi/replay_gui.py`）**：
   - **移动走法**：起始格以高可见度红色虚线框（`#E60000`）标注离开位置，落点格以加粗红色实线框紧扣最新位置，两格之间绘制带箭头的实线方向轨迹，清晰展现远距离滑行、进营或吃子过程；
   - **翻棋走法**：最新翻开的暗子所在格以加粗红色实线框精准锁定；
   - **视觉设计**：采用鲜明纯正的亮红色（`#E60000`），对比木纹棋盘对比度极高，方便一眼锁定 AI 刚刚出手的着法；
   - **状态栏协同提示**：轮到用户行动时，状态栏直接同步追加 `[AI刚走]: 连长 (5,2)->(5,3) 吃 蓝排长` 文案，实现视觉轨迹与文字描述双重直观反馈；
   - 适配新局（`new_game`）、悔棋（`undo`）与对局状态转移（`do_action`），确保标记即时刷新且无残留。
2. **测试回归**：
   - 全量自动化测试 166 项全绿通过（166 passed, 3 skipped, 0 failed）。

## [2026-09-07] — P1/P2 阶段：双格式复盘回放、单步专家点评标注与算法改进全量汇总数据管线

阶段归属：**P1 阶段（复盘数据解析与人机对战交互）** 与 **P2 阶段（人工战术打标、知识蒸馏与回归防线，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心改进与技术实现
1. **双格式统一复盘管理器（`junqi/replay_manager.py`）**：
   - 原生 App `.sav` 二进制复盘与人机对战 JSON（`games/*.json`）统一映射为 `ReplaySession`；
   - 自动逐步推演生成各步完整盘面快照（`GameState`）、动作（`Action`）、吃子/同尽判定及人类可读走法描述（`describe_action`）。
2. **单步战术点评与持久化系统（`junqi/review_storage.py`）**：
   - 建立标准点评数据结构，涵盖定性评级（专家恶手、自杀弃营、炸弹乱撞、盲目翻棋、错失绝杀、人类更优等）、推荐纠偏正着、心得评论与序列化局面；
   - 支持按文件与步数增量保存、修改、查询与删除，统一持久化至 `reviews/annotations.json`。
3. **可视化复盘与决策点评工作台（`junqi/replay_gui.py`）**：
   - 原版切片棋盘展示当前盘面，高亮着法轨迹与推荐正着（支持棋盘直接点选推荐走法）；
   - 步进控制条（开局/上一手/下一手/终局/自动播放/滑块调节）；
   - 集成 AI 即时智能研判：一键调用专家引擎与 P4 混合模型，展示当前局面 Top-3 推荐走法与胜负和概率；
   - 在对战主界面（`junqi/gui.py`）及 CLI（`python -m junqi replay_gui`）增加一键复盘入口。
4. **全量点评汇总与算法改进数据管线（`junqi/review_summary.py`）**：
   - **诊断报告**：自动统计高频战术失误分布与典型纠正案例，生成 `reports/review_summary_report.md`；
   - **蒸馏训练集导出**：将人工纠偏与正着走法转化为高质量训练样本 `datasets/user_review_labeled.json`，与现有价值蒸馏管线 100% 格式对齐，可直接运行 `train_value_distill.py` 微调；
   - **自动化回归测试套件生成**：自动生成 `tests/test_user_reviewed_tactics.py`，将恶手批评固化为 pytest 断言，彻底防范历史失误再犯。
5. **全量回归验证**：
   - 新增 `tests/test_replay_review.py`，全量单元测试 166 项全部通过（166 passed, 3 skipped, 0 failed）。

## [2026-09-07] — P0/P1 阶段：连续 40 步未吃子判和规则调整与 GUI 原生 APK 假想敌对战支持

阶段归属：**P0 阶段（规则对齐与配置校准）** 与 **P1 阶段（人机对战交互与假想敌接入，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心改进与技术实现
1. **连续未吃子判和时限调整为 40 手（`RuleConfig.no_capture_draw_plies = 40`）**：
   - 更新 `junqi/config.py` 和 `configs/rules.yaml` 中 `no_capture_draw_plies` 默认值为 40；
   - 同步更新神经网络通道特征编码（`junqi/encoder.py` 通道 31）、结构性死锁检测（`junqi/analysis.py` `is_dead_draw`）、专家搜索时钟衰减（`junqi/eval_expert.py`）以及胜率预测映射；
   - 更新 GUI 状态提示与终局文案：`REASON_CN["no_capture"] = "连续40步未吃子，判和"`；
   - 适配单元测试（`test_rl.py`、`test_endgame_deadlock_fixes.py`、`test_camp_tactics_fix.py`）中对 quiet 计数及历史复盘回放的兼容。
2. **GUI 对弈面板集成原生 APK 模型（`ApkNativeAgent`）人机选择**：
   - 重构 `junqi/gui.py` 控制面板为 2x2 网格单选组件，新增 `原生APK` 引擎模式（支持 `P4混合智能`、`原生APK`、`专家搜索`、`混合智能`）；
   - 在 AI 落子执行（`ai_move`）与着法推荐提示（`ask_hint`）中接入 `ApkNativeAgent`；
   - 将界面搜索算力档位（“快”、“标准”、“深算”）与 APK 原生难度级别精准映射为 `beginner`（初级）、`intermediate`（中级）、`advanced`（高级）；
   - 零依赖无缝对齐官方 libjunqi.so 逆向引擎的 2 的幂次子力梯度、动态炸弹折算、PVS 零窗口搜索与 Zobrist TT 剪枝体系。
3. **全量回归验证**：
   - 运行全量自动化测试套件 `pytest tests/`，162 项测试全部通过（162 passed, 3 skipped, 0 failed）。

## [2026-09-07] — P0/P1 阶段：前线中桥（第 2 列）铁路轨道贯通 Bug 修复与全量复盘回放验证

阶段归属：**P0 阶段（规则与几何拓扑正确性）** 与 **P1 阶段（复盘数据校验与传统引擎对齐，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心病灶剖析与技术解法
1. **中桥铁路阻断 Bug 彻底根除（`junqi/rules.py`）**：
   - **历史遗留缺陷**：旧版本在 `CROSS_BLOCKED` 集合中错误包含了 `frozenset(((5, 2), (6, 2)))`，将前线中桥误判为“仅公路可通、铁路滑行阻断”；
   - **客观棋理实证**：中国军棋中桥（第 2 列山界过河通道）为中铁桥，与左铁桥（第 0 列）、右铁桥（第 4 列）三座铁桥共同构成全场南北大贯通的铁路网络。此前将 `(5, 2) <-> (6, 2)` 设为阻断，导致工兵无法借由中桥转弯飞过河，同时 GUI 界面仅绘制实线公路而非虚线铁轨；
   - **实战复盘证据**：1,072 局真人实战样本中存在多达 132 手工兵经由 `(5, 2) <-> (6, 2)` 中铁桥转弯跨河着法（如 `(6, 1) -> (1, 0)`、`(7, 0) -> (5, 2)` 等），此前因误阻断导致数十局回放报非法走法；
   - **修复方案**：从 `CROSS_BLOCKED` 中彻底剔除 `((5, 2), (6, 2))`，仅保留无桥隔断的 `col1` 与 `col3`（山界河流）；
   - **回放与测试验证**：
     - 全量 1,072 局真人实战复盘（`.sav`）回放成功率由 91.5% 跃升至 **100.0%（1072/1072 全绿通过）**；
     - GUI 与打标面板（`label_gui.py`、`gui.py`）中路第 2 列双方轨道（(5, 2) 与 (6, 2)）虚线铁轨正确接通并完美绘制；
     - 全量自动化单元测试通过（`pytest tests/` 162 passed, 3 skipped, 0 failed）。

## [2026-09-07] — P2 阶段：基于 6,000 实战交火局面的专家搜索价值蒸馏预热与测试集验证

阶段归属：**P2 阶段（行为克隆与搜索蒸馏，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §5 P2 & S2）**。

### 核心改进与技术实现
1. **实战预打标数据直读与训练管线升级（`junqi/train_value_distill.py`、`junqi/__main__.py`）**：
   - 增加 `--data` 选项支持，直接无损挂载 `datasets/distill_tactical_labeled.json`（包含 6,000 个中后盘深度推演局及 36 条人工核验标注）；
   - 训练模块支持自动反序列化与高速特征编码（6,000 局面全量载入仅耗时 0.26 秒），彻底解除蒸馏对在线随机采样的依赖。
2. **三分类交叉熵 + 期望值 MSE 联合蒸馏目标**：
   - 价值头不仅监督胜/和/负离散分类（Cross-Entropy），同时对标量期望胜率 $\hat{v} = P(\text{Win}) - P(\text{Loss})$ 与归一化战术评分 $\tanh(\text{score} / 600.0)$ 施加 MSE 约束；
   - 训练收敛迅速，验证集三分类准确率达到 **84.2%**，MSE 误差低至 **0.0576**。
3. **严格冻结主干与 Policy Head（含 BatchNorm 动量锁定）**：
   - 训练时主干置为 `net.eval()` 且仅开启 `net.value_head.train()`，经自动化比对确认非价值头参数 100% 字节级不变，绝无任何策略漂移；
   - 更新并同步生成 `models/value_distilled.pt` 与 `models/pool/value_distilled.pt`。
4. **独立实战测试集验证（200 局，9,634 plies）**：
   - Value 三分类准确率由原始 BC 的 **42.54%** 提升至 **52.45%**（+9.91%）；
   - Value 期望 MSE 损失由 **0.8403** 下降至 **0.5642**（误差下降 32.8%）；
   - Top-1 命中率 **24.39%**、Top-3 命中率 **45.61%**、Policy CE **3.1880** 与非法移动率 **0.00%** 全量保持，无任何退化。

## [2026-09-07] — P1/P2 阶段：专家搜索实战战术缺陷根因整改与全量回归验证

阶段归属：**P1 阶段（传统搜索增强与专家评估校准）** 与 **P2 阶段（实战交火打标与决策质量把关，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）**。

### 核心病灶剖析与技术解法
1. **彻底解决四大技术根因**：
   - **根因 1（序列化状态丢失导致中盘乱换阵营）**：`junqi/tactical_sampler.py` 中 `state_to_dict` / `dict_to_state` 丢失了 `first_flip_done` 字段，导致残局盘面被反序列化后 `first_flip_done=False`，搜索树几率翻棋节点误判为“开局首翻”强行反转红蓝阵营（估值跳变超 130 分）。现已完备记录、推断与恢复 `first_flip_done`。
   - **根因 2（弃营虚假势能倒挂）**：`junqi/eval_expert.py` 中旧有空营势能导致棋子弃营将营地变空反而大幅加分（炸弹/大子弃营乱窜）。现已重构为空营净控制权与 2 步通畅中继推进评估，上限严格封顶（4.0~6.0 分），远低于实占行营保护分（15.0 分）与炸弹驻营要塞分（27.0 分），彻底杜绝弃营动机。
   - **根因 3（地雷主动攻击 Bug 与伪 HQ 扣分）**：修正威胁与攻击计算中将不可移动的 `Rank.LEI` 误算为主动攻击者的严重错误；解除无锁营规则下的伪 HQ 扣分；增加炸弹入营要塞加分（+12.0）。
   - **根因 4（超时截断降级与打标断点续传）**：`junqi/tactical_sampler.py` 默认 `time_limit_ms=0` 保证 Depth 3 完整计算；新增 `--relabel-file` 与 `--overwrite` 支持，在 100% 完好保留用户已标记的 36 条人工核验与批注的前提下执行专家全量重算。
2. **落地核心实战战术准则**：
   - **安全出营判定（`junqi/search.py`）**：严格落实用户准则“出营击杀若不会导致丢营，严禁扣分”。出营扑杀时自动排查落点被歼敌子，若下回合敌军无法侵入行营，则不扣失营分且赋予 `camp_outstrike_bias` (+400,000) 行营特权。
   - **敌前出营送死重罚（`junqi/search.py`）**：营内弱子在敌大子近身窥视下无故弃营给予严厉惩罚（-350,000）。
   - **后手火力护航机制（Battery Support, `junqi/eval_expert.py`）**：被威胁子后方若有己方更高军衔或炸弹护航（敌大吃小会被反杀），威胁自动折现或归零。
   - **战术确定性优先准则（`junqi/search.py`）**：翻棋几率期望与确定性战术走法差距 <= 0.5 分时，坚决优先由确定性走法胜出，杜绝盲目翻暗棋赌概率；修正根节点 alpha-beta 窗口误更新并确保 `best_action` 稳健置于 `root_scores[0]`。
3. **关键局面验证与全量回归**：
   - 6 个复核关键局面验证结果全部命中战术正手：
     - **#3**：`(6, 2) 司令 -> (5, 2) 吃营长`（**100% MATCH**，纠正原炸弹乱窜）；
     - **#8**：`(3, 2) 团长 -> (2, 1) 进营`（**100% MATCH**，纠正原盲目翻棋）；
     - **#23**：`(10, 4) 工兵 -> (7, 4) 吃地雷`（同动工兵避险且飞雷，杜绝炸弹乱动）；
     - **#24**：`(1, 2) 旅长 -> (1, 1) 吃营长`（**100% MATCH**，充分理解后手反杀）；
     - **#32**：`(10, 3) 排长 -> (9, 3) 进营`（纠正盲目翻暗子，安全控制要塞）；
     - **#33**：`(5, 4) 旅长 -> (4, 3) 进营`（**100% MATCH**）。
   - 全量自动化测试回归：**162 passed, 3 skipped, 0 failed**（100% 绿色通过）。

## [2026-09-07] — P2 阶段：实战交火局面采样打标与交互式核验 GUI 工作台

阶段归属：**P2 阶段（行为克隆与搜索蒸馏，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §5 P2）**。

### 核心改进与技术实现
1. **实战交火与活跃接触局面均匀采样器（`junqi/tactical_sampler.py`）**：
   - 遍历 835 局经过清洗的高质量实战复盘，识别出 47,213 个活跃交火候选手（排除开局无明子对峙与非走步局面）；
   - 精选 6,000 个中后盘交火盘面（按 55% 中盘 3,300 局 + 45% 残局 2,700 局配比），涵盖行营争夺、主力决战与工兵破雷等核心实战战术；
   - 实现了 `GameState` 与 `Action` 的完整无损 JSON 序列化/反序列化（`state_to_dict` / `dict_to_state`），支持断点续传与增量缓存。
2. **多进程专家深度搜索伪标签生成（`label_positions_parallel`）**：
   - 8 进程并行（Star1 期望极大极小 + QSearch，depth=3），实时产出专家推荐最佳着法、战术评分与胜/和/负三分类标签；
   - 对比人类实战决策与专家推荐，自动标记人机吻合与分歧（`human_expert_match`）。
3. **专用交互式打标与抽查核验 GUI 工作台（`junqi/label_gui.py`、`junqi/__main__.py`）**：
   - 原生 Tkinter 开发，复用 APK 贴图与棋盘拓扑；
   - **动态双走法光标与轨迹指示**：青蓝色高亮标出人类实战着法，橙金色高亮标出专家推荐着法，直观对比人机思考差异；
   - **招法合理性裁决体系 (Action Judgment)**：重构核验逻辑，由“虚无的终局胜负猜想”全面转向“走法合理性裁决”：
     - `⭐ 专家更优 (1/E)`：采纳推荐，作为正向策略蒸馏样本；
     - `👤 人类更佳 (2/H)`：实战人类大局观更好，抑制专家短视盲区；
     - `🤝 两手皆可 (3/B)`：常规正手，等权候选；
     - `❌ 专家恶手 (4/X)`：战术硬伤/送死，坚决剔除防带偏；
     - `❓ 标记存疑/跳过 (0/S)`；
   - **态势评估科学正名**：将评分正名为“优势局面 / 均势拉锯 / 劣势局面”控制力参考；
   - **按需 5 层超深度推演 (选项 B)**：新增 `⚡ 启动 5 层超深度推演 (Depth 5)` 异步多线程按钮（快捷键 F5），支持对复杂交火局随时调用 Depth 5 超算，结果与棋盘箭头实时无缝覆写刷新；
   - **智能高效筛选**：支持“仅看人机分歧局”、“仅看未核验局”快速跳转定位；
   - 注册至顶层 CLI：`python -m junqi label_gui`。
4. **单元测试与质量保障**：
   - 新增 `tests/test_tactical_sampler.py`（盘面序列化往返、动作序列化、活跃交火判定、评分转换等 4 项单测全通过）。

## [2026-09-07] — P2 阶段：专家搜索价值蒸馏预热 (Value Distillation) 与多进程加速

阶段归属：**P2 阶段（行为克隆与搜索蒸馏，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §5 P2 & S2）**。

### 核心改进与技术实现
1. **多进程并发专家搜索打标加速（`junqi/train_value_distill.py`、`junqi/__main__.py`）**：
   - 引入独立子进程打标函数 `_label_worker` 与 `--workers` 支持，利用多核 CPU 并发（8 Workers）对 1200 个分阶段局面（开局 300、中盘 480、残局 420）进行深度专家搜索（Star1 期望极大极小 + QSearch，depth=3）；
   - 产生高质量密集伪标签分布：Win=247, Draw=832, Loss=121；
2. **冻结主干与策略头，仅优化价值头**：
   - 冻结 `in_conv`、`blocks` 与 `policy_head`，确保行为克隆学到的人类走子翻棋大局观不受破坏；
   - 仅微调 `value_head`，验证准确率由初始 63.9% 稳步提升至 **74.4%**，生成带完整元数据与基座策略指标的权重文件 [`models/value_distilled.pt`](models/value_distilled.pt)；
3. **独立测试集无偏评测对比（`reports/p2_distilled_report.md`）**：
   - 在 9,634 plies 独立测试集上评测：
     - **Value 三分类准确率**：从 `bc_best.pt` 的 42.54% 大幅提升至 **50.25%**（提升 **+7.71%** 🚀，对比随机基准提升 +16.92%）；
     - **Value 期望 MSE 损失**：从 0.8403 显著降低至 **0.5261**（**误差缩减 37.4%** 🚀）；
     - **Policy Top-1/Top-3**：保持 23.21% / 44.06%，非法走法率严格为 **0.00%**；
4. **全量回归验证**：
   - `pytest tests/`：**158 passed, 3 skipped, 0 failed**（53.86s）。

## [2026-09-06] — P2 阶段：复盘数据清洗重构 (datasets/p1_v2) 与行为克隆模型 (bc_best.pt) 重训验收

阶段归属：**P2 阶段（行为克隆与数据集清洗重构，依据 AI_TRAINING_AND_HUMAN_PLAY_PLAN.md §5 P1 与 P2）**。严格遵守公共 Policy 不透视暗子、真实终局真值赋权等硬约束。

### 核心改进与技术实现
1. **真实复盘数据清洗与短步数异常局剔除（`junqi/dataset.py`）**：
   - **全面纳入实战复盘**：将复盘库扩充至 **1072 局**（包含 1000 局官方 DES-ECB 解密 `list.cfg` 终局真值对局，以及近期新增的 72 局 2026-08-30 至 2026-09-06 实战对局）；
   - **异常短步数（开局运气失衡秒退）清洗**：
     - 数据实证表明军棋正规杀棋（吃旗/全歼）最早在第 52 手才出现；20 手以内的对局 100% 为开局失衡直接认输（code 21，平均时长仅 38~50s）或早期逃跑强退（code 20）；
     - 引入可配置清洗阈值 `min_plies = 20`（默认 20 手，约 10 回合）：滤除 **121 局** 开局崩盘秒退局与 **116 局** 损坏/步法非法对局，保留 **835 局** 战术成熟的高质量对局；
     - 杜绝未充分展开的开局运气局被强行赋予终局胜负标签，彻底消除 Value 网络的开局噪声偏置；
   - **高质量数据集 `datasets/p1_v2` 导出**：
     - 按对局确定性洗牌（`seed=2026`）严格切分：Train 668 局 (76,292 plies), Val 83 局 (9,647 plies), Test 84 局 (9,634 plies)；
     - 总样本数 95,573 plies（Policy 样本 95,573，高置信度 Value 样本 80,253）；
     - 自动生成 SHA-256 哈希校验与规范 `metadata.json`。
2. **评估管线与战术斩杀优化（`junqi/eval_bc.py`、`junqi/ai.py`）**：
   - 修复 `eval_bc.py` 遍历 DataLoader 时解包 7 个元素的 `ValueError`；
   - 完善 Value 头三分类交叉熵损失与期望胜率标量 \(P(win) - P(loss)\) MSE 计算，并引入胜/和/负分类准确率统计；
   - 优化 `HybridAgent.choose_actions`：当走法能直接吃掉对方军旗一步制胜时，赋予绝对最高战术优先级（`WIN_SCORE + 5000.0`），避免与困毙并列导致受先验波动影响。
3. **行为克隆 GPU 重训与测试集独立验收（`junqi/train_bc.py`、`reports/p2_bc_report.md`）**：
   - 备份旧版权重至 `models/bc_best_legacy_20260901.pt`；
   - 在 RTX 4080 SUPER (CUDA) 上基于 `datasets/p1_v2` 训练 20 Epochs（耗时仅 100.0s），最优验证集模型保存至 `models/bc_best.pt`；
   - **独立测试集（9,634 plies）全面评测指标**：
     - **Top-1 准确率**：**24.39%**（对比随机基线 2.34%，提升 **+22.05%** 🚀）；
     - **Top-3 准确率**：**45.61%**（对比随机基线 7.02%，提升 **+38.59%** 🚀）；
     - **Top-5 准确率**：**58.49%**（对比随机基线 11.50%）；
     - **Policy 交叉熵损失**：**3.1880**（收敛良好）；
     - **Value 三分类准确率**：**42.54%**（大幅超越基准 33.33%）；
     - **非法走法预测率**：**0.00%**（100% 严格遵守规则）；
     - **分阶段命中率**：开局 Top-1 **31.75%** / Top-3 **53.37%**；中盘 Top-1 **19.52%** / Top-3 **40.24%**；残局 Top-1 **16.30%** / Top-3 **37.17%**。
4. **全量回归测试**：
   - 运行项目全量测试套件：**158 passed, 3 skipped, 0 失败**（耗时 52.20s）。

## [2026-09-06] — P1 实战重大战术缺陷修复：空行营安全推进中继与行营阻断防弃营送死

阶段归属：**P1 阶段（传统搜索与估值增强）**。严格遵循 `AGENTS.md`（“严格遵循‘首翻子力即据点，依托行营辐射拓荒’与行营单向打击特权”）。

### 核心病灶剖析与技术解法（基于实战对局 `games/game_20260906_230423.json` 复盘）
1. **第 19 手师长暴露却 12 步迟钝不进营（空行营安全中继推进 Heuristic）**：
   - **根本病灶**：红师长暴露在 (6,1) 时，中营 (8,2) 为空。因远端左路军长瞄准团长的吃子威胁遮蔽（Threat Dominance），根节点所有走法评分全部坍缩为 -74.25 分，且走法排序与评估函数中缺乏向空行营推进的中继启发，导致 AI 误选 `翻(1,3)`，将主力师长暴露长达 12 手未动；
   - **重构方案**：
     - 在 `junqi/search.py` 的 `_score_action` 中新增 `camp_staging` 优先级（大子向通往空行营的安全中继站挺进时赋予 200,000+ 极高走法排序优先级，高于任意翻棋）；
     - 在 `junqi/eval_expert.py` 中为大子通向空营的安全中继点赋予动态中继推进分（+10.8 分），打破威胁遮蔽下的平分迟钝。实测第 19 手最优走法果断修正为 `移(6,1)->(6,2)`。
2. **第 231~232 手排长主动弃营导致大子进驻全盘崩溃（严惩弃营送死与阻断守护）**：
   - **根本病灶**：红排长在 (2,3) 营内成功阻挡蓝师长进营；但在深度 3 截断下，AI 在第 231 手走出 `(2,3)->(2,2)` 弃营；而静态搜索（QSearch）在深度 3 未能延伸追踪蓝师长第 232 手进营后的第 234 手吃子，产生致命地平线盲区；
   - **重构方案**：
     - 在 `junqi/search.py` 的 `_score_action` 中新增**严惩敌大子近身窥视下弃营（Anti-Camp Abandonment）**：任何受敌大子贴身威胁时离开行营的自杀弃营走法直接判为 `-300,000` 严重惩罚，确保即使全盘走法平分也绝不选出营送死；
     - 实测第 231 手 AI 彻底杜绝走出 `(2,3)->(2,2)`，转而选择安全进驻空营 `(7,2)->(7,3)`。
3. **单元测试与全量回归**：
   - 新增 `tests/test_camp_tactics_fix.py`，覆盖第 19 手师长挺进、第 231 手防弃营、中继推进启发分与弃营重罚；
   - 运行全量测试套件：**158 passed, 3 skipped，0 失败**。

## [2026-09-06] — P1 传统搜索估值增强与 P4 官方 APK 逆向假想敌 (ApkNativeAgent) 上线

阶段归属：**P1 阶段（传统搜索增强与估值基线）** 与 **P4 阶段（基准评测与假想敌对弈体系）**。严格遵循 `AGENTS.md` 与 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`（“公共 Policy/传统估值严禁读取真实暗子身份”、“严禁添加吃子中间奖励”）。

### 核心改进与技术实现
1. **P1 传统博弈树估值与搜索增强（`junqi/config.py`、`junqi/eval_expert.py`、`junqi/search.py`）**：
   - **动态炸弹定价机制（0x600ca 权威公式）**：在 `EvalWeights` 中引入 `use_dynamic_bomb` 与 `bomb_ratio`（默认 1/3），基于全盘（明子+暗子期望池）中敌方存活最大军衔实时缩放炸弹价值（敌有司令时值 853，仅剩师长时值 213）；
   - **等比子力阶梯（0x124094 权威分值）**：引入 `EvalWeights.apk_weights()` 预设，采用官方等比价值阶梯（司令 2560 至排长 30），工兵高权值（80），地雷护旗阵地追加 80 分加成；
   - **PVS (Principal Variation Search / NegaScout) 零窗口剪枝**：在 `ExpertSearchEngine._negamax` 中全面替换普通全窗遍历，对非主变例走法先做零窗口探测 `[-alpha-1, -alpha]`，探测击穿再做全窗重搜，大幅降低分支因子；
   - **修复迭代加深根节点 dummy action 虚假先验**：消除根节点未搜索前将 `acts[0]` 误当成 TT Move 的假高分，确保首层以启发式真实排序（如一步吃旗 900,000）优先展开。
2. **P4 官方 APK 原生算法 1:1 复刻假想敌（`junqi/apk_agent.py`、`junqi/ai.py`、`junqi/benchmark.py`）**：
   - **假想敌智能体 `ApkNativeAgent`**：纯 Python/Cython 原生复刻，提供初级（depth=2, 100ms）、中级（depth=3, 300ms）、高级（depth=4, 1000ms）三档官方预设；严格遵守公共信息屏障（暗子统一遮罩为代号 13），100% 不透视；
   - **统一 Agent 决策接口**：为所有 Agent（`Agent`、`ExpertAgent`、`ApkNativeAgent`）统一补充 `select_action` 接口；
   - **自动化基准对抗接口**：在 `benchmark.py` 中新增 `run_apk_challenge(candidate_agent, n_games=40, level="advanced", ...)`，一键对决原版假想敌，输出胜/和/负率、Elo 分差与对局报告。
3. **单元测试与回归覆盖（`tests/test_p1_apk_search.py`、`tests/test_p4_apk_agent.py`）**：
   - 新增 7 项定向单元测试，覆盖动态炸弹定价缩放、地雷护旗加分、PVS 搜索绝杀、ApkNativeAgent 三档初始化、信息屏障验证、战术一步扛旗与轻量对战；
   - 全部 7 项测试通过（`Ran 7 tests in 3.849s, OK`）。

## [2026-09-06] — 自博弈挖掘失误针对性修复与人类高手复盘分歧率扫描工具

阶段归属：**P1 阶段（传统搜索增强与战术规则验证）** 与 **P2 阶段（混合决策引擎策略验证与自博弈评估）**。严格遵循 `AGENTS.md`（“每项实现必须有对应测试或固定评测证据”）。

### 核心病灶剖析与技术解法
1. **针对自博弈报告捕获失误的规则级修复（`junqi/hybrid_engine.py`）**：
   - **炸弹主动攻击廉价小子自爆（严重贱卖）**：在 `res == "both_die"` 分支，针对 `mover.rank == Rank.ZHA` 且目标非 `(QI, SI, JUN, SHI)` 施加 `-350.0` 严惩，彻底制止炸弹撞排长/工兵等自爆；
   - **弃营出击面临致命反杀（Fatal Threat in Camp Exit）**：在 `leaves_camp` 分支，只要检测到走步后次手目标格面临反杀（`fatal_threat`），无论出于何种目的，统一顶格扣除 `-400.0` 分；
   - **行营龟缩拒不翻棋（消除 48 次 `camp_turtling`）**：为翻棋动作引入**据点辐射拓荒战略加分**——若翻开的暗子邻接己方已占领的行营，赋予 `+15.0` 战术激励，驱使 AI 主动由据点向外拓荒，破除行营无谓来回踱步。
2. **人类高手复盘分歧率与胜率断崖自动化扫描器（`scripts/scan_replays.py`）**：
   - 自动批量加载 `军旗复盘/*.sav` 样本库；
   - 逐手回放人类动作并与 AI 决策对比，实时统计 Top-1 / Top-3 吻合率与分歧点（Divergence）；
   - 结合全盘真值演进探测胜率断崖（Valuation Cliff）；
   - 实测 5 局（451 手）：Top-1 吻合率 48.34%，Top-3 吻合率 65.85%，捕获 167 处战术分歧点，0 处致命断崖；
   - 自动输出结构化数据 `reports/replays/replay_blunders_<timestamp>.json` 与诊断报告 `reports/replays/replay_divergence_report_<timestamp>.md`。
3. **单元测试与全量回归**：
   - 在 `tests/test_replay_review_fixes.py` 中新增 `test_bomb_suicide_on_minor_penalized` 与 `test_camp_adjacent_flip_bonus`；
   - 运行全量测试套件：**147 passed, 3 skipped，0 失败**。

## [2026-09-06] — 自动化战术漏洞挖掘管道与全行营拓扑模糊测试体系

阶段归属：**P1 阶段（传统搜索增强与战术规则验证）** 与 **P2 阶段（混合决策引擎策略验证与自博弈评估）**。严格遵循 `AGENTS.md`（“每项实现必须有对应测试或固定评测证据”）。

### 新增工具与自动化基建
1. **全盘 10 个行营参数化拓扑模糊测试（`tests/test_camp_topology_fuzz.py`）**：
   - 彻底摆脱“单点局部人工试错”，利用参数化网状遍历覆盖全盘 10 个行营坐标、不同驻防军阶（工兵/排长/营长/司令）与不同诱饵兵种；
   - 断言行营“据点庇护”、“占营优于吃小子”与“反杀保护”在全盘任意对称位置上 100% 成立，1.6 秒内完成全拓扑验证。
2. **高速无头自博弈战术失误挖掘引擎（`scripts/mine_blunders.py`）**：
   - 脱离 GUI 进行高并发对弈，集成 5 大战术病态探针（弃营丢营、大子白送/炸弹贱卖、工兵自杀、无意义往复踱步、行营龟缩拒不翻棋）；
   - 实时拦截病态走法，自动生成结构化 JSON 错题库（`reports/blunders/blunders_<timestamp>.json`）与 Markdown 诊断复盘报告（`reports/blunders/blunder_report_<timestamp>.md`）；
   - 实测 2 局 `hybrid2` vs `expert2` 自动捕获 71 处战术异常（含师长撞军长、炸弹炸工兵、司令进敌方反扑网等极端实战案例），为后续战术优化提供全自动流水线。

## [2026-09-06] — 行营战略据点保护与“占营优于吃小子”战术硬约束重构

阶段归属：**P1 阶段（传统搜索增强与启发式战术修正）** 与 **P2 阶段（混合决策引擎策略修正）**。严格遵循 `AGENTS.md` 硬约束（“严格遵循‘首翻子力即据点，依托行营辐射拓荒’与行营单向打击特权”）。

### 核心病灶剖析与技术解法（基于对局 `games/game_20260906_192146.json` 第 8 手复盘）
1. **行营据点弃守与贪吃诱饵反被杀（`junqi/hybrid_engine.py`）**：
   - **根本病灶**：原战术规则 `_apply_tactical_rules` 在判断吃子时，只要满足 `battle(mover, target) == "attacker_wins"`（如工兵挖雷），无条件赋予 `+50.0 + 棋子价值 * 0.5`（工兵挖雷额外 `+30.0`），缺乏对“行营据点庇护”与“次手战术反扑”的全局感知。导致实战第 8 手红工兵占据核心行营 `(7, 3)` 庇护所时，为了贪吃铁路上的蓝地雷 `(6, 4)` 冒失出营，次手立刻被蓝连长 `(6, 3)` 反杀吃掉，并导致原本由 AI 占领的据点行营反遭敌军入驻占领；
   - **重构方案**：
     - **行营战略进出感知**：区分 `leaves_camp`（弃营）、`enters_camp`（占营）与 `camp_to_camp`（营间机动），对主动占领行营给予 `+25.0`，营间调动给予 `+15.0`；
     - **1-ply 战术反杀网检测（Fatal Threat）**：预测走步后敌方次手所有合法走法，若敌方在目标格能立即反杀己方（尤其是以小换大或大换大），判定为陷阱走法；吃小亏大或弃营被反杀给予 `-300.0` 重罚；
     - **丢营风险拦截（Camp Invadable）**：若离开行营后，该行营下一手即面临敌军合法入驻，判定为严重失守风险，扣除 `100.0` 分；
     - **“占营优于吃小子”铁律（Camp Hegemony）**：离开行营去吃小子（排/连/营/地雷），若己方为中小子直接 `-150.0`，若为大子但面临反杀/丢营亦 `-150.0`；无目的闲走弃营 `-80.0`；
     - 使得第 8 手 `走(7, 3)->(6, 4)` 综合评分从 `+95.02` 暴跌至 `-549.98`，彻底杜绝 AI 弃营贪吃廉价小子与白送；
2. **回归与覆盖测试**：
   - 在 `tests/test_replay_review_fixes.py` 中新增 `test_camp_preservation_over_minor_piece_bait`，精准复现 `game_20260906_192146` 第 8 手棋局，断言弃营吃小子被扣除大分且 AI 拒绝出营；
   - 调整 `test_winning_captures_not_penalized` 棋盘布局，隔离炸弹铁路通道与工兵挖雷兵站，验证安全胜势吃子仍受保护且 142 项测试全绿。

## [2026-09-06] — 残局结构性必和误判与 AI 决策卡死彻底修复

阶段归属：**P0 阶段（规则正确性与信息边界契约）** 与 **P2 阶段（专家评估与残局死局判定）**。严格遵循 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 `AGENTS.md`。

### 核心病灶剖析与技术解法
1. **残局结构性必和误判（`junqi/analysis.py`）**：
   - **根本病灶**：原 `is_dead_draw` 逻辑过于粗暴：① 在双无工兵分支下，简单判定若弱势方 `len(combat_my) >= 3` 或 `my_in_camps >= 1` 即判定为必和；② 原型 3 无条件判定 `not my_can_flag and not opp_can_flag` 即必和。导致在优势方拥有双师长、弱势方仅有营连排且多子裸露在外的胜势围剿局面下，被误判为 `红胜 0.0% | 和 100.0% | 蓝胜 0.0%`；
   - **重构方案**：
     - 若场上仍有未翻开暗子，绝不提前判定结构性死锁；
     - 严格约束无敌大子“歼灭不能”拓扑：优势方必须仅有单单一颗压制大子，且弱势方所有可动子力均已安全驻守行营（或在极端长局 `ply >= 140` 或 `quiet >= 25` 下多子在营对峙）；若优势方存在多颗大子（如双师长）或弱势方有子裸露在外，绝不判和；
     - 原型 3 改为真正的 BFS 地雷物理阻断连通性检测，仅当地雷彻底将双方棋子割裂为互不连通的子图时才判和；
     - 时钟判和严格遵循 `RuleConfig.no_capture_draw_plies`（70步）。
2. **AI 决策卡死在“AI 思考中…”（`junqi/hybrid_engine.py` & `junqi/gui.py`）**：
   - **根本病灶**：`is_dead_draw` 误判为和棋后，`HybridDecisionEngine.evaluate_position` 返回的字典遗漏了 `action_scores` 键，导致 `choose_actions` 返回空列表 `[]`；GUI 工作线程推入动作 `None`，UI 的 `_poll()` 因 `act is None` 直接跳过走法应用且未更新状态，导致界面永久死锁在“AI 思考中…”；
   - **重构方案**：
     - `HybridDecisionEngine.evaluate_position` 在和棋分支完备填充 `action_scores = [(a, 0.0) for a in acts]`；
     - `HybridDecisionEngine.choose_actions` 增加合法动作兜底：若评分列表为空且存在合法走法，强制兜底为 `[(acts[0], 0.0)]`，绝不返回空列表；全明子局面采样世界优化为单世界 `[{}]`；
     - `gui.py` 的 `ai_move()` 与 `ask_hint()` 增加顶级 `try...except` 异常捕获与合法动作安全兜底，且在 `_poll()` 中处理 `act is None` 时强制刷新界面状态并重置 `self.busy = False`，杜绝卡死。
3. **新增单元测试**：
   - 新增 `tests/test_endgame_deadlock_fixes.py`，完整覆盖截图实况局面（蓝方双师长压制营连排绝非必和、且专家估值正确评判蓝大优红大劣、AI 毫秒级生成跑营走法）、1v1 理论必和、地雷全线物理断连与 70 步限步，并通过全量 141 项回归测试。

## [2026-09-06] — 实战对局审查根因整改：胜势吃子硬门、公共 Policy 信息解耦与几率终局修复

阶段归属：**P0 阶段（规则正确性与信息边界契约）** 与 **P1 阶段（传统搜索增强与混合引擎推理约束）**。依据 `reports/expert_engine_v1/REPLAY_REVIEW_2026-09-06.md`，严格遵循 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 `AGENTS.md`。

### 核心病灶剖析与技术解法
1. **胜势吃子硬门保护与自杀判定彻底修复（`junqi/hybrid_engine.py`）**：
   - **根本病灶**：原代码在自杀检查中错误使用 `mover.rank > tgt.rank`（大子吃小子）判定自杀并扣除 100 分，直接导致军长吃排长、工兵挖雷等大量胜势吃子被判定为负分打压；
   - **重构方案**：彻底废除反向的等级比较，统一调用 `rules.battle(mover.rank, tgt.rank)`；
   - 对真实自杀（`defender_wins`）给予 `-500.0` 严惩；对胜势吃子（`attacker_wins`）给予 `+50.0 ~ +120.0` 战术保护分（工兵挖雷额外加 `+30.0`），确保合法胜势吃子绝不被无意义闲棋或翻棋掩盖；炸弹兑高价值大子给予 `+80.0` 战术换子加分。
2. **公共 Policy 与采样世界 Value 彻底解耦（`junqi/hybrid_engine.py`）**：
   - **信息边界合规**：严格践行“公共 Policy 不得读取真实暗子身份；采样世界只能用于 Value 评估”的硬约束；
   - **执行隔离**：将公共局面通过 `world=None` 单次前向推理计算唯一合规的公共 Policy（`action_scores`）；K 个采样世界仅作为批量张量输入网络的 Value 头计算胜率和期望标量，彻底消除暗子底牌漂移；新增 `test_policy_invariance_across_sampled_worlds` 验证世界不变性。
3. **翻棋几率节点统一结算终局价值（`junqi/search.py`）**：
   - 修复在几率节点 `_evaluate_chance_flip` 深层截断及全树分支中，翻开暗子后遗漏困毙终局判定的缺陷；构造翻后状态优先调用 `child.is_terminal()`，若困毙则回传精确的 `WIN_SCORE` 终局价值。
4. **GUI 引擎观测路由与对局审计元数据增强（`junqi/gui.py`）**：
   - 修复第 544 行 `self.engine_mode`（实际控件变量为 `self.ai_engine`）导致胜率显示与当前选中引擎脱钩的 Bug；
   - 对局记录（`games/game_*.json`）新增 `engine_type`、`depth`、`samples`、`model_sha256` 以及 AI 思考的 top-3 候选动作与评分快照，使对局可完全复现与回溯审计。
5. **单元测试与实战局面验证**：
   - 新增 `tests/test_replay_review_fixes.py`，全量覆盖吃子加分、自杀扣分、Policy 世界不变性与翻后困毙；
   - 实测验证 `141119` 第 5 手（红军长吃蓝排长评分 59.99 高居第 1）与 `164701` 第 141 手（蓝工兵挖红地雷评分 95.00 高居第 1）。

## [2026-09-06] — GUI 初始窗口尺寸与布局紧凑化重构（P4 阶段）

阶段归属：**P4 阶段（GUI 人机交互与可视化工程）**。遵循 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 `AGENTS.md`。

### 核心病灶与技术解法
1. **移除强制全屏最大化（`root.state("zoomed")`）**：
   - 原代码在 `junqi/gui.py` 的 `main()` 中调用了 `root.state("zoomed")`，在 Windows 高分屏（1080p/2K/4K）下强行最大化铺满屏幕；
   - 因棋盘与贴图为固定像素（520x756），且 Canvas 左浮动、Panel 右浮动，导致棋盘被甩在屏幕最左侧、面板被甩在数千像素外的最右侧，中间留下巨大空白断层。
   - 彻底移除 `root.state("zoomed")`。
2. **规范窗口初始几何尺寸与屏幕居中**：
   - 设定基准尺寸 `WIN_W = 840, WIN_H = 760`；
   - 启动时自动获取当前屏幕分辨率，精确计算居中坐标 `(x, y)` 并通过 `root.geometry()` 居中弹出；
   - 设置 `root.resizable(False, False)`，禁用失真拉伸与误触全屏。
3. **面板紧凑并列布局与对局记录滚动条支持**：
   - Panel 改为贴合 Canvas 右侧并列排列（`side="left"`），消除大缝隙；
   - 对局记录 Listbox 嵌入 `Scrollbar` 容器并启用纵向自适应伸缩，彻底解决长步数记录浏览与不同 DPI 字体下高度轻微溢出问题。
4. **统一 `cli.py` 启动入口**：
   - 将 `cli.py` 中 `args.command == 'gui'` 统一路由至 `junqi.gui.launch_gui`。

## [2026-09-06] — 残局机制性必和断言器（Dead Draw Assertion Engine）与 GUI 胜率预测脱钩重构

阶段归属：**P2 阶段（专家评估与残局攻坚）** 与 **P4 阶段（GUI 交互与胜率展现）**。遵循 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 `AGENTS.md`。

### 核心病灶与技术解法

1. **结构性必和死锁断言器（`junqi/analysis.py` -> `is_dead_draw`）**：
   - 彻底打破“单纯对子力加权乘折减系数”的局限，依据军棋四大底层机制（行营免死特权、公路同速距离守恒、70步无吃子限步时钟、图论连通性）实现硬性拓扑裁决；
   - 覆盖四大典型必和原型：
     - **原型 1（双无工兵死锁）**：双方工兵全灭且有雷护旗，拔旗通路100%封死；且双方无法全歼对方（如单方无敌司令 vs 对方多子扎营防守）；
     - **原型 2（1v1 追逐死锁）**：单大子追单小子，防守方身处行营、或距离最近行营 $\le 2$、或位于底线 1/2/3 列安全往复区，在 5x5 Mini-Junqi 穷举与大盘上证实 100% 走满 70 步和棋；
     - **原型 3（双向军旗死区）**：双方军旗皆处于不可攻破状态；
     - **原型 4（时钟极限逼近）**：`quiet >= 50` 逼近 70 步判和时限。
2. **专家评估函数前置拦截与时钟强衰减（`junqi/eval_expert.py`）**：
   - 评估入口直连 `is_dead_draw(state)`：一旦命中必和，估值瞬间截断为严格的 `0.0`，彻底消灭 +327 分的伪优势泡沫；
   - 修复工兵灭绝下的死棋估值：双无工兵时，不可移动的地雷与军旗不再算入机动攻击物质分；
   - 引入 70 步限步二次方衰减：当 `quiet >= 20` 且局势僵持时，估值按 `(1 - quiet/70)^2` 动态向 0 平滑衰减。
3. **GUI 胜率预测引擎路由解耦与顶层必和拦截（`junqi/gui.py`）**：
   - 严格根据单选框（`engine_mode`）路由，当用户选择“专家搜索”时，100% 走专家评估与和棋概率模型，不再盲目绕道离线神经网络；
   - GUI 胜率最顶层接入必和拦截：若触发 `is_dead_draw`，胜率直接显示 `红胜 0.0% | 和 100.0% | 蓝胜 0.0%`；
   - 混合引擎 `junqi/hybrid_engine.py` 同步接入必和拦截，保证全架构端到端一致性。
4. **自动化测试与对局验证**：
   - 新增 `tests/test_dead_draw_detection.py`，实测 `games/game_20260906_152706.json`（第 152 手）准确判定为 `is_dead_draw=True`，估值 = 0.0，胜率和率 = 100%；
   - 全量回归测试：**134 passed, 3 skipped**（137 项全部绿色通过）。

阶段归属：**P1/P2 阶段（传统搜索增强与专家评估校准）**。遵循 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 `AGENTS.md`。

### 核心病灶剖析与技术重构突破

1. **根治几率节点 Star1 剪枝窗口污染与算力幻觉（`junqi/search.py`）**：
   - **根本病灶**：在 `_evaluate_chance_flip` 几率节点中，原代码将父节点期望边界 `(-beta, -alpha)` 直接传入子节点 `_negamax`，由于几率期望是带权累加和并非单一极值，对手在子节点中只要找到一个高于 `-\alpha` 的常规走法（如 +10 分）便直接触发 Beta 剪枝，导致对手最佳反制（如吃子、占营 +40 分）被强制截断，使 AI 误以为“在敌方半场替敌翻棋收益极高”（计算出的负分被严重低估为 -1.8 分甚至正分），从而产生开局跨界替敌翻棋的荒谬行为。
   - **重构方案**：几率子节点 Negamax 采用全窗口（`-WIN_SCORE, WIN_SCORE`）搜索，彻底杜绝剪枝窗引起的对手反制失真；保留几率节点层面的 Star1 Fail-Low/Fail-High 极值截断，计算速度依然保持在毫秒级。

2. **杜绝暗子窥探与 6 分支概率截断偏色（`junqi/search.py`）**：
   - 彻底修复 `depth <= 1` 时调用 `state.apply(flip_act)` 偷看暗子真实底牌的违规行为，改为由公共信念状态对全部可能暗子求精确期望，100% 遵从 `AGENTS.md` 公共 Policy 禁偷看原则；
   - 移除原有硬编码 `outcomes[:6]` 截断（原字典遍历导致前 6 种包含 4 颗红子、仅 2 颗蓝子，造成 67% vs 33% 严重红方偏向并丢弃所有将官大子），恢复全暗子池真实概率分布期望。

3. **开局领地咽喉与防越界惩罚（`junqi/search.py`）**：
   - 在已定色局面的走法启发排序 `_score_action` 中，针对无己方部队就位掩护的敌方半场暗子赋予 `-100_000.0` 严厉压制，杜绝“己方未开、替敌翻棋”；若残局已有先头部队压境（`friendly_guards > 0`），则动态恢复推进翻棋权。

4. **调谐行营威慑与吃子收益，彻底根治“缩营不杀”（`junqi/config.py`, `junqi/eval_expert.py`）**：
   - 原估值中 `attack_camp` 高达 0.80，叠加大营围杀 `camp_siege=6.0` 与占营分 `camp_occ=10.0`，导致大子在营内空喊威胁高达 30.4 分，而出营实际吃子扣除离营与威胁分后反倒亏损 -12.4 分，引发 AI 宁可在营里发呆乱翻也不出营吃子的死锁；
   - 调谐 `EvalWeights.attack_camp` 为 0.20，`camp_siege` 为 3.0，且行营围杀严格限定为“营内子力足以战胜或对兑邻接敌明子”（杜绝排长在营里给邻格司令加“围杀分”的荒谬现象）；真实出营扑杀实测由原净亏损 -12.4 分转为净胜 **+9.4 分**，AI 100% 执行扑杀且吃完后顺势回营。

5. **根节点真实 Minimax 估值回传与 GUI 规范化（`junqi/search.py`, `junqi/ai.py`）**：
   - `SearchStats` 新增 `root_scores` 收集根节点各候选走法的真实 Minimax 估值；
   - `ExpertAgent.choose_actions(topn > 1)` 优先返回真实搜索分（如 +15, +8, -11），彻底消除此前 GUI 提示显示 `+1100000` 庞大启发式数字的异常。

6. **全量自动化测试与真实对战验证**：
   - 新增 `test_opening_stronghold_into_camp`（测试据点 100% 进营）与 `test_camp_outstrike_tactical_kill`（测试行营大子必扑杀敌小子）；
   - 全量回归测试通过：**130 passed, 3 skipped**（133 项测试 100% 绿色通过）；
   - 4 局真实发牌人机/自战模拟实证：AI 首翻 100% 命中八大黄金据点，翻开后 100% 走子进营，敌子靠近 100% 扑杀吃子，吃完 100% 回营固守。

## [2026-09-06] — P1 1000 局官方复盘大数据挖掘与实战棋理/规则体系实证校准

阶段归属：**P1 阶段（复盘数据挖掘、实战棋理总结与规则体系校准）**。遵循 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 `AGENTS.md`。

### 核心逆向突破与实证大盘

1. **破译 App 对局历史数据库 `list.cfg`**：
   - 逆向分析 `libjunqi.so` 证实 `军旗复盘/list.cfg` 采用 DES-ECB 加密，成功解密全部 1000 局官方终局判定（包含真实胜负、认输/逃跑/吃光/协议和棋原因码、用时等）；
   - 与移动端界面 7 盘记录逐行比对，判定结果 **100% 吻合**。全盘 1000 局权威胜率：核心玩家 458 胜、241 负、301 和。
2. **实战终局生态颠覆性发现**：
   - 常规拔旗/吃光仅占 **6.9%**，人类主动认输/退房占 **45.6%**，强退占 **15.0%**，协商求和占 **26.5%**，限步判和占 **3.1%**；
   - 决胜局平均手数仅 **89.5 手**（中位数 85 手），和棋局平均高达 **155.5 手**（中位数 150 手）；
   - **单司令决定论证伪**：261 局明确分出司令阵亡先后的对局中，先死司令一方告负占 **49.8%**，逆转获胜占 **50.2%**，胜负完全均等，彻底推翻“司令先死即必败”的传统偏见；
   - 炸弹击杀中坚小子（连/排/工/营）占 **47.5%**，实证揭示“小子占营/前线翻出敌炸后，利用刚翻开不可移动的时差，主动出营贴身拆弹”这一低成本消灭核威慑的高频实战定式。

### 文档校准变更

- `docs/03-RulesAndStrategy/JUNQI_RULES_AND_STRATEGY_GUIDE.md`：
  - 新增第 1.6 节《真实对局终局生态（基于 1000 局官方实战数据实证）》；
  - 校准第 2 章开局策略：确立《首翻子力据点化与四角行营辐射拓荒定式》（破除挑小子进营误区，基于军长据点实操三向博弈），确立**行营单向打击特权**与被翻子力战术死锁机制，增补 1000 局前 20 手大数据（进营率 82.7%、邻营翻棋率 96.2%、**前 20 手整整 50.1% 的吃子直接源自行营扑杀**）；新增《占营比例与胜率量化实证矩阵（5:5、6:4、7:3、8:2 等）》，给出边际胜率与引擎评估参数映射；
  - 校准第 3 章战术体系：增补 3.3 炸弹实战消耗与“小子贴身拆弹”定式、3.4 司令战损梯队接管定理；
  - 校准第 4 章残局与胜负判定：增补 4.1~4.3 认输折叠线（Resignation Collapse）与 301 局和棋典型定式。
- `docs/03-RulesAndStrategy/JUNQI_CHESS_INTELLECT_V2.md`：
  - 新增第 2.8 节《实战开局定式：首翻子力据点化与行营连锁拓荒（基于 1000 局官方实证）》，包含 5:5、6:4、7:3、8:2 占营比胜率矩阵与量化影响因子；
  - 校准第 4 章：引入 80~90 手决胜黄金窗口期模型，新增第 4.6 节《实战认输折叠线与终端时界建模》；
  - 校准第 5 章：新增第 5.3 节《实战子力价值校准专题》，建立“二线梯队火力网补偿系数”与工兵“挖雷期权”非线性折现模型；
  - 扩充第 7 章：新增定式 4（双无工兵死和定式）、定式 5（行营避险死守定式）、定式 6（军旗贴身绝杀定式）、定式 7（单大子巡场定式）。
- `scripts/mine_replays_report.py` 与 `reports/replays_1000_mining_report.md`：
  - 新增完整的 1000 局端到端自动化挖掘脚本与全面量化分析总览报告。
- **工程结构清理与旧模型归档**：
  - `models/` 目录清理：移除 2.41GB 庞大历史经验池缓存 `candidate_latest_buffer.pkl`、门控快照 `_candidate_gate.pt`、历史候选与测试模型目录（`models/test/`, `test_parallel/`, `test_v2/`），总计释放约 **2.62 GB** 空间，保留 `best.pt`、`bc_best.pt`、`value_distilled.pt`、`pool/`、`releases/` 等全部核心资产；
  - 根目录结构规整：将散落根目录的历史专家引擎报告归入 `reports/expert_engine_v1/`、仿真测试脚本归入 `scripts/playtest/`、复盘初筛中间工具归入 `scripts/replay_tools/`、中间缓存数据归入 `reports/replay_caches/`，清理临时日志文件，使根目录恢复清爽标准布局；
  - 全量回归测试：124 项测试（121 passed, 3 skipped）全量通过。
- **全局方案与代理执行契约校准**：
  - `AGENTS.md`：根据破译的官方 `list.cfg` 澄清第二条硬约束，精准界定认输/常规终局（Value=±1）、协议和棋（Value=0）与早期逃跑强退样本边界，增补“行营单向打击特权与据点拓荒”战略硬约束；
  - `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`：全面校准第 3.2 节复盘数据真实终局分布（可用真实胜负 Value 样本从原估 68 局跃升至近 800 局），修正第 6 节 Value 标签协议为官方三级体系，更新第 2.1 节开局进营率（82.7%）与行营扑杀吃子（50.1%）参数，更新第 8 节禁止事项。

### 算法引擎与数据管线重构落地（P1/P2 阶段）

1. **复盘行为克隆数据管线解封（`junqi/dataset.py`）**：
   - 引入 DES-ECB 解密与 `load_list_cfg_metadata` 元数据加载器，无缝绑定对局元数据；
   - 彻底废除“未自然拔旗即丢弃 Value”的旧逻辑，按官方三级终局体系精确注标：
     - code 21（认输）与 code 1（常规胜利）：按胜者注入 \(z \in \{+1, -1\}\)；
     - code 40/42/43（协议和棋/限步判和）：注入 \(z = 0\)；
     - code 20（早期逃跑/强退）：过滤丢弃 Value，仅用于策略 Imitation。
   - 真实有效 Value 样本由 68 局暴增至近 800 局，彻底根治 BC 价值头塌缩为 0 的历史顽疾。

2. **传统搜索与专家评估体系升级（`junqi/config.py`, `junqi/eval_expert.py`, `junqi/search.py`）**：
   - **二线梯队火力网接管补偿（`echelon_si_compensation`=18.0）**：实证司令先死胜负均等（50.2% vs 49.8%）。当司令战损但盘面已就位参战军长、双师长或炸弹时，给予梯队接管补偿，打破“司令先死即恐慌弃疗”的旧误区；
   - **占营比例非线性矩阵增益（`camp_matrix_weight`=12.0）**：实证占营比例非线性映射胜率（5:5 38.2% -> 6:4 52.1% -> 7:3 68.4% -> 8:2 83.3%），对净胜 \(\ge 2\) 营的结构性优势注入阶梯式非线性增益；
   - **行营单向打击与辐射拓荒启发（`camp_outstrike_bias`=25.0, `camp_adjacent_flip_bias`=30.0）**：将行营出击吃子（实证占开局吃子 50.1%）与四角行营辐射翻棋（实证邻营翻棋率 96.2%）内嵌至走法排序；
   - **小子主动贴身拆弹排序（`bomb_suicide_exchange`=35.0）**：实证炸弹杀伤 47.5% 为小子，利用敌炸刚翻开无法移动的时差，优先以连/排/工/营主动兑换敌方高危炸弹；
   - **残局工兵死锁折现**：全盘工兵残缺（\(\le 2\) 颗，实证占和棋 65.4%）且地雷守旗时，估值自动向和棋折现收敛；
   - 修复专家模块类型适配器导入（`expert/rule_validator.py`, `expert/tactical_analyzer.py`）。

3. **单元测试与全量回归验证（`tests/`）**：
   - 新增 `tests/test_p1_discovery_refactor.py`（7 项针对性实证实装测试，覆盖 DES 终局注入、行营出击、梯队补偿、拆弹排序等）；
   - 全量回归测试通过：**128 passed, 3 skipped, 0 failed**（耗时 25.89s），100% 绿色通过。


## [2026-09-01] — 用户实测“变弱”复诊：确认系基线切换预期行为 + 蒸馏步骤被跳过，补齐两处防护

阶段归属：**P3 收尾 / P4 准入复审**。

### 复诊结论（`_verify_rebase_state.py` 实证）

- `best.pt` 确已被 BC 重建（与 `bc_best.pt` 仅差 S2 新增的随机初始化 `aux_head` 键），rebase 生效；经验池 162k 样本为 S1 时期 3 轮新训自产，非旧数据残留；
- “变弱”的真实构成：① GUI 加载的 `best.pt` 从“全和塌缩旧模型（靠拖和显得不败）”换成了价值头未校准的 BC 基线；② 新训候选启动前**未执行 `distill_value` 蒸馏预热**（推荐流程被跳过），`ref_vs_search2` 0.139→0.278→0.222 处预期初期谷底；③ S2 升温/课程使采样更激进，短期观感更松。
- 处置：已启动正式蒸馏（1200 样本）生成 `models/value_distilled.pt`；后续训练将自动热启动自蒸馏产物。

### 变更

- `junqi/train_rl.py`：`--rebase-baseline` 清理清单补上 `_candidate_gate.pt`（防止旧门控快照残留）并注明经验池必须同步清空的原因；BC 热启动分支在检测不到 `value_distilled.pt` 时打印显式警告（防止再次跳过蒸馏直接训练导致初期变弱）。

## [2026-09-01] — S2 价值信号修复实施（公共模式样本、专家蒸馏、决胜课程、辅助锚定、温度上调）

阶段归属：**P3 收尾 / P4 准入复审**。依据 [`docs/TRAINING_ROOT_CAUSE_REVIEW.md`](TRAINING_ROOT_CAUSE_REVIEW.md) §S2；长期挂机禁令仍保持，实施后需按 S3 复验。

### 变更

- `junqi/train_rl.py`：
  1. **公共模式 Value 样本**：自对弈每手根局面（公共模式编码）以终局 z 为标签同步入池；`sample_value_batch` 改为“类配额 × 模式配额”双层抽样，公共/世界按 `PUBLIC_VALUE_RATIO=0.5` 混合，稀有类跨模式补足不被淹没；旧 3 元组样本按世界模式兼容；修复“训练全世界模式、评测全公共模式”的分布失配；
  2. **决胜课程**：残局课程子力失衡幅度扩至 ±0.6，且 |mb|≥0.3 时禁用堡垒，优先生成可破局局面（制造 Win/Loss 样本）；
  3. **中盘注入**（`--midgame-prob` 默认 0.1）：按概率从 `eval_sets/midgame.jsonl` 抽取起始局面（Lc0 开局多样性类比，文件缺失自动退回完整发牌）；
  4. **温度表上调**：{开局 1.0→1.2，中盘 0.6→1.0，尾盘 0.2→0.5}，恢复趋和局面下的探索；
  5. **辅助回归损失**（权重 0.1）：监督编码器通道 25 的 `material_diff`（无需额外存储），为主干提供密集锚定信号；`train_epoch` 返回四元组，日志新增 `aux_loss`；`load_checkpoint` 改 `strict=False` 兼容旧检查点；
  6. 热启动链新增优先级：`candidate_latest` > `value_distilled.pt` > `bc_best.pt` > `best.pt`。
- `junqi/net.py`：新增 `aux_head`（Tanh 回归，输出 [-1,1]）与 `forward_with_aux`（训练专用）；`forward` 保持 2 元组接口不变，推理链路零改动。
- `junqi/train_value_distill.py`（新增）+ `distill_value` 子命令：**专家价值蒸馏预热**——随机推进/残局生成器采样局面，`ExpertAgent`（Star1+QSearch，深度/时限可配）打 W/D/L 伪标签，**仅训练价值头**（主干与策略头冻结，保护 BC 策略能力），按验证集最优落盘 `models/value_distilled.pt`。属监督信号非奖励塑形，不改变终局回报。
- 合规：未触碰 `RuleConfig`/APK 规则；Value 标签仍为纯终局结果 `z∈{+1,0,−1}`；和棋语义不变。

### 验证

- 新增 `tests/test_s2_fixes.py` 12 项（温度表、课程/中盘注入确定性、公共/世界样本结构与混合比、3 元组兼容、辅助头形状/值域/接口兼容、蒸馏映射/冒烟/冻结验证）；
- 全量单测 **124/124 通过**（含修复的 `sample_value_batch` 稀有类淹没回归用例）。
- 推荐后续流程：`python -m junqi distill_value --samples 1200`（约数分钟）→ 短周期训练复验（S3）。

## [2026-09-01] — 训练停滞根因审查落盘 + S0/S1 整改实施（靶场修正、基线重建、门控与对手池解锁）

阶段归属：**P3 收尾 / P4 准入复审**。完整分析见 [`docs/TRAINING_ROOT_CAUSE_REVIEW.md`](TRAINING_ROOT_CAUSE_REVIEW.md)；实证脚本 `_verify_root_cause.py`；P4.4 长期挂机维持不通过。

### 审查核心结论（新增实证）

- `best.pt` 在 50 题靶场 **50 题全预测 Win**（0 Draw），发布基线自身类别塌缩；候选全 Draw 塌缩；门控实为两个塌缩模型对抗；
- 经验池 46,709 条 Value 样本中 Draw 占 **85.6%**（Win 10.5% / Loss 3.9%）；
- 旧门控判据（子阶段 Wilson 下界 >0.5）在每阶段 20 局样本下需得分 ≥0.75，数学上不可达，构成“永不晋升→池永不扩张→永不提升”死锁；
- 对手池注入依赖晋升而从不生效；对手走子被写成 one-hot 策略目标（违背 AlphaZero 语义）；
- Value 训练样本全为世界模式，而靶场/GUI/混合引擎全在公共模式评测，分布失配。

### S0 变更（干净基线与测量修复）

- `junqi/benchmark.py`：中盘 15 题 `true_value` 改为与 `true_val_class` 一致的极端值（Win→+1 / Loss→−1 / Draw→0），消除连续真值与三分类头数学互斥导致的不可达 MAE 目标；新增 **Brier 分数**与**预测熵**监控；塌缩告警阈值 90%→**70%**；看板同步新增两行指标。
- `junqi/train_rl.py`：无检查点热启动优先 `bc_best.pt`（策略头 Top-1 81.5% 可用），不再继承旧缺陷 `best.pt` 的塌缩权重；新增 `--rebase-baseline`：备份旧 `best.pt` 为 `best_legacy.pt`、用 BC 模型重建发布基线并清空候选进度。
- 基线快照落盘 `metrics/baseline.json`（三模型哈希 + 修正后靶场指标；收紧阈值后三模型均正确触发塌缩告警）。
- `__main__.py`：`train_rl` 新增 `--rebase-baseline`；`--eval-games` 默认 10→**32**；`--ref-games` 帮助文本改为“晋升硬条件”。

### S1 变更（门控与对手池解锁）

- `junqi/train_rl.py`：
  1. 新增 `OPP_MIX` 显式配比（50% 镜像 / 25% best / 10% expert / 10% greedy / 5% random）与纯函数 `opponent_type_for`；已发布模型权重**无条件**注入 Worker，“25% best 对抗”首次真实生效；
  2. Worker 返回逐局对手类型，主循环汇总并写入 `elo_history.jsonl` 的 `opponent_mix` 字段；
  3. **删除对手走子的 one-hot 策略目标**：对手手不再产生任何训练样本，策略目标仅来自训练方自身 MCTS 访问分布；编码也仅在训练方走子时执行；
  4. `decide_promotion` 修订：改善条件改为**整体得分 Wilson 下界 >0.5**；新增 `ref_score/prev_ref_score` 硬条件（≥0.5 且不低于上轮，未跑参考对抗时跳过）；`strict_stages=True` 保留给每场景 ≥200 局的正式晋升协议；
  5. **Elo 与晋升解耦**：每轮按门控整体得分更新，不再恒 1500。
- 未触碰：`RuleConfig`/APK 规则、终局奖励 `z∈{+1,0,−1}`、和棋标签语义（符合硬约束）。

### 验证

- 新增 `tests/test_s0_s1_fixes.py` 13 项（标签一致性、Brier/熵、塌缩阈值、晋升新判据含 ref 硬条件与严格协议、对手配比边界/经验分布、对手样本隔离、含对手可复现性）；
- 全量单测 **112/112 通过**；
- 端到端冒烟（`models_smoke`，1 epoch×4 局，已清理）：bc 热启动、对手占比日志、阶段拆分门控、拒绝晋升时 Elo 仍更新均正常。
- P4 复审表状态未变：Value 健康度仍未通过（候选 MAE 0.57 / 准确率 20%），**长期挂机禁令保持**；下一步按审查报告 S2（价值信号修复）实施。

## [2026-09-01] — P4.4 长期训练双门槛与熔断规范落盘

阶段归属：**P3 收尾 / P4 准入复审**。

### 变更

- 在 `docs/P4_EXECUTION_PLAN.md` 明确区分“允许启动长期训练”和“允许候选覆盖 `best.pt`”两个独立开关，补充量化硬门槛、每场景至少 200 局的正式晋升协议及 Wilson 条件。
- 新增长期训练自动熔断阈值：重复率、Value 单类占比、MAE/准确率退化、胜负信号、对手实际占比以及非法动作/NaN/断点恢复异常。
- 记录从当前状态到长期训练的分级动作：选定干净基线、补齐对手日志与自动暂停、64 局短测、中等规模 3-5 轮验证、120-200 局长期放量、独立正式晋升。
- 根据最新 Epoch 6 证据更正准入状态：重复问题专项修复有效，但 Value MAE `0.5690`、准确率 `20.0%`、Win 预测为 0，候选仍未晋升，因此长期训练禁令保持。
- 同步更新 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 与 README；本次仅修改文档，未修改训练代码或启动训练。

## [2026-09-01] — P3 收尾与 P4 准入复审问题专项整改与冒烟闭环

阶段归属：**P3 收尾 / P4 准入整改**。

### 核心整改与实证数据

1. **打通 `history_counts` 与 `avoid` 全链路**：
   - `junqi/mcts.py`：扩展 `MCTS.search` 支持 `history_counts` 与 `avoid`，在树内模拟中动态追踪 `sim_seen`，命中 3 次重复立即判定和棋并回传 0 价值；在根节点严厉惩罚命中 `avoid` 的走法，彻底封锁第 3 次重复；
   - `junqi/ai.py`：`NNAgent.choose_actions` 与 `predict_state` 全面透传 `avoid` 与 `history_counts`；
   - `junqi/selfplay.py`：所有 `Strategy` 子类均支持透传 `history_counts=seen`。
2. **重构 Value 经验池类别均衡采样（破除 Draw 塌缩）**：
   - `junqi/train_rl.py`：`StratifiedReplayBuffer` 将 Value 流改造为 Win / Draw / Loss 三个独立分桶，实现 1/3 类别均衡抽样，并支持完整序列化持久化（`_buffer.pkl`），彻底杜绝和棋样本主导梯度的数学闭环。
3. **训练 Worker 接入多样化对手池**：
   - `junqi/train_rl.py`：自博弈 Worker 混合 50% 镜像对弈、25% best 对抗、20% 专家/贪心搜索（`ExpertStrategy`/`AgentStrategy`）与 5% 随机扰动，打破对称镜像循环。
4. **校准贝叶斯信念先验与持续追踪**：
   - `junqi/belief.py`：将初始先验校准为翻棋发牌的精确均匀边缘分布（`state.marginal()`），清理暗子攻击等不合规逻辑；
   - `junqi/hybrid_engine.py`：支持外部持续信念追踪，避免单步评估暴力 reset。
5. **32 局冒烟实证与全量测试证据**：
   - **重复和棋率**：从整改前的 **88.5%（85/96）断崖式下降至 0%（0/32）**，门控全部对局均由规则 `no_capture` 或 `immobilized` 终结；
   - **Policy Loss**：单调持续降至 **`2.7214`**（优化完成总 Loss `2.9545`）；
   - **Value 预测分布**：50 题靶场预测分布恢复为 `胜 0 / 和 35 / 负 15`，成功打破 50/50 纯 Draw 塌缩；
   - **单元测试**：新增 4 项专项测试，全量规模达 **99 / 99 项，100% 通过**。


## [2026-09-01] — P4.4 长期挂机准入复审

阶段归属：**P3 收尾 / P4 准入复审**。

### 变更

- 将 P4.4 长期挂机状态更正为**未通过准入**：Epoch 3-5 门控合计 1 胜 / 95 和 / 0 负，85 局因重复结束，所有候选均未晋升。
- 记录候选 Value Draw 塌缩：50 题全部预测 Draw，MAE `0.4223`，表面 `50.0%` 准确率仅来自题库中 25 道 Draw。
- 在 `docs/P4_EXECUTION_PLAN.md` 补充 NN 丢弃 `avoid`、MCTS 历史盲区、Draw 标签反馈环、训练对手池未接入及 P4 原型未进入训练闭环等根因。
- 同步纠正 README、总执行方案和知识文档中“循环已根除”“P3.3 已验收”的过度结论；本次仅做诊断与文档更新，未修改训练代码，未启动长期训练。

## [P4.3 GUI 实时三分类胜率预测与智能求和/认输交互上线] - 2026-09-01

阶段归属：**P4.3**（GUI 人机界面与交互增强）。

### 核心实现与交互功能

1. **GUI 动态三色态势评估进度条（`junqi/gui.py`）**：
   - 接入 `HybridDecisionEngine.evaluate_position`，以三色分段（红胜/橙红、和棋/灰、蓝胜/蓝）直观展示盘面态势；
   - 动态更新当前局面三分类概率文本（`胜率预测: 红胜 xx% | 和 xx% | 蓝胜 xx%`）。
2. **AI 智能求和判定机制**：
   - 玩家点击求和时，AI 根据当前后验胜率与死区和棋概率智能决策：AI 预估胜率 $\le 55\%$ 或和率 $\ge 45\%$ 时接受和棋，优势明显时礼貌拒绝并提示对局。
3. **AI 引擎级别选择器升级**：
   - GUI 选项中原生支持 **“P4混合智能 (P4 Hybrid)”** 作为最高难度推荐引擎。


## [P4.1 贝叶斯暗子信念跟踪器与 P4.2 多世界混合决策引擎上线] - 2026-09-01

阶段归属：**P4.1**（贝叶斯暗子信念跟踪）与 **P4.2**（多世界混合决策引擎）。

### 核心实现与交付物

1. **P4.1 贝叶斯暗子信念跟踪器（`junqi/belief.py`）**：
   - 实现了逐格暗子后验概率分布矩阵 $P(\text{PieceType} \mid pos)$ 与严格概率归一化；
   - 注入布局空间先验（底二排/大本营地雷与军旗高发权重，前锋排地雷清零）；
   - 支持翻棋观测、战斗结果推断的贝叶斯动态更新；
   - 实现了高效加权无放回确定化多世界采样器 `sample_world` 与 `sample_k_worlds`。
2. **P4.2 在线多世界采样与混合决策引擎（`junqi/hybrid_engine.py`）**：
   - 支持采样 $K$ 个可能世界打包执行 GPU 张量批处理推理；
   - 集成一步吃旗（Instant Flag Capture）、困毙终结（Immobilization Win）、自杀送子剪枝与循环判和惩罚等轻量战术规则；
   - 提供 `evaluate_position` 态势评估接口，输出胜/和/负概率、期望估值与推荐动作。
3. **专项单测与全量回归（`tests/test_p4_hybrid.py`）**：
   - 新增 6 项 P4 专项单元测试，全量单测规模扩展至 **95 项，持续保持 100% 通过**。


## [P3.3 门控自博弈强化实证 Epoch 3~5 训练与 50 题靶场全面上线] - 2026-09-01

阶段归属：**P3.3**（门控强化实证）与 **P3 准入补证**。

### 核心训练进展与实证指标

1. **原生 38 通道与 50 题靶场闭环验证（`junqi/train_rl.py`, `junqi/benchmark.py`）**：
   - 全面在原生 38 通道 + 三分类 Softmax Value Head 架构下完成 Epoch 3 ~ 5 自博弈强化学习；
   - 回放经验池分层沉淀：总 Policy 样本达 **19,318 条**，总 Value 样本达 **153,989 条**。
2. **多维指标记录（后续复审确认不构成收敛证据）**：
   - **Policy Loss**：单调持续下降（Epoch 1: `3.5615` → Epoch 3: `3.3420` → Epoch 4: `3.2040` → Epoch 5: **`3.0234`**）；
   - **Value 损失**：降至 **`0.2025`**；
   - **50 题固定标杆靶场**：50 题全覆盖（15 开局 + 15 中盘 + 20 尾盘），但候选对 50 题全部预测 Draw；`50.0%` 准确率来自其中恰有 25 道 Draw；
   - **门控对抗结果**：Epoch 3-5 合计 1 胜 / 95 和 / 0 负，其中 85 局重复判和，所有候选均 `promoted=false`；“0 败”不能解释为稳定性或棋力提升。
3. **架构与工程质量保障**：
   - 彻底统一所有磁盘检查点（`best.pt`, `candidate_latest.pt`, `bc_best.pt`）为原生 38 通道三分类；
   - 全量单元测试持续保持 **89 / 89 100% 通过**。


## [P3.1 架构升级与 P3.2 实验靶场/多维指标看板上线] - 2026-08-31

阶段归属：**P3.1**（网络/MCTS基础架构升级）与 **P3.2**（实验靶场与多维看板建设）。

### 核心实现与产出

1. **输入特征 38 通道与重复计数注入（`junqi/encoder.py`）**：
   - 依据 2023 JAIST CLAP 框架实证，将状态编码扩展为 **38 通道**；
   - 新增通道 36（`seen >= 1`）与通道 37（`seen >= 2`），为神经网络提供重复历史输入接口；后续复审确认 MCTS/NN 门控尚未把真实历史传入搜索链路。
2. **神经网络 Value 头改造为三分类概率头（`junqi/net.py`）**：
   - Value 输出重构为 3 维 Softmax：$[P_{\text{win}}, P_{\text{draw}}, P_{\text{loss}}]$；
   - 彻底解耦“动态均势（$V=0$）”与“死区和棋（$V=0$）”，期望胜负标量由 $P_{\text{win}} - P_{\text{loss}}$ 换算；
   - `JunqiNet.load_from_file` 与 `forward` 具备双向通道裁剪与补零能力，实现与旧版 36 通道/1 维模型的 100% 向后兼容。
3. **MCTS 探索超参数对齐黄金区间（`junqi/mcts.py`）**：
   - 严格采用 2018 论文推荐的黄金收敛参数：$c_{\text{puct}} = 0.6$，根节点 Dirichlet 噪声 $\alpha = 0.15, \epsilon = 0.20$；
   - 降低高 $c_{\text{puct}}$ 导致随机漫步的风险；后续 1 胜 95 和结果证明，超参调整不能替代历史感知和树内重复终局建模。
4. **军棋 AI 实验靶场与指标看板（`junqi/benchmark.py` & `metrics/`）**：
   - 构建包含 50 个经典残局、中盘战术、开局发牌的固定测试集（附带 Star1 专家引擎计算的标准真值）；
   - 自动化追踪：Value MAE（残局绝对误差）、三分类准确率、Elo 梯队分差、和棋率与重复率；
   - 命令行支持 `python -m junqi benchmark --model ...`，自动输出结构化 JSON 与 Markdown 仪表盘至 `metrics/`。
5. **单元测试扩充**：全量单元测试扩充至 **89 / 89 项 100% 通过**。


## [规则修正与 UI 胜率预测/求和功能上线] - 2026-08-31

### 核心变更与修正

1. **铁路网边界修正（`junqi/rules.py`）**：
   - 修正两侧纵向铁路（col 0 与 col 4）：纵向铁路仅在 row 1..10 之间运行，**不到达双方大本营底线（row 0 与 row 11）**；
   - 棋盘渲染与走法生成自动对齐，杜绝沿铁路线直达底线的非标走法。
2. **工兵路径阻挡修正（`junqi/state.py`）**：
   - 修正工兵铁路飞行规则为标准军棋规则：**工兵不可越过轨道上的棋子移动**；遇到空轨道可继续通行与转弯，遇到障碍物（己方子/暗子/敌子）必须阻挡，仅可对敌子作为攻击终点，不可穿透。
   - `RuleConfig` 中新增 `engineer_can_fly_over_pieces` 配置项（默认 `False` 严格遵守标准规则；`True` 保留对旧版 APK 特殊复盘回放的兼容）。
3. **GUI 胜率预测（`junqi/gui.py`）**：
   - 右侧操作面板新增实时 **胜率预测组件**（红蓝胜率百分比 + 彩色动态胜率对比条 + 50% 居中基准线）；
   - 支持神经网络 Value 输出与专家估值 Sigmoid 映射双模式。
4. **GUI 人机求和选项（`junqi/gui.py`）**：
   - 新增 **【求和】** 交互按钮；
   - 用户提出求和时，AI 根据当前局面的胜率评估做智能决策：
     - AI 胜率 $\le 58\%$（局势胶着/劣势）：AI 同意和棋，判定为“双方协议和棋 (`draw_agreement`)”；
     - AI 胜率 $> 58\%$（优势明显）：AI 礼貌拒绝和棋并提示胜率，继续对局。
5. **单测全绿**：全量单元测试扩充至 **85 / 85 项 100% 通过**。


## [P2 阶段完成：行为克隆 Policy 训练与 HybridAgent 混合引擎上线] - 2026-08-31

阶段归属：**P2**（依据 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §5 P2）。

### 核心实现与产出

1. **行为克隆训练管线 (`junqi/train_bc.py`)**：
   - 基于 P1 导出的 16.9 万样本训练公共 Policy-Value ResNet，在 RTX 4080 SUPER 上 15 轮训练仅耗时 162 秒；
   - 严格在合法动作上做 Masked Softmax，Value 仅在真实终局样本上反向传播；
   - 产出最佳权重 `models/bc_best.pt`，验证集 Top-1 准确率达 **87.17%**，Top-3 达 **91.15%**。
2. **独立测试集评测 (`junqi/eval_bc.py` & `reports/p2_bc_report.md`)**：
   - 在 200 局独立测试集（22,626 plies）上进行无偏评测：
     - **Top-1 准确率**：**81.51%**（对比随机基准 2.34% 提升近 40 倍）；
     - **Top-3 准确率**：**87.05%**；
     - **Top-5 准确率**：**90.01%**；
     - **非法动作预测率**：**0.00%**。
3. **混合引擎代理 (`HybridAgent` in `junqi/ai.py`)**：
   - 实现“深度神经网络全局大局观先验 + 专家搜索引擎 (Star1 + QSearch) 浅层战术把关”；
   - 彻底杜绝贪吃陷阱与漏看一步吃旗，在实战对抗中以 **58.3% 得分率** 战胜传统基线且保持对 Expert2 零败率；
   - 在图形界面 (`junqi/gui.py`) 中升级为默认人机对战引擎。
4. **单测全绿**：全量单元测试扩充至 **83 / 83 项 100% 通过**。


## [P1 验收完成：复盘数据集标准导出与基线就绪] - 2026-08-31

阶段归属：**P1**（依据 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §5 P1 与 §6 数据规范）。

### 完成内容

1. **`junqi/dataset.py`（标准数据集导出与切分）**：
   - 严格按对局（Game）切分 Train (80%, 1600局/16.9万plies) / Val (10%, 200局/2.0万plies) / Test (10%, 200局/2.2万plies)，禁止按 ply 随机切分；
   - 提取 212,548 个公共 Policy 样本（(36,12,5) 状态张量、3650 维动作掩码、阶段标签）；
   - 严格执行 Value 标签规范：仅明确胜负与规则和棋写入 Value (23,156 样本)，未终局与特殊中止局标记 `has_value=False`，彻底防止虚假和棋污染；
   - 产出数据集附带版本 `1.0.0`、种子 `2026`、SHA-256 哈希与 `metadata.json` 溯源记录；
2. **基线指标与验收报告**：
   - 完成合法随机动作基线评测（Top-1: 2.34%, Top-3: 7.02%, CE: 3.790）；
   - 生成完整验收报告 `reports/p1_dataset_report.md`；
   - 新增 `tests/test_dataset.py` 单测，全量单元测试 **80/80 全部通过**。

**P1 阶段所有任务与验收指标全部达成，正式具备进入 P2（行为克隆与搜索蒸馏）的全部条件。**


## [借鉴暗棋 CDC 专家算法升级传统搜索引擎与评估体系] - 2026-08-31

阶段归属：**P1**（依据 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §4.2 第一条线与 §5 P1 阶段）。

### 算法重构与新增（全面吸收计算机暗棋/半棋博弈研究成果）

1. **`junqi/zobrist.py`（64 位 Zobrist 哈希系统）**：
   - 覆盖 60 格棋盘 × 12 军衔 × 2 色 × 明暗状态 + 轮次 + 暗子池指纹。
   - 提供公共视角与完全信息视角的快速哈希计算与增量签名。
2. **`junqi/tt.py`（高效置换表）**：
   - 支持 `FLAG_EXACT`、`FLAG_LOWER_BOUND`、`FLAG_UPPER_BOUND` 边界剪枝。
   - 深度优先替换策略，支持 PV 最佳着法跨深度迁移与排序。
3. **`junqi/eval_expert.py`（专家级动态评估体系）**：
   - **动态制霸矩阵 (Dominance)**：双方司令/军长/工兵/地雷生存状态动态联动（敌司令阵亡司令称霸、工兵全灭地雷军旗永久安全、炸弹联动）。
   - **机动力与死子惩罚 (Mobility)**：精准统计畅通步数，重度惩罚无法动弹的受困大子，奖励铁路通达。
   - **局部翻棋安全指数 (Flip Safety)**：分析暗子邻域己方护卫与敌方虎口态势。
   - **行营控制与死区和棋势能**：集成 `camp_zone`、`camp_siege`、`fortress_score`。
4. **`junqi/search.py`（专家级现代搜索核心）**：
   - **Star1 期望极大极小 (Expectiminimax)**：几率节点概率分支与 Star1 上下界剪枝，杜绝 PIMC 策略融合与透视眼幻觉。
   - **静态搜索 (Quiescence Search)**：叶子节点扩展吃子与营地避险，Stand-Pat 剪枝彻底消除地平线效应与贪吃诱饵陷阱。
   - **多级走法排序**：`TT Move` → `MVV-LVA 吃子` → `安全翻棋` → `杀手着法` → `历史启发` → `普通走步`。
   - **迭代加深 (IDS) 与时间管理**：支持毫秒级时间预算（`time_limit_ms`）与防超时回退。
5. **系统兼容与对接（`junqi/ai.py` & `junqi/selfplay.py`）**：
   - 新增 `ExpertAgent` 与 `ExpertStrategy`（支持 `expert2`, `expert3` 等策略规格）。
   - 保持与原有 `Agent`、`NNAgent`、`gui.py`、`calculator.py` 的 100% 向后兼容。

### 验证

- 新增 `tests/test_zobrist_tt.py`、`tests/test_expert_eval.py`、`tests/test_search_expert.py`（12 项单测）。
- 全量单元测试 `python -m unittest discover tests`：**77/77 全部通过**。
- 性能实测：
  - 开局搜索单步耗时由 1647ms 降至 **11.6ms（提速 140x+）**；
  - 中盘搜索单步耗时由 1917ms 降至 **947ms（提速 2x+）**；
  - 静态搜索成功识破并规避司令贪吃排长诱饵被炸的地平线陷阱。


## [拟合权重晋级验证通过] - 2026-08-31（第四批）

阶段归属：**P1**（基线 §3.2 / §7）。

- `search2(拟合权重) vs search2(默认权重)` 镜像 200 局、先后手各半、固定种子 20260831：
  **胜 39 / 和 142 / 负 19，纯得分 0.550，不败率 Wilson 下界 0.856 → 晋级判定通过**；
- 结果已存 `reports/fit_weights.json` 的 `metrics.mirror_verify`；
- 注意：该验证运行于 A2 增强合入前的旧估值口径（进程内存），双方对称受影响，结论仍有效；
- **暂未写入 `config.py` 默认值**：待 A3 验收（三项增强镜像）完成后一次性合入，
  避免两个变量同时改动导致无法归因。


## [A2 传统搜索显式判断增强] - 2026-08-31（第三批）

阶段归属：**P1/P2**（基线 §4.2 第一条线：传统搜索增强）。红测先行（`tests/test_ai.py` 6 项）。

### 新增：三类显式判断（`ai.py` `evaluate()` + `config.py` `EvalWeights`）

| 权重键 | 默认值 | 含义 |
|---|---|---|
| `camp_zone` | 2.0 | 行营势力：已方/敌方活动明子贴近空行营的净控制差（占营/扩张准备） |
| `fortress` | 25.0 | 死区势能：双方 `fortress_score` 差——劣势方封死死区≈锁定和棋，估值注入“和棋势能” |
| `hidden_tempo` | 6.0 | 暗子时差：活动明子数差（暗子激活需先翻后走两回合） |

- 序列化向后兼容：`to_dict` 含新键；`from_dict` 对旧字典（无新键）取默认值。
- `fit_weights.json` 旧权重可直接加载，不受影响。

### 顺手修复：`evaluate()` 暗子期望口径双重缺陷（§2.2 信息边界）

- 旧实现在公共局面直接读暗子真实 `pc.rank` 计期望（`hidden_val`）——**暗子身份泄漏**，
  且“存活数 − 明子数”口径双减阵亡子造成系统性低估；
- 现改用 `remaining_types()` 精确剩余池（纯公共信息）按颜色摊派；
- PIMC 全翻开世界里两口径重合，世界内搜索行为不变（故此前未被镜像对局暴露）。
- 探针实证：5 子尾盘公共局面旧口径估值偏差约 −60 分量级。
- 新增权重键默认值非零：`greedy/search2/search3` 行为自本版本起含增强项；
  “旧行为”基线可用三新键置 0 复原（A3 镜像验证口径）。

### 验证

- `tests/test_ai.py` 6 项（三项方向性 + 关闭开关对照 + 序列化往返/旧字典兼容 + PIMC 冒烟）；
- `python -m unittest discover tests`：**65/65 通过**。
- A3 效果验收（`search2(增强) vs search2(新键置0)` ≥200 局镜像）待后台拟合权重验证完成后执行。


## [P0 收尾：训练闭环修复] - 2026-08-30（第二批）

阶段归属：**P0**（§3.1.3 / §3.1.4 / §3.1.5）。至此 P0 五项正确性问题全部修复。

### 修复与新增（junqi/train_rl.py V2.3、junqi/mcts.py、junqi/__main__.py）

- **§3.1.3 candidate/best 分离**：门控失败不再回滚候选模型训练进度；
  `best.pt` 仅在晋升时更新。新增 `save_checkpoint`/`load_checkpoint`：
  每轮保存完整检查点 `candidate_latest.pt`（网络 + 优化器 + Python/PyTorch
  随机源状态 + 元数据），启动时自动断点续训；`--fresh` 可忽略检查点。
  注：经验回放池不随检查点持久化，续训时重建（已在代码注释标明）。
- **§3.1.4 门控重设计**：废弃 40 局同模型镜像（历史上 3 轮 0胜0负40和、无区分度）。
  新门控 `evaluate_gate`：候选 vs 已发布 best 在 opening/midgame/endgame 三套固定
  评测集 + 随机完整发牌上对抗，先后手各半、跨轮固定种子；按阶段拆分胜/和/负、
  终局原因；晋升判定 `decide_promotion`（Wilson 区间）：整体不败率下界 ≥0.5、
  无阶段得分 <0.3、至少一阶段得分下界 >0.5；另附 vs search2 参考对抗（`--ref-games`，
  仅记录不阻塞）。正式晋级建议 `--eval-games ≥200`（§7 协议）。
  对局循环复用 `selfplay.play_game`，规则/循环判和口径与正式评测一致。
- **§3.1.5 随机源统一**：`mcts.py` Dirichlet 噪声改由实验种子派生的独立
  `np.random.Generator`；主循环池采样改用种子化 `main_rng`；回放池采样注入种子化 rng；
  Worker 子进程内 Python/NumPy/PyTorch 三源由 `base_seed` 统一派生。
- `__main__.py`：train_rl 子命令新增 `--ref-games`、`--fresh`，`--eval-games` 语义更新。

### 新增测试（5 项）

- `test_mcts_dirichlet_determinism` / `test_selfplay_seed_determinism`：
  同种子搜索/自对弈逐字节可复现（含残局课程分支）；
- `test_checkpoint_roundtrip`：完整检查点无损往返；
- `test_wilson_lower_bound` / `test_decide_promotion_rules`：Wilson 数学性质与
  晋升三条件（含“全和镜像不得晋升”“单阶段退化一票否决”回归用例）。

### 验证

- `python -m unittest discover tests`：**59/59 通过**。
- 端到端冒烟（CPU，2 局自对弈 + 8 局门控）：门控在全和场景正确拒绝晋升且保留候选进度；
  二次启动自动从检查点续训（epoch 2 起）。冒烟产物已清理。
- `elo_history.jsonl` 新增字段：`stage_scores`、`gate_reasons`、`ref_vs_search2`、
  `promoted`、`gate_decision`。

### 训练禁令状态更新

P0 五项（终局价值、跨世界合法性、candidate/best 分离、门控重设计、随机源统一）均已完成，
**解除“禁止长时间自训练”禁令**。后续训练要求：
- 现有 `models/best.pt` 系旧缺陷下训练 3 轮的产物，重启训练建议 `--fresh` 从随机初始化开始；
- 晋级结论必须基于新门控的分阶段报告，禁止仅凭 loss 或小样本宣称变强（第 8 节）。


## [P0 正确性修复] - 2026-08-30

阶段归属：**P0**（依据 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` §3.1 / §5）。

### 修复

- **mcts.py：终局价值视角反号（§3.1.1）**
  `GameState.apply()` 定胜者后切换 `turn`，旧版终局价值误用 `state.turn` 判视角，
  一步吃旗被反向传播成 −1。新增 `_terminal_value()`：统一为走子者（叶子轮次方）
  视角，+1 胜 / −1 负 / 0 和。
- **mcts.py：极低温度动作选择对零访问终局子节点失明**
  终局价值挂在零访问子节点上（终局态不入搜索路径），纯 `argmax(visits)`
  无法选出一步致胜着法。改为 Q 值优先、访问量决胜。
- **mcts.py：跨采样世界合法动作污染（§3.1.2）**
  选择阶段现按当前采样世界的 `legal_actions()` 过滤候选子节点（ISMCTS 可用性
  语义）；展开阶段子节点取跨世界并集，已存在子节点保留统计量不覆盖。
- **state.py：`legal_actions()` 等价动作去重**
  同一(起点,终点)可由公路一步与铁路滑行/工兵飞行重复生成，重复实例会在
  列表采样与动作统计中撕裂权重，现以 `dict.fromkeys` 去重。

### 新增测试（tests/test_rl.py，`TestMCTSCorrectnessP0`，9 项）

- 终局价值直检：吃旗 / 困毙 / 70 步判和各 1 项；
- 行为级：强制吃旗、强制困毙（唯一致胜分支）、强制拖和；
- 跨世界合法性：真实网络 15 种子 × 150 模拟 + 桩网络 1000 固定种子 × 30 模拟，
  钩住 `apply` 校验，非法动作必须为零；
- 循环局面键一致性（自对弈层循环判和依赖）。

### 验证

- `python -m unittest discover tests`：**54/54 通过**（原 45 项 + 新增 9 项）。
- 红测流程留痕：修复前新测试 2 失败 + 4 错误（`_terminal_value` 缺失、
  吃旗不被选中、15 种子搜索捕获 20 次非法模拟动作）。

### 未解决风险（后续阶段）

- §3.1.3 candidate/best 分离与完整 checkpoint（P0 剩余项）；
- §3.1.4 门控样本量与对手选择重设计；
- §3.1.5 随机源统一（`np.random.dirichlet` 仍用全局源）；
- 本修复不解除训练禁令的其余条件：P0 全部验收通过前仍禁止长时间自训练。
