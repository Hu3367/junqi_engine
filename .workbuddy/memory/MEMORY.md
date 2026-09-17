# junqi_engine 长期记忆 · 索引

> **详情分册在同目录 `MEMORY_DETAILS.md`**（含全部陷阱、行号、工具与实测数字）——需要动手前先读它。
> 流水账见 `.workbuddy/memory/<日期>.md`、`docs/CHANGELOG.md`、`reviews/`。本文件只放**不可违背的告警与入口**。

## ⚠️ 四条硬告警
1. **跨数据集版本评测泄漏（最高优先级）**：同一批 1072 局 `.sav` 反复重导出，同 seed 但列表长度不同 ⇒ 划分全变，旧版 train 覆盖新版 test 约 80%。
   ⇒ **任何"用数据集 B 评模型 A"的数字，若 A 的训练语料与 B.test 重叠一律无效；唯一诚实基线是模型训练期记录的 val 指标。** BC 诚实天花板 ≈ Top-1 **24–25%**（52.90%/74.61%、0.735/0.288 等均已作废）。已修：`split_files` 清单 + `datasets/canonical_test.json`(96 局冻结) + `datasets/p1_v4`(leak_free)。审计工具 `scripts/audit_dataset_leakage.py`。
2. **`requires_grad=False` ≠ 冻结**：BatchNorm `running_mean/var` 在 `training=True` 时无条件更新。冻结某头训练时用 `net.eval()` + `net.value_head.train()`，**禁止** `net.train()`。
3. **`seed` 在搜索链路上是死参数**：`ExpertSearchEngine.rng` 与 `ExpertAgent.rng` 全仓只赋值、从不读取 ⇒ 专家搜索完全确定性，别用它论证可复现性。
4. **"改动是否影响引擎"必须 A/B 实测**：读代码推断会得出**相反**结论。工具 `scripts/ab_search_compare.py --ref-commit <旧提交>`。

## 当前卡点（2026-09-17 收尾）
- **P2 ④ 已修复并通过**：`junqi/ai.py:538` 的 `if avoid and a in avoid:` 是**孤例错误**（`avoid` 装 `position_key()` 局面键、`a` 是 `Action` dataclass）⇒ 混合引擎从不规避重复。改为 `position_key(nxt) in avoid` 且限定走子后，重复判和 **24/60=40% → 2/60=3.3%**（对照 expert2 6.7%）。守卫：`test_hybrid_agent_penalizes_repetition_position`（自校准，修复前必红）、审计脚本 `scripts/audit_p2_game_quality.py`。
- **P2 ③ 已用修复后引擎复跑并通过（非劣）**：0.5225 → **0.5475**，Wilson [0.4783,0.6149]，配对检验 p 0.0463 → **0.0002**；SPRT `accept_h0` → `continue` ⇒ **仍不可宣称"显著更强"**。⚠️ 两数是两次独立门控的对照、非两版引擎对杀，差值未经检验。
- **P2 ② 已测但未通过**：决策级全覆盖（3 阶段 × 200 快照 × 2 引擎，`scripts/audit_p2_phase_stability.py`）异常/返回空/非法**全 0** ✅；对局级 `endgame` ✅、`opening` ❌（往复踱步 **2.80×**、龟缩拒翻 **1.84×** 于 expert2）、`midgame` ❌（**仅**因 `--abs-cap 2.0` 标定失当，对照 expert2 也 6.01）。
- ✅ **已证明上述劣势非本次修复引入**（单变量对照副本 A/B，见 `MEMORY_DETAILS.md`「引擎改动纪律」）：修复使三阶段**每项都改善**（开局重复判和 75% → 8.3%）。**开局棋风是独立待办**（提高 `top_k` / 加强营间闲走惩罚 / 阶段化 `prior_weight`）；`--abs-cap` 需按阶段标定。
- 报告：`reviews/P2_ACCEPTANCE_2_AND_GATE_RERUN_2026-09-17.md`（②③）、`reviews/P2_ACCEPTANCE_34_AND_DATA_SPLIT_2026-09-17.md`（④ 修复）、`reviews/BC_ACCEPTANCE_VERDICT_2026-09-17.md`、`reviews/CODE_REVIEW_P2_DISTILL_2026-09-17.md`。

## 入口速查
- 跑任何 Python：`run.bat <子命令>` / venv 在 `E:\Local code\军棋\venv_junqi_engine`（**别用 PATH 上的 python**）。重建 C++：`python scripts/build_cpp.py --clean`（**不要 `pip install -e .`**）。
- 数据集默认 `datasets/p1_v3`（<3.0.0 被 `check_p1_version` 拒载）；门控唯一真源 `junqi/eval_gate.py::run_gate`；Value 标签唯一真源 `dataset.terminal_label_from_meta`。
- ⚠️ `models/value_distilled_v2.pt` 已于 21:56 重建，诚实 val 平衡准确率 **0.3247 < 1/3 随机**，而 `train_rl` 热启动**优先取它** ⇒ 启动 P3 前先改这个选择依据。
- **删文件一律用 Python 文件系统操作，禁止 `git rm`**（曾因 SIGTERM 留下 `.git/index.lock` 导致 96 文件消失）。
