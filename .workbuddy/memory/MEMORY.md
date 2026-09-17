# junqi_engine 长期记忆（决策结论与陷阱）

> 详细历史见 `docs/CHANGELOG.md`、`docs/05-ExecutionPlans/CPP_EXPERT_ENGINE_PORT_PLAN.md`、`.workbuddy/memory/<日期>.md`。
> 本文件只留**决策相关的结论与陷阱**。

## 环境
- venv 在工程外 `E:\Local code\军棋\venv_junqi_engine`（Py3.11+torch2.5.1）。**别用 PATH 上的 python**，一律 `run.bat <子命令>`。
- 重建 C++ 扩展用 `python scripts/build_cpp.py --clean`，**不要 `pip install -e .`**（reg.exe 被拦 → io.h/rc.exe 找不到）。
- 含非 ASCII 的 `.ps1` 必须带 UTF-8 BOM（PS5.1 否则按 GBK 读 → 乱码+假语法错误）。
- 本 bash 环境 shim 损坏（dirname/cp/ls 不可用、PowerShell stdout 不回传）⇒ 脚本写成文件再跑。
- Edit 工具会模糊匹配并插错缩进 ⇒ 每次 Edit 后读回确认。

## C++ 移植（三切片已收官）
- 边界：C++ 持有 `evaluate_expert`/`legal_actions`/`_score_action`/`_qsearch`/`_negamax`/`_evaluate_chance_flip`；迭代加深根循环与根语义（degraded/avoid/root_scores/_is_tactical）留在 Python。开关 `use_cpp_eval/qsearch/search` 默认全开、异常永久降级。提速 19~43×。
- `src_cpp/src/eval_expert_tables.cpp` 是 `scripts/gen_expert_tables.py` 的生成物；改 Python 几何常量必须重跑（`--check` 守卫）。⚠️ `get_road_neighbors()` 与 Python `NEIGHBORS` 集合一致但**顺序不同**。
- 热路径只用 `core_bridge.encode_state_blob`（紧凑 blob）。`state_to_cpp` ≈50µs 不可用于热路径。**blob 必须传 winner**（None→-2），漏传会把已终局子状态当未终局。
- ⚠️ pybind11 基类须先注册（ExpertQSearch 先于 ExpertSearch），否则 import 失败 → HAS_CPP_CORE=False → C++ 静默退回 Python、等价性测试假通过 ⇒ **验证脚本必须 `assert HAS_CPP_CORE`**。
- C++ 侧刻意不做 QTT；`tests/test_p1_qsearch_tt.py` 须**同时**关 use_cpp_qsearch 与 use_cpp_search 才覆盖 Python `_qsearch`。
- ⚠️ 新增搜索分支必须与同函数既有约定逐条对齐：P1.A 长尾分支曾漏负号 → (ply=1, depth≥3) 10/10 分歧、最大 257.6 分。
- ⚠️ 等价性契约边界：单节点语义（子节点用全新引擎）逐位一致；但 depth≥3 子树可差 O(10) 分（两侧 zobrist 键独立 ⇒ TT 碰撞/淘汰模式独立）。契约为 `depth≤2 逐位一致 + 任意深度决策等价`，由 `tests/test_p4_cpp_search.py::TestKnownZobristResidue` 固化；彻底对齐需把 C++ zobrist 改为 Python 生成（约 3000 uint64，**未做**）。任何改变递归层次/入口的改动后必须重跑 ply×depth 矩阵（`scratch/probe_review_clean_ab.py`）。
- ⚠️ 翻棋排序打分两侧不等（C++ `_score_action` FLIP 分支少 `apk_flip_bonus`）。"走法顺序不影响 minimax 值"的旧注释在 TT+PVS 下**不成立**。

## "改动是否影响引擎"必须 A/B 实测
- 工具 `scripts/ab_search_compare.py --ref-commit <旧提交>`。读代码推断会得出**相反**结论。
- 改 `search.py` 后复跑 `scratch/perf_baseline.py verify`（capture 须在改动前做）；要**同时**比节点数与决策，只看 max_depth 会误判。
- 有意改变搜索行为后必须**立刻重新 capture**，否则守卫只制造误报。
- 证明"非本次引入"只能用 HEAD 对照（工程只读复制到 `E:/Local code/军棋/_review_base` 覆写后重编），**不要 stash/pop 用户工作区**。

## 评测门控纪律
- 先证明靶场有分辨力：同模型空测应恰 0.5000 且决定率不为 0。
- `gate --init-set` 必需（随机发牌/中盘开局下自对局 100% 循环判和；`eval_sets/endgame.jsonl` 把决定率拉到 30%）。
- 多重比较：单次 60 局的 p 不能单独下结论；分辨 0.05 得分率差需 ~1500 局。
- 指标可比性：`val_KL` 跨 T 不可比；跨轮用 `val_teacher_top1`/`val_base_top1`/`eval_bc`。
- 蒸馏前先验证教师前提：`gate --a expert3 --b nn --model-b models/best.pt`（实测 40 局零负、0.7125）。
- 镜像配对对确定性策略无区分力（得分率恒 0.5000），分辨力用 `paired_delta_test`。`mirror_state` 只翻位置+交换座位标签，**不换颜色**。

## P2 搜索蒸馏：已证伪，停止调参
四项干预（T 120→20、基座锚点、教师置信度过滤、修靶场）让中间指标单调改善（人类 top1 0.204→0.408、基座 0.529），但对局强度无稳健提升（门控 0.46~0.57，CI ±0.12）。原因：① 教师优势来自搜索深度，单次前向 Policy 表达不了；② C++ 提速 19~43× 反使直接跑搜索更划算；③ 蒸馏是"用人类模仿换搜索模仿"，无净收益。**建议直接加深 hybrid 搜索（无需训练），不要再用扫温度/锚点系数。** 相关代码已落地且默认关闭：`--anchor-weight`(0)、`--tac-min-spread`(0)、`gate --init-set`、`val_base_top1`。

## SearchStats.degraded
`time_limit_ms>0` 的调用方**应先查 `stats.degraded`**。True ⇒ score 是真实值**下界**（仍是合法决策，别丢弃）、`root_scores` 被截断、未搜完的层只写 `FLAG_LOWER_BOUND`（绝不写 EXACT）、本层零动作完成时不写 TT 不记 `max_depth`。`self.stopped`（子搜索超时）是另一条路径：整层作废。
**⚠️ 生产链路仍无消费者**：`ExpertAgent.choose_actions` 不暴露 stats ⇒ 蒸馏教师把**截断**的
`root_scores` 全量 softmax 重归一化（把概率给"恰好搜到的动作"）、价值伪标签把下界当精确分。
`search.py:985/1168-1170` 两处注释都要求消费方查 degraded，实际无人查。

## 蒸馏/克隆链路的已定证陷阱（2026-09-17 审查，详见 `reviews/CODE_REVIEW_P2_DISTILL_2026-09-17.md`）

- **⚠️ `requires_grad=False` 不等于冻结**：BatchNorm 的 `running_mean/var` 在 `training=True`
  时**无条件更新**。`train_value_head_only` 用 `net.train()` ⇒ 主干/策略头 BN 统计被改写
  （实测 `policy_head.1.running_mean` Δ=0.455，而权重 Δ=0），策略输出分布随之漂移
  ⇒ "冻结主干与策略头"的承诺不成立。**正确写法**（同文件另一处）：`net.eval()` +
  `net.value_head.train()`。改任何"冻结后训练某头"的代码前先查这一点。
- **⚠️ `seed` 在搜索链路上是死参数**：`ExpertSearchEngine.rng`（`search.py:161`）与
  `ExpertAgent.rng`（`ai.py:298`）全仓**只赋值、从不读取** ⇒ 专家搜索完全确定性。
  `--seed` 只影响**局面采样**，不影响教师打分。别用它论证可复现性。
- **⚠️ 教师打标 serial≠mp**：serial 复用一个 agent（TT 跨轮保留、2^18 条目），
  mp 每 task 新建 ⇒ TT 淘汰模式依赖局面顺序 ⇒ 同 seed 下 `--workers 0` 与 `>1` 教师分布不同。
  要等价需**逐局面 `tt.clear()`**（与"单节点等价性＝全新引擎"口径一致）。
- **W/D/L 码表散落 4 份**（`dataset.py:69`、`train_bc.py:68/203`、`eval_bc.py:78`），后 3 处
  **不可达**（`__getitem__` 恒返 7 元组 ⇒ `len(batch_data)==7` 恒真）。违反"唯一真源"硬约束。
- **`train_bc`/`eval_bc` 绕过 `check_p1_version`**（走 `NpzReplayDataset` 裸 `np.load`；
  该守卫只有 `load_p1_arrays` 一个调用点）⇒ 可静默在 code-24 口径不同的旧数据集上训练。
- **`train_value_distill` 两分支回归目标量纲不一致**：预打标用真实 `expert_score`
  ⇒ `tanh(1e6/600)=1.0`；采样分支写死 `±600.0` ⇒ `tanh(1)=0.7616`，差 **0.2384**，
  系统性压低价值头置信度（`600.0` 像是把 `SCORE_SCALE` 当成了分数）。
- **实测否定的优化点（勿重复劳动）**：`_batch` 每批重算 `legal_action_mask` 实测仅
  **0.040ms/次**（1200×6 epoch ≈ 0.3s），相对教师打标（≈6 min）可忽略，**不要优化**。

## 训练产物状态
- 复现：`python scripts/audit_artifacts.py --probe`。
- `models/best.pt` = `value_distilled_v2.pt`（唯一健康 Value 基座 0.735/0.288）；原 best 与 `models/pool/bc_best.pt` 逐位相同（Value 头退化：Draw 恒 0）。三个作废经验池与冒烟池已删，`datasets/p1_v3` 标签可用。
- 待办：小规模正式复跑自对弈（≥300 局/轮、sims ≥20）→ 蒸馏 → `gate --seeds 100 --promote-to-best`。终局偏 `immobilized`、候选对 search2 得分 0.083~0.167 是棋力真问题、非 bug。

## 危险操作禁令
- **删除文件一律用 Python 文件系统操作，禁止 `git rm`**（曾被 SIGTERM 中断留下 `.git/index.lock`，`junqi/`+`scripts/` 共 96 文件消失；恢复：删 stale lock → `git checkout HEAD -- junqi scripts`）。
- 不跑长 `&&` 链的 git 写命令；git 前后确认无 `.git/index.lock`。
- 不用 `dangerouslyDisableSandbox`；不主动删 `models/`、`军旗复盘/`、`datasets/`。

## 项目硬约束（AGENTS.md / AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）
- 改代码前先读方案并**声明阶段 P0–P4**；**先加/改测试再改实现**；报告改动文件、种子、测试结果、未解决风险。
- 禁止：真实暗子身份进公共 Policy；吃子/挖雷等中间奖励；三个独立阶段模型；只看 loss 覆盖 `best.pt`；P0 未过就长训练。
- Value 标签严格按官方 `list.cfg`：1/21/22/23→±1；40/42/43→0；20/24→不赋。`dataset.terminal_label_from_meta` 是唯一真源，统计走 `outcome_bucket_from_meta`，**禁止各写一份码表**。数据集默认 `datasets/p1_v3`，<3.0.0 被 `check_p1_version` 拒载。
- 门控唯一真源 `junqi/eval_gate.py::run_gate`（配对同牌 + Wilson + 三元 SPRT）。

## 架构要点
- `junqi/search.py::ExpertSearchEngine`：`self.tt`(depth) 与 `self.qtt`(depth_left) **必须分离**（共用会让 qsearch 条目被 `_negamax` 命中 → 错分）；`qsearch_depth` 默认 16，**未经对局级 A/B 不得下调**；`_qsearch` 声明 **Captures Only**，引入非吃子动作须同步更新声明与终止性论证。`junqi/expert/` 已删除，不要重建。
- `junqi/net.py::save()` 写**包装字典** `{"model_state": ...}`；load 后必须先 `JunqiNet.unwrap_state_dict()`，否则 `strict=False` 静默零载入（历史"best 对手是随机网络"根因）。
- 热启动**禁止**重绑 `net = JunqiNet.load_from_file(...)`（optimizer 脱钩、权重零更新），用 `train_rl.warmstart_candidate(...)`。
- `StratifiedReplayBuffer` 契约：policy `(state, mask, target, phase)`、value `(state, z_cls, is_world)`。
- `junqi/fit_weights.py` 全部**按特征名索引**（曾 25 维错位）。三条不变式有测试守卫：`_project(default_vector())==default_vector()`、`to_eval_weights(default_vector())==EvalWeights()`、每特征都能写回（例外仅 `flip_bias`）。**`mobility` 不能列入**（EvalWeights 无此字段，且 `evaluate_expert` 的机动力项是另一套定义、系数硬编码）。
- `scripts/archive/`、`tests/utils/` 是归档目录，不进 pytest 收集，不要在其上继续开发。
