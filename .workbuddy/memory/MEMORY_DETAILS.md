# junqi_engine 长期记忆 · 详情（决策结论与陷阱）

> 本文件是 `MEMORY.md` 的**详情分册**，不被自动注入。要查具体陷阱/数字时读这里；索引与最高优先级告警见同目录 `MEMORY.md`。
> 流水账见 `.workbuddy/memory/<日期>.md`、`docs/CHANGELOG.md`、`reviews/`。

## 环境
- venv 在工程外 `E:\Local code\军棋\venv_junqi_engine`（Py3.11+torch2.5.1）。**别用 PATH 上的 python**，一律 `run.bat <子命令>`。
- 重建 C++ 扩展用 `python scripts/build_cpp.py --clean`，**不要 `pip install -e .`**（reg.exe 被拦 → io.h 找不到）。
- 本 bash 环境 shim 损坏（dirname/cp/ls 不可用、PowerShell stdout 不回传）⇒ 脚本写成文件再跑；含非 ASCII 的 `.ps1` 必须带 UTF-8 BOM。Edit 工具会模糊匹配插错缩进 ⇒ 改完读回。

## ⚠️ 最高优先级：跨数据集版本评测泄漏（2026-09-17 定证）
同一批 1072 局 `.sav` 反复重导出（p1_v1 38ch / p1_v2 有效 835 / p1_v3 有效 951），同 `seed=2026` 但**列表长度不同 ⇒ 划分全变**；旧版 train 覆盖新版 test 约 80%。实测 Top-1：`bc_best.pt`(p1_v3.train) 在 p1_v3/test **24.90%**〔未见〕、p1_v2/test 48.86%〔见过〕；`bc_best_legacy`(p1_v2.train) 反之 52.41%/24.39%；`pool/bc_best`(p1_v1.train) 三划分 86~89%〔几乎全覆盖〕。
- **任何"用数据集 B 评模型 A"的数字，若 A 的训练语料与 B.test 重叠一律无效。唯一不可伪造的诚实基线是模型训练期记录的 val 指标。**
- 已作废：`eval_bc_best.md` 52.90%/74.61%、`eval_bc_distilled20k_*`、`train_rl.py:1280-1282` 注释的 0.735/0.288（诚实 = p1_v2/val **0.4467/0.9366**）。BC 诚实天花板 ≈ **Top-1 24–25%**，加 epoch 无用（val 第 4 轮见顶 25.36% 后单调恶化，train 记到 98.63%）。
- ✅ 已修：`metadata.json` 落盘 `split_files` 清单；`datasets/p1_v3` 反推成功（760/95/96 局，plies 逐项吻合）→ `leak_free=True`；`datasets/canonical_test.json` 冻结 96 局；`datasets/p1_v4` 冻结模式导出 `leak_free=True`。工具 `scripts/audit_dataset_leakage.py`。
- 残余：只冻结了 **test**，train/val 仍 WARN（模型选择不可比）；p1_v1/v2 无清单不可反推，需人工录单。

## P2 验收现状（2026-09-17 第二十一批后）
- ①满足但相对上一版诚实值无提升。
- ②**已测（第二十一批）**：拆两层证据 —— **决策级全量覆盖**（新脚本 `scripts/audit_p2_phase_stability.py`，3 阶段 × 200 快照 × 2 引擎，异常/返回空/非法**全 0**，42 秒）；**对局级**（`audit_p2_game_quality.py`，各 N=12 × 2 配置）：`endgame` ✅、`opening` ❌（往复踱步 **2.80×**、龟缩拒翻 **1.84×** 于对照）、`midgame` ❌（**仅**龟缩拒翻超 `--abs-cap 2.0`，而**对照 expert2 也是 6.0071 ⇒ 上限标定失当**）。判定**未通过**，但缺口是**既有棋风**、已修复减轻、不阻塞 P3。
- ③`hybrid2 vs search2` 200 局：修复前 0.5225 → **修复后 0.5475**（Wilson [0.4783,0.6149]，δ=5% 非劣成立）；配对检验 p 0.0463 → **0.0002**；SPRT `accept_h0` → `continue`（**仍未达 `accept_h1`**）⇒ **非劣通过**，**不可宣称"显著更强"**。⚠️ 两数是**两次独立门控**的对照、非两版引擎对杀，差值未经假设检验。
- ④**已修复并通过**：非法 0/5798 ✅、送旗 1/5798 ✅、无意义循环 **2/60=3.3%** ✅（对照 expert2 4/60=6.7%、修复前 24/60=40%）。修复前数据留档 `reports/p2_game_quality_prefix_20260917.*`。
- ✅ **④ 根因（已修，第二十批）**：`junqi/ai.py:538` 写 `if avoid and a in avoid:`，而 `avoid` 是 `position_key()` 的**局面键**集合，`a` 是 `Action` dataclass ⇒ **恒不命中，混合引擎从不规避重复**。其余 5 处消费者（`ai.py:208/451`、`mcts.py:155`、`hybrid_engine.py:195`、`apk_engine.py:371`）写法都正确 ⇒ **孤例错误**。
  修法：`nxt = state.apply(a) if a.kind == "move" else None`，判 `position_key(nxt) in avoid`。**只判走子的理由**：翻子会永久增加公开信息，而 `position_key` 记每格 `(color, rank)`/`None` ⇒ 翻子后的键不可能与历史键重合，重复判和在数学上不可能由翻子造成。
  守卫：`tests/test_hybrid_agent.py::test_hybrid_agent_penalizes_repetition_position`（**自校准**：先用无 avoid 的打分挑出"本来最优的走子"再要求它垫底 ⇒ 修复前必红，不会假通过）。审计脚本 `scripts/audit_p2_game_quality.py`（探针按**事件段**折叠计数；非法类硬性为 0；其余对 expert2 做 Wilson 非劣性判定）。
- ✅ **修复无回归（第二十一批单变量 A/B）**：复制最小运行时子集到 `E:\Local code\军棋\_p2_prefix_base`，用 `git show HEAD:junqi/ai.py` 覆写副本（与工作区**仅此一文件不同**），同种子同局面重跑 ⇒ 三阶段**每项都改善**：开局 往复踱步 1.4053→1.2000、龟缩拒翻 11.8767→7.9429、重复判和 9/12(75%)→1/12(8.3%)；中盘 0.1018→0.0821、7.8411→6.3218、7/12(58.3%)→1/12。**开局探针劣势是既有棋风**（修复前比现在还重 17~50%）。
- 报告：`reviews/P2_ACCEPTANCE_2_AND_GATE_RERUN_2026-09-17.md`（②③）、`reviews/P2_ACCEPTANCE_34_AND_DATA_SPLIT_2026-09-17.md`（§7 = ④ 修复与复测）、`reviews/BC_ACCEPTANCE_VERDICT_2026-09-17.md`。

## 训练产物状态
- 复现：`python scripts/audit_artifacts.py --probe`。
- `models/bc_best.pt` = epoch 4；策略 Top-1 诚实 24.90%（p1_v3/test）；**Value 头对未见对局零判别力**（胜负 AUC 0.4899、MSE 0.7686 > 平凡基线 0.4517）。
- ⚠️ `models/value_distilled_v2.pt` 已于 21:56 重建（base=新 `bc_best.pt`）：诚实 val 平衡准确率 **0.3247 < 1/3 随机**；而 `train_rl` 热启动**优先取该文件** ⇒ 启动 P3 会从"策略 24.9% + 价值低于随机"起步。旧版备份 `value_distilled_v2_legacy_20260913.pt`（诚实 0.4467）= 现在的 `models/best.pt`。
- 待办：改 `train_rl` 热启动选择依据；小规模正式复跑自对弈（≥300 局/轮、sims ≥20）→ 蒸馏 → `gate --seeds 100 --promote-to-best`。终局偏 `immobilized`、候选对 search2 得分 0.083~0.167 是棋力真问题、非 bug。

## P2 搜索蒸馏：已证伪，停止调参
四项干预（T 120→20、基座锚点、置信度过滤、修靶场）只让中间指标改善，对局强度无稳健提升（门控 0.46~0.57，CI ±0.12）。原因：① 教师优势来自搜索深度，单次前向 Policy 表达不了；② C++ 提速 19~43× 反使直接跑搜索更划算；③ 蒸馏＝用人类模仿换搜索模仿，无净收益。**建议直接加深 hybrid 搜索（无需训练），不要再扫温度/锚点系数。** 相关代码已落地且默认关闭：`--anchor-weight`(0)、`--tac-min-spread`(0)、`gate --init-set`、`val_base_top1`。

## 已定证陷阱（详情见 `reviews/CODE_REVIEW_P2_DISTILL_2026-09-17.md`）
- **`requires_grad=False` ≠ 冻结**：BatchNorm 的 `running_mean/var` 在 `training=True` 时**无条件更新**。`train_value_head_only` 用 `net.train()` ⇒ BN 统计被改写（`policy_head.1.running_mean` Δ=0.455，权重 Δ=0），策略分布漂移。**正确写法**：`net.eval()` + `net.value_head.train()`。
- **`seed` 在搜索链路上是死参数**：`ExpertSearchEngine.rng`(`search.py:161`) 与 `ExpertAgent.rng`(`ai.py:298`) 全仓**只赋值、从不读取** ⇒ 专家搜索完全确定性，`--seed` 只影响局面采样。
- **教师打标 serial≠mp**：serial 复用 agent（TT 跨轮保留 2^18 条目）、mp 每 task 新建 ⇒ TT 淘汰依赖局面顺序 ⇒ 同 seed 下 `--workers 0` 与 `>1` 分布不同；要等价需逐局面 `tt.clear()`。
- **`SearchStats.degraded` 生产链路无消费者**：`ExpertAgent.choose_actions` 不暴露 stats ⇒ 蒸馏教师把截断的 `root_scores` 全量 softmax 重归一化、价值伪标签把**下界**当精确分。`time_limit_ms>0` 的调用方应先查 degraded。
- **W/D/L 码表散落 4 份**（`dataset.py:69`、`train_bc.py:68/203`、`eval_bc.py:78`），后 3 处**不可达** ⇒ 违反"唯一真源"。`train_value_distill` 两分支量纲不一致（`tanh(1e6/600)=1.0` vs 0.7616，差 0.2384）。
- ✅ `check_p1_version` 已下沉进 `NpzReplayDataset.__init__`（p1_v2 已被拒载），但挡不住跨版本对局重叠。
- **实测否定的优化点（勿重复劳动）**：`_batch` 每批重算 `legal_action_mask` 仅 **0.040ms/次**，相对教师打标（≈6 min）可忽略。

## C++ 移植（三切片已收官）
- 边界：C++ 持 `evaluate_expert`/`legal_actions`/`_score_action`/`_qsearch`/`_negamax`/`_evaluate_chance_flip`；迭代加深根循环与根语义（degraded/avoid/root_scores）留 Python。开关 `use_cpp_*` 默认全开、异常永久降级。提速 19~43×。
- ⚠️ pybind11 基类须先注册（ExpertQSearch 先于 ExpertSearch），否则 import 失败 → HAS_CPP_CORE=False → **C++ 静默退回 Python、等价性测试假通过** ⇒ 验证脚本必须 `assert HAS_CPP_CORE`。
- 热路径只用 `core_bridge.encode_state_blob`（`state_to_cpp` ≈50µs 不可用）。**blob 必须传 winner**（None→-2），漏传会把终局子状态当未终局。
- ⚠️ 等价性契约：单节点语义逐位一致；depth≥3 子树可差 O(10) 分（两侧 zobrist 键独立 ⇒ TT 淘汰模式独立）。契约为 `depth≤2 逐位一致 + 任意深度决策等价`（`tests/test_p4_cpp_search.py::TestKnownZobristResidue` 固化）。改递归层次/入口后必须重跑 ply×depth 矩阵（`scratch/probe_review_clean_ab.py`）。
- `src_cpp/src/eval_expert_tables.cpp` 是 `scripts/gen_expert_tables.py` 生成物，改 Python 几何常量必须重跑（`--check` 守卫）。C++ 侧刻意不做 QTT ⇒ `tests/test_p1_qsearch_tt.py` 须**同时**关 use_cpp_qsearch 与 use_cpp_search。
- ⚠️ 翻棋排序打分两侧不等（C++ `_score_action` FLIP 少 `apk_flip_bonus`）；**"走法顺序不影响 minimax 值"在 TT+PVS 下不成立**。

## 引擎改动纪律
- **"改动是否影响引擎"必须 A/B 实测，读代码推断会得出相反结论**：`scripts/ab_search_compare.py --ref-commit <旧提交>`；改 `search.py` 后复跑 `scratch/perf_baseline.py verify`（capture 须在改动前做），要**同时**比节点数与决策。
- 有意改变搜索行为后**立刻重新 capture**，否则守卫只制造误报。证明"非本次引入"只用 HEAD 对照（只读复制到 `_review_base` 重编），**不要 stash/pop 用户工作区**。
- 门控纪律：先证明靶场有分辨力（空测应恰 0.5000 且决定率≠0）；`gate --init-set` 必需（否则随机开局自对局 100% 循环判和）；单次 60 局不能下结论（分辨 0.05 需 ~1500 局）；`val_KL` 跨 T 不可比；镜像配对对确定性策略恒 0.5000（分辨力用 `paired_delta_test`），`mirror_state` **不换颜色**。
- **单变量对照副本 A/B（判定"缺陷是否本次引入"的标准手法，2026-09-17 定）**：
  复制工程最小运行时子集到副本目录（`junqi/`、`scripts/`、`eval_sets/`、`configs/`、
  `junqi_core.cp311-win_amd64.pyd`、`models/bc_best.pt`；`import junqi_core` 靠工程根目录的 `.pyd`），
  再用 `git show HEAD:<被改文件>` 覆写副本里的同一文件 ⇒ 副本与工作区**仅此一文件不同**。
  同种子同起始局面重跑同一评测即可。**务必断言"副本含旧写法、原工程含新写法"**，
  否则还原失败会得出假结论。对照引擎（不受本次改动影响的那些）的数字可直接取自主运行。
  ⚠️ 不要用 stash/pop 动用户工作区。
- **"修复 → 复测"要分三步**：① 被修的东西是否达标；② 修复可能波及的别处是否退化；
  ③ 区分"既有问题"与"本次引入"（只能靠上一条的单变量副本）。②③ 不因 ① 通过而免除。
- **探针类的绝对阈值必须按靶场标定**：`audit_p2_game_quality.py --abs-cap 2.0/100` 是在**残局**靶场
  标定的（对照 `expert2` 在 endgame 恰为 0.0000），套到开局/中盘会让**任何**引擎判 FAIL
  （对照分别为 4.32 / 6.01）。看到"候选超限"**先看对照是否也超限**。
- **阶段集合的对局成本由对手决定**：开局单局 50–90 秒，但候选单步中位仅 19ms ——
  瓶颈是 `search2`（samples=4 的世界采样在暗子多时最贵）。压测预算要按对手估，不是按候选。

## 危险操作禁令
- **删文件一律用 Python 文件系统操作，禁止 `git rm`**（曾被 SIGTERM 中断留下 `.git/index.lock`，`junqi/`+`scripts/` 共 96 文件消失；恢复：删 stale lock → `git checkout HEAD -- junqi scripts`）。
- 不跑长 `&&` 链的 git 写命令；git 前后确认无 `.git/index.lock`。不用 `dangerouslyDisableSandbox`；不主动删 `models/`、`军旗复盘/`、`datasets/`。

## 项目硬约束（AGENTS.md / AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）
- 改代码前先读方案并**声明阶段 P0–P4**；**先加/改测试再改实现**；报告改动文件、种子、测试结果、未解决风险。
- 禁止：真实暗子身份进公共 Policy；吃子/挖雷等中间奖励；三个独立阶段模型；只看 loss 覆盖 `best.pt`；P0 未过就长训练。
- Value 标签按官方 `list.cfg`：1/21/22/23→±1；40/42/43→0；20/24→不赋。`dataset.terminal_label_from_meta` 是唯一真源，统计走 `outcome_bucket_from_meta`，**禁止各写一份码表**。数据集默认 `datasets/p1_v3`，<3.0.0 被 `check_p1_version` 拒载。门控唯一真源 `junqi/eval_gate.py::run_gate`。

## 架构要点
- `search.py::ExpertSearchEngine`：`self.tt`(depth) 与 `self.qtt`(depth_left) **必须分离**（共用会让 qsearch 条目被 `_negamax` 命中 → 错分）；`qsearch_depth` 默认 16，**未经对局级 A/B 不得下调**；`_qsearch` 声明 **Captures Only**。`junqi/expert/` 已删除，不要重建。
- `net.py::save()` 写**包装字典** `{"model_state": ...}`，load 后必须先 `JunqiNet.unwrap_state_dict()`，否则 `strict=False` 静默零载入（历史"best 对手是随机网络"根因）。热启动**禁止**重绑 `net = JunqiNet.load_from_file(...)`（optimizer 脱钩），用 `train_rl.warmstart_candidate(...)`。
- `StratifiedReplayBuffer` 契约：policy `(state, mask, target, phase)`、value `(state, z_cls, is_world)`。
- `fit_weights.py` 全部**按特征名索引**（曾 25 维错位），三条不变式有守卫；**`mobility` 不能列入**。
- 策略工厂 `selfplay.py::make_strategy`：`hybrid2` → `HybridStrategy(depth=2)` → `HybridAgent`；`p4hybrid` → `HybridDecisionEngine`。
- `scripts/archive/`、`tests/utils/` 是归档目录，不进 pytest 收集。
