# junqi_engine 项目长期记忆（跨会话约定）

## 环境与启动（最容易踩坑）

- **虚拟环境不在工程内**：位于同级 `E:\Local code\军棋\venv_junqi_engine`（2026-09-15 移出，
  原 `junqi_engine/venv/`）。Python 3.11.9 + torch 2.5.1+cu121 + pytest 9.1.1。
  `venv/` 在 `.gitignore` 内，移动不影响版本控制。
- **激活虚拟环境**：`.\activate_env.ps1`（PowerShell）或 `activate_env.cmd`（cmd）——
  根目录脚本自动定位，不必记 `..`。手写路径注意是 **`..\venv_junqi_engine\...`**
  （上一级，不是 `.\`）；激活后脚本会做 `import torch` 依赖自检。
- **一律用工程根 `run.bat <子命令>` 启动**（自动定位解释器，无需先激活）：
  `run.bat test` / `run.bat gui` / `run.bat train_rl --epochs 5 --games 24`。
  等价完整路径：`../venv_junqi_engine/Scripts/python.exe -m junqi <子命令>`。
- **不要用 PATH 上的 `python`**：PATH 上是系统 Python（3.13/3.14，**无 torch/numpy**）。
  用它跑 `python -m junqi ...` 会在包导入阶段失败（`junqi/__init__.py` 经 ai/hybrid_engine
  连带 `import torch`）。现在会给出中文可操作提示，但根本解法是用 run.bat。
- **⚠️ 残留 `(venv)` 激活陷阱**：venv 被移动/删除后，旧会话仍显示 `(venv)`，
  但 PATH 首项指向已失效目录 → `python` 静默落到别的解释器 → 报 `No module named 'torch'`。
  处置：`deactivate` 或**重开终端**，再用 `activate_env.ps1` 激活。
  `activate_env.*` 已内置检测与清理。
- **⚠️ PowerShell 5.1 编码陷阱**：PS 5.1 对**无 BOM** 的 UTF-8 `.ps1` 按本地编码（GBK）读取，
  中文注释会变乱码并抛出"缺少右 }"这类假语法错误。
  → **仓库内任何含非 ASCII 的 `.ps1` 必须带 UTF-8 BOM**（已有测试守卫）。
- **本 bash 环境 shim 已损坏**：`dirname: command not found`、`cd: null directory`；
  且 **shell 里带反斜杠的 Python `-c` 字符串会被吞掉**（会让 `str.count()` 恒为 0）。
  → 含路径的脚本请写成文件再执行；路径用正斜杠或 `chr(92)` 拼接。
  → coreutils（ls/grep/head/tail/wc）不可用，文件/内容检索用 Read/Glob/Grep 工具或 Python。
  → PowerShell 工具的 stdout 不落回上下文，验证 PS 脚本要把输出重定向到文件再读。
- **Edit 工具会模糊匹配并插入错误缩进**（2026-09-15 踩过）：`old_string` 缩进写少 4 个空格
  仍会"匹配成功"，然后把整段按**错的**缩进写回 → `IndentationError`。
  → **每次 Edit 后读回确认缩进**，尤其复用旧代码块当锚点时。
  → 改 `junqi/selfplay.py::play_game` 时注意 `s0`/`s1` 已是**策略对象**，局部变量别重名。
- **⚠️ 重建 C++ 扩展一律用 `python scripts/build_cpp.py`**（`--clean` 全量），
  **不要用 `pip install -e .`**：本机 `reg.exe` 被安全策略拦截，setuptools 靠注册表
  定位 Windows SDK 会失败 → 先报 `C1083: 无法打开包括文件 'io.h'`（缺 ucrt 头），
  再报 `LNK1158: cannot run 'rc.exe'`。build_cpp.py 自动探测 SDK/MSVC/rc.exe
  并注入 `INCLUDE`/`LIB`/`PATH`（distutils 会读这两个环境变量）。

## C++ 移植（切片 1 已交付，2026-09-15）

- 计划与验收见 `docs/05-ExecutionPlans/CPP_EXPERT_ENGINE_PORT_PLAN.md`；
  CHANGELOG「第十二批」有完整实测数据。状态：切片 1 完成，切片 2/3 待决策。
- `evaluate_expert` + `fortress_score` + `is_dead_draw` 已移植到
  `src_cpp/src/eval_expert.cpp`；残局单步 4236→1225ms（3.46×），估值 5.8~9.0×。
  开关 `ExpertSearchEngine(use_cpp_eval=...)`，默认开，异常自动永久降级回 Python。
- **顺序敏感常量表是生成物**：`src_cpp/src/eval_expert_tables.cpp` 由
  `scripts/gen_expert_tables.py` 从 Python 真源生成（NEIGHBORS 的 set 迭代序、
  `sorted(CAMPS, key=中营优先)` 的 frozenset 迭代序）。**改了 Python 端几何常量必须重跑
  生成器**，`--check` 会校验是否过期（有测试守卫）。
  ⚠️ 别复用 `src_cpp` 里既有的 `get_road_neighbors()` —— 它按 (上,左,下,右) 构造，
  **顺序与 Python NEIGHBORS 不同**（(0,1)：C++ [0,6,2] vs Python [0,2,6]），
  而 `my_reach[0]` 取首元素 ⇒ 是语义差异不是浮点误差。
- **阵亡子必须显式序列化，不可由棋盘反推**：`_evaluate_chance_flip` 的子状态会替换
  暗子身份但 dead 不变，`board ∪ dead = 完整编制` 不变式**不成立**（实测 12 次派生为负）。
  有测试 `test_dead_counts_are_not_derivable_from_board` 固定此事实。
- **桥接成本实测（修正了原计划预估）**：C++ 计算 2.4~3.8 µs（优于原估 5 µs），
  序列化 + pybind 编组 6.0~6.4 µs（原估 1~2 µs，慢 3 倍）。
  ⇒ 别再花力气做序列化微优化，收益在切片 2/3（整棵子树搬进 C++ 后该开销归零）。
- 浮点顺序敏感度已量化：idx 升序 vs board dict 序，max_abs 差 **5.684e-14**
  （default）/ 2.274e-13（apk），比 1e-9 阈值低 4 个量级 ⇒ 无需传迭代序。
  残留：节点数可能差 <2%（并列比较翻转），决策不变。

## C++ 移植（切片 2 已交付，2026-09-15）

- 残局 4250 → 245ms（**17.36×**，切片 1 后为 1228ms）；中盘 9.84×/7.74×；开局 3.60×
  （qnodes=0，其 424ms 全在机会节点/`_negamax` ⇒ 切片 3 目标）。
  开关 `ExpertSearchEngine(use_cpp_qsearch=...)`，默认开。
- **走法顺序不必对齐**（重要简化）：探针 300 局面 → 集合不一致 **0**、顺序不一致 269
  （Python 铁路走法返回 `set`，迭代序由 tuple 哈希决定）。alpha-beta **返回值与遍历
  顺序无关**，只需集合一致 ⇒ 不必改 Python 侧 `legal_actions`。
- **C++ 侧刻意不实现 QTT**：纯缓存，µs 级节点下收益为负。代价 `qnodes` 更高
  （36721 vs 25780），有测试固定该事实防误判回归。`tests/test_p1_qsearch_tt.py`
  已改为显式 `use_cpp_qsearch=False` 继续覆盖 Python 侧 QTT。
- **`JunqiBoard::apply` 与 Python `GameState.apply` 有差异**：BOTH_DIE 撞军旗时 Python
  判攻方胜，C++ 未判。切片 2 用独立的 `apply_expert()` 对齐，**没改 board.cpp**（会影响 APK 引擎）。
- **热路径跨语言必须用单个紧凑 blob**（`core_bridge.encode_state_blob`：
  60B 棋盘 + 阵亡子 + 18B 头 `<7hi`）。`state_to_cpp` 要 50 次 set_piece ≈ 50µs，
  **绝不能用在热路径**。
- **工作量校准**：计划估 150 行、实际 ~570 行（还依赖 `apply` 与 235 行的
  `_score_action`）。**后续切片按 3~4× 倍率估算。**

## 现有训练产物的真实状态（2026-09-15 清理后，决定重训范围时必看）

用 `python scripts/audit_artifacts.py --probe` 可复现。

**2026-09-15 已做的清理与重置**：
- 三个作废的自对弈经验池（`models/ models_b3/ models_b3opp/` 下的
  `candidate_latest_buffer.pkl`）已删除，释放 11.18 GB。
  `models/evidence_collapsed_20260914/`(3.58 GB) 保留。
- **`models/best.pt` 已用 `models/value_distilled_v2.pt` 覆盖**（md5 `26675f331…`）。
  原 best.pt 的权重与 `models/pool/bc_best.pt` 逐位相同（＝BC 基线副本，
  Value 头退化：平衡 acc 0.330、Draw 恒为 0）；旧内容在 pool/ 下天然保留。
- 冒烟训练产生的 2.37 GB 池也已删除；`models/` 现无 `.pkl`。

**当前产物状态**：
- `value_distilled_v2.pt` = `best.pt`：唯一健康 Value 基座（0.735 / 0.288）。
- `candidate_latest.pt`：修复后首跑（seed 42，2 轮 × 60 局，sims 10）的候选，
  epoch=2、elo=1456.5、Value 平衡 acc 0.619（未塌缩）；与热启动源 106/106 键均不同
  → 证明 R1 已修（此前恒为零更新）。
- `_candidate_gate.pt`：同一轮的门控快照（`net.save()` 格式，可直接喂 gate 的 `--model-a`）。
- `search_distilled_smoke_20260915.pt`：蒸馏链路冒烟产物（仅 80 局面，无质量意义）。
- `reports/gate_smoke_20260915.json`：首次有效配对门控报告（24 局，promote=False）。
- **`datasets/p1_v3` 标签可用**，无需重导。

**待办（下一轮）**：小规模正式复跑自对弈（建议 ≥300 局/轮、sims ≥20）→ 搜索蒸馏
（建议 1200 局面 / depth 3）→ `gate --seeds 100 --promote-to-best` 正式晋级。
注意终局分布异常偏向 `immobilized`（71.7%/53.3%，而人类复盘认输 45.6%），
以及候选对 search2 参考得分仅 0.083~0.167 —— 这两点是"数据质量/棋力"层面的真实问题，
不是本轮修复范围内的 bug。

## 判断"改动是否影响引擎"必须做 A/B 实测（2026-09-15 血的教训）

用 `python scripts/ab_search_compare.py --ref-commit <旧提交>`：导出旧版 `junqi` 包 →
两个独立进程跑同一批固定局面 → 输出"默认深度 / 限时深度 / APK 引擎"三分项对照。
**读代码推断会得出相反结论**：C1 想修"限时搜索浪费预算"，第一版用实测比值外推，
实测反而在 endgame 局面少搜一层（1000ms 预算只用 171ms）。改成
`下一层估计 = max(本层耗时, 累计耗时/2)`（不做比值外推）后才正确。

已确立的基线（改动 `search.py` 后必须复跑）：
- 默认 `depth=2, time_limit=0` 必须**完全一致**（动作/分值/节点数）；
- 限时 `1000ms` 判据见下条，**不要只看 `max_depth`**；
- 当前状态：默认 17/17 一致、APK 17/17 一致、限时 更深 3 / 一致 14 / 更浅(仅记录口径) 0
  / 更浅(回归) 0、决策变化 3 处（均在"更深"的局面上）。

**⚠️ 只看 `max_depth` 会误判（2026-09-15 第二次踩到）**：修 C13 时我额外把降级结果
"退回上一次完整层"，工具报 5 处"更浅(回归)"；逐层追踪后确认是**真回归**——
`deal0` 处女局面 d=1 的全部翻棋候选**同分 0.0**（无信息），d=2 才有区分度。
工具现已改为同时比对**节点数 + 决策（动作/分值）**，并把"节点与决策都没变、只有
`max_depth` 记法不同"单列为"更浅(仅记录口径)"。判定改动的正确顺序始终是：
**逐层 trace / A/B 实测 → 结论**，绝不靠读代码推断。

## `SearchStats.degraded` 的正确含义（2026-09-15，C13 后）

`time_limit_ms > 0` 的调用方**应先查 `stats.degraded`**。为 True 表示预算不足
（根循环在层内被打断，或首层未完成），此时：
- `score` 来自"最后一次有产出的层"，是该层部分最优——真实值的**下界**估计，
  仍是合法决策，**不要**因此丢弃它（退回浅一层会丢信息）；
- `root_scores` 是**截断**的候选集合（不覆盖全部合法动作），逐个分值与排序有效，
  GUI top-N 与蒸馏 softmax 应结合本标记判断；
- 未搜完的层**只写 `FLAG_LOWER_BOUND`** 进 TT，绝不写 `EXACT`
  （根节点局面键可能作为子树在后续搜索中被查询，写 EXACT 会污染真实分数与 PV）；
- 本层一个动作都没搜完时**不写 TT、不记 `max_depth`**（旧实现会把 `-inf` 写成 EXACT）。
`self.stopped`（子搜索内部超时）是另一条路径：整层作废，语义不变。
**遗留**：`degraded` 目前在生产链路无消费者，`ExpertAgent.choose_actions` 不暴露
`stats`；要让它影响训练数据需改教师侧（`train_search_distill`），未做，先问用户。

## 教训：局部 import 的 NameError（2026-09-15 自查出来）

`junqi/train_value_distill.py` 只在函数内 `import json`，而模块级新增的守卫函数也用了
`json.load` → `load_p1_arrays` NameError → 会**直接崩掉 train_rl 的每轮重锚与 Value 探针**。
**判定导入是否可用，必须看 import 的所在作用域，不能用 `'import xxx' in src` 做子串判断**
（我正是这样误判的）。现已有精确到函数作用域的静态扫描测试守卫。同理，写测试要覆盖
"带 metadata 的真实数据集"路径，否则会绕过版本守卫分支。

## 评测口径的真相（2026-09-15 实测，别重复踩坑）

- **镜像标定对确定性策略无区分力**：`eval_gate._run_pair` 的两局（同牌、先后手互换）在
  **确定性策略下逐字段完全相同** ⇒ 配对必然抵消任何座位优势 ⇒ 裁决得分率**恒为 0.5000**。
  这是结构性必然，**不能当作"裁判无偏"的证据**。要检验 arbiter 偏差必须用有随机性的策略
  （只有 nn_mcts 的 r0≠r1）。一般化：配对下 `P(A=1,B=0) − P(A=0,B=1) = P(A) − P(B) ≡ 0`。
- **`evaluate_expert` 在真实对局终局上精确镜像对称**（`Δ − Δ_sym ≡ 0`，实测 60/60）。
  此前基于 `gate_calib_mirror_fixed` 0.6000 的"座位标签偏差"归因**已被推翻** ——
  n=80 时 Wilson 半宽 ±0.105，0.6 与 0.5 **不显著**。真问题是**统计功效**。
- 分辨力优先用 `eval_gate.paired_delta_test`（`d = Δ(r0) − Δ(r1)`，保留幅度 + 配对差分
  自动消座位优势），优于把 Δ 压成 0/0.5/1 的二元裁决；`run_gate` 报告已有 `paired_delta` 段。
- `state.mirror_state` **只翻转位置 + 交换座位标签，不换颜色**；换颜色会得到"恰好取负"的
  假象，那是测量陷阱不是对称化。

## 危险操作禁令（血的教训）

- **删除文件一律用 Python 文件系统操作**（`os.remove` / `shutil.rmtree`），
  **禁止用 `git rm`**：2026-09-15 用 `git rm -r` 时命令被 SIGTERM 中断，遗留
  `.git/index.lock`，随后 `junqi/` 与 `scripts/` 共 96 个文件从工作区消失。
  恢复靠：删 stale lock → `git checkout HEAD -- junqi scripts`（本环境会自动提交，
  所以 HEAD 含最新改动）。核验文件是否还在用关键词检查，不要只信 `git status`。
- **不跑长 `&&` 链的 git 写命令**；git 操作前后确认无 `.git/index.lock`。
- 不用 `dangerouslyDisableSandbox`；不主动删除 `models/`、`军旗复盘/`、`datasets/` 等数据。

## 项目硬约束（来自 AGENTS.md / AI_TRAINING_AND_HUMAN_PLAY_PLAN.md）

- 改代码前先读 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`，并声明本次修改所属阶段 P0–P4。
- **先加/改测试，再改实现**；完成后报告：改动文件、实验种子、测试结果、未解决风险。
- 禁止：真实暗子身份进公共 Policy；吃子/挖雷等中间奖励；三个独立阶段模型；
  只看 loss 覆盖 `best.pt`；P0 未过就长训练。
- Value 标签严格按官方 `list.cfg` 终局码：1/21/22/23 → ±1；40/42/43 → 0；
  20/24 → 不赋 Value（`dataset.terminal_label_from_meta` 是唯一真源，
  统计口径必须走 `outcome_bucket_from_meta`，禁止各写一遍码表）。
- 数据集默认目录 `datasets/p1_v3`（`dataset.DEFAULT_P1_DIR`），
  低于 3.0.0 的旧数据集会被 `check_p1_version` 拒载。
- 门控唯一真源是 `junqi/eval_gate.py::run_gate`（配对同牌 + Wilson + 三元 SPRT）；
  训练主循环的 `train_rl.evaluate_gate` 只是它的薄封装。

## 架构要点

- 在线传统搜索引擎是 **`junqi/search.py::ExpertSearchEngine`**（`ai.py` / `hybrid_engine.py` 复用）。
  · **它有两张置换表，且必须分离**：`self.tt`（主表，`depth` = 剩余搜索深度）与
    `self.qtt`（QSearch 专用，`depth_left` = 剩余吃子链长度）。共用一张表会让 qsearch 写入的
    `depth_left=10` 条目被 `_negamax` 的 `depth=2` 查询命中（`entry.depth >= depth` ⇒ 10>=2）
    → **返回错误分数**。`use_qtt=False` 可完全退回原实现。
  · **`qsearch_depth` 不要凭直觉下调**：残局 `depth=2` 下 `qd=4` 会改变决策，
    而 `depth=3` 下 `qd=4` 与 `16` 决策相同 —— 两档结论矛盾，敏感性随深度/局面变化，
    未经对局级 A/B 不得改默认值 16。
  · 改动 `search.py` 后跑 `scratch/perf_baseline.py verify`（**capture 必须在改动前做**）。
    残局加速基准（2026-09-15）：QTT 开启 6038ms → 4188ms（1.44×），qnodes −27.5%。
  `junqi/expert/` 已于 2026-09-15 **彻底删除**（不可导入、fan-in 0），
  记录见 `docs/06-References/DEPRECATED_EXPERT_PACKAGE.md`。不要重建同名包。
- `junqi/net.py::save()` 写出的是**包装字典** `{"model_state": ...}`，不是裸 `state_dict`。
  `torch.load` 之后必须先 `JunqiNet.unwrap_state_dict()`，否则 `load_state_dict(strict=False)`
  会静默零载入（这正是历史上"best 对手是随机网络"的根因）。
- 热启动**禁止** `net = JunqiNet.load_from_file(...)` 重绑变量（会让 optimizer 脱钩、
  权重零更新）；用 `train_rl.warmstart_candidate(net, optimizer, path, ...)`。
- `StratifiedReplayBuffer` 样本契约：policy `(state, mask, target, phase)`、
  value `(state, z_cls, is_world)`，索引写死。
- `scripts/archive/` 与 `tests/utils/` 是**归档脚本**（见各自 README），
  不是 API、不进 pytest 收集范围；不要在其上继续开发。
- `junqi/benchmark.py::create_benchmark_suite`、`run_training` 等 21 个函数 >120 行，
  属已知结构性债务；拆分需单独评估，不要顺手改。
