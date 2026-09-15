# C++ 移植 ExpertSearchEngine：可行性与立项方案

**归属阶段**：P4（工程基础设施）/ P1（传统搜索性能）
**状态**：**切片 1、2 均已完成并通过等价性验收（2026-09-15）；切片 3 待决策**
**日期**：2026-09-15（立项） / 2026-09-15（切片 1 落地）
**依据**：`AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`（AGENTS.md 指定执行基线）
**触发**：用户问"专家引擎还有提升空间吗" → 建议顺序第 4 项

---

## 一、结论

> **2026-09-15 更新：切片 1 已交付并通过验收**，实测残局单步 4236 → 1225 ms（3.46×），
> 估值函数 5.8~9.0×，`perf_baseline` 逐位一致（worst 5.684e-14）。
> 同时**修正了本文原头寸模型**：桥接开销实测 6 µs 而非原估 1~2 µs，
> C++ 计算 2.4~3.8 µs 而非原估 5 µs ⇒ 收益大头在切片 2/3，不在继续优化编码。
> 详见第五节与 `docs/CHANGELOG.md` 第十二批。

**可行，收益显著，但不建议一次全量移植。**

- 语言级节点吞吐比实测 **21.9×**（同一算法、同一评估函数，仅切换实现语言）；
- C++ 侧**已有可复用的地基**：`board` / `rules` / `zobrist` / `eval_apk` / `apk_engine`
  与 Python↔C++ 桥接（`junqi/core_bridge.py`，含自动探测与透明降级）；
- 但 `ExpertSearchEngine` 的评估函数（`eval_expert.py`，500 行特征工程）在 C++ 侧
  **没有任何对应物** —— 这是工作量主体，也是收益主体。

**建议按三个切片递进，先做切片 1（`evaluate_expert`），理由是它同时是最大瓶颈与最易验证的一环。**

---

## 二、现状盘点

### 2.1 C++ 侧已具备

| 文件 | 内容 | 可直接复用 |
|---|---|---|
| `include/board.h` / `src/board.cpp` | 棋盘、棋子、走法生成 | ✅ |
| `include/rules.h` / `src/rules.cpp` | 邻接、行营、铁路、`battle` 判定 | ✅ |
| `include/zobrist.h` / `src/zobrist.cpp` | 64 位哈希 | ✅ |
| `include/eval_apk.h` / `src/eval_apk.cpp` | **APK 评估函数**（2560 等比表 + 动态炸弹 + 行营偏置） | ⚠ 仅可作结构参考，特征集与 `evaluate_expert` 不同 |
| `include/apk_engine.h` / `src/apk_engine.cpp` | APK 搜索（PVS + 纯吃子 QSearch + Delta 剪枝 + TT） | ⚠ 搜索骨架可参考，评估需替换 |
| `bindings/python_bindings.cpp` | pybind11 绑定 | ✅ 加新类即可 |
| `CMakeLists.txt` / `setup.py` | 构建体系 | ✅ |

### 2.2 Python 侧现状（本次实测）

`ExpertSearchEngine`（`junqi/search.py`）残局 ply=90、depth=2：

| 指标 | 数值 |
|---|---|
| 单步耗时 | 6038 ms（QTT 前） / **4188 ms**（QTT 后，第 3 项） |
| nodes / qnodes | 759 / 35538（qnodes 是 43~47×） |
| `evaluate_expert` | **104.68 µs / 次**，占搜索耗时约 91% |
| `legal_actions` | 56.15 µs / 次（11859 次 ≈ 0.67 s） |
| `compute_zobrist` | 7.41 µs / 次 |

**瓶颈结论**：`evaluate_expert` 是单点主导（CHANGELOG 第七节独立复现过同一结论：
"91% 耗时在 evaluate_expert，49.6 万次调用"）。

---

## 三、头寸量化（实测，`scratch/measure_cpp_speedup.py`）

方法：用**同一算法（APK 引擎）+ 同一评估函数**跑同一局面，只切换实现语言，
因此 nps 之比就是"移植能拿到的**语言级加速**"，不含算法差异干扰。

| 局面 | 轨道 | 墙钟 ms | 内部 ms | nodes | qnodes | nps |
|---|---|---|---|---|---|---|
| midgame (ply=30) | C++ | 3.09 | 0.70 | 115 | 379 | **708,041** |
| midgame (ply=30) | Python | 15.21 | 15.18 | 115 | 376 | 32,348 |
| endgame (ply=90) | C++ | 4.69 | 1.72 | 139 | 858 | **579,415** |
| endgame (ply=90) | Python | 68.99 | 68.97 | 124 | 2031 | 31,247 |

- **语言级节点吞吐比（中位）21.9×；墙钟比 4.9~14.7×**（墙钟含状态转换与引擎构造开销）。
- ⚠ **顺带发现**：endgame 的 C++/Python 节点数差异较大（qnodes 858 vs 2031，2.4×），
  说明 APK 双轨**当前并不完全等价**。这不是本次移植的障碍，但
  `docs/CHANGELOG.md` 里"双轨 100% 对齐"的说法在**当前代码状态下已不成立**，需单独核查。

**预期收益（保守）**：`evaluate_expert` 从 104.68 µs 降到 ~5 µs（按 20× 计），
残局单步 4188 ms → **约 600~900 ms**；教师打标 / 自对弈全线同比例受益。

---

## 四、移植范围分解

| # | 组件 | Python 位置 | 现状 | 估行数（C++） | 难度 |
|---|---|---|---|---|---|
| 1 | `evaluate_expert` 特征工程 | `eval_expert.py` 500 行 | ✅ **已完成**（切片 1，`src_cpp/src/eval_expert.cpp` ~700 行） | 600~800 | 高（特征多，逐位等价难） |
| 2 | `fortress_score` / `is_dead_draw` | `analysis.py` | ✅ **已完成**（随切片 1 一并移植） | 150~250 | 中（BFS + 缓存） |
| 3 | `_qsearch` | `search.py` 95 行 | 骨架可参考 `apk_engine` | 150 | 中 |
| 4 | `_negamax` + PVS + TT | `search.py` | 骨架可参考 | 200 | 中 |
| 5 | Star1 机会节点 + 期望 | `search.py` 110 行 | **无**（APK 引擎没有暗子期望） | 250 | **高** |
| 6 | IDS + 时限 + killer/history | `search.py` | APK 引擎部分具备 | 200 | 中 |
| 7 | `_score_action` 走法排序 | `search.py` 235 行 | 无 | 250 | 中（大量硬编码常量） |
| 8 | Python 绑定 + 桥接 | `core_bridge.py` | ✅ 已有模式 | 100 | 低 |

**合计约 1900~2200 行 C++**（不含测试）。

---

## 五、分期计划（建议）

### 切片 1：`evaluate_expert` → C++（**已完成，2026-09-15**）

- **为什么是它**：单点占 91% 耗时，且验证方式最干净（纯函数，逐位比对即可）。
- **接口**：Python 侧把 `GameState` 序列化为紧凑字节（**实测改用 60 格 × 1 字节**，
  而非原估计的 50 格 × 8 字节 —— 每格只需 `rank(5bit) | color(1bit) | revealed(1bit)`），
  另加 dead 字节流与少量标量；C++ 解析后返回 `float`。
- **收益（实测）**：残局单步 **4236 ms → 1225 ms（3.46×）**；估值函数 99.64 µs →
  8.4~10.2 µs（5.8~9.0×）。**低于原估的 600~900 ms**，原因见下方"桥接成本重估"。
- **验收**：✅ `scratch/perf_baseline.py verify --cpp` → `EQUIVALENT`；
  560 次估值比对 worst_abs_diff = **5.684e-14**（阈值 1e-9）；
  30 次 `depth=2` 搜索动作 + 分值全一致；全量测试 518 passed / 3 skipped。
- **回滚**：`ExpertSearchEngine(use_cpp_eval=False)` 与 `core_bridge` 的既有降级路径
  （异常时还会自动永久关闭 C++ 快路径）。

#### 桥接成本重估（**修正原计划的头寸模型**）

原计划假定"转换 + 跨语言调用约 1~2 µs"，实测为 **6.06~6.44 µs**，高约 3 倍；
而 C++ 计算实测 **2.38~3.80 µs**，比原估 ~5 µs 更快。即：

| 段 | 原估 | 实测 |
|---|---|---|
| C++ 计算 | ~5 µs | **2.38~3.80 µs** |
| 序列化 + pybind 编组 | 1~2 µs | **6.06~6.44 µs** |

**这直接改变切片 2/3 的论证**：继续做序列化微优化（如增量编码、参数打包）最多再省
1~2 µs，属一次性收益；而切片 2/3 把整棵 qsearch 子树搬进 C++ 后，**每节点序列化
彻底消失**，该开销自然归零。故本切片不做序列化微优化，把预算留给切片 2/3。

#### 已固化为测试的两个陷阱

1. **邻接表顺序**：C++ `get_road_neighbors()` 与 Python `NEIGHBORS` **顺序不同**
   （(0,1)：C++ `[0,6,2]` vs Python `[0,2,6]`），而 `my_reach[0]` 取首元素 ⇒
   顺序差异是**语义**差异。改用 `scripts/gen_expert_tables.py` 从 Python 真源生成
   常量表，并有逐格比对测试 + `--check` 防漂移。
2. **阵亡子不可由棋盘反推**：曾想省掉 `encode_dead`，用
   `dead = COMPOSITION − board_counts` 推导。实测该不变式在
   `_evaluate_chance_flip` 的子状态上**不成立**（12 次派生为负）。已放弃，
   并留 `test_dead_counts_are_not_derivable_from_board` 固定此事实。

### 切片 2：`QSearch` + `legal_actions` → C++（**已完成，2026-09-15**）

- 依赖切片 1（已完成）；把 `_qsearch` 整棵递归搬进 C++，Python 只在根节点跨语言一次。
- **验收**：✅ `perf_baseline verify` → `EQUIVALENT`；三档（纯 Python / 仅 C++ 估值 /
  完整 C++）决策全部一致；全量 526 passed / 3 skipped。
- **实测（depth=2）**：残局 **4250 → 245 ms（17.36×）**，中盘 9.84× / 7.74×，开局 3.60×。
  切片 1 预测的"每节点 6 µs 序列化归零"完全兑现：切片 1 后 1228 ms → 切片 2 后 245 ms。
- **实际工作量远超原估**：计划第 4 节给 `_qsearch` 估 150 行，实际 ~570 行 ——
  因为 `_qsearch` 还依赖 `legal_actions`、`apply`、`_score_action`（235 行硬编码常量）。
  **后续切片请按此校准倍率（约 3~4×）估算。**

#### 三个关键设计决策（已固化成测试）

1. **走法顺序不必对齐**：探针 300 局面 —— 集合不一致 **0**，顺序不一致 **269**
   （Python 铁路走法来自 `set`）。但 alpha-beta 返回值与遍历顺序无关，
   故只需集合一致。省掉了改 Python 侧 `legal_actions` 的麻烦。
2. **C++ 侧刻意不实现 QTT**：纯缓存，µs 级节点下收益为负。代价是 `qnodes` 更高
   （残局 36721 vs 25780）。已加测试防止误判为回归。
3. **单独实现 `apply_expert`**：`JunqiBoard::apply` 与 Python 有一处差异
   （BOTH_DIE 撞旗时 Python 判攻方胜）。改 `board.cpp` 会影响 APK 引擎，故独立实现。

### 切片 3：完整 `ExpertSearchEngine`（Star1 + IDS）（**下一个应做的切片**）

- **切片 2 后的论证**：开局 ply=0 在切片 2 后仍要 424 ms，而它 qnodes=0 ——
  即这 424 ms **全部**在机会节点（`_evaluate_chance_flip`）与 `_negamax` 上。
  切片 3 是唯一能继续削减它的路径，且收益面覆盖开局/中盘（切片 2 主要惠及有吃子的局面）。

- 依赖切片 1-2；新增 `junqi_core.ExpertSearchEngine`，`core_bridge` 加 `search_expert_auto`。
- **验收**：`ab_search_compare.py` 的默认深度 17/17 一致；限时路径记录深度差异（允许更深）。
- **注意**：Star1 语义复杂（暗子期望 + 全窗口子搜索），`_evaluate_chance_flip` 的
  `depth<=1 or ply_depth>=1` 截断必须原样保留。切片 2 完成并实测后再决定是否上切片 3。

---

## 六、等价性验证方案（复用现有工具，不另造）

| 层次 | 工具 | 判据 |
|---|---|---|
| 评估函数 | `scratch/perf_baseline.py`（220 随机 + 60 残局） | `eval0`/`eval1`/`dead`/`fort0`/`fort1` 逐位一致 |
| 搜索决策 | 同上（30 次 depth=2 搜索） | 动作 + 分值逐位一致 |
| 限时行为 | `scratch/bench_qtt_timed.py` 模式 | 记录深度差异，**更深允许、更浅排查** |
| 端到端 | `scripts/ab_search_compare.py --ref-commit <旧>` | 默认深度一致性 + 决策差异明细 |
| 对局级 | `junqi/eval_gate.py::run_gate` + `paired_delta_test` | 双轨对局置换应无显著差异（第 1 项交付物） |

⚠ **capture 必须在改代码之前做**（第 3 项已确立的纪律）。

---

## 七、回退策略

1. **编译期**：`junqi_core` 未编译 → `core_bridge.HAS_CPP_CORE=False` → 透明降级（既有能力）。
2. **运行期**：每个切片都带独立开关（`use_cpp_eval` / `use_cpp_qsearch` / `use_cpp_expert`），
   默认值由该切片的等价性验收结果决定，未通过者默认 `False`。
3. **异常**：`search_apk_auto` 已有 `try/except → Python 降级` 模式，新入口沿用。

---

## 八、风险与未决

| 风险 | 说明 | 缓解 / **切片 1 后的实测结论** |
|---|---|---|
| 逐位等价难 | `evaluate_expert` 含浮点累加顺序、字典迭代序、缓存副作用 | ✅ **已量化**：顺序敏感度 max 5.684e-14（default）/ 2.274e-13（apk），比阈值 1e-9 低 4 个数量级（探针 `scratch/probe_eval_order_sensitivity.py`）。**残留**：节点数可能差 <2%（并列比较翻转），决策不变 |
| **邻接表顺序是语义差异** | 切片 1 中新发现，原计划未识别 | ✅ 已用 `scripts/gen_expert_tables.py` 从 Python 真源生成 + 逐格比对测试 + `--check` 防漂移 |
| **阵亡子不可由棋盘反推** | 切片 1 中新发现，原计划未识别 | ✅ 已放弃该"优化"，留 `test_dead_counts_are_not_derivable_from_board` 固定 |
| Star1 语义复杂 | 暗子期望 + 全窗口子搜索，移植易错 | 切片 3 单独验证；`_evaluate_chance_flip` 的 `depth<=1 or ply_depth>=1` 截断必须原样保留 |
| APK 双轨已漂移 | 实测 qnodes 2.4× 差异，与 CHANGELOG 记载不符 | **单独立项核查**，不混入本次移植（切片 1 未触碰 APK 引擎） |
| 构建环境 | 需 MSVC + pybind11；`pip install -e .` 会重新编译 | ✅ 已解决：本机 `reg.exe` 被拦截导致 setuptools 找不到 SDK，加 `scripts/build_cpp.py` 自动注入 `INCLUDE`/`LIB`/`PATH` |
| 收益上限 | 21.9× 是**语言级上界**；C++ 版评估函数本身也变重 | ✅ 已重估：纯计算已达 2.4~3.8 µs（优于预估），瓶颈转移到桥接（6 µs）⇒ 由切片 2/3 结构性消除 |

---

## 九、待决策

### 已解决（2026-09-15）

1. ✅ **是否开写切片 1** —— 已开写并交付，等价性验收通过（见第五节）。
2. ✅ **切片顺序** —— 先搬评估的顺序是对的：它单点占 91%，且验证最干净
   （纯函数、280 状态逐位比对），确实最快拿到可验证收益。
3. ⬜ **"APK 双轨漂移"核查**（第三节发现）—— **仍未做**，与本次移植相互独立。
   切片 1 未触碰 APK 引擎，该问题原样保留。

### 仍待决策

4. ✅ **是否继续切片 2** —— 已交付（2026-09-15）。残局 4250 → 245 ms（17.36×）。
   原担心的"`legal_actions` 铁路滑行需要对齐顺序"经探针证实**不必对齐**
   （集合一致即可，alpha-beta 返回值与遍历顺序无关），风险显著低于预估。
5. **是否做切片 3（Star1 机会节点 + `_negamax` + IDS → C++）？**
   这是当前**收益最大的剩余项**：开局 ply=0 在切片 2 后仍需 424 ms 且 qnodes=0，
   耗时全在机会节点与 `_negamax` 上，切片 1/2 都碰不到它。
   风险点仍是 Star1 语义（`_evaluate_chance_flip` 的 `depth<=1 or ply_depth>=1`
   截断必须原样保留）与 `_negamax` 的 PVS/杀手/历史/路径重复检测。
   按切片 2 的实际/估算倍率（3~4×），预计 ~1500~2000 行 C++。
5. ✅ **限时（IDS）路径复核** —— **已完成（2026-09-15）**。
   `scratch/bench_cpp_timed.py`（800 ms 预算，`max_depth=8`）：
   更深 2 / 一致 2 / **更浅 0**，四局面决策全一致 ⇒ 无回归。
   中盘两局面 C++ 多搜一层（2→3）后仍选同一动作。
   判读提醒：残局 `nodes` 28 vs 513 但 `max_depth` 同为 1，**不限时下两者
   nodes/qnodes 完全相同**，差异来自 C++ 更快从而在预算内多加了一层并被 `self.stopped`
   整层作废 —— 属预期，不是 C++ 引入的问题。切片 2/3 后仍建议重跑本脚本。
6. **是否需要"严格逐位"模式？**
   若未来要求训练完全可复现（连节点数都一致），需让 C++ 复刻 Python 的 board dict
   迭代序 —— 代价是每次多传一个顺序数组（约 +2~4 µs，加速比从 3.46× 降到 ~2.6×）。
   当前判定：**不需要**，5.7e-14 的差异远小于任何实际决策阈值。
