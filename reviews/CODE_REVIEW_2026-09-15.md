# junqi_engine 全面代码审查报告

> 审查日期：2026-09-15
> 审查方式：初检为**只读审查**；随后按优先级逐条修复（见下方「修复状态」）
> 审查范围：`junqi/` 全部 54 个模块、`scripts/` 34 个脚本、`tests/` 39 个测试文件、核心文档与配置
> 阶段归属：本次审查覆盖 P0–P4

## 修复状态（2026-09-15 收口）

全部条目已处置。测试：**298 passed / 3 skipped（修复前基线）→ 403 passed / 3 skipped**。
详细改动见 `docs/CHANGELOG.md` 同日两条记录。

| 编号 | 结论 | 处置 |
|---|---|---|
| R1 热启动 optimizer 脱钩 | 致命 | ✅ 已修（`warmstart_candidate` 就地载入） |
| R2 best 对手为随机网络 | 致命 | ✅ 已修（`unwrap_state_dict` + 缺键降级为 None） |
| R3 门控双实现 | 高 | ✅ 已修（训练主循环委托 `eval_gate.run_gate`） |
| R4 `junqi/expert/` 僵尸包 | 高 | ✅ 标记废弃 + 守卫测试（未删除，见 DEPRECATED.md） |
| R5 方案与代码认输局口径冲突 | 中 | ✅ 已回写方案（含原因/影响/验证/回滚） |
| R6 数据集版本口径分裂 | 中 | ✅ 统一 `DEFAULT_P1_DIR=p1_v3` + 版本守卫 |
| C1 IDS 早停语义错误 | 高 | ✅ 已修（`should_stop_ids`） |
| C2 超时返回 `-inf` 污染蒸馏 | 高 | ✅ 已修（降级返回 0.0 + `degraded` 标记） |
| C3 根节点上界当精确分 | 中 | ✅ 已修（`exact_root_scores` + `root_scores_bounded`） |
| C4 topn 回退用 1e5 排序分 | 中 | ✅ 已修 |
| C5 `Action("pass")` TypeError | 低 | ✅ 已修（返回 None） |
| C6 MCTS 重复阈值硬编码 | 中 | ✅ 已修（`tree_repetition_limit`） |
| C7 code 24 计入 decided_win | 中 | ✅ 已修（`outcome_bucket_from_meta` 同源） |
| C8 异常静默吞掉 | 中 | ✅ 已修（计数 + 告警 + 记录到对局） |
| C9 伪 Elo 随机游走 | 中 | ✅ 已修（`elo_update_from_score` 标准公式） |
| C10 长函数/死代码 | 中 | 🟡 部分：删除死类；超长函数拆分未做（收益 < 回归风险） |
| C11 文档漂移 | 中 | ✅ 已修（README 计数/YAML 声明、CHANGELOG） |
| C12 高危路径零测试 | 中 | ✅ 已修（train_bc / eval_bc / fit_weights / load_p1_arrays） |
| P1 经验池每轮全量落盘 | 高 | ✅ 已修（节流 + 原子写） |
| P2 MCTS batch=1 / 重复 legal_actions | 中 | ✅ 已修（`acts` 透传 + avoid 复用） |
| P3 position_key 重复构造 | 中 | ✅ 已修（MCTS 根避免重算；hybrid 复用 nxt） |
| P4 门控每局反序列化模型 | 中 | ✅ 已修（`load_net_cached`） |
| P5 叶子白算 Zobrist、闭包重建 | 低 | ✅ 已修 |
| P6 apk TT 错命中 + 无界 | 中 | ✅ 已修（完整键校验 + 容量上界） |
| P7 `venv/` 4.7GB 在工程内 | 低 | ⬜ 未处置（运营决策，需人工确认） |

**附带发现并修复**：`scripts/cleanup_models.py` 的删除模式含 `*_distilled.pt`，会删掉
P3 热启动链首选的 `models/value_distilled_v2.pt`，导致训练静默退回未校准的 BC 价值头。
现改为显式保护名单 + 默认 dry-run。

---

## 0. 总体结论

1. **P0/P3 的"已完成"标记与代码实际状态不符**：当前自博弈训练闭环在数学上是无效的——存在两个会让训练输出恒等于输入的缺陷（R1、R2）。这不是调参问题。
2. **门控存在双实现**，且训练主循环只使用了统计效力不足的那套（R3），因此历史上所有由轮内门控得出的"模型变强/未变强"结论都不成立。
3. **核心规则层（`rules.py` / `state.py` / `config.py` / `encoder.py`）结构干净、无循环依赖**，且传统估值函数**未**被 1e5 量级启发分污染——这是可信的地基，不需要重构。

### 修复优先级

| 顺序 | 编号 | 一句话 | 严重度 |
|---|---|---|---|
| 1 | R1 | 热启动重绑 `net`，optimizer 与网络脱钩 → 权重零更新 | 致命 |
| 2 | R2 | best 对手权重加载格式不匹配 → 25% 对局对手是随机网络 | 致命 |
| 3 | R3 | 门控双实现，训练主循环用弱的那套 | 高 |
| 4 | P1 | 回放池每轮全量 pickle 落盘（单文件 3.8 GB） | 高 |
| 5 | R4 | `junqi/expert/` 11 模块僵尸包 | 高 |
| 6 | R5–R12 | 文档漂移、异常吞掉、伪 Elo、统计口径分裂等 | 中 |

---

## 1. 路线与方案层问题

### R1 · 热启动导致 optimizer 与网络脱钩，首轮起权重零更新

- **严重度**：致命 ｜ **影响范围**：整个 P3 自训练
- **证据**：
  - `junqi/train_rl.py:1113-1114` `net = JunqiNet().to(device)` → `optimizer = torch.optim.AdamW(net.parameters(), ...)`
  - `junqi/train_rl.py:1162 / 1174 / 1179` 三处 `net = JunqiNet.load_from_file(...)` **重新绑定变量名**
  - `junqi/train_rl.py:1016-1039` `train_epoch` 对新 `net` 做 forward/backward
- **机理**：`optimizer` 的 `param_groups` 仍持有被丢弃的旧对象引用。新 `net` 反传后新参数有 grad，旧参数 `grad is None` → AdamW 跳过 → **权重永不更新**。仅断点续训路径（`:1143-1152` 走 `load_checkpoint` 就地载入）正确。
- **触发条件**：首次训练、`--fresh`、`--rebase-baseline` 之后（此时 ckpt 被删除）。
- **唯一可观测征兆**：loss 恒定不变。
- **修复方向**：改为 `load_state_dict` 就地载入；或重绑后重建 optimizer 并恢复其 state。

### R2 · 自博弈的 "best 对手" 实际是随机初始化网络

- **严重度**：致命 ｜ **影响范围**：自博弈数据质量、所有 Policy/Value 目标
- **证据**：
  - `junqi/train_rl.py:1217` `opp_dict = torch.load(opp_path, ...)`
  - `junqi/net.py:252-259` `net.save()` 写出的格式为 `{"model_state": ..., "in_channels": ..., "num_blocks": ..., "channels": ...}`
  - `junqi/train_rl.py:529-537` worker 内 `if "in_conv.0.weight" in opp_net_dict`（包装字典下为 False）→ `net1.load_state_dict(opp_net_dict, strict=False)` **键名无一匹配，静默通过**
  - `junqi/train_rl.py:556` `if opp_type == "best": target_net1 = net1`
- **后果**：OPP_MIX 中 best 占 0.25（diverse 预设 0.30），即 1/4 对局的"强对手"是未训练网络。`opponent_mix` 日志看不出异常。
- **修复方向**：worker 内复用 `JunqiNet.load_from_file` 语义；加载后断言 `missing_keys` 为空。

### R3 · 门控双实现，训练主循环用的是弱的那套

- **严重度**：高 ｜ **影响范围**：P3/P4 晋升判定与所有实力结论
- **证据**：
  - `junqi/eval_gate.py:149-158` `run_gate`：**配对同牌**（同一 seed 跑 A 先手 + B 先手）、显式座位计分、Wilson 区间、三元 SPRT、终局原因拆分、裁决判分——质量合格
  - `junqi/train_rl.py:596-760` 自建 `wilson_lower_bound` / `decide_promotion` / `inloop_gate_decision` / `_gate_game_job` / `_tally` / `evaluate_gate`，且 `train_rl.py` **不 import `eval_gate`**
  - `junqi/train_rl.py:747-748` 两个方向用 `seed + i` 与 `seed + 100_000 + i` → **非配对同牌**
  - `junqi/train_rl.py:693` `device="cpu"` 硬编码（即便有 GPU）
  - `junqi/train_rl.py:679-683` 默认 `inloop_gate_promote=False` → 轮内判定恒返回 False
- **修复方向**：训练主循环直接复用 `eval_gate.run_gate`，删除 `train_rl` 内的重复实现。

### R4 · `junqi/expert/` 11 个模块是僵尸包

- **严重度**：高（认知负担 + 误导新代理）｜ **影响范围**：架构可信度、README 准确性
- **证据**（AST 静态扫描 + 成员存在性核验）：
  - 引用的成员在 `state.py` / `config.py` 中**全部不存在**：`Move`、`get_piece_at`、`current_turn`、`turn_count`、`get_pieces`、`PIECE_RANKS`、`is_my_base`
  - 命中模块：`expert_engine / tactical_analyzer / threat_detection / search_optimizer / mobility_calculator / conditional_value / hidden_piece_belief / tempo_tracker / rule_validator / move_adapter`（10/11）
  - `state.py` 只定义 `Action`，**从未定义 `Move`**
  - 内部依赖图 fan-in = 0，`junqi/` 全库除 `expert/` 内部自引用外无任何调用点
  - `expert/expert_engine.py:305` `self._score_move(m, None, None, None, None, None)` → 对 `None` 取属性 → 必然 AttributeError
- **结论**：真实在线引擎是 `junqi/search.py` 的 `ExpertSearchEngine`；不存在"双胞胎引擎"，是被放弃的第三套实现。
- **修复方向**：整包标 Deprecated 或删除；同步 README 中"传统搜索专家模型深度对齐"的描述。

### R5 · 方案文档与代码冲突（AGENTS.md:20 要求先报备）

- **严重度**：中 ｜ **影响范围**：后续所有代理的修改依据
- `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md:244`：「自博弈认输局…双方按 ±1 计入 Value」
- `junqi/train_rl.py:980-982`：`if resigned_seat is not None: value_samples = []; public_value_samples = []`
- **代码是对的**（防 Value 自证回路；实证：认输开启时三分类准确率 60%→16%，关闭后 1 轮恢复），但方案未回写，后续代理会按文档改回去。

### R6 · 数据集版本口径分裂

- **严重度**：中 ｜ **影响范围**：Value 训练、BC 训练、所有指标可比性
- `junqi/train_rl.py:102` `ANCHOR_VAL_DIR = "datasets/p1_v3"`
- 但下列**默认路径仍是旧版本**：`junqi/train_value_distill.py:489`（p1_v2）、`junqi/__main__.py:183/188/192/193/210`（p1_v1）、`junqi/train_bc.py:126-127`（p1_v1）、`junqi/eval_bc.py:22/212`、`junqi/dataset.py:291/546/552`（p1_v2）
- `datasets/p1_v2/metadata.json` created_at = 2026-09-06，**早于** `junqi/dataset.py:179` 记录的「2026-09-13 修正：code 24 此前被无条件视为明确胜负」

---

## 2. 代码细节与健壮性

| # | 问题 | 证据 | 影响范围 | 严重度 |
|---|---|---|---|---|
| C1 | IDS 早停判据语义错误：总耗时 > 25% 预算即停，而非"预估下一层超限" | `search.py:769-771` | 传统搜索、GUI 提示、搜索蒸馏教师 | 高 |
| C2 | 超时时返回 `(acts[0], -inf)`；`-inf` 被 `hybrid_engine.py:262` 取负成 `+inf` 参与排序，并被全量 softmax 蒸馏 | `search.py:774`、`hybrid_engine.py:262`、`train_search_distill.py:49/73` | 蒸馏标签污染 | 高 |
| C3 | 根节点后续动作用收窄 alpha 窗口搜索，fail-low 的**上界**被当精确分写入 `stats.root_scores` | `search.py:698-714, 763` | GUI top-3 提示 | 中 |
| C4 | topn 回退路径把 1e5 量级 move-ordering 分当估值分返回（还 `+= 100_000`） | `ai.py:329-336` | GUI 提示 | 中 |
| C5 | `Action("pass")` 缺必填参数 `frm`，触发即 TypeError | `hybrid_engine.py:213` | 兜底分支 | 低 |
| C6 | MCTS 树内重复判和阈值硬编码 `>= 3`，未读 `cfg.repetition_draw_count`（生成夹具已改 4） | `mcts.py:184` | 搜索与规则夹具不一致 | 中 |
| C7 | 统计口径与标签口径漂移：`dataset.py:421` 把 code 24（断线）计入 `decided_win`，而 `:185` 判为无 Value | `dataset.py:421` vs `:183-189` | `metadata.json` 中 `decided_win=488` 虚高 | 中 |
| C8 | 异常静默吞掉：池加载失败、重锚失败仍继续训练、专家估值失败置 None（裁决判分静默退化 0.5） | `train_rl.py:390/897/1284`、`selfplay.py:233/319` | 故障不可观测 | 中 |
| C9 | 伪 Elo：`current_elo += 16*(ov_score-0.5)*2` 是随机游走，却被写入 `elo_history.jsonl` 当实力曲线 | `train_rl.py:1340` | 所有基于 Elo 的结论 | 中 |
| C10 | 8 处 `except…: pass`、23 处 `except Exception`、20 处 TODO；21 个超长函数 | 全库统计 | 可维护性 | 中 |
| C11 | 文档漂移：README 测试计数四处自相矛盾（188/225/258/实际 297）；`configs/*.yaml` 全库零读取点，"修改后无需重启生效"为假 | `README.md:13/86/210/317`、`README.md:182-189` | 违反 AGENTS.md:18 | 中 |
| C12 | 高危路径零测试：`train_bc.py`、`eval_bc.py`、`fit_weights.py`、热启动优先级链（`:1154-1156`）在 tests 中均无命中 | tests 全量检索 | 回归风险 | 中 |

---

## 3. 性能瓶颈

- **P1（最严重）每轮无条件把整个回放池 pickle 落盘**
  `train_rl.py:805-807` + `:1343`。实测 `models/candidate_latest_buffer.pkl` **单文件 3.8 GB**（×3 实验目录 ≈ 11.4 GB），另有 `models/evidence_collapsed_20260914/` 3.7 GB。这是当前磁盘占用与单轮耗时的主要来源。

- **P2 MCTS 每次模拟 batch=1 前向**
  `mcts.py:196` 在 `for _ in range(simulations)` 内做单状态推理（sims=20~60 即 20~60 次前向/手），仅根节点用了 `predict_batch`（`:112`）。另 `net.py:164 + :177` 单次 `predict_state` 把 `legal_actions()` 算了两遍。

- **P3 `position_key()` 在候选循环里逐动作构造**
  `state.py:19-30` 对整盘 50 子排序建元组，被 `mcts.py:122/227`、`search.py:711`、`hybrid_engine.py:270/296` 反复调用；`hybrid_engine` 每个动作先 `state.apply(a)`（`:246`）再 `apply(a)`（`:270`）算 key。

- **P4 门控每局从磁盘反序列化模型 + 强制 CPU + 每场景重建进程池**
  `train_rl.py:688-693`、`selfplay.py:206`、`train_rl.py:722-726`。默认 256 局/轮 ≈ 512 次模型加载。

- **P5 Zobrist 与闭包开销**
  `search.py:552` 在 `depth<=0` 判定之前无条件算 Zobrist，叶子节点白算一次；`_is_tactical` 闭包定义在 `for a in ordered_acts` 循环体内部（`search.py:719`），每动作每深度重复创建。

- **P6 `apk_engine` 置换表错命中 + 无界内存**
  `apk_engine.py:267-268` 只存掩码后的 18 位哈希，`:446` 取值不校验原 key（对比 `tt.py:80` 有校验）；`self.tt` 是无界 dict 且全库无 `clear()`。

- **P7 工程体积**
  `venv/` 4.7 GB 位于工程目录内。

---

## 4. 已核查确认**没有**问题（避免重复排查）

- **估值函数未被 1e5 启发分污染**：`bomb_suicide_exchange`、`camp_outstrike_bias` 只出现在 `search.py:124/132` 的 `_score_action()` 排序路径，与终局分 `WIN_SCORE = 1_000_000`（`state.py:16`）分界清晰。
- **P1 数据集合规**：按**对局**切分（`dataset.py:374-383`，非 ply 随机）；编码 `world=None` 公共视角（`dataset.py:249`）；终局码映射本体正确（`dataset.py:183-189`：1/21/22/23 → ±1，40/42/43 → 0，20/24 → 无 Value）。
- **MCTS 终局价值符号与反向传播视角一致**（`mcts.py:50-61, 212-218`），P0 第 1 项修复真实有效。
- **架构无循环依赖**（AST 全量 DFS 核验）；核心模块 fan-in 健康：`state.py` 43、`config.py` 37、`rules.py` 28。
- **`eval_gate.run_gate` 统计口径合格**：配对同牌、先后手各半、Wilson 区间、三元 SPRT、终局原因拆分、裁决判分。问题只在于它没被训练循环用上。

---

## 5. 附：静态度量基线（供后续回归对比）

| 指标 | 值 |
|---|---|
| `junqi/` 模块数 | 54 |
| 内部循环依赖 | 0 |
| 死模块（fan-in = 0 且非入口） | `junqi.mini_agents`、`junqi.expert.__init__` |
| 超长函数（> 120 行） | 21 |
| `except Exception` | 23 |
| `except …: pass` | 8 |
| TODO/FIXME/HACK | 20 |
| 测试文件 / test_ 函数 | 39 / 297 |
| `models/` 磁盘占用 | 8.06 GB |
| `venv/` 磁盘占用 | 4.69 GB |
