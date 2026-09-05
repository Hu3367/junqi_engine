# 军棋翻棋深度强化学习训练计划 V2（实现规格书）

> **文档定位**：本文档是决策完备（decision-complete）的实现规格书，供实现者（人或 AI 编码助手）
> 直接按步实施，无需再做架构决策。它取代 `RL_TRAINING_ROADMAP.md` 中阶段二/三的相关条目。
>
> **代码基线**：`junqi_engine/junqi/`（encoder.py / mcts.py / net.py / train_rl.py / state.py / rules.py / config.py），
> 38 项单元测试全过状态。规则默认值以 `config.RuleConfig`（APK 逆向对齐）为准，**不得修改规则**。

---

## 一、修订背景：相对 V1 路线图的三个关键认知修正

### 1.1 规则认知修正（来自人类棋手输入）

- 翻子**无限制**：双方任意回合可翻任意暗子，翻出的子可能属于对方；仅明子可移动。
- **信息推理量级很小**：暗子池构成 = 总构成 − 明子 − 死子，`state.remaining_types()` 已精确可算。
  难点不在贝叶斯推理，而在**长程结构性决策**（死区构筑、行营经营、和棋线管理）。
- 对局天然分为三个战术阶段（详见 §3）：
  1. **开局翻棋**：翻子位置选择决定后期胜率；
  2. **中期占营/绞杀**：以行营为势力范围，翻大子附近暗子，己方扩营、敌方绞杀；
  3. **尾盘**：按精确子力对比分流——劣势方布置**死区**拖和（如用对方明地雷+己方子围住己方雷，
     对方永远无法破除自己的雷），优势方防死区并优先绞杀剩余暗子（暗子行动需两回合：先翻后走，
     高手利用此时差布局）。

### 1.2 实现缺陷清单（按修复优先级）

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| D1 | **MCTS 叶子编码与采样世界脱节**：每次模拟采样确定化世界，但叶子送网编码中暗子仍是 `revealed=False`，采样到的暗子身份对网络完全不可见；同一路径各世界的叶子张量几乎相同 | `mcts.py` 模拟循环 + `encoder.encode_state_np` | 价值网络在搜索内对暗子世界失明，网络引导近似失效（一级缺陷，不修复则算力全部浪费） |
| D2 | 死子信息缺失：通道 29 只有工兵/地雷计数，"对手已死哪些大子"完全丢失 | `encoder.py` | 子力判断（尾盘核心）无法学习 |
| D3 | 通道 25/26 是全局标量平铺，均匀洗牌下信息量≈0 | `encoder.py` | 占用通道容量 |
| D4 | 无评估门控：每 epoch 无条件覆盖 `best.pt` | `train_rl.py` | 训练退化无法察觉 |
| D5 | 种子 `int(time.time()*1000)%1000000+g` 不可复现 | `train_rl.py` | 实验不可对比 |
| D6 | Dirichlet 噪声含死代码 `noise = rng.gauss(0,1)`，且 α=0.3 平摊到过百动作上探索过弱 | `mcts.py` | 探索不足 |
| D7 | batch=1 单状态推理 + Python 串行模拟，吞吐极低 | `mcts.py` / `net.py` | 自对弈规模上不去 |

### 1.3 明确禁止项（来自阶段战术分析）

- **禁止任何吃子/挖雷类奖励塑形**（如"挖雷 +0.1"）：尾盘劣势方的正确策略恰恰是**停止吃子**、
  让 70 步无吃子计数器跑满判和并构筑死区。任何奖励吃子的信号都会系统性破坏拖和策略。
  终局目标 z 必须保持纯结果值（赢 +1 / 输 −1 / 和 0）。
- **禁止为三个阶段训练三个独立模型**：单一网络 + 阶段特征通道 + 课程采样配比即可，
  分模型会破坏阶段过渡（何时从争营转入筑垒）的连续价值判断。

---

## 二、总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  静态分析层（新增，手工可算信号直接注入，不让网络自己"悟"）           │
│  analysis.py:  detect_phase() / material_diff() / fortress_score()  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  编码层  encoder.py:  encode_state_np(state, seat, world=None)      │
│  36 通道 [36,12,5]，双模式：公共模式(玩家视角) / 世界模式(采样世界)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  JunqiNet (net.py): 36 通道输入，其余结构不变 (6 ResBlock × 128ch)   │
│  Policy 头学公共模式 π；Value 头学世界模式 V                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  mcts.py: 修复 D1——世界沿搜索路径传播，叶子以世界模式编码送网；       │
│  每次模拟在根重新采样世界（保留），搜索统计跨世界共享（ISMCTS 语义）   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  train_rl.py: 双流样本（policy=公共模式/根, value=世界模式/叶子）+    │
│  阶段标签 + 评估门控 + 历史模型池 + 课程采样                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、阶段判定（新增 `junqi/analysis.py`）

```python
# junqi/analysis.py —— 全部为纯函数，只读 GameState，不修改任何现有文件语义

PHASE_OPENING, PHASE_MIDGAME, PHASE_ENDGAME = 0, 1, 2

def detect_phase(state: GameState) -> int:
    """阶段判定（阈值集中在此，允许后续微调）：
    - 暗子 >= 20：开局翻棋期
    - 6 <= 暗子 < 20 且行营未全部被占：中期占营期
    - 其余（暗子 < 6，或 10 个行营全部有子）：尾盘期
    """

def hidden_count(state: GameState) -> int:
    return len(state.hidden_positions())

def camps_occupied(state: GameState) -> int:
    """CAMPS 中被棋子占据的格数（0~10）。"""
```

注意：阶段是**局面属性**，允许随对局回退（行营腾出后重新计入争营），不记录历史。

---

## 四、静态分析器（`junqi/analysis.py` 续）

### 4.1 子力差 `material_diff`

```python
def material_diff(state: GameState, seat: int) -> float:
    """己方存活子力 − 对方存活子力，归一化到 [-1, 1]。
    存活 = 棋盘明子 + 暗子池期望（按 remaining_types() 精确份额）。
    子力价值表沿用 config.EvalWeights().piece（司令100 ... 军旗50）。
    归一化常数 MATERIAL_MAX = 1120（单方全子力约 1119）。
    实现要点：暗子池中每类 (color, rank) 的数量 × 价值，按颜色分摊给双方；
    明子直接按位置归属。座次视角：返回 seat 方减对方。
    """
```

### 4.2 死区/堡垒分析 `fortress_score`（v1 启发式，保持简单）

```python
def fortress_score(state: GameState, seat: int) -> float:
    """评估 seat 方以军旗为核心的死区完备度，返回 [0,1]。

    算法（BFS 可达性）：
    1. 若 seat 方军旗未翻开（场上无 revealed 己方 QI）→ 返回 0.0
       （暗子身份连自己都不知道，筑垒无从谈起）。
    2. 定义"永久墙"（对方永远无法通过/拔除的格子）：
       a. 对方已翻开的明地雷——对方不能吃自己的子，非工兵攻雷攻方阵亡、
          雷存活，故对方明雷对对方是永久墙；
       b. 己方已翻开的地雷，且对方工兵已全灭（含暗子池：
          remaining_types() 中对方工兵数为 0 才算全灭，保守处理）。
    3. 从所有"对方明子占据格"出发，沿 NEIGHBORS 公路邻接（含行营斜道）做 BFS，
       只能通过空格；任何棋子占据格（含永久墙）都阻挡通行。
       （v1 不建模铁路长距离滑行，保持简单；铁路可达性作为 v2 改进项。）
    4. 若军旗格不可达 → 返回 1.0（死区已成）。
       否则返回软分：0.3 × (军旗四邻中永久墙格数 / 军旗四邻总数)。
    """
```

测试要求（`tests/test_rl.py` 新增）：构造 2 个手工局面——①己方旗被对方明雷+己方明子
完全围死且对方无工兵 → 分数 ≥ 0.9；②旗完全暴露 → 分数 ≤ 0.3。断言单调性即可。

---

## 五、特征编码规格（重写 `encoder.py` 编码部分）

### 5.1 新函数签名

```python
NUM_CHANNELS = 36   # 原 31 → 36

def encode_state_np(state: GameState, seat: int | None = None,
                    world: dict | None = None) -> np.ndarray:
    """world 为 sample_world() 返回的 {pos -> Piece} 暗子身份指派。
    - world is None  → 公共模式（玩家真实视角，暗子只见掩码）
    - world 给定     → 世界模式（暗子按 world 身份写入明子 one-hot 通道）
    两个模式的通道布局完全一致，仅内容不同，另有模式标志通道区分。
    """
```

### 5.2 通道布局（36 通道）

| 通道 | 内容 | 备注 |
|---|---|---|
| 0~11 | 己方 12 军衔 one-hot | 世界模式下：暗子按 `world` 身份并入（视为已知明子） |
| 12~23 | 敌方 12 军衔 one-hot | 同上 |
| 24 | 暗子掩码 | 世界模式下该通道为全 0 |
| 25 | 子力差平面 | 广播 `material_diff(state, seat)`，全局同值 |
| 26 | 己方死区潜力 | 广播 `fortress_score(state, seat)` |
| 27 | 对方死区潜力 | 广播 `fortress_score(state, other_seat)` |
| 28 | 铁路网 | 原通道 27 |
| 29 | 行营(1.0)/大本营(0.5) | 原通道 28 |
| 30 | 工兵/地雷存活率 | 原通道 29，保留 |
| 31 | 无吃子倒计时 `quiet/70` | 原通道 30，保留（拖和策略的关键输入） |
| 32 | 总步数进度 `ply/1000` | 新增（与 31 解耦） |
| 33 | 暗子数量 `hidden_count/50` | 新增（显式阶段信号） |
| 34 | 阶段标号 `phase/2` | 新增 |
| 35 | 模式标志 | 世界模式=1.0，公共模式=0.0，全平面广播 |

**删除**原通道 25/26（全局后验标量热力图，信息量≈0，见缺陷 D3）。
首翻定色前（`my_color is None`）沿用现有绝对颜色编码分支，逻辑不变。

`legal_action_mask` 不变。`encode_state`（torch 版）同步增加 `world` 参数透传。

---

## 六、MCTS 修复规格（缺陷 D1，最高优先级）

文件：`junqi/mcts.py`。核心原则：**同一模拟内，世界沿搜索路径传播；叶子以世界模式编码。**

### 6.1 模拟循环重构

```python
for _ in range(self.simulations):
    # 根节点采样世界（保留现状），但世界要随路径传播
    world = root_state.sample_world(rng, reveal=False)
    sim_root = root_state.instantiate(world)
    ...
    # Select 阶段：每走一步，若当前节点对应局面仍有暗子且世界未覆盖新翻开的格，
    # 无需重采样——world 在 instantiate 后已固定全部暗子身份；
    # apply(flip) 翻开的就是 world 中的真实身份（instantiate 已写入）。
    ...
    # Expand & Evaluate：
    policy_map, value = self.net.predict_state(
        state, seat=state.turn, world=world, device=self.device)   # 世界模式
    # 收集叶子价值样本（见 §7.2）
    leaf_samples.append((encode_state_np(state, seat=state.turn, world=world),
                         state.turn))
```

关键验证点：`GameState.instantiate(world)` 已把身份写入暗子但 `revealed=False`；
`apply(Action("flip", pos))` 会 `replace(pc, revealed=True)`，翻出的即世界身份。
因此**不需要沿路径重新采样世界**，禁止保留旧代码中"每次模拟重新 instantiate 后
树节点仍指向旧 state"的混用写法——子节点展开时的 `state.legal_actions()` 必须在
当前世界的 instantiated state 上调用。

### 6.2 根节点先验

根节点展开时对 **K=4 个预采样世界**分别做世界模式前向，策略先验取平均：

```python
worlds = [root_state.sample_world(rng, reveal=False) for _ in range(4)]
policies = [self.net.predict_state(root_state, world=w) for w in worlds]
prior[a] = mean(policies[k][a] for k in range(4))
```

价值先验不参与根节点，只做探索偏置用。模拟循环中每个世界从 `worlds` 中随机选取
（允许重复，不足时补采），减少每模拟的采样开销。

### 6.3 其他修复

- 删除死代码 `noise = rng.gauss(0, 1)`（D6）；Dirichlet 改为
  `alpha = 10.0 / len(root.children)`，混合后对先验重新归一化。
- `predict_state` 增加 `seat`、`world` 参数，内部改用新 `encode_state`。
- 反向传播（backprop）的视角符号逻辑保持现状（已正确：叶子价值 = 叶子轮次方视角）。

---

## 七、训练流水线规格（重写 `train_rl.py` 主循环）

### 7.1 双流样本

| 流 | 张量模式 | 来源 | 目标 |
|---|---|---|---|
| Policy 样本 | 公共模式（`world=None`） | 自对弈每手根局面 | `π` = MCTS 访问分布，交叉熵 |
| Value 样本 | 世界模式 | MCTS 叶子（每手上限随机抽 8 个叶子） | `z` = 终局结果，按叶子轮次方视角 ±1/0，MSE |

`play_selfplay_game` 返回值扩展为 `(policy_samples, value_samples)`，每个样本附
`phase` 标签（`analysis.detect_phase`）。`MCTS.search` 增加返回叶子记录列表。

### 7.2 回放池与采样

- 容量 200,000（原 30,000 过小）；两条流各一个 `deque`。
- 采样按阶段加权：Policy 流按阶段权重 {开局 0.2, 中期 0.5, 尾盘 0.3} 抽取；
  Value 流均匀抽取（叶子本身已遍布各深度）。实现：按 `phase` 维护三个索引桶。

### 7.3 温度策略（按阶段，替代按 ply）

```python
TEMP_BY_PHASE = {PHASE_OPENING: 1.0, PHASE_MIDGAME: 0.6, PHASE_ENDGAME: 0.2}
```

### 7.4 评估门控（修复 D4，强制实现）

每个 epoch 结束后：

1. 新模型与 `best.pt` 对战 100 局，`sims=200`、温度 0（贪心）、双方先后手各半；
2. 得分 = 胜 1 / 和 0.5 / 负 0；**得分 ≥ 0.55 才晋升**为新 `best.pt`，否则回滚；
3. 记录每模型 Elo（简化版：以 400 为尺度、K=16 的增量更新），写入
   `out_dir/elo_history.jsonl`。

### 7.5 历史模型池（轻量 League）

- 维护最近 8 个已晋升的 snapshot（`out_dir/pool/step_{n}.pt`）；
- 每局自对弈以 80% 概率双方都用当前模型、20% 概率一方随机抽取池内模型；
- 目的：防止策略循环克制与灾难性遗忘。

### 7.6 可复现种子（修复 D5）

```python
# 命令行新增 --seed（默认 42）
game_seed = seed * 1_000_000 + epoch * 10_000 + game_idx
```

### 7.7 损失与优化器

保持 `L = CE(π, p) + MSE(z, v)`，无权重衰减改动；**不加入任何中间奖励项**。
学习率 1e-3 AdamW 保留，但晋升回滚时不回滚优化器状态。

---

## 八、课程学习与残局生成器

### 8.1 残局生成器（新增 `junqi/endgame_gen.py`）

```python
def gen_endgame(rng, cfg, *,
                material_balance: float = 0.0,   # [-1,1]，正=seat0 优势
                hidden_k: int = 4,               # 保留暗子数
                my_engineers: int = 1, opp_engineers: int = 0,
                fortress: bool = False) -> GameState:
    """生成 10~15 子尾盘局面：
    - seat0 军旗翻开，置于下半场，周围堆 2~3 颗己方明地雷；
    - fortress=True 时，用地雷+对方明雷把旗围成死区结构（供价值网络学习 1.0 目标）；
    - 行营预先被双方大子占据；
    - 剩余子力按 material_balance 配置（总子数 10~15）；
    - hidden_k 个子保持暗置（身份正常分配，供翻子训练）。
    返回合法 GameState（seat_color 已定、turn=0、quiet 可随机 0~40）。
    """
```

### 8.2 评测集（固定化，随仓库提交）

用生成器 + `deal()` 截断生成三套评测集，`state.to_json()` 存入：

| 文件 | 内容 | 数量 |
|---|---|---|
| `junqi_engine/eval_sets/opening.jsonl` | 首翻后 ≤5 手的开局局面 | 200 |
| `junqi_engine/eval_sets/midgame.jsonl` | 暗子 6~20 的中盘局面 | 200 |
| `junqi_engine/eval_sets/endgame.jsonl` | `gen_endgame` 尾盘局面（含堡垒/非堡垒各半） | 200 |

### 8.3 分阶段评测命令（扩展 `selfplay.py`）

```bash
.\venv\Scripts\python.exe -m junqi selfplay --a nn_mcts_200 --b search2 \
    --eval-set eval_sets/endgame.jsonl --model-a models/best.pt
```

`--eval-set` 从 jsonl 加载起始局面逐局评测，输出每套集合的胜率/和率。
**训练监控只看三套评测集分数 + Elo，不看训练 loss。**

### 8.4 课程调度

- **P1 阶段**（前 2 万局）：自对弈正常全局进行，但尾盘评测集分数若 < 40%，
  将 30% 的自对弈替换为"从 `gen_endgame` 局面续弈"的半局训练；
- 中期/开局不做人工干预（数据天然充足）。

---

## 九、吞吐工程（第二批实现，阻塞项修复后）

1. **叶子批量推理**：模拟循环中把同一次迭代内待展开叶子聚合成 batch 一次前向；
   `JunqiNet.forward` 已支持批量，只需改调用侧。
2. **多进程自对弈**：`train_rl` 用 `multiprocessing`（Windows 用 spawn）起 8 个
   worker 进程，各自在 CPU 上做走法生成、叶子请求通过队列聚合到主进程 GPU。
3. 目标吞吐：≥ 500 局/小时（当前约几十局/小时）。走法生成
   （`legal_actions` 铁路 BFS）是下一个瓶颈，必要时将 `state.apply/legal_actions`
   热路径用 `__slots__`+局部变量优化，不做重写。

---

## 十、实现步骤（按序执行，每步有验收）

| 步骤 | 内容 | 改动文件 | 验收标准 |
|---|---|---|---|
| **P0-1** | 新增 `analysis.py`：`detect_phase / material_diff / fortress_score / camps_occupied / hidden_count` | `junqi/analysis.py`（新建） | 新增单测：阶段判定边界、子力差对称性 `f(st,0) == -f(st,1)`、堡垒单调性（§4.2 两用例） |
| **P0-2** | 重写编码：36 通道、`world` 参数、双模式；同步 `net.py` `in_channels=36` | `encoder.py`、`net.py` | 形状断言 `(36,12,5)`；世界模式/公共模式在 `world=真实暗子身份` 时明子通道一致；旧 38 测试中涉及 31 通道的断言同步更新 |
| **P0-3** | MCTS 修复（§6 全部）：世界传播、叶子世界模式编码、根先验 4 世界平均、Dirichlet 修正 | `mcts.py`、`net.py`（`predict_state` 签名） | 一致性测试：`search` 内部叶子编码 == `encode_state_np(state, world=w)`；10 局 smoke 自对弈无异常 |
| **P0-4** | 训练流水线（§7 全部）：双流样本、阶段桶采样、门控、模型池、种子、温度表 | `train_rl.py` | 门控回滚路径有单测（伪造低分场景）；跑 `--epochs 2 --games 10` 端到端无错 |
| **P1-1** | `endgame_gen.py` + 三套评测集生成脚本 + `selfplay --eval-set` | `endgame_gen.py`（新建）、`selfplay.py` | 评测集文件入库；`--eval-set` 三套集合各跑 20 局抽测 |
| **P1-2** | 首轮正式训练：2 万局（约 1~2 天连续），观察三套评测集曲线 | — | 尾盘评测集胜率 ≥ 50%（对 search2）；Elo 单调不降 |
| **P2-1** | 叶子批量推理 + 多进程自对弈（§9） | `mcts.py`、`train_rl.py` | 吞吐实测 ≥ 500 局/小时；结果与串行版对拍（同种子 5 局动作序列一致） |
| **P2-2** | 规模化训练：10 万局 + 课程调度（§8.4） | — | 三套评测集全面超越 search2；对战人类规则基线（greedy）≥ 85% |

每步完成后运行：`.\venv\Scripts\python.exe -m unittest discover tests -v`，全过方可进入下一步。

---

## 十一、算力与预期（RTX 4080 SUPER 16GB）

| 里程碑 | 数据量 | 预计耗时 | 预期水平 |
|---|---|---|---|
| P0 修复完成 | — | 实现 2~3 天 | 训练信号恢复有效（先决条件） |
| P1-2 | 2 万局 | 单卡 1~2 天 | 三阶段均胜 search2 |
| P2-2 | 10 万局 | 单卡 3~5 天 | 稳定击败 95% 休闲/熟练玩家 |
| 后续（可选） | 50 万局+ | 单卡 2~3 周 | 强业余水平；对顶尖高手仍受限于死区博弈深度 |

网络容量（6 ResBlock × 128 通道）对本问题偏富余，无需扩容；显存占用 < 2GB，
剩余显存全部留给批量推理。

---

## 十二、验收总标准

1. `python -m unittest discover tests -v` 全过（含新增测试）；
2. 三套分阶段评测集对 `search2` 胜率均 ≥ 60%；
3. 评估门控生效：`elo_history.jsonl` 无回退式覆盖记录；
4. 同种子重跑 5 局动作序列完全一致（可复现性）；
5. 价值头在尾盘堡垒局面（`fortress_score=1.0` 的生成局面）上输出 |V| < 0.3
   （学会了"死区 ≈ 和棋"，而不是判负）——这是本方案成败的最直接信号。
