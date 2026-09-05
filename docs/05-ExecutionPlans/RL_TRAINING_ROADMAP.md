# 军棋翻棋深度强化学习（AlphaZero / MCTS）提升方案与演进路线图

> 本文档系统汇总了针对军棋翻棋 AI 引擎的深度强化学习自训练方案，包括已完成的核心架构建设与中远期演进规划。
>
> ⚠️ **注意**：本文档的阶段二/三规划已被 [`RL_TRAINING_PLAN_V2.md`](RL_TRAINING_PLAN_V2.md)（修订版实现规格书）取代。
> V2 修正了 MCTS 叶子编码与采样世界脱节的一级缺陷、引入了三阶段课程学习与死区静态分析器，
> 并明确禁止吃子类奖励塑形。**后续实现请以 V2 为准**，本文档仅作为 Phase 1 已完成工作的历史存档。

---

## 一、 背景与架构演进

| 维度 | 原有架构（旧版本） | 深度强化学习架构（当前版本） |
|---|---|---|
| **决策算法** | PIMC（完全信息蒙特卡洛采样 + 浅层 Alpha-Beta） | **AlphaZero 范式（双头 Policy-Value ResNet + PUCT MCTS）** |
| **博弈缺陷** | 存在严重的“策略融合（Strategy Fusion）”与全知假象 | **基于后验概率的确定化采样 + 神经网络先验引导** |
| **特征容量** | 十几个手工标量权重，无法捕捉空间拓扑与战术动态 | **31 通道立体空间张量**，全盘感知铁路、行营、威胁分布与暗子信念 |
| **自训练机制** | 随机局部扰动爬山（Hill Climbing），在和棋下高方差震荡 | **经验回放（Replay Buffer）+ Policy 交叉熵与 Value 均方差 GPU 梯度下降** |
| **硬件利用** | 纯 CPU 单核解释执行 | **NVIDIA GeForce RTX 4080 SUPER (16GB) CUDA 加速** |

---

## 二、 已完成内容（Phase 1：AlphaZero 闭环建设）

```
                               ┌────────────────────────────────────────────────────────┐
                               │                 JunqiNet 双头残差网络                   │
                               │  输入: 31通道状态张量 [B, 31, 12, 5]                    │
                               │  ├─ Policy Head: 3650 维合法走法概率分布 P(a|s)         │
                               │  └─ Value Head: 局面胜率估值 V(s) ∈ [-1, +1]           │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │ 引导搜索展开与叶子估值
                                                          ▼
┌───────────────────────────────┐              ┌────────────────────────────────────────┐
│   junqi.train_rl 自训练流水线  │              │        junqi.mcts PUCT 树搜索引擎      │
│  1. 多并发自对弈对局生成       │  自对弈数据   │  1. 结合 GameState 采样隐藏世界        │
│  2. 经验回放 (Replay Buffer)   │<─────────────│  2. Dirichlet 噪声注入增强探索         │
│  3. GPU 梯度优化 (Loss 反传)   │              │  3. 输出动作访问概率分布作为训练目标 π   │
│  4. 自动保存 Checkpoints       │              └────────────────────────────────────────┘
└───────────────────────────────┘
```

### 1. 运行环境与硬件适配
- [x] 构建了基于 Python 3.11 的独立虚拟环境 `venv`。
- [x] 安装了支持 CUDA 12.1 加速的 `PyTorch` (2.5.1+cu121) 与 `NumPy`。
- [x] 成功识别并启用了本地 **RTX 4080 SUPER 16GB** GPU。

### 2. 状态空间张量化与动作编码 (`junqi/encoder.py`)
- [x] **31 通道立体特征张量**（Shape: `[31, 12, 5]`）：
  - 通道 0~11: 己方 12 类军衔明子位置分布（One-Hot）。
  - 通道 12~23: 敌方 12 类军衔明子位置分布（One-Hot）。
  - 通道 24: 暗子位置掩码。
  - 通道 25: 暗子中敌方高危子（司/军/师/炸/雷）的 `marginal()` 边缘后验概率热力图。
  - 通道 26: 暗子中工兵（己方/敌方）的后验概率热力图。
  - 通道 27: 铁路网络拓扑通道。
  - 通道 28: 行营（1.0）与大本营（0.5）掩码通道。
  - 通道 29: 双方工兵与地雷存活全局状态矩阵。
  - 通道 30: 连续无吃子步数进度（`quiet / 70.0`）。
- [x] **3650 维动作空间映射**：
  - 翻暗子动作：50 维离散索引（$0 \sim 49$）。
  - 走子动作：$60 \times 60 = 3600$ 维起点到终点索引（$50 \sim 3649$）。
  - 合法动作布尔掩码生成器（`legal_action_mask`）。

### 3. 双头残差神经网络 (`junqi/net.py`)
- [x] **网络主干**：6 个残差块（Residual Block），128 个特征通道。
- [x] **策略头 (Policy Head)**：$1\times 1$ 卷积 + 全连接层，输出 3650 维 Logits，结合 Mask 计算合法走法概率。
- [x] **价值头 (Value Head)**：$1\times 1$ 卷积 + 全连接层 + Tanh，输出局面估值 $V(s) \in [-1, 1]$。
- [x] 模型序列化与加载接口（支持 `weights_only=True` 安全加载）。

### 4. 神经网络引导的 MCTS (`junqi/mcts.py`)
- [x] 基于 **PUCT（Predictor + UCB）** 算法的蒙特卡洛树搜索。
- [x] 支持在每次模拟中根据后验分布确定化采样暗子（解决不完全信息下的策略引导）。
- [x] 支持自对弈探索的 **Dirichlet 噪声注入**（$\alpha=0.3, \epsilon=0.25$）。
- [x] 支持根据温度参数（Temperature $\tau$）输出动作分布 $\pi$。

### 5. 自对弈强化学习训练流水线 (`junqi/train_rl.py`)
- [x] 自对弈数据采样（对局回放数据构建）。
- [x] 经验回放池（Replay Buffer，容量 30,000+）。
- [x] 联合损失函数优化：$\mathcal{L} = (z - v)^2 - \boldsymbol{\pi}^\top \log \boldsymbol{p} + c \|\theta\|^2$。
- [x] 命令行支持：`python -m junqi train_rl --epochs 10 --games 50 --sims 60`。

### 6. 系统集成与自动化测试
- [x] 在 `junqi/ai.py` 中新增 `NNAgent`。
- [x] 在 `junqi/selfplay.py` 中支持 `nn` 与 `nn_mcts` 策略及模型对战评测。
- [x] 编写并全量通过了 38 项单元测试（[`tests/test_rl.py`](file:///e:/Local%20code/军棋/junqi_engine/tests/test_rl.py) + [`tests/test_rules.py`](file:///e:/Local%20code/军棋/junqi_engine/tests/test_rules.py)）。

---

## 三、 待完成与后续演进计划

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             演进路线图 (Roadmap)                          │
├────────────────────────────────┬─────────────────────────────────────────┤
│ Phase 2: 吞吐加速与模型对抗     │ Phase 3: 深度博弈与产品化部署            │
│ ├─ Batched 并行 MCTS (GPU 满载) │ ├─ 残局课程学习 (Curriculum Learning)   │
│ ├─ 历史模型池 (League Training) │ ├─ CFR / 信息集子博弈精细化求解          │
│ └─ 内在奖励塑形 (挖雷/控线)      │ └─ ONNX / TensorRT 导出与 GUI 档位集成   │
└────────────────────────────────┴─────────────────────────────────────────┘
```

### 阶段二：吞吐加速与对抗机制优化（近期计划）

1. **异步批处理 MCTS（Batched GPU MCTS）**
   - **目标**：充分压榨 RTX 4080 SUPER 的 16GB 显存与 Tensor Core 算力。
   - **方案**：将多个并发对局（例如 16~32 个并行 Worker）的叶子节点评估聚合成一个 Batch 送入 GPU 推理，消除 Python 单步调用的等待开销，预计自对弈速率提升 **5~10 倍**。
2. **历史模型池与对抗评级机制（League Training & Elo Rating）**
   - **目标**：防止自对弈陷入策略循环克制（Rock-Paper-Scissors cycle）和灾难性遗忘。
   - **方案**：
     - 维护历史 Snapshot 模型池。
     - 采样时 80% 与当前最新模型对弈，20% 与历史高分模型对弈。
     - 自动跟踪记录模型迭代的 Elo Rating 积分曲线。
3. **奖励塑形与和棋优化（Reward Shaping）**
   - **目标**：解决军棋翻棋 70 步无吃子判和带来的稀疏奖励问题。
   - **方案**：
     - **挖雷阶段性奖励**：工兵每挖掉一颗地雷给予 $+0.1$ 阶段性正反馈。
     - **子力差势能引导**：在和棋终局中，按双方歼敌子力差线性折算微量收益，引导 AI 主动进攻而非消极躲避。

---

### 阶段三：深度博弈与产品化落地（中远期计划）

1. **残局课程学习（Curriculum Learning）**
   - **目标**：快速掌握残局杀王、行营控盘与暗子排查定式。
   - **方案**：
     - **阶段 1**：生成 10~15 子中后盘残局（双方工兵、司令、地雷残局）进行强化自对弈。
     - **阶段 2**：逐步增加棋子数直至 50 子全盘。
2. **CFR / 不完全信息子博弈分析（Subgame Solving）**
   - **目标**：在关键暗子残局中计算纳什均衡策略（如暗子真伪试探、空城计与护旗博弈）。
3. **ONNX / TensorRT 高性能导出与部署**
   - **目标**：将训练好的 PyTorch 模型导出为静态图，实现 $<1\text{ ms}$ 的推理延迟。
4. **GUI 人机对战无缝接入**
   - **目标**：在图形界面（`python -m junqi gui`）中增加“深度学习 AI（快手 / 大师）”档位，供用户直接对战。

---

## 四、 快速使用与实战命令指南

### 1. 启动自强化训练
```bash
# 在 RTX 4080 SUPER 上启动多进程并发强化自训练（8 Worker 并发，混入 30% 残局课程）
.\venv\Scripts\python.exe -m junqi train_rl --epochs 10 --games 80 --sims 60 --eval-games 40 --workers 8 --curriculum-prob 0.3 --batch-size 256 --out-dir models
```

### 2. 评测自训练 AI 与传统搜索 AI
```bash
# 评测训练好的深度学习 AI 与传统搜索 (search2) 的实战胜率
.\venv\Scripts\python.exe -m junqi selfplay --a nn_mcts_60 --b search2 --games 50 --model-a models/best.pt

# 在残局评测集上评测 AI 的死区拖和与终局实力
.\venv\Scripts\python.exe -m junqi selfplay --a nn_mcts_60 --b search2 --games 50 --eval-set eval_sets/endgame.jsonl --model-a models/best.pt

# 评测纯神经网络直接走子（毫秒级极速响应）
.\venv\Scripts\python.exe -m junqi selfplay --a nn --b greedy --games 50 --model-a models/best.pt
```

### 3. 运行完整单元测试套件
```bash
.\venv\Scripts\python.exe -m unittest discover tests -v
```
