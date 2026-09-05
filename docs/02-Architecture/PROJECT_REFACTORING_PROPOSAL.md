# 军棋翻棋引擎 - 全面重构建议方案

> **创建日期**: 2026-09-05  
> **当前状态**: 功能完备但存在架构债务与算法瓶颈  
> **重构目标**: 提升可维护性、建立科学评估体系、加速训练迭代  
> **适用版本**: junqi_engine v1.x → v2.0

---

## 📋 执行摘要

经过深度代码审计与算法分析，本项目存在 **5 大核心痛点**：

| # | 问题描述 | 影响范围 | 优先级 | 解决路径 |
|---|----------|---------|--------|---------|
| 1 | **棋力差** - AI 未掌握基本棋理 | P3 训练失效 | 🔴 P0 | 系统化规则知识文档 + 人工审核 |
| 2 | **评估浅** - ELO 评分系统不完善 | 模型晋级无法衡量 | 🔴 P0 | 多维等级分框架 + 循环赛平台 |
| 3 | **黑盒化** - 训练过程不可解释 | 无法调试优化 | 🟠 P1 | MCTS 搜索可视化 + 决策日志 |
| 4 | **无版本管理** - 模型迭代混乱 | 实验无法追溯 | 🟠 P1 | 轻量级模型管理平台 |
| 5 | **验证慢** - 12×5棋盘训练周期长 | 探索成本高 | 🟢 P2 | 小棋盘课程学习 (6×6→12×5) |

**推荐实施顺序**: 先解决 P0（规则知识 + 评估体系），再推进 P1（可视化管理），最后考虑 P2（小棋盘加速）。

---

## 一、问题 1: 棋力差 - 未掌握基本棋理

### 现状诊断

**根本原因**: 
- ❌ 现有 `AI_TRAINING_AND_HUMAN_PLAY_PLAN.md` 仅定义了战略原则，缺少**具体的战术模式库**
- ❌ `junqi/analysis.py` 中的阶段识别逻辑过于简单（仅依赖暗子数量）
- ❌ Value Head 网络缺乏明确的战术特征监督信号

**证据**:
```python
# junqi/analysis.py lines 36-48
def detect_phase(state: GameState) -> int:
    h = hidden_count(state)
    c = camps_occupied(state)
    if h >= 20: return PHASE_OPENING   # 仅靠暗子数量判断
    if 6 <= h < 20 and c < 10: return PHASE_MIDGAME
    return PHASE_ENDGAME
```

这个判定逻辑忽略了：
- 行营控制度（中央行营 vs 边角行营价值差异巨大）
- 子力对比（多司令的一方即使暗子多也应是优势）
- 铁路控制权（是否打通关键运输线）

### 解决方案：构建《军棋规则与棋理知识手册》

**已完成的产出**: 
我刚刚生成了 **《军棋翻棋玩法规则与策略完整汇总文档》**（见 [docs/JUNQI_RULES_AND_STRATEGY_GUIDE.md](docs/JUNQI_RULES_AND_STRATEGY_GUIDE.md)），内容涵盖：

#### 核心章节目录
1. **基础规则定义** - 棋盘几何、棋子等级、战斗结算、APK 对齐开关
2. **开局阶段策略** - 翻棋节奏、行营争夺、前线通道控制、经典定式
3. **中盘阶段策略** - 子力交换决策树、攻防选择、炸弹使用时机、工兵挖雷
4. **尾盘阶段策略** - 死区判定、雷阵突破、行营封锁、长距离调度、逼和技巧
5. **关键知识点** - 势力范围计算、两回合时差利用、循环局面规避、子力动态价值
6. **代码证据索引** - 所有规则在源码中的位置

**下一步行动**:
```markdown
✅ [已完成] 自动生成完整规则文档  
⏳ [待人工审核] 组织 2-3 位军旗高手 review 文档准确性  
⏳ [需补充] 从 1000 局复盘中提取统计规律（开局频率、各阶段平均步数等）  
⏳ [需实现] 将规则知识注入训练数据（如标注"这是正确的翻棋位置"）
```

**预期效果**:
- **短期**: 人工审核后的知识手册可作为新开发者的入职培训材料
- **中期**: 基于知识库构造合成数据集，强化 Policy Head 学习正确模式
- **长期**: 引入符号规则约束（Rule-guided RL），避免网络学习非法战术

---

## 二、问题 2: 评估浅 - 等级分制度缺失

### 现状分析

**当前实现**:
- `models/elo_history.jsonl` 记录 ELO 变化，但恒为 1500（仅在晋升时更新）
- `junqi/selfplay.py` 有基本的胜负统计，但未使用 Wilson 区间门控
- `train_rl.py` 的 `decide_promotion()` 完全依赖胜率阈值（0.75），导致和棋率>90% 时死锁

**数学缺陷**:
```python
# train_rl.py lines 823-847
def decide_promotion(self, wins, draws, losses):
    total = wins + draws + losses
    win_rate = wins / total
    if win_rate > self.cfg.promote_threshold:  # 默认 0.75
        return True
    return False
```

**问题**:
1. **未考虑样本量**: 1 胜 0 负 (100%) vs 100 胜 25 和 25 负 (50%) 都可能是偶然
2. **未处理高和棋率**: 军棋规则天然容易和棋，强行要求胜率>75% 不现实
3. **未区分对手强度**: 击败 Random 不算本事，需要 Challenge 基线模型

### 设计方案：多维度等级分系统

**已完成的设计文档**: [docs/ELO_RATING_SYSTEM_DESIGN.md](docs/ELO_RATING_SYSTEM_DESIGN.md)

#### 核心特性

##### 1. 混合 ELO-Glicko-2 模型

**ELO Rating** (传统):
```math
R_{new} = R_{old} + K \times (Score - Expected)
```
其中 `Expected = 1 / (1 + 10^((R_opponent - R_self)/400))`

**Glicko-2 RD** (Rating Deviation - 不确定度):
```math\nRD_{new} = sqrt(RD_old^2 + Var_game)
Var_game = (1 / sum(1/(RD_i^2)))^-1
```
**优势**: 新增模型初始 RD 很大，随着参赛次数增加 RD 下降，评分趋于稳定

##### 2. 阶段分离评分

针对军棋三个阶段的特点，分别记录能力指标：

| 阶段 | 权重 | 评估方法 | 达标阈值 |
|------|------|---------|---------|
| Opening (暗子≥20) | 15% | 翻棋安全性 Top-3 命中率 | > 60% |
| Midgame (6≤暗子<20) | 45% | 中盘残局胜率 (vs search2) | > 55% |
| Endgame (暗子<6) | 40% | 死区构筑成功率 + 破阵效率 | Fortress > 0.7 |

**综合评分** = 0.15×Opening_Elo + 0.45×Midgame_Elo + 0.4×Endgame_Elo

##### 3. 高和棋率应对机制

**三档选项**:

| 版本 | 处理方式 | 优点 | 缺点 |
|------|---------|------|------|
| **简易版** | 超基准和棋折 0.3 分<br>(如 1500 分基线，和棋得 0.3 而非 0.5) | 快速上线<br>代码改动<200 行 | 主观性强<br>缺乏理论支持 |
| **标准版** | 重新比赛机制<br>(交换颜色重赛，最多 2 次) | 公平<br>接近职业裁判标准 | 计算成本×2~3 |
| **专业版** | 状态加权评分<br>(考虑子力差、死区完备度) | 最科学<br>认可高质量和棋 | 复杂度高<br>需大量调参 |

**推荐路径**: 从简易版启动，2 周后升级到标准版

##### 4. 对抗平台架构

**Round-Robin 循环赛制**:
```python
class Tournament:
    def __init__(self):
        self.participants = []  # List[ModelSnapshot]
        self.rounds = []        # List[List[Tuple[ModelA, ModelB]]]
    
    def add_model(self, model_path, elo=1500, rd=350):
        self.participants.append(ModelInfo(model_path, elo, rd))
    
    def schedule_round_robin(self):
        """生成单循环赛程"""
        from itertools import combinations
        for a, b in combinations(self.participants, 2):
            self.rounds.append([(a, b)])
    
    def play_match(self, model_a, model_b, games=40):
        """一对多对局，返回 (wins_a, draws, wins_b)"""
        results = parallel_play_games(
            model_a=model_a.path,
            model_b=model_b.path,
            n=games,
            workers=8,
            fixed_seeds=True  # 确保可复现
        )
        return results
```

**Champion-Challenger SPRT 检验**:
```python
class SPRT:
    """Sequential Probability Ratio Test
    持续监控对局结果，一旦统计显著就停止并决定是否晋升
    """
    H0: new_model <= champion (H0: 没有显著提升)
    H1: new_model > champion  (H1: 显著优于)
    alpha = 0.05  # Type I error rate
    beta = 0.10   # Type II error rate
    
    def update(self, result):  # result = 'win'|'draw'|'loss'
        log_likelihood_ratio += likelihood(result)
        if log_likelihood_ratio > B:
            return "ACCEPT_H1", "显著优于冠军，可晋升"
        elif log_likelihood_ratio < A:
            return "ACCEPT_H0", "未达到显著优势"
        else:
            return "CONTINUE", f"还需{remaining_games}局"
```

##### 5. 军棋特色适配

**不完全信息权重因子**:
```python
if revealed_ratio < 0.3:  # 暗子占比>70%
    base_rating *= 0.85  # 降低开局评分权重（运气成分大）
elif revealed_ratio > 0.7:  # 大部分明子
    base_rating *= 1.15  # 提高尾盘评分权重（技术主导）
```

**死区防守质量加分**:
```python
if fortress_completeness > 0.85 and score == 0.5:  # 铁壁 + 和棋
    score *= 1.1  # 和棋视为成功，给予额外认可
```

### 集成方案

**目录结构扩展**:
```
models/
├── checkpoints/          # [新建] 训练快照
│   ├── epoch_001.pt
│   └── latest.pt
├── releases/             # [重写] 发布模型
│   ├── v1.0_best.pt      # 第一版发布
│   └── current_best.pt   # 软链接到最新发布的
├── tournament/           # [新建] 赛事数据
│   ├── season_1/
│   │   ├── round_1.jsonl
│   │   └── standings.csv
│   └── championship/
├── rating/               # [新建] 评级系统
│   ├── player_profile.json
│   └── rd_curves.jsonl
└── validation/           # [新建] 门控测试集
    ├── opening_suite.jsonl  # 50 局开局题
    ├── midgame_suite.jsonl  # 50 局中盘题
    └── endgame_suite.jsonl  # 50 局尾盘题
```

**代码修改清单**:
1. `junqi/train_rl.py`: 替换 `decide_promotion()` 为 SPRT 检验
2. `junqi/selfplay.py`: 增加 `TournamentMatch.play()` 支持并行循环赛
3. `junqi/benchmark.py`: 添加阶段分离评分接口
4. `scripts/run_tournament.py`: 新脚本，运行全自动锦标赛

**实施时间预估**:
- 简易版 (ELO+Wilson): **2-3 周**
- 标准版 (Glicko-2+Rematch): **4-6 周**
- 专业版 (SPRT+自动报告): **8-10 周**

**推荐起点**: 简易版，快速上线验证效果

---

## 三、问题 3: 思考过程不可解释

### 问题本质

当前训练是典型的"黑盒"：
- ✅ 知道输入输出（状态张量 → Policy logits）
- ❌ 不知道中间发生了什么（MCTS 树的构建过程、Value 预测的依据）

**后果**:
- 发现模型退步时无法定位原因
- 无法验证 AI 是否真的学到了战术，还是走捷径
- 难以向人类用户解释"为什么 AI 要这么走"

### 解决方案：MCTS 搜索过程可视化

**设计要点**:

#### 1. MCTS 访问热力图

对于根节点的所有合法动作，展示 MCTS 访问次数分布：

```
当前局面：红方行动 (子力劣势 -0.2)

合法动作及 MCTS 访问统计:
┌─────────────────────────┬──────────────┬─────────────┐
│ 动作描述                │ 访问次数     │ 访问概率    │
├─────────────────────────┼──────────────┼─────────────┤
│ 翻 (5,2) 暗子          │ 1,245        │ 24.9%       │
│ 移 (11,3) 军旗 → (11,1)│ 89           │ 1.8%        │
│ 移 (8,2) 排长 → (7,2)  │ 2,103        │ 42.1%       │ ← 最高频
│ 移 (6,0) 工兵 → (5,0)  │ 987          │ 19.7%       │
│ ...                     │ ...          │ ...         │
└─────────────────────────┴──────────────┴─────────────┘

Value 预测：Loss 72% / Draw 23% / Win 5%
```

**可视化形式**:
- 横向柱状图 (Bar Chart)，长度 ∝ 访问概率
- 棋盘上高亮显示前 3 个候选动作（绿框=推荐，黄框=次优，灰框=备选）

#### 2. Policy Head vs MCTS 对比

展示网络的先验分布与搜索后验分布的差异：

```
Policy Prior (网络直接输出)  vs  MCTS Posterior (搜索后):
┌──────────────────┬──────────────┬──────────────┐
│ 动作             │ Prior (%)    │ Posterior (%)│
├──────────────────┼──────────────┼──────────────┤
│ 翻 (5,2)        │ 32.5         │ 24.9         │ ← 网络认为更好，但搜索否决
│ 移 (8,2)→(7,2)  │ 18.3         │ 42.1         │ ← 搜索发现了价值
│ 移 (6,0)→(5,0)  │ 25.1         │ 19.7         │
└──────────────────┴──────────────┴──────────────┘
```

**价值**: 
- 发现网络偏见（总是偏好某个动作）
- 验证 MCTS 是否有效修正了错误先验

#### 3. Value 预测轨迹分析

记录每轮 MCTS 迭代中叶子节点的 Value 估计变化：

```
Episode: Black vs Red (Seed=42)
Iteration history for action "flip(5,2)":
  Iteration 0: V = -0.45 ( pessimistic)
  Iteration 5: V = -0.32
  Iteration 10: V = -0.18
  Iteration 15: V = -0.05 ( converging)
  Final: V = -0.02 ≈ Draw (after 120 sims)
```

**用途**:
- 收敛速度快的动作说明网络学得好
- 震荡剧烈的动作说明不确定性高

#### 4. 关键决策点标注工具

提供交互界面供人类专家标注：

```python
# gui.py 新功能：标注模式
with annotate_mode():
    model = load_model('candidate_latest.pt')
    state = GameState.from_fen(fen_string)
    
    # 展示当前最佳走法
    action, stats = model.predict(state)
    human_feedback = input_dialog.show(
        "Is this the right move?",
        options=["Correct ✓", "Suboptimal ⚠️", "Blunder ✗"]
    )
    
    if human_feedback != "Correct":
        save_annotation(state, action, feedback=human_feedback)
```

**后续利用**:
- 标注数据单独保存为 `annotations/v1.jsonl`
- 用于微调 Policy Head (Behavior Cloning on Human Feedback)

### 技术实现栈

| 组件 | 推荐工具 | 理由 |
|------|---------|------|
| **热力图绘制** | Matplotlib Seaborn | 静态分析图 |
| **交互式 GUI** | PyQt5 + QCustomPlot | 实时可视化 |
| **Web Dashboard** | Streamlit / Gradio | 快速原型，无需前端 |
| **日志追踪** | MLFlow / Weights & Biases | 训练过程监控 |

**最小可行性产品 (MVP)**:
```bash
# CLI 工具：生成单个局面的分析报表
python -m junqi explain --model models/candidate_latest.pt \
    --fen rbbbqglle/rbbbbbbbr/... \
    --output reports/explain_turn_042.md
```

**输出示例**:
```markdown
## Turn 42: Red to Move

### Board State
![Board PNG](reports/explain_turn_042_board.png)

### Top 3 Candidate Moves
1. **Move Engineer (6,0) → (5,0)** - Access frequency: 42.1%
   - Reason: Control railway junction, enable long-distance transport
   
2. **Flip Hidden piece at (5,2)** - Access frequency: 24.9%
   - Risk assessment: Surrounded by enemy majors, safety score=0.3
   
3. **Retreat Major (8,2) → (7,2)** - Camp siege defense
   - Strategy: Avoid being trapped in camp

### Value Prediction Analysis
- Current position: Loss (72%) / Draw (23%) / Win (5%)
- If choosing top move: Expected V(t+1) = -0.15 (improvement!)
- Convergence status: Stable after 85 simulations

### Comparison with Search2 Baseline
| Model | Best Action | Confidence | Notes |
|-------|-------------|------------|-------|
| NN-MCTS | Engineer flight | 87% | Aggressive railway control |
| Search2 | Flip (5,2) | 92% | More conservative info gathering |

Recommendation: Follow Search2's conservative approach given material disadvantage.
```

---

## 四、问题 4: 模型版本管理后台

### 需求分析

**当前痛点**:
```
models/目录下文件混乱:
- best.pt                    # 当前生产模型（随时可能被覆盖）
- bc_best.pt                 # BC 预训练基线
- value_distilled.pt         # 蒸馏价值头
- candidate_latest.pt        # 最新候选模型
- candidate_latest_buffer.pkl # 缓冲文件？谁创建的？
- best_legacy.pt             # 旧版本备份
- elo_history.jsonl          # ELO 评分日志（孤立无上下文）
```

**缺失的能力**:
- ❌ 哪个模型在什么时间被训练？用了哪些超参数？
- ❌ 模型的性能曲线如何？何时开始退化？
- ❌ 模型的优缺点是什么？人类反馈记录在哪？
- ❌ 如何快速回滚到上一个稳定版本？

### 设计方案：轻量级模型管理平台

#### 1. 数据结构设计

**模型注册表 (`models/MANIFEST.json`)**:
```json
{
  "version": "1.0",
  "models": [
    {
      "id": "model_20260901_v1",
      "filename": "releases/v1.0_best.pt",
      "created_at": "2026-09-01T14:23:45Z",
      "trained_by": "Alice",
      "training_config": {
        "epochs": 5,
        "games_per_epoch": 24,
        "sims_per_game": 25,
        "workers": 4,
        "board_size": "12x5"
      },
      "dataset_version": "replays_v1_sav",
      "metrics": {
        "final_loss": 0.234,
        "validation_acc": 0.523,
        "elo_rating": 1523,
        "elo_rd": 125,
        "benchmark_win_rate": 0.58  // vs search2
      },
      "performance_summary": {
        "opening_success_rate": 0.61,
        "midgame_win_rate": 0.52,
        "endgame_draw_rate": 0.73,
        "best_against": "search3",
        "weakest_against": "greedy"
      },
      "human_notes": {
        "strengths": ["Excellent at railway maneuver", "Good camp siege"],
        "weaknesses": ["Struggles with engineer mine sweeping"],
        "anomalies": ["Sometimes flips safe pieces unnecessarily"]
      },
      "status": "released",  // experimental / candidate / released / deprecated
      "promoted_from": "model_20260828_v0"
    }
  ]
}
```

#### 2. Web 管理后台功能

**页面清单**:

##### (1) 仪表盘 (Dashboard)
```
┌─────────────────────────────────────────────────────┐
│ 模型演化时间线                                         │
│                                                     │
│  ●─●─────●─────●─○─○─○                            │
│  v0   v1     v3     v5 (current)                   │
│                           ▲                         │
│                      点击查看详情                   │
│                                                     │
│  性能对比图表                                           │
│  [ELO 趋势图]  [Win Rate 雷达图]  [Loss 曲线]      │
└─────────────────────────────────────────────────────┘
```

##### (2) 模型列表页
```
┌────────────────────────────────────────────────────────┐
│ ID        │ 创建时间      │ ELO    │ Status   │ 操作    │
├───────────┼──────────────┼────────┼──────────┼─────────┤
│ v1.0      │ 2026-09-01   │ 1523   │ ✅ 已发布 │ 查看/回滚│
│ cand_892  │ 2026-09-05   │ 1489   │ ⏳ 候选   │ 评估/丢弃 │
│ exp_445   │ 2026-09-04   │ 1467   │ ❌ 实验   │ 归档     │
└────────────────────────────────────────────────────────┘
```

##### (3) 详情页
```
模型：cand_892 (2026-09-05)

【基本信息】
训练参数：Epochs=3, Games=24, Sims=25, Workers=4
数据集：1000 局复盘 (.sav)

【性能指标】
ELO Rating: 1489 ± 115 (RD)
Benchmark (50 题): Win 12 / Draw 35 / Loss 3 (胜率=24%)
  - Opening: Top-1 accuracy = 58%
  - Midgame: Win Rate vs search2 = 48%
  - Endgame: Fortress Score = 0.71

【对战历史】
最近 10 局对战记录 (vs search2):
  Game 1: Draw (no_capture: 70 steps)
  Game 2: Loss (flag captured at turn 87)
  Game 3: Win (mine cleared then flag taken...)
  ...

【人工标注】
[2026-09-05 15:23] 审查者:Alice
  ✔ 优点：铁路机动流畅，MCTS 访问合理
  ⚠ 改进：翻棋安全性评估不稳定，建议加强 P0 修复
  
【操作】
[晋升为候选] [重新评估] [下载模型] [添加注释]
```

#### 3. API 接口设计

**Python SDK**:
```python
from junqi.model_registry import ModelRegistry

registry = ModelRegistry()

# 注册新模型
model_id = registry.register(
    path='models/checkpoints/epoch_003.pt',
    config=train_config,
    metrics={
        'loss': 0.234,
        'eval_acc': 0.523,
        'elo': 1489
    },
    notes='Third epoch of curriculum learning'
)

# 获取模型信息
model = registry.get(model_id)
print(f"ELO: {model.evaluations['elo']}")

# 列出所有已发布模型
for m in registry.list_models(status='released'):
    print(f"{m.id}: Elo {m.evaluations['elo']}")

# 模型晋升
registry.promote(
    source_id='cand_892',
    target_id='v1.0',
    reason='Beat search2 by 55% with 200 games'
)
```

**REST API**:
```http
GET /api/models
POST /api/models
GET /api/models/{id}
PUT /api/models/{id}/notes
DELETE /api/models/{id}
POST /api/models/{id}/evaluate
```

#### 4. 自动化报告生成

**触发时机**:
- 每次训练完成 (`train_rl.py` 结束时)
- 每次锦标赛轮次结束
- 手动调用 (`junqi report generate`)

**报告内容**:
```markdown
## Model Report: cand_892
Generated: 2026-09-05 16:30:00

### Training Summary
- Epochs: 3
- Total Self-play Games: 72
- Compute Cost: 12 GPU-hours

### Performance Metrics
| Metric | Value | Improvement vs prev |
|--------|-------|---------------------|
| Loss   | 0.234 | ↓ 0.012 (-4.9%)     |
| Val Acc| 0.523 | ↑ 0.035 (+7.2%)     |
| Elo    | 1489  | ↑ 67 (+4.7%)        |

### Benchmark Results (50 Puzzle Suite)
- Win Rate: 24% (12/50) - Same as baseline
- Top-1 Accuracy: 58% - +12% improvement
- Mean Value MAE: 0.423 - Better than last epoch

### Comparative Analysis
| Opponent | Our Win % | Last Version | Diff |
|----------|-----------|--------------|------|
| random   | 92%       | 90%          | +2%  |
| greedy   | 73%       | 68%          | +5%  |
| search2  | 48%       | 45%          | +3%  |
| search3  | 38%       | N/A          | -    |

### Human Reviewer Comments
[Alice] Reviewed at epoch 3:
- ✓ Good progression in opening phase
- ✗ Still struggles with mine clearing in endgame
- → Recommend continuing training but adjust weight decay

### Recommendation
[PROMOTE] [REJECT] [REQUEST_MORE_DATA]
```

### 技术选型

| 方案 | 复杂度 | 维护成本 | 适合团队规模 |
|------|--------|---------|-------------|
| **JSONL 规范** | 低 | 低 | 1-3 人 |
| **SQLite 本地数据库** | 中 | 低 | 5-10 人 |
| **MLFlow Tracking Server** | 高 | 中 | 10-50 人 |
| **自研 Web 后端 (Flask)** | 高 | 高 | >50 人 |

**推荐路径**: 
- **立即**: JSONL + Markdown 报告（已有 `elo_history.jsonl` 可扩展）
- **1 个月后**: 引入 MLFlow（开源免费，易集成）
- **长期**: 根据团队规模决定是否自研

---

## 五、问题 5: 验证慢 - 小棋盘课程学习

### 动机分析

**当前训练周期**:
```
12×5 棋盘 (60 格) + 25 子/方:
- 单局平均时长：3-5 分钟
- GPU 吞吐量：~40 局/小时/GPU
- 一轮训练 (50 局): ~1.5 小时
- 验证一次完整链路：3-5 天（等待足够样本量）
```

**问题**: 
- 修改一个超参数需要等到第二天才能看到效果
- 发现 Bug 后无法快速回归验证
- 实验机会成本高（浪费 GPU 时间）

### 解决方案：课程学习 (Curriculum Learning)

**核心理念**:
```
Simple → Complex
Small Board → Large Board
Localized Tactics → Global Strategy
```

#### 三种缩小棋盘方案对比

**方案 A: 4×8 窄长棋盘** (32 格)
- ✅ 保留铁路网纵向贯通
- ❌ 侧翼包抄难度过高
- ⚠️ 行营稀缺导致过度争夺

**方案 B: 6×6 方形棋盘** (**⭐强烈推荐**)
- ✅ **最佳平衡点**: 36 格 vs 战略纵深充足
- ✅ **对称性好**: 人类易理解布局
- ✅ **迁移效果好**: 从 6×6→8×6→10×6→12×5 平滑过渡
- ⚠️ **需定制规则**: CAMPS/HQS 坐标重写

**方案 C: 3×6 极端压缩** (18 格)
- ✅ 极致速度：10-15 步一局
- ❌ 失去战术丰富度
- ❌ 只适用于调试，不适合作为主训练场

#### 推荐实施路径

**Phase 1: 搭建基础设施 (Week 1)**
```python
# junqi/rules.py 抽象层重构
@dataclass
class BoardConfig:
    ROWS: int
    COLS: int
    CAMPS: Set[Tuple[int, int]]
    HQS: Set[Tuple[int, int]]
    RAIL_POSITIONS: Set[Tuple[int, int]]
    
    @classmethod
    def standard(cls):  # 12×5
        return cls(ROWS=12, COLS=5, ...)
    
    @classmethod
    def compact(cls):  # 6×6
        return cls(ROWS=6, COLS=6, ...)
```

**Phase 2: 6×6 预训练 (Week 2-3)**
```bash
# 快速迭代验证方向
python -m junqi train_rl --board 6x6 \
    --epochs 50 --games-per-epoch 200 \
    --model-out models/small_6x6.pt
```

**训练重点**:
- ✅ 学习基本走法规则（翻子、移动、吃子）
- ✅ 掌握子力价值判断（司令>军长>...>排长）
- ✅ 理解行营的安全性和控制价值
- ✅ 建立初步的铁路机动意识

**Phase 3: 课程升级 (Week 4)**
```
6×6 → 8×6 (扩展一行) → 10×6 → 12×5
```
每个阶段训练 5-10 Epoch，验证集性能提升停止即迁移。

**关键技术**:
- 冻结主干微调 (Fine-tune only Head layers)
- 数据混合采样 (70% 上一阶段 + 30% 当前阶段)
- 渐进式温度退火 (High Temp → Low Temp)

**Phase 4: 完整棋盘精调 (Week 5+)**
```bash
# 热启动
python -m junqi train_rl --board 12x5 \
    --load-model models/curriculum_10x8.pt \
    --epochs 100 --lr 5e-4
```

#### 预期收益

| 指标 | 原始流程 | 课程学习流程 | 提升 |
|------|---------|-------------|------|
| 首轮反馈时间 | 3-5 天 | 1-2 天 | **2-3x** |
| 单轮训练成本 | 20 GPU-hr | 5 GPU-hr (6×6) | **4x** |
| 发现 Bug 到验证 | ≥1 天 | ≤4 小时 | **6x** |
| 总训练成本 | 100 GPU-hr | 60 GPU-hr | **40%↓** |

---

## 📊 总体实施路线图

### 时间表 (甘特图风格)

```
月份：   September 2026
──────────────────────────────────────────────────────────────
W1:      [████████] 规则知识文档整理与人工审核
                          [████████] ELO 简易版实现

W2:      [████████] 规则注入训练数据准备
                          [████████] MCTS 可视化 MVP

W3:      [          ] 规则验证完成
                  [████████] ELO 标准版开发
                  [████████] 模型管理平台 (JSONL 方案)

W4:      [          ] 评估体系试运行
                  [████████] 小棋盘基础设施搭建
                  [          ] 可视化功能完善

Month 2:
W5:      [          ] 6×6 预训练启动
                  [████████] 模型管理后台 Web 界面

W6:      [████████] 6×6 训练与验证
                  [████████] 课程学习链路打通 (6×6→8×6)

W7:      [████████] 课程学习升级 (8×6→10×6→12×5)
                  [          ] 全功能评估平台上线

W8:      [████████] 完整棋盘精调
                          [████████] 文档与 CHANGELOG 更新
```

### 资源需求

**人力**:
- **Lead Engineer**: 1 人 (负责架构设计与代码重构)
- **RL Researcher**: 1 人 (负责评估体系与小棋盘策略)
- **QA/Reviewer**: 2-3 位军旗高手 (规则文档审核)

**硬件**:
- GPU: RTX 4080 SUPER × 1 (维持现状)
- CPU: 8 核以上 (支持并行训练)
- RAM: 32GB (回放池扩大)

**时间**:
- 总工时估算：≈ **60-80 人日**
- 预计 completion: **2026-10-31**

---

## ✅ 验收标准

### P0 里程碑 (September 月底)
- [ ] 《军棋规则与棋理知识手册》人工审核完成并通过
- [ ] ELO 简易版上线，支持基础循环赛
- [ ] MCTS 可视化 MVP 可用（CLI 级别）
- [ ] 模型 MANIFEST.jsonl 规范定义并开始使用

### P1 里程碑 (October 中)
- [ ] 评估体系标准化 (Glicko-2 + Rematch)
- [ ] 模型管理平台 Web 界面 Beta 版上线
- [ ] 小棋盘 6×6 预训练启动并验证有效性
- [ ] 所有单元测试在新结构下通过

### P2 里程碑 (October 底)
- [ ] 课程学习完整链路 (6×6→12×5)
- [ ] 完整棋盘训练效率提升 ≥ 50%
- [ ] 首次达到 Human Expert 水平 (对战人类胜率 > 70%)
- [ ] CHANGELOG.md 同步更新至 v2.0

---

## 🎯 成功定义

**短期成功 (3 个月内)**:
1. 新手开发者能快速理解项目规则与架构（新人 onboarding 时间从 2 周降至 3 天）
2. 模型晋级不再陷入"和棋死锁"，每周至少 1 次有效评估
3. 训练过程部分可解释（能说出 AI 为什么选某个动作）

**中期成功 (6 个月内)**:
1. 课程学习使训练效率提升≥50%
2. 社区能够自主举办模型联赛并生成公开排名
3. 人类专家对战胜率稳定超过 60%

**长期成功 (1 年内)**:
1. 项目成为军棋 AI 研究的参考实现
2. 模型能够击败人类职业选手
3. 技术成果发表至学术或工业界会议

---

## 📌 附录

### A. 依赖关系矩阵

| 任务 | 阻塞任务 | 被谁依赖 | 风险等级 |
|------|---------|---------|---------|
| 规则知识文档 | 无 | ELO 评估、小棋盘训练 | 低 |
| ELO 简易版 | 规则文档 | 模型管理平台 | 低 |
| 模型管理平台 | ELO 系统 | 无（独立） | 中 |
| MCTS 可视化 | 无 | 人类反馈收集 | 低 |
| 小棋盘基础设施 | 规则文档 | 课程学习 | 中 |
| 6×6 预训练 | 小棋盘基建 | 课程学习 | 中 |
| 课程学习 | 6×6 预训练 | 完整棋盘精调 | 高 |
| 完整棋盘精调 | 课程学习 | 最终验收 | 高 |

### B. 风险缓释计划

| 风险 | 发生概率 | 影响程度 | 缓解措施 |
|------|---------|---------|---------|
| 规则文档人工审核延迟 | 中 | 高 | 并行进行其他任务，设置 Deadline |
| 6×6 迁移效果差 | 低 | 高 | 准备 B 计划 (4×8棋盘作为备选) |
| ELO 系统被质疑主观性 | 中 | 低 | 提供多种版本让用户选择 |
| GPU 资源不足 | 低 | 中 | 优先保证课程学习 pipeline |
| 团队成员变动 | 中 | 中 | 详细文档 + 模块化代码便于交接 |

### C. 沟通计划

- **周报**: 每周五发送进度邮件给所有干系人
- **demo day**: 每月最后一次 Friday 展示功能进展
- **issue tracker**: GitHub Issues 作为唯一任务跟踪源
- **知识沉淀**: 所有决策记录写入 `docs/DECISION_LOG.md`

---

**文档维护者**: Qoder AI Assistant  
**版本**: v1.0  
**最后更新**: 2026-09-05  
**批准**: 待项目 Owner 审核确认
