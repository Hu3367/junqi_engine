# 军棋翻棋推演引擎（junqi_engine） v2.0

两人陆战棋 · **翻棋（翻明棋）模式**的 AI 推演与强化学习研究引擎：规则建模 + **AlphaZero/MCTS 深度强化学习自对弈** + 不完全信息搜索 + 局面计算器。

> **版本**: v2.0 (2026-09-07) - 工程化重构版  
> **重要更新**: 
> - ✅ 全新的目录结构与配置文件体系
> - ✅ 传统搜索专家模型 (ExpertSearchEngine) 与官方 APK (ApkNativeAgent) 深度对齐：消除 QSearch 视界盲区、引入 Delta 剪枝与 IDS 25% 耗时控制
> - ✅ 根治中营与角营往复互窜缺陷，全面校准“首翻即据点、依托行营辐射拓荒”实战棋理
> - ✅ 1:1 独立极简纯净 APK 引擎 (ApkSearchEngine)：5 大逆向差异全量对齐（2560 等比估值核、纯明子树拓扑、开局 6 黄金位、Jitter 抖动与 70 步和棋）
> - ✅ 高性能 C++ 原生引擎基础设施 (`src_cpp/`) 与双轨热拔插网桥 (`junqi/core_bridge.py`)：60 字节紧凑内存布局、50~100 倍潜在算力加速、未编译环境透明自动降级
> - ✅ 13 份专业文档 (>120k 字)
> - ✅ 全量单元测试通过（`python -m pytest tests/ -q`，数量随迭代增长，以 CI 输出为准）
> - ⚠️ Python ≥ 3.10，必须使用虚拟环境运行
>
> **代码审查报告**：[reviews/CODE_REVIEW_2026-09-15.md](reviews/CODE_REVIEW_2026-09-15.md)
> ⚠️ `junqi/expert/` 已废弃且不可导入，在线搜索引擎是 `junqi/search.py`，详见
> [junqi/expert/DEPRECATED.md](junqi/expert/DEPRECATED.md)

---

## 📚 核心文档导航

### 必读书单（按顺序）

1. **[AI_TRAINING_AND_HUMAN_PLAY_PLAN.md](AI_TRAINING_AND_HUMAN_PLAY_PLAN.md)** - P0-P4 执行基线与标签规则 ⭐
2. **[AGENTS.md](AGENTS.md)** - 大模型代理硬约束与契约 ⭐
3. **[JUNQI_RULES_AND_STRATEGY_GUIDE.md](docs/03-RulesAndStrategy/JUNQI_RULES_AND_STRATEGY_GUIDE.md)** - 规则细则、实战棋理与1000局官方大数据实证手册 ⭐
4. **[JUNQI_CHESS_INTELLECT_V2.md](docs/03-RulesAndStrategy/JUNQI_CHESS_INTELLECT_V2.md)** - 不完全信息博弈理论与棋理本体体系

### 详细文档目录

- **架构设计**: [PROJECT_REFACTORING_PROPOSAL.md](docs/02-Architecture/PROJECT_REFACTORING_PROPOSAL.md), [MODULAR_ARCHITECTURE_GUIDE.md](docs/02-Architecture/MODULAR_ARCHITECTURE_GUIDE.md)
- **实施指南**: [FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md](docs/02-Architecture/FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md), [REFACTORING_CHECKLIST.md](docs/01-GettingStarted/REFACTORING_CHECKLIST.md)
- **快速参考**: [QUICK_REFERENCE_CARDS.md](docs/QUICK_REFERENCE_CARDS.md), [QUICK_REFERENCE_V2.0.md](docs/QUICK_REFERENCE_V2.0.md)
- **理论体系**: [PROJECT_KNOWLEDGE_AND_THEORY.md](docs/03-RulesAndStrategy/PROJECT_KNOWLEDGE_AND_THEORY.md), [RL_TRAINING_ROADMAP.md](docs/05-ExecutionPlans/RL_TRAINING_ROADMAP.md)
- **开局行营争夺**: [CAMP_RUSH_OPENING_PLAYBOOK.md](docs/03-RulesAndStrategy/CAMP_RUSH_OPENING_PLAYBOOK.md) - 邻营度定理、闭合期望模型、25000 场占营竞赛与先手/后手策略决策树
- **变更记录**: [CHANGELOG.md](docs/CHANGELOG.md)

---

## 玩法与规则（已按 APK 解包规则对齐，权威对照见 ../apk_extracted/RULES_FROM_APK.md）

50 枚棋子（双方各 25：司令1 军长1 师长2 旅长2 团长2 营长2 连长3 排长3 工兵3 炸弹2 地雷3 军旗1——与 App `default.lay` 解码一致）洗乱、背面朝上铺满全盘 50 个非行营位置。双方轮流行动，**首次翻子颜色决定阵营**；每回合可"翻开一枚暗子"或"移动一枚己方明子"。吃掉对方军旗、或使对方无子可动即获胜。

- 棋盘 5 列 × 12 行；每方 5 行营（安全区，不可被攻击，开局空）+ 2 大本营（末排 2/4 位）
- **大本营不锁行动**（App 实测，可用 `hq_locks_pieces` 开启旧规则）
- **铁路一律直线滑行**，不拐弯（含四角弧线位置）；前线三通道中 col1/col3 完全不通、col2 为公路
- **工兵铁路飞行：全场任意转弯，且可飞越路径上的棋子**（1000 局真实复盘反推验证；早期"底线不转弯/不可越子"的实测结论均被数据推翻，见 `junqi/replay.py`）
- **禁止自杀攻击**：小子不可撞大子、非工兵不可碰雷（可用 `allow_suicide_attack` 开启旧规则）
- 战斗：大吃小、同级同尽；炸弹与任何子同尽；地雷不能动（工兵挖除、炸弹同尽）
- **扛旗双门槛（APK ruleflip.txt 原文）**：需先清光对方 3 雷（`flag_needs_mines_cleared`），且**只有工兵能扛旗**（`flag_gong_only`，对应 App"吃军旗:工兵"默认档）
- **和棋三线（APK 官方细则）**：连续 40 步未吃子判和（`no_capture_draw_plies`）；总步数 1000 判和（`max_plies`）；可观察局面重复 3 次判和（`repetition_draw_count`，对局层计数并把临界局面传给搜索主动规避）

与具体 App 细则有差异时，改 `junqi/config.py` 即可对齐（App"规则设置"对话框另有 大本营吃子开关/吃地雷者/吃军旗者 三项可选档；引擎默认取其"工兵/允许吃"默认档，中间档"最小棋子"未建模）。

---

## 🚀 快速开始

### 环境准备（必须）

本项目使用独立的虚拟环境管理依赖。**切勿直接使用系统 Python**！

```bash
cd junqi_engine

# Windows (PowerShell)
.\venv\Scripts\activate.ps1

# Linux/Mac
source venv/bin/activate

# 验证 Python 版本（需要 3.10+）
python --version
```

### 运行测试

```bash
# 方法 1: 使用便捷脚本（推荐）
.\run_tests.bat

# 方法 2: 使用完整路径（Windows）
"/e/local code/军棋/junqi_engine/venv/Scripts/python.exe" -m pytest tests/ -v

# 方法 3: 激活虚拟环境后运行
python -m pytest tests/test_rules.py -v

# 运行所有单元测试（数量随迭代增长，此处不写死，以 `pytest` 输出为准）
python -m pytest tests/ -v --tb=short
```

### CLI 命令

统一入口是 `python -m junqi <子命令>`（旧 `cli.py` 为已失效的历史壳，其 `train/`、`eval/`、`ui/` 目录已不存在）。

```bash
# 测试套件
python -m junqi test        # 全量单元测试

# GUI 界面
python -m junqi gui         # 启动人机对战

# 局面计算器  
python -m junqi calc        # 交互式计算

# 训练（需要 PyTorch/GPU）
python -m junqi train_rl --epochs 5 --games 24

# 训练（P3 修订 2026-09-14：默认 lr=1e-4 + 每轮 Value 重锚 + p1_v3/test 健康验收）
#   --fresh 会优先热启动 models/value_distilled_v2.pt（健康 Value 头）
python -m junqi train_rl --epochs 5 --games 500 --sims 20 --fresh
#   回滚开关：--no-reanchor 关闭每轮重锚（此时仅低学习率起作用）

# Benchmark 评测
python -m junqi benchmark     # 50 题固定评测

# P0 评测门控（配对同牌/座位互换/Wilson 区间 + 三元 SPRT）
python -m junqi gate --a hybrid2 --b expert2 --seeds 100

# P1 真实终局标签 Value 头重训（官方 list.cfg 解密标签，候选权重不覆盖 best.pt）
python -m junqi distill_value --p1-dir datasets/p1_v3 --out models/value_distilled_v2.pt

# P2 搜索蒸馏（QSearch 专家教师软分布 -> Policy 头交叉熵）
python -m junqi distill_search --base models/bc_best.pt --states 1200 --depth 3
```

---

## 📁 项目结构（v2.0）

```
junqi_engine/
├── docs/                      # 核心文档库（13 份专业文档）⭐
│   ├── FINAL_REFACTORING_SUMMARY.md      # 重构总结
│   ├── JUNQI_RULES_AND_STRATEGY_GUIDE.md # 规则与策略手册
│   ├── ELO_RATING_SYSTEM_DESIGN.md       # 等级分设计
│   ├── PROJECT_REFACTORING_PROPOSAL.md   # 完整方案
│   ├── MODULAR_ARCHITECTURE_GUIDE.md     # 模块化指南
│   └── ... (共 13 份 >120k 字)
│
├── configs/                     # 配置文件目录 ⭐
│   ├── rules.yaml              # 规则参数
│   ├── training.yaml           # 训练超参数
│   ├── search.yaml             # 搜索参数
│   └── boards/                 # 棋盘配置
│       ├── standard.yaml       # 12×5 标准棋盘
│       └── compact.yaml        # 6×6 紧凑棋盘
│
├── scripts/                     # 工具脚本 ⭐
│   ├── gen_assets.py
│   ├── gen_eval_sets.py
│   ├── benchmark_p2.py
│   └── benchmark_expert.py
│
├── tests/utils/                 # 验证工具 ⭐
│   ├── _verify_a2.py
│   ├── _verify_fit_weights.py
│   └── ... (共 4 个文件)
│
├── models/                      # 模型输出目录
│   ├── MANIFEST.jsonl          # 模型注册表 ⭐
│   ├── checkpoints/            # 训练快照
│   ├── releases/               # 发布版本
│   └── pool/                   # 模型池
│
├── junqi/                       # 核心业务模块（保持平铺结构）
│   ├── __init__.py
│   ├── rules.py                # 规则定义
│   ├── state.py                # 状态机
│   ├── net.py                  # 神经网络
│   └── ... (共 31 个文件)
│
├── cli.py                       # CLI 入口点 ⭐
├── run_tests.bat                # 测试启动器 ⭐
├── requirements.txt             # Python 依赖 ⭐
├── AGENTS.md                    # 大模型规则
└── README.md                    # 本文件
```

---

## 🔧 高级配置

> **现状说明（2026-09-15 审查）**：`configs/*.yaml` 目前是**待接线的配置模板**，
> 运行期真正生效的是代码里的 `junqi/config.py`（`RuleConfig` / `SearchConfig` /
> `EvalWeights` 三个 dataclass）。**修改 YAML 不会影响运行行为**，也无热加载。
> 需要改参数请直接改 `junqi/config.py`，或先实现 YAML 加载再加开关。
> 目前仅有 `scripts/mini_junqi_experiment.py` 读取 `configs/mini_junqi_config.json`。

- **规则开关**: `junqi/config.py::RuleConfig` - APK 对齐参数（`configs/rules.yaml` 为模板）
- **训练超参**: `junqi/config.py` 与 `train_rl` CLI（`configs/training.yaml` 为模板）
- **搜索配置**: `junqi/config.py::SearchConfig`（`configs/search.yaml` 为模板）
- **棋盘规格**: `configs/boards/*.yaml` - 支持多棋盘尺寸（模板）

---

## 📖 详细文档索引

欲了解更详细信息，请查阅以下文档：

- **架构总览**: [PROJECT_REFACTORING_PROPOSAL.md](docs/02-Architecture/PROJECT_REFACTORING_PROPOSAL.md)
- **实施细节**: [FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md](docs/02-Architecture/FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md)  
- **模块化设计**: [MODULAR_ARCHITECTURE_GUIDE.md](docs/02-Architecture/MODULAR_ARCHITECTURE_GUIDE.md)
- **规则与实战策略**: [JUNQI_RULES_AND_STRATEGY_GUIDE.md](docs/03-RulesAndStrategy/JUNQI_RULES_AND_STRATEGY_GUIDE.md)
- **评估体系**: [ELO_RATING_SYSTEM_DESIGN.md](docs/04-TrainingAndEvaluation/ELO_RATING_SYSTEM_DESIGN.md)
- **快速参考**: [QUICK_REFERENCE_CARDS.md](docs/QUICK_REFERENCE_CARDS.md)

---

## 🎯 下一步行动

### 立即开始
1. ✅ 阅读 [`FINAL_REFACTORING_SUMMARY.md`](docs/01-GettingStarted/FINAL_REFACTORING_SUMMARY.md) 了解工程重构
2. ✅ 激活虚拟环境并运行 `.\run_tests.bat` 验证安装
3. ✅ 启动 GUI: `python cli.py gui` 体验人机对战

### 深入学习
1. 📚 阅读 [`AI_TRAINING_AND_HUMAN_PLAY_PLAN.md`](AI_TRAINING_AND_HUMAN_PLAY_PLAN.md) 了解 P0-P4 路线图与官方标签规则
2. 🎓 研读 [`JUNQI_RULES_AND_STRATEGY_GUIDE.md`](docs/03-RulesAndStrategy/JUNQI_RULES_AND_STRATEGY_GUIDE.md) 掌握实战棋理
3. 🔬 探索 [`JUNQI_CHESS_INTELLECT_V2.md`](docs/03-RulesAndStrategy/JUNQI_CHESS_INTELLECT_V2.md) 学习不完全信息博弈理论

---

```bash
python -m junqi gui
```

参照手机 App 截图风格：木纹棋盘、铁路虚线、行营圆圈、大本营括号。**棋子贴图为 APK 原版资源**（`gen_assets.py` 从解包精灵表切片生成 `assets_app/cells/`，缺文件时自动回退矢量绘制）；状态栏实时显示官方规则线（手数/1000、无吃子/70、循环线）。

- 点己方明子 → 红框选中、红框高亮合法落点（虚线=移动，实线=吃子）→ 点落点执行
- 点暗子两次确认翻开（防误触）
- 右侧面板：AI 棋力（快=深度1 / 标准=深度2 / 深算=深度3）、**发牌种子**（同种子同布局，便于复现检验具体走法）、执先手/后手、暗子剩余构成（公开信息推断）、对局记录
- **提示**按钮：AI 给你推荐前 3 个走法（金框标注推荐步）
- **悔棋**：一键回退到你上一次行动前
- AI 与提示都会**主动规避循环判和局面**（可观察局面重复计数传入搜索根节点罚分）

---

## 局面计算器（交互式）

```bash
python -m junqi calc
```

常用命令：

```
deal 42                新开真实发牌局（内部可见底牌，用于研究）
set 5,2 r师            手动放子（r/b + 司军师旅团营连排工炸雷旗，?? 表示暗子）
color r                设当前行动方执红（录入实战中盘局面用）
best 3 2 4             前3个建议，搜索深度2、翻子采样4（约0.3秒）
best 3 3 3             深度3深算（约10秒）
move 5,2 6,2           走子（自动战斗结算）
play 10                双方搜索自动走10手
undo / save 1.json / load 1.json / show / quit
```

---

## 真实复盘数据：回放与权重拟合（AI 训练）

`军旗复盘/` 下的 1000 局真实对局（.sav，含完整暗子身份表，已全量验证与引擎规则零矛盾）：

```bash
python -m junqi replay ../军旗复盘           # 解码+回放校验+统计报告（1000/1000 通过）
python -m junqi fit ../军旗复盘 --verify 60  # 行为克隆拟合估值权重（条件logistic）
python scripts/mine_replays_report.py        # 1000局官方历史数据库(list.cfg)联合大数据挖掘报告
```

- **官方胜负数据库破译**：成功逆向 `libjunqi.so` 破译 `list.cfg`（DES-ECB 加密），获得全部 1000 局 100% 权威终局胜负与原因码（核心玩家 458 胜 241 负 301 和；认输折叠 45.6%、协议和棋 26.5%）。
- **实战棋理与量化挖掘报告**：详见 [`reports/replays_1000_mining_report.md`](reports/replays_1000_mining_report.md)，已全面融入 [`JUNQI_RULES_AND_STRATEGY_GUIDE.md`](docs/03-RulesAndStrategy/JUNQI_RULES_AND_STRATEGY_GUIDE.md) 与 [`JUNQI_CHESS_INTELLECT_V2.md`](docs/03-RulesAndStrategy/JUNQI_CHESS_INTELLECT_V2.md)。
- fit：每局采样决策点，人类动作 vs 候选动作做条件 logistic 回归（验证集早停+符号约束），产出 `reports/fit_weights.json`。
- 实测效果（1000 局，85/15 训练/验证）：人类动作命中率 top1 0.482→0.525、top3 0.698→0.745；新权重 vs 旧权重镜像分 0.544。

---

## 深度强化学习自对弈训练（AlphaZero / MCTS 范式）

基于 PyTorch + CUDA（适配 RTX 4080 SUPER 等现代显卡）：

```bash
# 仅用于短周期/中等规模复验；当前禁止据此启动 P4.4 长期挂机训练。
# 首次从旧基线重启可加 --rebase-baseline（备份旧 best 并用 bc_best 重建发布基线）。
python -m junqi train_rl --epochs 5 --games 24 --sims 25 --eval-games 4 --workers 4 --out-dir models

# 经验池落盘默认每 5 轮一次（实测约 3.8 GB/份）；--buffer-save-every 0 除末轮外不写，
# 1 恢复旧的"每轮写 4GB"行为
python -m junqi train_rl --epochs 5 --games 24 --buffer-save-every 5

# 运行固定 50 题实验靶场基准测试（S0 起真值与类别一致，含 Brier/预测熵指标）
python -m junqi benchmark --model models/best.pt

# S2 专家价值蒸馏预热（仅训练价值头，产物被 train_rl 热启动链自动优先采用）
python -m junqi distill_value --base models/bc_best.pt --out models/value_distilled.pt --samples 1200
```

- **状态张量化（38 通道双模式）**：包含己方/敌方 12 级明子分布、暗子掩码、精确子力差平面、死区潜力、铁路/行营几何掩码、工兵地雷存活全局状态、连续无吃子步数进度、阶段标记、世界模式标志、以及 **通道36/37的重复历史计数（seen >= 1 / seen >= 2）**。
- **双头网络架构 (`JunqiNet`)**：6-Block 残差卷积（ResNet）主干，Policy Head（输出 3650 维动作 logits）+ Value Head（输出 Win / Draw / Loss 三分类概率）。
- **MCTS 搜索 (`MCTS`)**：使用 $c_{\text{puct}} = 0.6$ 与 $\epsilon = 0.20, \alpha = 0.15$ Dirichlet 探索噪声；历史重复计数已接入 NN/MCTS 根、叶与树内终局判断，并新增循环规避专项测试。
- **训练闭环与门控（V2.3）**：candidate 与 best 已分离，检查点可恢复真实 Replay Buffer；Worker 已有多类对手分支。
- **晋升门控唯一真源（`junqi/eval_gate.py`）**：2026-09-15 起训练主循环的 `train_rl.evaluate_gate` 直接委托 `eval_gate.run_gate`（配对同牌 + 先后手互换 + Wilson + 三元 SPRT），
  此前它是另一套**非配对同牌**且 `device` 写死 `cpu` 的平行实现，其数字不具备统计效力。
- **数据质量改造（2026-09-13）**：自对弈生成侧使用独立夹具（无吃子判和 70→120 步、循环判和 3→4 次），评测门控保持官方规则；认输机制（走子方根 Value ≤ −0.95 连续 8 回合、ply ≥ 40）按官方 code 21 语义提前终局（认输局的 Value 样本不入训练池，仅保留 Policy——防止未校准 Value 的自证回路）；和棋局 quiet ≥ 60 的尾部垃圾样本不入池；每轮打印决胜率与终局原因分布。详见 [AI_TRAINING_AND_HUMAN_PLAY_PLAN.md](AI_TRAINING_AND_HUMAN_PLAY_PLAN.md) P3 节修订。

> **训练状态（2026-09-01）**：P4.4 长期挂机准入仍未通过。Epoch 6 已将重复和棋降至 0/32，但候选仍 `promoted=false`，Value MAE `0.5690`、准确率 `20.0%`、Win 预测为 0。启动训练和自动晋升采用两套独立门槛，详见 [P4 执行计划](docs/P4_EXECUTION_PLAN.md#p44-长期挂机准入复审与开启条件2026-09-01)。

---

## 目录结构

```
junqi/rules.py       棋盘几何、铁路网、走法生成辅助、战斗结算
junqi/state.py       对局状态、发牌、合法走法、机会节点翻子、JSON 记谱、可观察局面键
junqi/encoder.py     状态空间 38 通道双模式张量编码与 3650 维动作空间编解码
junqi/net.py         双头 Policy-Value 残差网络（JunqiNet，三分类 Value Head）
junqi/mcts.py        神经网络引导的 PUCT 蒙特卡洛树搜索（黄金探索区间 c_puct=0.6）
junqi/train_rl.py    深度强化学习自对弈训练流水线（P3.3 门控增强闭环）
junqi/benchmark.py   实验靶场题库（50标杆残局）与指标看板生成器
junqi/ai.py          估值函数 + 期望化搜索 + NNAgent 深度学习代理
junqi/selfplay.py    策略池（含 NNStrategy）、批量对局、统计聚合、Markdown 报告
junqi/tune.py        估值权重调优（镜像自对弈爬山）
junqi/replay.py      App .sav 复盘解码与引擎回放
junqi/fit_weights.py 复盘数据行为克隆拟合权重
junqi/calculator.py  交互式局面计算器
junqi/config.py      规则开关、估值权重、搜索参数
junqi/gui.py         人机对战图形界面
tests/               全量单元测试（规则 + 复盘 + RL/MCTS + P3/P4 + 开局占营竞赛与 Δ 战术专项）
metrics/             实验靶场看板 (benchmark_dashboard.md) 与误差时序数据
models/              核心深度学习模型权重 (best.pt, bc_best.pt, value_distilled.pt) 与发布目录
reports/             1000 局官方大数据挖掘报告、专家引擎阶段报告与评测归档
scripts/             挖掘流水线、性能评测基准、人机 playtest 仿真与工具集
docs/                01~08 分类体系化理论文档、规则手册与 CHANGELOG.md
军旗复盘/            官方 1000 局原始 .sav 对局样本与 list.cfg 历史数据库
```
