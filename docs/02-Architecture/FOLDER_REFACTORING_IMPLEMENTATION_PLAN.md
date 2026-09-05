# 项目文件夹结构重构实施方案

> **目标**: 解决当前代码组织混乱问题  
> **预计耗时**: 2-3 小时（Phase 1+2 可快速完成）  
> **风险等级**: Phase 1 低风险，Phase 3 中风险  
> **回滚方案**: 使用 Git revert 或手动恢复

---

## 📊 当前问题分析

### 主要痛点

1. **根目录污染**: `_verify_*.py`, `benchmark_*.py` 散落
2. **模型目录混乱**: `models/`, `models_v3/`, `models/test*` 混杂
3. **缺少分类意识**: 工具脚本、配置、资源未分组
4. **测试分散**: `tests/`外存在独立基准测试

### 重构目标结构

```
junqi_engine/
├── .gitignore                    # [现有] Git 忽略配置
├── LICENSE                       # [新增] 开源协议
├── README.md                     # [现有] 项目说明
├── AGENTS.md                     # [现有] 大模型规则
├── AI_TRAINING_AND_HUMAN_PLAY_PLAN.md  # [现有] 执行基线
│
├── docs/                         # [现有] 技术文档
│   ├── CHANGELOG.md
│   ├── P4_EXECUTION_PLAN.md
│   ├── PROJECT_KNOWLEDGE_AND_THEORY.md
│   ├── JUNQI_RULES_AND_STRATEGY_GUIDE.md  # [新建] 规则手册
│   ├── PROJECT_REFACTORING_PROPOSAL.md  # [新建] 重构提案
│   ├── REFACTORING_CHECKLIST.md         # [新建] 行动计划
│   ├── ELO_RATING_SYSTEM_DESIGN.md      # [新建] 等级分设计
│   ├── QUICK_REFERENCE_CARDS.md         # [新建] 快速参考
│   ├── PROJECT_STRUCTURE_REFactoring_REPORT.md # [已有] 审计报告
│   └── *.pdf                     # 学术论文
│
├── configs/                      # [新增] 配置文件目录
│   ├── rules.yaml                # 规则配置
│   ├── training.yaml             # 训练超参数
│   ├── search.yaml               # 搜索参数
│   ├── weights.yaml              # 专家权重
│   └── boards/                   # 棋盘配置
│       ├── standard.yaml         # 12×5标准棋盘
│       ├── compact.yaml          # 6×6紧凑型
│       └── narrow.yaml           # 4×8窄长型
│
├── scripts/                      # [新增] 实用工具脚本
│   ├── gen_assets.py             # 从 gen_assets.py 迁移
│   ├── gen_eval_sets.py          # 从 gen_eval_sets.py 迁移
│   ├── benchmark_p2.py           # 从 benchmark_p2.py 迁移
│   ├── benchmark_expert.py       # 从 benchmark_expert.py 迁移
│   ├── run_tournament.py         # 锦标赛运行器 [新建]
│   └── analyze_replays.py        # 复盘分析 [新建]
│
├── junqi/                        # [现有] 核心业务模块
│   ├── __init__.py
│   ├── __main__.py
│   ├── ai.py
│   ├── belief.py
│   ├── benchmark.py
│   ├── calculator.py
│   ├── config.py
│   ├── dataset.py
│   ├── encoder.py
│   ├── gui.py
│   ├── mcts.py
│   ├── net.py
│   ├── replay.py
│   ├── rules.py
│   ├── search.py
│   ├── selfplay.py
│   ├── state.py
│   ├── train_bc.py
│   ├── train_rl.py
│   ├── train_value_distill.py
│   ├── tt.py
│   ├── zobrist.py
│   ├── tune.py
│   ├── analysis.py
│   ├── analyze.py
│   ├── endgame_gen.py
│   ├── eval_bc.py
│   ├── eval_expert.py
│   ├── fit_weights.py
│   ├── hybrid_engine.py
│   └── cli.py                    # [合并] 从 __main__.py 拆分
│
├── core/                         # [新增] 基石模块（从 junqi/ 迁移）
│   ├── rules.py                  # 规则定义
│   ├── state.py                  # 状态机
│   ├── encoder.py                # 编码
│   ├── zobrist.py                # Zobrist 哈希
│   └── tt.py                     # 转位表
│
├── model/                        # [新增] 模型定义（从 junqi/ 迁移）
│   ├── net.py                    # JunqiNet
│   ├── policy_head.py            # Policy Head 子类 [新建]
│   └── value_head.py             # Value Head 子类 [新建]
│
├── search/                       # [新增] 搜索算法（从 junqi/ 迁移）
│   ├── mcts.py                   # MCTS
│   ├── search_agent.py           # ExpertSearchAgent
│   ├── hybrid_engine.py          # 混合引擎
│   └── utils.py                  # 搜索辅助函数 [新建]
│
├── train/                        # [新增] 训练流水线（从 junqi/ 迁移）
│   ├── train_rl.py               # RL 训练
│   ├── train_bc.py               # BC 训练
│   ├── train_value_distill.py    # 价值蒸馏
│   ├── selfplay.py               # 自对弈
│   └── checkpoint.py             # 检查点管理 [新建]
│
├── data/                         # [新增] 数据处理（从 junqi/ 迁移）
│   ├── dataset.py                # 数据集
│   ├── replay.py                 # 复盘解析
│   ├── endgame_gen.py            # 残局生成
│   └── fit_weights.py            # 权重拟合
│
├── eval/                         # [新增] 评估工具（从 junqi/ 迁移）
│   ├── benchmark.py              # 50 题评测
│   ├── eval_bc.py                # BC 评估
│   └── eval_expert.py            # 专家评估
│
├── ui/                           # [新增] 用户界面（从 junqi/ 迁移）
│   ├── gui.py                    # GUI 界面
│   ├── calculator.py             # 局面计算器
│   └── widgets.py                # 自定义组件 [新建]
│
├── util/                         # [新增] 工具库（从 junqi/ 迁移）
│   ├── config.py                 # 全局配置加载
│   ├── belief.py                 # 信念跟踪
│   ├── analysis.py               # 分析工具
│   ├── tune.py                   # 调优工具
│   └── common.py                 # 公共函数 [新建]
│
├── tests/                        # [现有] 测试套件
│   ├── __init__.py
│   ├── test_rules.py
│   ├── test_state.py
│   ├── test_encoder.py
│   ├── test_net.py
│   ├── test_mcts.py
│   ├── test_search.py
│   ├── test_ai.py
│   ├── test_rl.py
│   ├── test_replay.py
│   ├── test_dataset.py
│   ├── test_zobrist_tt.py
│   ├── test_hybrid_agent.py
│   ├── test_p3_benchmark.py
│   ├── test_p4_audit_fixes.py
│   ├── test_p4_hybrid.py
│   ├── test_s0_s1_fixes.py
│   ├── test_s2_fixes.py
│   ├── test_expert_eval.py
│   ├── test_replay.py
│   └── utils/                    # [新建] 验证工具集
│       ├── verify_a2.py          # 从根目录迁移
│       ├── verify_fit_weights.py # 从根目录迁移
│       ├── verify_rebase_state.py # 从根目录迁移
│       ├── verify_root_cause.py  # 从根目录迁移
│       └── helpers.py            # 测试辅助函数 [新建]
│
├── models/                       # [重写] 训练产出目录
│   ├── MANIFEST.jsonl            # [新建] 模型注册表
│   ├── checkpoints/              # [新建] 训练快照
│   │   ├── epoch_001.pt
│   │   ├── epoch_002.pt
│   │   └── latest.pt -> latest_epoch.pt
│   ├── releases/                 # [新建] 发布模型
│   │   ├── v1.0_best.pt
│   │   ├── v1.1_best.pt
│   │   └── current_best.pt (symbolic link)
│   ├── pool/                     # [现有] 模型池
│   │   ├── search2_baseline.pt
│   │   ├── bc_best.pt
│   │   └── nn_mcts_v3.pt
│   ├── rating/                   # [新建] 评级数据
│   │   ├── player_profile.json
│   │   └── rd_curves.jsonl
│   ├── tournament/               # [新建] 赛事数据
│   │   └── season_1/
│   │       ├── round_1.jsonl
│   │       └── standings.csv
│   └── validation/               # [新建] 门控测试集
│       ├── opening_suite.jsonl
│       ├── midgame_suite.jsonl
│       └── endgame_suite.jsonl
│
├── datasets/                     # [现有] 数据集
│   └── p1_v1/                    # 第一版数据
│
├── eval_sets/                    # [现有] 评估集
│   └── benchmark_50.jsonl        # 50 题题库
│
├── reports/                      # [现有] 报告输出
│   ├── benchmark_dashboard.md
│   ├── explain_turn_*.md         # MCTS 可视化报告
│   └── big/                      # 大型报告
│
├── metrics/                      # [现有] 指标数据
│   ├── benchmark_results.json
│   └── elo_history.jsonl
│
├── assets_app/                   # [现有] 棋子贴图资源
│   └── cells/                    # 单元格图像
│
├── assets_gui/                   # [新增] GUI 专用资源
│   ├── backgrounds/              # 棋盘背景
│   ├── fonts/                    # 字体文件
│   └── icons/                    # 图标
│
├── benchmarks/                   # [新增] 基准测试数据
│   └── p2_results.json
│
├── notebooks/                    # [新增] Jupyter Notebook
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_debug.ipynb
│   └── 03_analysis_visualization.ipynb
│
├── scripts_backup/               # [新增] 临时备份区（清理时创建）
│
└── venv/                         # [排除] Python 虚拟环境 (.gitignore)
```

---

## 🚀 实施步骤详解

### Step 1: 准备工作（5 分钟）

```bash
# 1. 创建备份目录
mkdir -p scripts_backup

# 2. 保存当前 Git 状态
git status --porcelain > /tmp/pre_refactor_status.txt
git diff > /tmp/pre_refactor_diff.patch
```

### Step 2: Phase 1 - 根目录清理（20 分钟）

#### A. 移动验证脚本

```bash
# 创建 tests/utils 目录
mkdir -p tests/utils

# 移动验证脚本
mv _verify_a2.py tests/utils/
mv _verify_fit_weights.py tests/utils/
mv _verify_rebase_state.py tests/utils/
mv _verify_root_cause.py tests/utils/
```

#### B. 创建 scripts 目录并迁移工具脚本

```bash
mkdir -p scripts

# 移动生成类脚本
mv gen_assets.py scripts/
mv gen_eval_sets.py scripts/

# 移动基准测试脚本
mv benchmark_expert.py scripts/
mv benchmark_p2.py scripts/
```

#### C. 清理无用目录

```bash
# 删除已废弃的模型目录（确认无用时）
rm -rf models_v3/

# 可选：保留但归档 test*/目录
# mv models/test models_archive/test_$(date +%Y%m%d)
# mv models/test_v2 models_archive/test_$(date +%Y%m%d)
# mv models/test_parallel models_archive/test_$(date +%Y%m%d)
```

### Step 3: Phase 2 - 模型目录标准化（30 分钟）

#### A. 创建新目录结构

```bash
cd models

# 创建新目录
mkdir -p checkpoints
mkdir -p releases
mkdir -p rating
mkdir -p tournament/season_1
mkdir -p validation

# 移动现有模型
mv best.pt releases/v1.0_current.pt
mv bc_best.pt pool/bc_best.pt
mv value_distilled.pt pool/value_distilled.pt
mv best_legacy.pt backups/
mv _candidate_gate.pt ../scripts_backup/  # 临时文件

# 将 pool 中的内容归位到 pool/
# 已有的 pool 直接保留

# 移动日志文件
mv elo_history.jsonl ../metrics/
```

#### B. 创建 MANIFEST.jsonl

```bash
cat > MANIFEST.jsonl << 'EOF'
{"id": "model_20260901_v1", "filename": "releases/v1.0_current.pt", "created_at": "2026-09-01T00:00:00Z", "status": "released"}
EOF
```

### Step 4: Phase 3 - 模块化重构（需仔细规划，建议分步进行）

⚠️ **此阶段风险较高，需要先创建详细映射表**

#### A. 创建模块映射文档

| 原路径 | 新路径 | 是否需要 | 备注 |
|--------|--------|---------|------|
| `junqi/rules.py` | `core/rules.py` | ✅ | 基石模块 |
| `junqi/state.py` | `core/state.py` | ✅ | 基石模块 |
| `junqi/encoder.py` | `core/encoder.py` | ✅ | 基石模块 |
| `junqi/net.py` | `model/net.py` | ✅ | 神经网络定义 |
| `junqi/mcts.py` | `search/mcts.py` | ✅ | 搜索算法 |
| `junqi/search.py` | `search/search_agent.py` | ✅ | 搜索代理 |
| `junqi/train_rl.py` | `train/train_rl.py` | ✅ | 训练流水线 |
| `junqi/gui.py` | `ui/gui.py` | ✅ | 用户界面 |
| `junqi/config.py` | `util/config.py` | ✅ | 配置加载 |
| `junqi/__main__.py` | `cli.py` | ⏳ | CLI 入口 |

#### B. 逐步重命名（每次一个子目录）

```bash
# 第一步：创建新目录
mkdir -p core model search train data eval ui util

# 第二步：移动文件（不要一次性全部移动）
# 先从一个小模块开始，比如 core

# 移动基石模块
mv junqi/rules.py core/
mv junqi/state.py core/
mv junqi/encoder.py core/
mv junqi/zobrist.py core/
mv junqi/tt.py core/

# 第三步：更新 import 语句
# 只修改那些确实引用了这些文件的文件
# 使用 find + grep + sed 批量替换
find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.git/*" \
    -exec sed -i 's/from junqi\.rules/from core.rules/g' {} \;

# 第四步：测试是否正常工作
python -m pytest tests/test_rules.py -v

# 如果通过，继续移动下一个模块
# 如果不通过，回滚并重试
```

---

## 🔍 验证清单

### Phase 1 完成后必须检查：

- [ ] 运行 `ls -la` 确认根目录清爽
- [ ] 运行 `python -m pytest tests/ -v` 确认所有测试通过
- [ ] 运行 `python -m junqi test` 确认 CLI 工作正常
- [ ] 尝试启动 GUI: `python -m junqi gui` 确认不报错

### Phase 2 完成后必须检查：

- [ ] `models/checkpoints/` 包含最新训练快照
- [ ] `models/releases/` 包含发布版本
- [ ] `MANIFEST.jsonl` 格式正确且可读取
- [ ] 旧模型文件已归档或清理

### Phase 3 完成后必须检查：

- [ ] 所有单元测试通过
- [ ] 训练脚本可正常运行
- [ ] GUI 启动无导入错误
- [ ] 文档中的示例代码依然有效

---

## 🛡️ 回滚方案

### 如果 Phase 1 出问题：

```bash
# 简单撤销
git reset --hard HEAD
# 然后手动将文件移回原位
mv tests/utils/_verify_a2.py .
# ... 其他文件同理
```

### 如果 Phase 3 出问题：

```bash
# 彻底回滚
git checkout .
# 或者使用备份
cp -r scripts_backup/junqi_backup junqi_new
```

---

## 📝 实施记录模板

每次重构后填写：

```markdown
## 重构记录

日期：2026-XX-XX
执行人：XXX

### 已完成的操作
- [操作 1]
- [操作 2]

### 遇到的问题
- [问题描述 + 解决方案]

### 测试结果
- 单元测试：✅ Pass / ❌ Fail
- GUI 启动：✅ OK / ❌ Error
- 训练脚本：✅ OK / ❌ Error

### 下一步计划
- [待办事项]
```

---

## 💡 最佳实践建议

1. **小步快跑**: 每次只移动一小部分文件，充分测试后再继续
2. **频繁提交**: 每次成功的修改都 git commit，便于回滚
3. **并行分支**: 在主仓库外创建一个 worktree 用于试验
4. **文档同步**: 修改的同时更新 README 和依赖图
5. **团队协作**: 通知所有成员重构进度，避免冲突

---

**准备就绪！是否开始 Phase 1？我会帮你执行每一步并等待你的确认。**
