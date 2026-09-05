# 模块化架构设计指南（v2.0）

> **版本**: v2.0  
> **状态**: 已实施（向后兼容包装层）  
> **建议**: 新用户直接使用新结构，旧代码可通过包装层过渡

---

## 🎯 架构目标

将原本平铺在 `junqi/`的 31 个 Python 文件按功能重新组织为独立子包，提高：
- ✅ **可维护性** - 明确模块边界
- ✅ **可读性** - 快速定位代码
- ✅ **可扩展性** - 便于添加新功能

---

## 📁 新目录结构

```
junqi_engine/
├── core/                    # 基石模块（游戏核心逻辑）
│   ├── rules.py            # 棋盘几何、战斗规则
│   ├── state.py            # 游戏状态机
│   ├── encoder.py          # 状态编码
│   ├── zobrist.py          # Zobrist 哈希
│   └── tt.py               # 转位表
│
├── model/                   # 模型定义（神经网络）
│   └── net.py              # JunqiNet (Policy + Value heads)
│
├── search/                  # 搜索算法
│   ├── mcts.py             # MCTS 搜索
│   ├── search.py           # ExpertSearchEngine
│   └── hybrid_engine.py    # 混合引擎
│
├── train/                   # 训练流水线
│   ├── train_rl.py         # RL 自博弈训练
│   ├── train_bc.py         # 监督学习训练
│   ├── train_value_distill.py # 价值蒸馏
│   └── selfplay.py         # 批量自对弈
│
├── data/                    # 数据处理
│   ├── dataset.py          # 数据集生成
│   ├── replay.py           # 复盘解析
│   ├── endgame_gen.py      # 残局生成
│   └── fit_weights.py      # 权重拟合
│
├── eval/                    # 评估工具
│   ├── benchmark.py        # 50 题评测
│   ├── eval_bc.py          # BC 模型评估
│   └── eval_expert.py      # 专家评估
│
├── ui/                      # 用户界面
│   ├── gui.py              # GUI 界面
│   └── calculator.py       # 局面计算器
│
├── util/                    # 通用工具
│   ├── ai.py               # AI 代理封装
│   ├── belief.py           # 信念跟踪
│   ├── analysis.py         # 分析工具
│   ├── tune.py             # 调优工具
│   ├── config.py           # 配置加载
│   └── common.py           # 公共函数
│
└── junqi/                   # 向后兼容的包装层
    └── __init__.py         # 从各子包重新导出所有符号
```

---

## 💡 使用方法

### 推荐方式（新项目）

直接导入具体的模块：

```python
# 游戏核心
from core.rules import Rank, battle
from core.state import GameState, Piece, Action
from core.encoder import StateEncoder

# 神经网络
from model.net import JunqiNet

# 搜索
from search.mcts import MCTS
from search.search import ExpertSearchEngine
from search.hybrid_engine import HybridDecisionEngine

# 训练
from train.train_rl import RLTrainer
from train.selfplay import SelfPlayGenerator

# 数据
from data.replay import parse_sav
from data.dataset import export_replay_dataset

# 评估
from eval.benchmark import evaluate_benchmark

# 工具
from util.config import RuleConfig, EvalWeights
from util.belief import BeliefTracker
```

### 兼容性方式（旧项目）

继续使用旧的导入方式（通过包装层自动重定向）：

```python
# 这些仍然有效，内部会转发到新位置
from junqi import GameState, RuleConfig
from junqi.rules import Rank
from junqi.state import deal
from junqi.ai import Agent
```

---

## ⚠️ 重要注意事项

### 1. Import 路径更新策略

如果你在 `util/`或`search/`等子包中编写新代码：

✅ **正确**:
```python
# 跨子包导入使用完整路径
from core.rules import Rank
from model.net import JunqiNet
```

❌ **错误**（会导致循环导入）:
```python
# 不要在子包内使用 relative import 调用上级
from .config import xxx  # 这可能无效
```

### 2. 测试文件更新

所有测试文件的 import 需要手动更新：

```bash
# 示例：更新 test_rules.py
sed -i 's/from junqi\.rules/from core.rules/g' tests/test_rules.py
sed -i 's/from junqi\.state/from core.state/g' tests/test_rules.py
```

### 3. CLI 入口更新

统一的命令行入口：

```bash
# 推荐使用 cli.py
python cli.py test
python cli.py gui
python cli.py calc
python cli.py train --epochs 10

# 或者仍可用 old way（通过包装层）
python -m junqi test
```

---

## 🔧 重构检查清单

完成重构后需要验证：

- [ ] 所有单元测试通过 (`python -m pytest tests/ -v`)
- [ ] GUI 能正常启动 (`python cli.py gui`)
- [ ] CLI 命令正常工作 (`python cli.py calc`)
- [ ] 训练脚本无导入错误 (`python cli.py train --epochs 1`)
- [ ] Benchmark 运行成功 (`python cli.py benchmark`)

---

## 📝 迁移示例

### 旧代码 → 新代码对照表

| 旧方式 | 新方式 | 说明 |
|--------|--------|------|
| `from junqi.rules import Rank` | `from core.rules import Rank` | 规则 → core |
| `from junqi.state import GameState` | `from core.state import GameState` | 状态 → core |
| `from junqi.ai import Agent` | `from util.ai import Agent` | AI 工具 → util |
| `from junqi.config import RuleConfig` | `from util.config import RuleConfig` | 配置 → util |
| `from junqi.mcts import MCTS` | `from search.mcts import MCTS` | MCTS → search |
| `from junqi.net import JunqiNet` | `from model.net import JunqiNet` | 网络 → model |
| `from junqi.train_rl import RLTrainer` | `from train.train_rl import RLTrainer` | 训练 → train |
| `from junqi.gui import launch_gui` | `from ui.gui import launch_gui` | GUI → ui |

---

## 🚀 未来规划

### v2.1 计划（待定）

1. **彻底移除包装层** - 完全弃用 `from junqi.xxx`的导入
2. **添加类型注解** - 为所有模块增加完整的 type hints
3. **文档自动生成** - 基于 docstring 生成 API 文档
4. **性能优化** - 减少模块加载时间

### v3.0 愿景

- 支持插件化架构
- 提供多种后端选择（CPU/GPU/TPU）
- 集成更多评估指标
- 提供移动端适配版本

---

## 📞 技术支持

遇到问题？参考以下资源：

1. **架构总览**: `docs/PROJECT_REFACTORING_PROPOSAL.md`
2. **实施计划**: `docs/FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md`
3. **快速参考**: `docs/QUICK_REFERENCE_CARDS.md`
4. **常见问题**: 查看 GitHub Issues

---

**最后更新**: 2026-09-05  
**维护者**: Junqi Engine Team

*祝编码愉快！🎉*
