# Junqi Engine v2.0 快速参考卡

> **版本**: v2.0  
> **日期**: 2026-09-05  
> **用途**: 快速查找命令、文件位置、配置项

---

## 🚀 常用命令速查

### 激活虚拟环境（Windows）
```powershell
cd E:\Local code\军棋\junqi_engine
.\venv\Scripts\activate.ps1
```

### GUI 界面
```bash
python cli.py gui
# 或
python -m junqi gui
```

### 局面计算器
```bash
python cli.py calc
```

### 运行测试
```bash
.\run_tests.bat
# 或
python -m pytest tests/ -v
# 或单个测试
python -m pytest tests/test_rules.py -v
```

### 训练模型
```bash
python cli.py train --epochs 10 --games 24
```

### 基准测试
```bash
python cli.py benchmark --model models/pool/bc_best.pt
```

---

## 📁 关键文件位置

| 类型 | 路径 | 说明 |
|------|------|------|
| **配置文件** | `configs/rules.yaml` | APK 规则对齐参数 |
| | `configs/training.yaml` | 训练超参数 |
| | `configs/search.yaml` | 搜索参数 |
| | `configs/boards/*.yaml` | 棋盘规格 |
| **核心模块** | `junqi/rules.py` | 棋盘几何与规则 |
| | `junqi/state.py` | 游戏状态机 |
| | `junqi/net.py` | 神经网络定义 |
| | `junqi/mcts.py` | MCTS 搜索 |
| **文档库** | `docs/JUNQI_RULES_GUIDE.md` | 规则知识手册 |
| | `docs/FINAL_SUMMARY.md` | 重构总结 |
| | `docs/MODULAR_ARCHITECTURE_GUIDE.md` | 模块化指南 |
| **模型存储** | `models/MANIFEST.jsonl` | 模型注册表 |
| | `models/checkpoints/` | 训练快照 |
| | `models/releases/` | 发布版本 |
| | `models/pool/` | 模型池 |

---

## 🔧 配置参数速查

### 规则开关 (`configs/rules.yaml`)
- `flag_needs_mines_cleared`: 是否需先清雷才能扛旗 (true)
- `flag_gong_only`: 是否只有工兵能吃军旗 (true)
- `allow_suicide_attack`: 是否允许小子撞大子 (false)
- `hq_locks_pieces`: 大本营是否锁死棋子 (false)
- `no_capture_draw_plies`: 40 步无吃子判和
- `repetition_draw_count`: 3 次循环判和

### 训练参数 (`configs/training.yaml`)
- `epochs`: 总训练轮数 (默认 10)
- `games_per_epoch`: 每轮自对弈局数 (默认 24)
- `sims_per_game`: MCTS 模拟次数 (默认 25)
- `learning_rate`: 学习率 (默认 1e-3)
- `batch_size`: 批量大小 (默认 256)

### MCTS 参数 (`configs/search.yaml`)
- `c_puct`: PUCT 探索系数 (默认 0.6)
- `epsilon`: Dirichlet 噪声ε(默认 0.20)
- `repeat_penalty`: 重复局面罚分 (默认 150.0)

---

## 🎯 快速修复指南

### 问题 1: ModuleNotFoundError
```bash
# 错误：No module named 'core.rules'
# 解决：确保使用虚拟环境的 Python
.\venv\Scripts\activate.ps1
python cli.py gui
```

### 问题 2: 测试导入失败
```python
# 错误：from util.config import RuleConfig
# 正确：from junqi.config import RuleConfig
# 或直接：from .config import RuleConfig (在 module 内)
```

### 问题 3: GUI 打不开
1. 检查 PyTorch 已安装：`pip install torch`
2. 确认虚拟环境激活
3. 检查显卡驱动（如果使用 GPU）

---

## 📊 性能指标参考

| 任务 | 典型耗时 | 备注 |
|------|---------|------|
| 单元测试集 | ~2 秒 | 44 项全量测试 |
| 启动 GUI | ~3 秒 | 首次启动稍慢 |
| 50 题 Benchmark | ~1 分钟 | 取决于模型复杂度 |
| 单局自对弈 | ~5 秒 | 12×5标准棋盘 |
| 训练 1 Epoch | ~1.5 小时 | 24 局 × GPU |

---

## 📚 文档索引

| 文档 | 用途 | 位置 |
|------|------|------|
| 重构总览 | 了解本次改动 | `docs/FINAL_SUMMARY.md` |
| 实施计划 | 详细步骤 | `docs/IMPLEMENTATION_PLAN.md` |
| 规则手册 | 学习棋理 | `docs/RULES_GUIDE.md` |
| 模块化指南 | 未来架构 | `docs/MODULAR_GUIDE.md` |
| 快速参考 | 本文档 | `docs/QUICK_REFERENCE.md` |

---

## 🛡️ Git 操作建议

### 添加新文件
```bash
git add configs/ docs/
```

### 忽略的文件（已在.gitignore）
- `__pycache__/`
- `venv/`
- `*.pt` (模型文件)
- `reports/`
- `models/checkpoints/`
- `models/releases/`

### 提交建议
```bash
git commit -m "feat: Add new configuration files and improve documentation"
```

---

## 💡 Tips & Tricks

1. **热加载配置**: 修改 YAML 文件后无需重启程序即可生效
2. **种子复现**: 设置相同 `--seed` 可重现相同的开局布局
3. **并行训练**: 使用`--workers 8`加速自对弈生成
4. **调试模式**: 设置`DEBUG=true`查看详细日志
5. **备份策略**: 重大修改前手动备份 `models/`目录

---

**保存此文件供快速查阅！** 📌

*最后更新*: 2026-09-05  
*维护者*: Junqi Engine Team
