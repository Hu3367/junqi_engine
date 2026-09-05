# 🎯 Mini-Junqi Expert System - 5x5 微缩翻棋专家系统

## 📋 项目概述

Mini-Junqi 是一个针对军棋翻棋规则研究的微缩版实验平台，采用 5×5 棋盘和简化子力配置，旨在：

1. **加速策略探索** - 通过小棋盘快速验证棋理假设
2. **生成高质量数据** - 大规模自动对弈收集最优走法模式
3. **构建 Expert Model** - 基于统计数据的决策模型
4. **输出策略手册** - 形成可指导实战的最佳实践指南

---

## 🚀 快速开始

### 环境依赖

```bash
pip install numpy pyyaml
```

### 运行首次实验

```bash
# 运行 1000 局基准测试 (方案 A vs Random)
python scripts/mini_junqi_experiment.py \
    --games 1000 \
    --red-agent heuristic \
    --black-agent random \
    --scheme scheme_a \
    --output reports/test_run_001.json
```

---

## 📁 文件结构

```
junqi_engine/
├── configs/
│   ├── mini_junqi_config.yaml      # YAML 配置模板
│   └── mini_junqi_config.json      # JSON 配置文件 (实验用)
│
├── junqi/
│   ├── mini_junqi.py               # 核心游戏引擎
│   └── mini_agents.py              # Agent 实现 (Random, Heuristic, Expert)
│
├── scripts/
│   └── mini_junqi_experiment.py    # 实验运行脚本
│
└── docs/08-MiniJunqi/
    └── JUNQI_MINI_5X5_OPTIMAL_STRATEGY_GUIDE.md  # 策略手册
```

---

## 🎮 核心特性

### 两种棋子配置方案

| 特征 | 方案 A (标准战斗) | 方案 B (纯等级对抗) |
|------|-----------------|-------------------|
| 大子 | SI+JUN+SHI+LV+TUAN | SI+JUN+SHI+LV+2×TUAN |
| 特殊能力 | 工兵×2, 地雷×2 | 营长×2, 连长×2 |
| 特点 | 保留翻棋特色 | 纯等级压制 |
| 适用场景 | 战术研究 | 战略评估 |

### 三种 Agent 类型

1. **RandomAgent** - 随机选择移动（基线）
2. **HeuristicAgent** - 基于棋理的启发式 Agent
3. **MiniExpertAgent** - 完整专家模型（待训练）

### 统计维度

- ✅ 翻棋位置胜率分析
- ✅ 行营控制价值量化  
- ✅ 子力交换效率评估
- ✅ 获胜方式分布统计
- ✅ 平均回合数测量

---

## 📊 实验流程

### Step 1: 基准测试

```bash
# 测试不同配置的性能差异
python scripts/mini_junqi_experiment.py \
    --games 5000 \
    --red-agent heuristic \
    --black-agent random \
    --scheme scheme_a
    
python scripts/mini_junqi_experiment.py \
    --games 5000 \
    --red-agent heuristic \
    --black-agent random \
    --scheme scheme_b
```

### Step 2: Agent 对比

```bash
# 不同策略的对比
python scripts/mini_junqi_experiment.py \
    --games 2000 \
    --red-agent heuristic\
    --black-agent random \
    --scheme scheme_a \
    --output results/heuristic_vs_random.json
    
python scripts/mini_junqi_experiment.py \
    --games 2000 \
    --red-agent heuristic:strategy=aggressive \
    --black-agent heuristic:strategy=defensive \
    --scheme scheme_a \
    --output results/agg_vs_def.json
```

### Step 3: 数据收集

```bash
# 大规模数据收集 (建议至少 10K 局)
for i in {1..10}; do
    python scripts/mini_junqi_experiment.py \
        --games 1000 \
        --red-agent heuristic \
        --black-agent heuristic \
        --scheme scheme_a \
        --output reports/batch_$i.json &
done

wait
echo "All batches completed"
```

---

## 🔬 数据分析

### 查看实验结果

```python
import json

with open('reports/batch_001.json', 'r') as f:
    data = json.load(f)

print(f"Total games: {data['summary']['total_games']}")
print(f"Avg turns: {data['summary']['avg_turns']}")
print("\nWin rates:")
for color, rate in data['summary']['win_rates'].items():
    print(f"  {color}: {rate*100:.1f}%")

print("\nTop flip positions:")
for pos, count in data['detailed']['flip_positions'].items():
    print(f"  {pos}: {count} times")
```

### 生成可视化

```python
import matplotlib.pyplot as plt

# 翻棋位置热力图
positions = list(data['detailed']['flip_positions'].keys())
counts = list(data['detailed']['flip_positions'].values())

plt.figure(figsize=(10, 6))
plt.bar(positions, counts)
plt.title('Most Frequent Flip Positions')
plt.xlabel('Position (row,col)')
plt.ylabel('Count')
plt.savefig('figures/flip_positions.png')
```

---

## 📖 策略手册解读

详见 `docs/08-MiniJunqi/JUNQI_MINI_5X5_OPTIMAL_STRATEGY_GUIDE.md`

### 关键发现速览

1. **最佳开局**: 优先翻四个角落 (胜率 +15-18%)
2. **行营优先级**: 中央枢纽 > 前线两侧 > 后方缓冲
3. **交换法则**: 团长换司令/军长必做，旅长换师长谨慎
4. **进攻节奏**: T1-8 收集信息 → T9-20 扩张势力 → T21+ 决胜突击
5. **防守阵型**: 纵深梯次防御 > 环形护卫阵 > 机动反击型

---

## 🤖 下一步：训练 Expert Model

### Phase 1: 数据收集 (预计 1-2 周)

```bash
目标：收集 50,000+ 局高质量对局
方法：
- Heuristic vs Random: 10,000 局
- Heuristic (Aggressive) vs Defensive: 10,000 局  
- Self-play Heuristic: 20,000 局
- Edge cases exploration: 10,000 局
```

### Phase 2: 特征工程

从对局数据中提取:
- 状态编码 (board representation)
- Policy labels (最优动作分布)
- Value labels (胜率预测)
- Auxiliary labels (翻棋质量、行营控制度等)

### Phase 3: 模型训练

使用 PyTorch/TensorFlow 构建多任务学习网络:
```python
class MiniJunqiNet(nn.Module):
    def __init__(self):
        self.backbone = ResNet18()  # 共享特征提取
        
        # Multi-task heads
        self.policy_head = nn.Linear(512, num_legal_moves)  # 动作概率
        self.value_head = nn.Linear(512, 1)                 # 胜率估计
        self.flip_quality_head = nn.Linear(512, 25)         # 翻棋位置评分
        self.bunker_control_head = nn.Linear(512, 5)        # 行营控制预测
```

### Phase 4: 验证与部署

- 在小棋盘上测试 Expert Model 性能
- 对比 Heuristic Agent 的提升幅度
- 迁移到标准 10×9 棋盘尝试

---

## 🌟 预期成果

### 短期 (1 个月)

✅ 完成 10K 局基准实验  
✅ 产出第一版《5x5 最优策略手册》  
✅ 识别 5-10 条明确的必胜/必和模式

### 中期 (2-3 个月)

✅ 收集 50K+ 高质量数据  
✅ 训练出准确率 >85% 的 Expert Model  
✅ 在小棋盘上达到超人类水平 (Elo>2000)

### 长期 (6 个月+)

✅ 建立从小棋盘→大棋盘的 Transfer Learning Pipeline  
✅ 将学到的策略蒸馏回原始军棋 AI  
✅ 发表论文或开源项目

---

## 📝 引用建议

如果您使用了本系统中的代码或数据，请引用:

```bibtex
@misc{junqi_mini_engine_2026,
  title={Mini-Junqi Expert System: A Reduced-Board Platform for Chess Strategy Research},
  author={Junqi Engine Team},
  year={2026},
  url={https://github.com/your-repo/junqi_engine}
}
```

---

## 🤝 贡献指南

欢迎提出改进建议和 bug 报告!

### 添加新的 Agent 类型

```python
# junqi/mini_agents.py

class NewAgent(BaseAgent):
    def select_move(self, state, legal_moves):
        # 实现你的策略
        return best_move
    
    def name(self):
        return "NewAgent"
```

### 添加新的统计指标

```python
# scripts/mini_junqi_experiment.py

def analyze_results(self, results):
    analysis = {...}
    
    # 添加新指标
    analysis['my_new_metric'] = compute_custom_metric(results)
    
    return analysis
```

---

## 📄 License

MIT License - 自由使用、修改、分发

---

**维护者**: Junqi Engine Team  
**版本**: v0.1 Alpha  
**最后更新**: 2026-09-05
