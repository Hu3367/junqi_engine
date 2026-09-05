# 项目重构行动清单 Checklist

> **目标**: 解决 5 大核心算法与工程问题  
> **时间框架**: 2026-09-05 ~ 2026-10-31  
> **优先级**: 🔴 P0 > 🟠 P1 > 🟢 P2

---

## 🔴 本周必须启动 (Week 1)

### P0-1: 规则知识文档人工审核 ⏳ 进行中

- [x] 自动生成《军棋翻棋玩法规则与策略完整汇总文档》
- [ ] **找 2-3 位军旗高手审核准确性** (`docs/JUNQI_RULES_AND_STRATEGY_GUIDE.md`)
- [ ] 补充复盘数据统计（开局定式频率、各阶段平均步数等）
- [ ] 标注"需人工验证"部分的知识盲区

📍 **交付物**: 审核签字版棋理手册 + 待办事项列表

---

### P0-2: ELO 简易版实现 💡 新任务

设计目标：**快速上线**验证效果，代码改动 < 200 行

#### 实施步骤:

- [ ] Step 1: 修改 `junqi/config.py` 添加 ELO 参数
```python
ELO_CONFIG = {
    'K_FACTOR': 32,           # 更新率
    'BASE_ELO': 1500,         # 起始分
    'WIN_SCORE': 1.0,
    'DRAW_SCORE': 0.3,        # 简化版：和棋只给 0.3 而非 0.5
    'LOSS_SCORE': 0.0,
}
```

- [ ] Step 2: 重写 `train_rl.py::decide_promotion()`
```python
def decide_promotion(self, wins, draws, losses):
    """使用 ELO 评分而非简单胜率"""
    total = wins + draws + losses
    if total < 20:  # 最小样本量要求
        return False, "样本不足"
    
    expected_win_rate = 0.5  # 假设对手也是 1500 分
    actual_score = (wins * ELO_CONFIG['WIN_SCORE'] + 
                   draws * ELO_CONFIG['DRAW_SCORE']) / total
    
    # Wilson 区间下界计算
    z = 1.96  # 95% 置信度
    std = math.sqrt(actual_score * (1 - actual_score) / total)
    wilson_lower = (actual_score + z**2/(2*total) - 
                   z*math.sqrt(std**2 + z**2/(4*total))) / (1 + z**2/total)
    
    if wilson_lower > 0.65 and total >= 40:
        return True, f"Wilson 下界{wilson_lower:.3f}>0.65"
    return False, f"Wilson 下界{wilson_lower:.3f}<0.65"
```

- [ ] Step 3: 扩展 `models/elo_history.jsonl` 格式
```json
{"model": "cand_892", "opponent": "search2", "result": "win", "elo_before": 1489, "elo_after": 1512, "date": "2026-09-05"}
```

- [ ] Step 4: 编写单元测试 `tests/test_elo_simple.py`
- [ ] Step 5: 在测试环境试运行 1 轮（40 局 vs search2）

✅ **完成标准**: ELO 评分能稳定输出，不再恒为 1500

⏰ **预计耗时**: 2-3 天  
👤 **负责人**: Lead Engineer

---

## 🟠 下周启动 (Week 2-3)

### P1-1: MCTS 可视化 MVP 🎨

设计目标：**CLI 工具**输出单局面分析报告

#### 实施步骤:

- [ ] Step 1: 新增 `junqi/explain.py` 模块
```python
def explain_move(model, state, top_k=5):
    """返回前 k 个候选动作的 MCTS 统计"""
    actions, visit_counts, values = model.mcts_search(state, sims=120)
    
    results = []
    for i, (action, visits, value) in enumerate(zip(actions, visit_counts, values)):
        results.append({
            'rank': i+1,
            'action': action,
            'visit_count': visits,
            'visit_prob': visits / sum(visit_counts),
            'value_estimate': value
        })
    
    return sorted(results, key=lambda x: x['visit_count'], reverse=True)[:top_k]
```

- [ ] Step 2: CLI 命令集成 (`junqi/__main__.py`)
```bash
python -m junqi explain --model models/candidate_latest.pt \
    --fen rbbbqglle/rbbbbbbbr/... --output reports/debug_turn.md
```

- [ ] Step 3: 生成 Markdown 报告模板（见主方案附录）
- [ ] Step 4: 人工验证报告合理性（对比 Human Expert 判断）

✅ **完成标准**: 能生成可读性良好的分析日志，帮助调试

⏰ **预计耗时**: 2-3 天  
👤 **负责人**: RL Researcher

---

### P1-2: 模型 MANIFEST 规范定义 📋

设计目标：**结构化记录**所有训练模型信息

#### 实施步骤:

- [ ] Step 1: 定义 JSONL Schema (`models/MANIFEST_SCHEMA.json`)
```json
{
  "type": "object",
  "properties": {
    "id": {"type": "string"},
    "filename": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},
    "training_config": {...},
    "metrics": {...},
    "human_notes": {"type": "object"}
  },
  "required": ["id", "filename", "metrics"]
}
```

- [ ] Step 2: 迁移现有模型到规范格式
```bash
python scripts/migrate_models_to_manifest.py
```

- [ ] Step 3: 修改 `train_rl.py` 自动写入新训练模型
- [ ] Step 4: 编写查询工具 `scripts/list_models.py`

✅ **完成标准**: 能够列出所有模型的元数据 + 性能摘要

⏰ **预计耗时**: 1-2 天  
👤 **负责人**: Lead Engineer

---

## 🟢 次周启动 (Week 4+)

### P2-1: 小棋盘基础设施搭建 🔧

设计目标：**抽象棋盘尺寸**,支持 6×6/4×8/12×5切换

#### 实施步骤:

- [ ] Step 1: `junqi/rules.py` 重构 BoardConfig 类
```python
@dataclass
class BoardConfig:
    ROWS: int
    COLS: int
    CAMPS: Set[Tuple[int, int]]
    HQS: Set[Tuple[int, int]]
    RAIL_POSITIONS: Set[Tuple[int, int]]
    
    @classmethod
    def standard(cls): ...   # 12×5
    @classmethod
    def compact(cls): ...    # 6×6
    @classmethod
    def narrow(cls): ...     # 4×8
```

- [ ] Step 2: 棋子配比缩容器 `junqi/scale_composition.py`
- [ ] Step 3: 不同棋盘的 Action Space 适配 (3650→1332→320)
- [ ] Step 4: 单元测试覆盖 (所有 test_* 通过)

✅ **完成标准**: `--board 6x6` 参数可用，训练不报错

⏰ **预计耗时**: 3-4 天  
👤 **负责人**: Lead Engineer

---

### P2-2: 6×6 预训练实验 🚀

设计目标:**快速反馈**验证课程学习有效性

#### 实施步骤:

- [ ] Step 1: 配置 6×6棋盘参数文件 `configs/board_6x6.yaml`
- [ ] Step 2: 运行首轮训练 (Epoch 0-10)
```bash
python -m junqi train_rl --board 6x6 --epochs 10 \
    --games-per-epoch 200 --workers 8
```

- [ ] Step 3: 评估指标收集
  - [ ] vs Random Baseline 胜率
  - [ ] vs Greedy (search d2) 胜率  
  - [ ] Value Head MAE
  - [ ] MCTS 收敛速度

- [ ] Step 4: 分析日志，定位问题

✅ **完成标准**: 确认 6×6训练有效（vs random > 85%）

⏰ **预计耗时**: 2-3 天（含等待训练完成）  
👤 **负责人**: RL Researcher

---

## 📊 关键里程碑检查点

### 🎯 Milestone 1: September 月底
- [ ] 规则知识手册审核完成 ✅
- [ ] ELO 简易版上线 ✅  
- [ ] MCTS 可视化 MVP ✅
- [ ] 模型 MANIFEST 规范 ✅

### 🎯 Milestone 2: October 中
- [ ] ELO 标准版 (Glicko-2) ✅
- [ ] 模型管理平台 Web Beta ✅
- [ ] 小棋盘 6×6有效训练 ✅
- [ ] 课程学习链路 (6×6→8×6) ✅

### 🎯 Milestone 3: October 底
- [ ] 完整棋盘课程学习贯通 ✅
- [ ] 训练效率提升≥50% ✅
- [ ] Human Expert 对战胜率>60% ✅
- [ ] CHANGELOG v2.0 发布 ✅

---

## 🛠️ 每日 Standup 模板

```
Date: 2026-09-XX
Task ID: P0-2 / P1-1 / ...

Yesterday:
- Completed: [列出已完成的任务]
- Blocked: [是否有阻塞点]

Today:
- Planned: [今天计划做什么]
- Risks: [潜在风险]

Help Needed:
[需要团队协助的事项]
```

---

## 📞 紧急联系人

| 角色 | 姓名 | 联系方式 | 负责领域 |
|------|------|---------|---------|
| Project Owner | ? | ? | 总体决策 |
| Lead Engineer | ? | ? | 架构重构 |
| RL Researcher | ? | ? | 训练策略 |
| QA Reviewer | ? | ? | 规则审核 |

---

## 📚 参考文档索引

1. [PROJECT_REFACTORING_PROPOSAL.md](./PROJECT_REFACTORING_PROPOSAL.md) - 完整重构方案
2. [JUNQI_RULES_AND_STRATEGY_GUIDE.md](./JUNQI_RULES_AND_STRATEGY_GUIDE.md) - 规则知识手册
3. [ELO_RATING_SYSTEM_DESIGN.md](./ELO_RATING_SYSTEM_DESIGN.md) - 等级分系统设计
4. [AI_TRAINING_AND_HUMAN_PLAY_PLAN.md](../AI_TRAINING_AND_HUMAN_PLAY_PLAN.md) - P0-P4执行基线
5. [AGENTS.md](../AGENTS.md) - 大模型代理规则

---

**最后更新**: 2026-09-05  
**版本**: v1.0  
**状态**: 待审核 → 待批准 → 待实施
