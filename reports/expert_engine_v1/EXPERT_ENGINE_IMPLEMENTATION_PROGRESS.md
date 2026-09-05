# Expert Engine V0 实现进度

## 当前状态

### ✅ 已完成

1. **专家模型设计文档** (docs/03-RulesAndStrategy/专家模型.md)
   - 定义了四层级架构
   - 明确了 Teacher → Student → Self-play 的完整流程
   - 建立了 Curriculum Dataset 的 12 个难度等级

2. **棋理体系 v2.0** (docs/03-RulesAndStrategy/JUNQI_CHESS_INTELLECT_V2.md)
   - 7 章完整内容 (~50,000 字)
   - 包含信息推理、空间控制、威胁体系、Tempo 经济学等核心概念

3. **Expert Engine 目录结构**
   ```
   junqi/expert/
   ├── __init__.py          ✅ 已创建
   ├── rule_validator.py    ✅ 已创建 (Expert-0)
   ├── tactical_analyzer.py ⏳ 待创建 (Expert-1)
   ├── principle_evaluator.py ⏳ 待创建 (Expert-2)
   ├── search_optimizer.py  ⏳ 待创建 (Expert-3)
   └── expert_engine.py     ⏳ 待创建 (统一接口)
   ```

### 🚧 进行中

- Expert-0: Rule Validator (规则验证器) - **80% 完成**
  - [x] 基本移动合法性验证
  - [x] 战斗规则实现
  - [x] 铁路路径检查
  - [ ] 循环局面检测（简化版）
  - [ ] 单元测试

### 📋 待完成

#### Expert-1: Tactical Analyzer (战术分析器)

需要实现：
- 立即吃子识别
- 强制吃子判断
- 连续吃链条发现
- 反吃风险检测
- 必杀/将军检测
- 战术 Puzzle 生成器

#### Expert-2: Principle Evaluator (棋理评估器)

根据棋理体系 v2.0 实现：
- Information: 不完全信息推理
  - HiddenPieceBelief (贝叶斯信念建模)
  - Entropy calculation (信息熵计算)
  - Reveal value quantification (信息价值量化)
  
- Space: 空间与势力控制
  - Mobility-1/2/3 calculation (多层 mobility)
  - Strategic node identification (战略节点识别)
  - Space advantage scoring (空间优势评分)
  
- Tempo: Tempo 经济学
  - Tempo gain/loss tracking (得失跟踪)
  - Phase-based strategy (分阶段策略)
  - Information-tempo tradeoff (信息-tempo 权衡)
  
- Threat: 威胁体系
  - 7-tier threat classification (七类威胁分类)
  - Threat response priority matrix (响应优先级矩阵)
  - Confidence scoring (置信度计算)
  
- Material: 条件子力价值
  - Position quality evaluation (位置质量评估)
  - Offensive pressure assessment (威胁贡献度)
  - Defensive necessity analysis (防守必要性)
  - Mobility bonus calculation (机动潜力)
  - Special ability activation (特殊能力激活)

#### Expert-3: Search Optimizer (搜索优化器)

- MCTS 集成 (带 ISMCTS/PIMC 支持不完全信息)
- Alpha-Beta 剪枝
- Beam Search
- Candidate move generation from chess principles
- Variance reduction techniques

#### Expert Engine: Unified Interface (统一接口)

- 整合四个层级的 API
- 决策生成引擎
- Policy + Value + Labels 输出
- State encoding/decoding

#### Dataset Generator (数据集生成器)

- Curriculum-based puzzle generator (基于课程的 Puzzle 生成器)
- Level 1-12 难度递增的训练题
- Label extraction utility (标签提取工具)

---

## 下一步行动计划

### 优先级 1: 完成 Expert-0 (1-2 小时)
1. 补充单元测试
2. 修复可能的边界条件问题
3. 验证 44/44 测试通过

### 优先级 2: 实现 Expert-1 (2-3 小时)
1. 立即吃子检测算法
2. 连环攻击识别
3. 战术风险评分
4. 简单的战术 Puzzle 生成

### 优先级 3: 实现 Expert-2 的核心模块 (4-6 小时)
1. 从第 1-2 章开始：HiddenPieceBelief 和 Mobility calculator
2. 然后是威胁检测系统
3. Tempo 计算器
4. 条件价值评估器

### 优先级 4: 构建统一接口 (2-3 小时)
1. ExpertEngine 主类
2. 决策流程编排
3. 输出格式定义

### 优先级 5: 创建最小可运行示例 (1-2 小时)
1. 单局面评估脚本
2. 自动生成 10 个训练样本
3. 验证 Pipeline 通顺

---

## 技术要点备忘

### 输入输出格式

```python
# Input
game_state: GameState
color: int (己方颜色)

# Output
{
    "policy": [...],      # 动作概率分布 (归一化)
    "value": float,       # 局面评估值 [-1, 1]
    "tactical_labels": {  # 战术标签
        "immediate_capture": bool,
        "forced_move": bool,
        "attack_opportunity": bool,
        ...
    },
    "strategic_labels": { # 战略标签
        "information_advantage": float,
        "space_advantage": float,
        "tempo_balance": float,
        "flag_pressure": float,
        "recommended_plan": str,
        ...
    }
}
```

### 关键点

1. **Expert-0 必须 100% 正确** - 这是所有上层功能的基础
2. **Expert-2 是核心** - 真正执行棋理体系的价值所在
3. **不要过早引入神经网络** - 先让纯规则 + 搜索系统跑通
4. **Curriculum Learning 很重要** - 从简单到复杂逐步训练
5. **标签质量 > 数据数量** - 一个高质量标注样本胜过 100 个噪声数据

---

*文档更新时间：2026-09-05*
*Next Steps: Complete Expert-1 and begin Expert-2 core modules*
