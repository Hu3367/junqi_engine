# 📊 训练与评估文档

本目录包含等级分系统、训练问题和修复方案，适合**RL 工程师**和**研究员**。

---

## 📋 文档清单

| 文件名 | 字数 | 内容 | 优先级 |
|--------|------|------|--------|
| **ELO_RATING_SYSTEM_DESIGN.md** | ~13k | 多维度等级分系统设计方案 | ⭐⭐⭐⭐⭐ |
| TRAINING_ROOT_CAUSE_REVIEW.md | ~12k | Value Head 坍塌原因分析 | ⭐⭐⭐⭐ |

---

## 💡 ELO 等级分系统设计要点

### 核心特性
1. **混合 ELO-Glicko-2 模型** - Rating + RD 不确定度建模
2. **阶段分离评分** - Opening/Midgame/Endgame 独立评分
3. **高和棋率应对** - 三种解决方案可选
4. **对抗平台架构** - Champion-Challenger SPRT 检验

### 实施选项
- **简易版**: ELO+Wilson 区间 (2-3 周)
- **标准版**: Glicko-2+Rematch (4-6 周)
- **专业版**: SPRT+Bayesian (8-10 周)

### 推荐路径
从简易版起步，验证效果后升级到标准版

---

## 🎯 Value Head 问题诊断

### 核心发现
- Win 预测全部为 0（类别塌缩）
- 训练和棋率 >90% 导致门控失效
- 样本量不足，无法证明稳定压制

### 整改措施
- Epoch 6: repetition=0，专项修复有效
- 候选门控：1 胜/30 和/1 负
- 仍需 Value MAE 和准确率改善

详见：[TRAINING_ROOT_CAUSE_REVIEW.md](./TRAINING_ROOT_CAUSE_REVIEW.md)

---

## 🔗 相关资源

- **执行计划**: [`docs/05-ExecutionPlans/`](../05-ExecutionPlans/)
- **规则知识**: [`docs/03-RulesAndStrategy/`](../03-RulesAndStrategy/)

---

**维护者**: Junqi Engine Team  
**最后更新**: 2026-09-05  
**版本**: 1.0
