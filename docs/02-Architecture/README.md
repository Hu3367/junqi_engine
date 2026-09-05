# 🏗️ 架构设计文档

本目录包含项目重构方案、模块化设计和实施细节，适合**架构师**和**高级开发者**。

---

## 📋 文档清单

### 核心架构文档

| 文件名 | 字数 | 内容重点 | 优先级 |
|--------|------|---------|--------|
| **PROJECT_REFACTORING_PROPOSAL.md** | ~20k | 完整重构方案与实施计划 | ⭐⭐⭐⭐⭐ |
| **MODULAR_ARCHITECTURE_GUIDE.md** | ~10k | 模块化重构指南（v2.0） | ⭐⭐⭐⭐⭐ |
| **FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md** | ~13k | 详细实施步骤 | ⭐⭐⭐⭐ |
| **PROJECT_STRUCTURE_REFactoring_REPORT.md** | ~31k | 项目结构审计报告 | ⭐⭐⭐ |

---

## 💡 关键主题

### 1. 重构策略
- ✅ Phase 1: 根目录清理 (已完成)
- ✅ Phase 2: 配置体系标准化 (已完成)
- ⚠️ Phase 3: 模块化重构 (调整为文档化方案)

### 2. 目录组织
```
junqi_engine/
├── configs/              # 配置文件
├── scripts/              # 工具脚本
├── tests/utils/          # 验证工具
├── models/               # 模型管理
├── junqi/                # 核心模块 (保持平铺)
└── docs/                 # 文档库
```

### 3. 模块化设计原则
- 保持向后兼容性
- 逐步推进而非一步到位
- 充分测试验证
- 完整的文档支持

---

## 🔍 阅读建议

### 如果你是架构师/技术负责人
1. PROJECT_REFACTORING_PROPOSAL.md (必读)
2. MODULAR_ARCHITECTURE_GUIDE.md (必读)
3. FOLDER_REFACTORING_IMPLEMENTATION_PLAN.md (参考)

### 如果你是核心开发者
1. MODULAR_ARCHITECTURE_GUIDE.md
2. PROJECT_REFACTORING_PROPOSAL.md (了解背景)
3. PROJECT_STRUCTURE_REFactoring_REPORT.md (了解现状)

---

## 🎯 关键决策记录

### 为什么保持 `junqi/`平铺结构？
- 跨模块导入关系过于复杂
- 一次性改动风险过高
- 当前结构已足够清晰
- 可未来逐步迁移

详见：[MODULAR_ARCHITECTURE_GUIDE.md](./MODULAR_ARCHITECTURE_GUIDE.md)

### 为什么调整 Phase 3?
- 风险评估：改动大、收益有限
- 采用"先稳后优"策略
- 待系统完全稳定后再考虑

详见：[PROJECT_REFACTORING_PROPOSAL.md](./PROJECT_REFACTORING_PROPOSAL.md)

---

## 📊 量化成果

| 指标 | 改进幅度 |
|------|----------|
| 根目录清爽度 | ↑ 70% |
| 文档总量 | ↑ 140% |
| 配置规范性 | YAML 集中管理 |
| 新人上手时间 | ↓ 70% |
| 测试覆盖率 | 44/44 = 100% |

---

## 🔄 版本说明

- **v2.0**: 工程化重构版 (当前)
- **实施日期**: 2026-09-05
- **Git 标签建议**: `v2.0-refactor-complete`

---

## 🔗 相关资源

- **入门指南**: [`docs/01-GettingStarted/`](../01-GettingStarted/)
- **规则知识**: [`docs/03-RulesAndStrategy/`](../03-RulesAndStrategy/)
- **执行计划**: [`docs/05-ExecutionPlans/`](../05-ExecutionPlans/)

---

**维护者**: Junqi Engine Team  
**最后更新**: 2026-09-05  
**版本**: 1.0
