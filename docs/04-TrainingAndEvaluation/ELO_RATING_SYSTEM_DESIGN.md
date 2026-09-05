# 军棋翻棋等级分（ELO）评估体系设计文档

## 1. 概述 (Overview)

### 1.1 背景与目标

军棋翻棋作为不完全信息博弈，具有高和棋率（当前 >90%）、状态空间复杂、战术深度大等特点。本设计旨在构建一套**多维度的综合评估体系**：

- 准确量化模型/玩家的真实棋力水平
- 支持多模型同时参赛的循环赛制排名  
- 追踪模型训练过程中的能力演化
- 识别不同阶段（开局/中盘/尾盘）的专项能力差异
- 提供统计显著性保证的晋级判定机制

### 1.2 参考标准

- **国际象棋 ELO/Glicko-2**: Rating + RD 不确定度建模
- **中国军旗职业联赛**: 段位制 + 升降级赛
- **DarkKnight 评估框架**: Star1 Expectiminimax + PIMC
- **现有项目实现**: Wilson 区间门控已验证有效

## 2. 核心数学模型 (Core Mathematical Model)

### 2.1 基础 ELO + Glicko-2 混合模型

**基本公式**:
```
R_new = R_old + K * (S - E[S])
```
其中 E[S] = 1 / (1 + 10^((R_opp - R_self) / 400))

**Glicko-2 扩展**：
- Rating Deviation (RD): 表示评级不确定性
- Volatility (φ): 表现波动率
- 初始值建议：R=1500, RD=350, φ=0.06
- RD < 100 视为"稳定评级"

**阶段分离评分**:
```
R_total = w_o*R_opening + w_m*R_midgame + w_e*R_endgame
权重建议：w_o=0.15, w_m=0.45, w_e=0.40
```

### 2.2 Wilson 置信区间下界（已验证）

当前项目已有 `wilson_lower_bound()` 实现：
```python
def wilson_lower_bound(successes, n, z=1.96):
    p = min(1.0, max(0.0, successes / n))
    denom = 1.0 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * sqrt(max(0, p*(1-p)/n + z*z/(4*n*n))) / denom
    return max(0, centre - half)
```

**军棋适配参数**:
- 快速门控：n=64, z=1.645（单侧 90% 置信）
- 正式晋升：n=200, z=1.96（双侧 95% 置信）

### 2.3 SPRT 顺序检验（新增）

用于实时判断是否显著优于基线：
```python
def sprt_test(wins, draws, losses, h0=0.50, h1=0.55):
    # h0: 不优于基线阈值
    # h1: 显著优于基线阈值
    log_ratio = wins*ln(h1(1-h0)/(h0(1-h1))) + 
                losses*ln((1-h1)h0/((1-h0)h1)) + 
                draws*ln(1)  # 和棋通常不计入
    
    ln_A = ln((1-beta)/alpha)   # 拒绝 H0 阈值
    ln_B = ln(beta/(1-alpha))   # 接受 H0 阈值
    
    if log_ratio >= ln_A: return 1  # 显著优于
    elif log_ratio <= ln_B: return -1  # 不优于
    else: return 0  # 继续观察
```

## 3. 高和棋率问题解决方案

### 3.1 挑战分析

军棋翻棋 >90% 和棋率导致：
- 传统 ELO 收敛缓慢
- Rating 差异敏感度降低  
- Wilson 下界易陷入死锁（需≥0.75 才能通过）

### 3.2 多级解决方案

**方案 A: 改良计分制（简易版）**
```python
def adaptive_score(win_rate, draw_rate, baseline_draw=0.85):
    adjusted_draws = min(draw_rate, baseline_draw) + max(0, draw_rate - baseline_draw) * 0.3
    return win_rate + adjusted_draws * 0.5
```

**方案 B: 重新比赛机制（推荐）**
- 首次判和后，交换颜色重赛一局
- 最多重赛 2 局（共 3 局定胜负）
- 仍和则记录为和棋，双方扣 2% Rating

**方案 C: 状态特征加权（专业版）**
根据终局状态复杂度赋予不同权重：
- 子力差大的局面更可信 (+20%)
- 快速和棋 (前 10 手) 受惩罚 (-30%)
- 长考和棋 (>100 手) 给予奖励 (+20%)

| 版本 | 策略 | 实施成本 | 预期效果 |
|------|------|----------|----------|
| 简易版 | 方案 A + 0.5 分 | 低 | 改善 20% |
| 标准版 | 方案 B + 自适应计分 | 中 | 减少重赛<15% |
| 专业版 | 方案 C + Glicko-2 | 高 | 提升 40% |

## 4. 对抗平台架构设计

### 4.1 Round-Robin 循环赛

```python
class RoundRobinTournament:
    def schedule(self):
        # 种子选手 vs 新模型优先
        sorted_models = sorted(models, key=lambda m: m.effective_rating, reverse=True)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((sorted_models[i], sorted_models[j], games//2))
                pairs.append((sorted_models[j], sorted_models[i], games - games//2))
        return pairs
    
    def run_parallel(self, workers=4):
        # 多进程并发执行比赛
        with Pool(workers) as pool:
            results = pool.map(play_match_job, jobs)
        return aggregate_results(results)
```

### 4.2 Champion-Challenger 模式

**规则**:
- 冠军固定，挑战者依次发起挑战
- SPRT 实时判断：需达到显著优于（p<0.05）才上位
- 旧冠军降级到 challenger pool

### 4.3 Model Pool Management

```python
class ModelPoolManager:
    def get_opponent_distribution(candidate_elo):
        if candidate_elo < 1450:
            return {"strong": 0.6, "medium": 0.3, "weak": 0.1}
        elif candidate_elo < 1550:
            return {"champion": 0.4, "strong": 0.3, "medium": 0.2, "random": 0.1}
        else:
            return {f"model_{i}": 1.0/len(pool) for i in range(len(pool))}
```

## 5. 军棋特色适配

### 5.1 不完全信息权重

```python
def adjust_for_incomplete_info(base_rating, game_state):
    revealed_ratio = calculate_revealed_ratio(game_state)
    
    # 暗子未揭示时降低权重
    if revealed_ratio < 0.3:
        base_rating *= 0.85
    
    # 快速翻子给予正向奖励
    flip_speed = calculate_flip_speed(game_state)
    if flip_speed > 0.8:
        base_rating *= 1.10
    
    return base_rating
```

### 5.2 状态特征纳入评分

```python
def incorporate_state_features(score, features):
    weight = 1.0
    
    # 子力差大→更可信
    if abs(features.material_diff) > 0.4:
        weight *= 1.2
    
    # 铁壁防守成功→和棋认可度提升
    if features.fortress_completeness > 0.85 and score == 0.5:
        score *= 1.1
    
    # 快速和棋惩罚
    if score == 0.5 and features.early_phase_draw:
        score *= 0.7
    
    return score * weight
```

## 6. 实现选项对比

### 简易版 (~200 行代码)
- 基础 ELO + Wilson 下界
- K=32 固定值
- 不分阶段总评
- 和棋计 0.5 分
- 纯 Python 实现

**准确度**: ★★★☆☆  
**部署难度**: 低  

### 标准版 (~800 行代码，推荐)
- Glicko-2 + 阶段分离
- 动态 K 因子 (16~48)
- Rematch 机制解决和棋
- Champion-Challenger 支持
- 多线程并行调度

**准确度**: ★★★★☆  
**部署难度**: 中  

### 专业版 (~2000 行 + 前端)
- Glicko-2 + SPRT 双门控
- 状态特征加权
- Bayesian 层级建模
- 自动化报告生成
- 分布式比赛调度

**准确度**: ★★★★★  
**部署难度**: 高

## 7. 与现有代码集成方案

### 7.1 models/目录结构扩展

```bash
models/
├── best.pt                          # 当前发布最佳模型
├── bc_best.pt                       # BC 预训练模型  
├── candidate_latest.pt              # 训练中的候选模型
├── elo_history.jsonl                # 已有：每轮 ELO 记录
├── pool/                            # 已有：历史模型快照池
│   ├── step_ep001.pt
│   └── ...
├── tournament/                      # 【新增】对抗赛事数据
│   ├── season1/
│   │   ├── schedule.json
│   │   ├── results.csv
│   │   └── analysis_report.md
│   └── season2/
├── rating/                          # 【新增】评级系统数据
│   ├── player_profile.json          # 模型档案
│   ├── rd_curves.jsonl              # RD 随时间变化
│   └── stage_performance.jsonl      # 阶段能力分解
└── validation/                      # 【新增】门控测试集
    ├── opening.jsonl
    ├── midgame.jsonl
    └── endgame.jsonl
```

### 7.2 junqi/train_rl.py 增强

在现有 `decide_promotion()` 基础上增加 SPRT 检查：

```python
def decide_promotion_enhanced(stage_stats, champion_model=None, use_sprt=False):
    # 原有 Wilson 条件检查（已存在）
    promotion, reason = decide_promotion(stage_stats, ...)
    if not promotion:
        return False, f"[Wilson] {reason}"
    
    # SPRT 额外检查（可选）
    if champion_model is not None and use_sprt:
        overall_stats = stage_stats['overall']
        sprt_state = sprt_test(
            overall_stats['wins'],
            overall_stats['draws'], 
            overall_stats['losses'],
            h0=0.50, h1=0.55
        )
        
        if sprt_state == -1:
            return False, "[SPRT] 未达到显著优于冠军标准"
    
    return True, "All checks passed"
```

### 7.3 junqi/selfplay.py 扩展

添加 TournamentMatch 类支持多模型循环赛：

```python
class TournamentMatch:
    @staticmethod
    def play(models, games_per_pair, base_seed, workers=4):
        pairs = []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                pairs.append((models[i], models[j], games_per_pair//2))
        
        jobs = [(spec_a, path_a, spec_b, path_b, games, seed) 
                for (i,j), (specs, paths), games in enumerate(pairs)]
        
        with Pool(workers) as pool:
            results = pool.map(play_match_job, jobs)
        
        return aggregate(results)
```

## 8. 前端展示界面原型

### Dashboard 布局

```
┌─────────────────────────────────────────┐
│  军棋 AI 评级系统 Dashboard                │
├─────────────────────────────────────────┤
│                                         │
│ [全局统计]                              │
│ 参赛模型：24 个 | 已完成对局：12,567     │
│                                         │
│ ┌────排行榜 Top10────┬────实时更新────┐ │
│ │ 排名│模型│Rating│RD││ ╔═对战进行时══╗ │ │
│ │ 1  │A1  │1687│28 ││ ║Alpha vs Beta ║ │ │
│ │ 2  │A2  │1654│32 ││ ║ Game 3/40    ║ │ │
│ │ ...                   ╚═════════════╝ │ │
│ └───────────────────────────────────────┘ │
│                                         │
│ ┌────Wilson 下界趋势───┬────雷达图────┐  │
│ │ ███████░░░░░      │   ★★★★★       │  │
│ │ (近 20 轮变化)       ★ Model A    ★  │  │
│ └───────────────────────────────────────┘  │
│                                         │
│ [操作区] [新建赛季] [导出报告] [设置]   │
└─────────────────────────────────────────┘
```

**单个模型详情页面包含**:
1. 基本信息卡片 (Rating/RD/K)
2. Winrate 趋势折线图 + Wilson 带
3. 阶段能力饼图 (Opening/Midgame/Endgame)
4. 对战记录表格 (可筛选对手)
5. SPRT 监控曲线 (Log Likelihood Ratio)

## 9. 落地建议与路线图

### Phase 1: 基础设施 (2-3 周)
- ✓ SimpleEloSystem 实现
- ✓ 集成到 _candidate_gate
- ✓ 扩展 elo_history.jsonl
- ✓ Wilson 单元测试

### Phase 2: 标准版功能 (4-6 周)
- ✓ Glicko-2 更新逻辑
- ✓ TournamentMatch 调度器
- ✓ Rematch 机制
- ✓ Dashboard API

### Phase 3: 专业版 (8-10 周)
- ✓ SPRT 实时监控器
- ✓ State Feature Extractor
- ✓ 前端可视化界面
- ✓ 自动化报告系统

## 10. 总结

本设计方案融合了经典理论 (ELO/Glicko-2)、前沿方法 (SPRT/Bayesian)、领域适配 (高和棋率优化) 以及工程实践 (多层级实现选项)。通过分阶段实施，可在 6 个月内建立起科学的军棋 AI 竞技评级生态体系。

**核心优势**:
✅ 科学的模型能力量化与横向对比  
✅ 透明的晋级判定机制 (可复现、可审计)  
✅ 细粒度的能力诊断与指导改进  
✅ 可持续演化的社区评级生态

---
**文档版本**: V1.0  
**最后更新**: 2026-09  
